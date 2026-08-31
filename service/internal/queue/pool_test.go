package queue

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lieri123/rosetta/service/internal/events"
	"github.com/lieri123/rosetta/service/internal/store"
)

func newStore(t *testing.T) *store.Store {
	t.Helper()
	database, err := store.Open(filepath.Join(t.TempDir(), "queue.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	return database
}

func enqueue(t *testing.T, database *store.Store, count int) int64 {
	t.Helper()
	ctx := context.Background()
	document, err := database.CreateDocument(ctx, "Notes", "notes.pdf")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < count; i++ {
		page, err := database.CreatePage(ctx, document.ID, i, "/tmp/page.png")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := database.EnqueueJob(ctx, document.ID, page.ID, store.KindPage); err != nil {
			t.Fatal(err)
		}
	}
	if err := database.SetDocumentStatus(ctx, document.ID, store.StatusRunning, ""); err != nil {
		t.Fatal(err)
	}
	return document.ID
}

func waitFor(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("condition not met within the deadline")
}

func TestPoolProcessesEveryJobExactlyOnce(t *testing.T) {
	database := newStore(t)
	const jobs = 40
	documentID := enqueue(t, database, jobs)

	var mu sync.Mutex
	seen := map[int64]int{}

	pool := &Pool{
		Store:    database,
		Workers:  6,
		IdlePoll: 10 * time.Millisecond,
		Handler: func(ctx context.Context, job store.Job) error {
			mu.Lock()
			seen[job.ID]++
			mu.Unlock()
			return nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pool.Start(ctx)

	waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(seen) == jobs
	})

	mu.Lock()
	defer mu.Unlock()
	for id, count := range seen {
		if count != 1 {
			t.Errorf("job %d ran %d times", id, count)
		}
	}

	waitFor(t, func() bool {
		document, err := database.GetDocument(ctx, documentID)
		return err == nil && document.Status == store.StatusDone
	})
}

func TestPoolRunsJobsConcurrently(t *testing.T) {
	// The reason the pool exists: pages spend nearly all their time waiting on
	// a subprocess and a remote API, so four slow pages should take about as
	// long as one.
	database := newStore(t)
	enqueue(t, database, 4)

	var inFlight, peak int64
	pool := &Pool{
		Store:    database,
		Workers:  4,
		IdlePoll: 10 * time.Millisecond,
		Handler: func(ctx context.Context, job store.Job) error {
			current := atomic.AddInt64(&inFlight, 1)
			for {
				observed := atomic.LoadInt64(&peak)
				if current <= observed || atomic.CompareAndSwapInt64(&peak, observed, current) {
					break
				}
			}
			time.Sleep(80 * time.Millisecond)
			atomic.AddInt64(&inFlight, -1)
			return nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	start := time.Now()
	pool.Start(ctx)
	waitFor(t, func() bool {
		pending, err := database.PendingJobs(ctx, 1)
		return err == nil && pending == 0
	})
	elapsed := time.Since(start)

	if atomic.LoadInt64(&peak) < 2 {
		t.Errorf("jobs never overlapped: peak concurrency %d", peak)
	}
	if elapsed > 300*time.Millisecond {
		t.Errorf("four 80ms jobs across four workers took %s", elapsed)
	}
}

func TestFailedJobMarksTheDocumentFailed(t *testing.T) {
	database := newStore(t)
	documentID := enqueue(t, database, 2)

	var count int64
	pool := &Pool{
		Store:    database,
		Workers:  1,
		IdlePoll: 10 * time.Millisecond,
		Handler: func(ctx context.Context, job store.Job) error {
			if atomic.AddInt64(&count, 1) == 1 {
				return errors.New("recognition exploded")
			}
			return nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pool.Start(ctx)

	waitFor(t, func() bool {
		document, err := database.GetDocument(ctx, documentID)
		return err == nil && document.Status == store.StatusFailed
	})

	// One page failing must not stop the other from being processed: a
	// fifty-page notebook should not lose forty-nine pages to one bad photo.
	if atomic.LoadInt64(&count) != 2 {
		t.Errorf("want both jobs attempted, got %d", count)
	}
}

func TestPoolPublishesCompletion(t *testing.T) {
	database := newStore(t)
	broker := events.NewBroker()
	documentID := enqueue(t, database, 1)

	stream, cancelSubscription := broker.Subscribe(documentID)
	defer cancelSubscription()

	pool := &Pool{
		Store: database, Broker: broker, Workers: 1, IdlePoll: 10 * time.Millisecond,
		Handler: func(ctx context.Context, job store.Job) error { return nil },
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pool.Start(ctx)

	select {
	case event := <-stream:
		if event.Type != "done" {
			t.Errorf("want a done event, got %q", event.Type)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no completion event")
	}
}

func TestPoolStopsWithItsContext(t *testing.T) {
	database := newStore(t)
	enqueue(t, database, 1)

	pool := &Pool{
		Store: database, Workers: 2, IdlePoll: 10 * time.Millisecond,
		Handler: func(ctx context.Context, job store.Job) error { return nil },
	}

	ctx, cancel := context.WithCancel(context.Background())
	pool.Start(ctx)
	cancel()

	done := make(chan struct{})
	go func() {
		pool.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("workers did not stop when the context was cancelled")
	}
}

func TestPoolRequeuesInterruptedJobsOnStart(t *testing.T) {
	database := newStore(t)
	enqueue(t, database, 1)
	ctx := context.Background()

	// Simulate a crash: the job is claimed and the process dies.
	if _, err := database.ClaimJob(ctx); err != nil {
		t.Fatal(err)
	}

	var ran int64
	pool := &Pool{
		Store: database, Workers: 1, IdlePoll: 10 * time.Millisecond,
		Handler: func(ctx context.Context, job store.Job) error {
			atomic.AddInt64(&ran, 1)
			return nil
		},
	}
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	pool.Start(runCtx)

	waitFor(t, func() bool { return atomic.LoadInt64(&ran) == 1 })
}
