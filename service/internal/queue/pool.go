// Package queue is the worker pool.
//
// Pages are independent, and every page spends nearly all its time waiting --
// on a subprocess, then on a remote API. Running them concurrently is what
// turns a fifty page notebook from a coffee break into a few seconds, and it
// is the one place in this system where Go's concurrency is doing real work
// rather than being a checkbox.
//
// Jobs come from the database rather than from an in-memory channel, so the
// queue survives a restart and no page is lost because the process died
// between the upload and the recognition.
package queue

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/lieri123/rosetta/service/internal/events"
	"github.com/lieri123/rosetta/service/internal/store"
)

type Handler func(ctx context.Context, job store.Job) error

type Pool struct {
	Store   *store.Store
	Handler Handler
	Broker  *events.Broker
	Workers int
	Logger  *log.Logger
	// IdlePoll is the safety net: workers are woken by Notify, and this only
	// matters if a wake-up is ever missed or a job is written by something
	// other than this process.
	IdlePoll time.Duration

	wake chan struct{}
	once sync.Once
	wg   sync.WaitGroup
}

func (p *Pool) init() {
	p.once.Do(func() {
		// Capacity one: many enqueues while workers are busy should collapse
		// into a single wake-up, not queue up behind each other.
		p.wake = make(chan struct{}, 1)
	})
}

// Notify tells the pool there may be new work.
func (p *Pool) Notify() {
	p.init()
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

func (p *Pool) Start(ctx context.Context) {
	p.init()

	if requeued, err := p.Store.RequeueRunning(ctx); err != nil {
		p.logf("could not requeue in-flight jobs: %v", err)
	} else if requeued > 0 {
		p.logf("requeued %d job(s) left running by a previous process", requeued)
	}

	workers := p.Workers
	if workers < 1 {
		workers = 1
	}
	idle := p.IdlePoll
	if idle <= 0 {
		idle = 2 * time.Second
	}

	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go p.run(ctx, i, idle)
	}
	p.Notify() // pick up anything already queued
}

// Wait blocks until every worker has stopped. Call after cancelling the
// context passed to Start.
func (p *Pool) Wait() { p.wg.Wait() }

func (p *Pool) run(ctx context.Context, id int, idle time.Duration) {
	defer p.wg.Done()

	timer := time.NewTimer(idle)
	defer timer.Stop()

	for {
		// Drain the queue before going back to sleep, so a burst of uploads is
		// worked through rather than one job per wake-up.
		for {
			if ctx.Err() != nil {
				return
			}
			job, err := p.Store.ClaimJob(ctx)
			if err != nil {
				p.logf("worker %d: claiming a job: %v", id, err)
				break
			}
			if job == nil {
				break
			}
			p.execute(ctx, id, *job)
		}

		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(idle)

		select {
		case <-ctx.Done():
			return
		case <-p.wake:
			// Re-arm the other workers: one wake-up signal is consumed by one
			// worker, and a batch of uploads should not be worked by one
			// worker while the others sleep.
			p.Notify()
		case <-timer.C:
		}
	}
}

func (p *Pool) execute(ctx context.Context, id int, job store.Job) {
	err := p.Handler(ctx, job)

	state := store.JobDone
	errText := ""
	if err != nil {
		state = store.JobFailed
		errText = err.Error()
		p.logf("worker %d: job %d failed: %v", id, job.ID, err)
	}

	if finishErr := p.Store.FinishJob(ctx, job.ID, state, errText); finishErr != nil {
		p.logf("worker %d: recording job %d: %v", id, job.ID, finishErr)
	}

	p.finishDocument(ctx, job.DocumentID)
}

// finishDocument flips a document to done or failed once nothing is left.
//
// Derived from the jobs table rather than from a counter incremented as
// workers finish: with several workers on one document, a counter is a race,
// and the table already knows the answer.
func (p *Pool) finishDocument(ctx context.Context, documentID int64) {
	pending, err := p.Store.PendingJobs(ctx, documentID)
	if err != nil {
		p.logf("checking pending jobs for document %d: %v", documentID, err)
		return
	}
	if pending > 0 {
		return
	}

	failed, err := p.Store.CountJobs(ctx, documentID, store.JobFailed)
	if err != nil {
		p.logf("counting failed jobs for document %d: %v", documentID, err)
		return
	}

	status := store.StatusDone
	message := "complete"
	if failed > 0 {
		status = store.StatusFailed
		message = "finished with failures"
	}

	if err := p.Store.SetDocumentStatus(ctx, documentID, status, ""); err != nil {
		p.logf("updating document %d: %v", documentID, err)
	}
	if p.Broker != nil {
		p.Broker.Publish(events.Event{
			Type: "done", DocumentID: documentID, Message: message,
		})
	}
}

func (p *Pool) logf(format string, args ...any) {
	if p.Logger != nil {
		p.Logger.Printf(format, args...)
	}
}
