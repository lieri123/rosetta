package store

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	database, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	return database
}

func TestDocumentLifecycle(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	document, err := s.CreateDocument(ctx, "Notes", "notes.png")
	if err != nil {
		t.Fatal(err)
	}
	if document.Status != StatusPending {
		t.Errorf("want a new document to be pending, got %q", document.Status)
	}

	if err := s.SetDocumentStatus(ctx, document.ID, StatusRunning, ""); err != nil {
		t.Fatal(err)
	}
	if err := s.SetDocumentPageCount(ctx, document.ID, 3); err != nil {
		t.Fatal(err)
	}

	loaded, err := s.GetDocument(ctx, document.ID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Status != StatusRunning || loaded.PageCount != 3 {
		t.Errorf("status/page count not persisted: %+v", loaded)
	}
}

func TestGetDocumentNotFound(t *testing.T) {
	if _, err := newStore(t).GetDocument(context.Background(), 999); err != ErrNotFound {
		t.Errorf("want ErrNotFound, got %v", err)
	}
}

func TestListDocumentsIsNewestFirst(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	s.CreateDocument(ctx, "first", "a.png")
	second, _ := s.CreateDocument(ctx, "second", "b.png")

	documents, err := s.ListDocuments(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(documents) != 2 || documents[0].ID != second.ID {
		t.Errorf("want the newest document first, got %+v", documents)
	}
}

func TestSavePageResultReplacesTokens(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.png")
	page, _ := s.CreatePage(ctx, document.ID, 0, "/tmp/page.png")

	first := []Token{{Text: "one", Original: "one", Confidence: 0.9, Tier: "none", Start: 0, End: 3}}
	if err := s.SavePageResult(ctx, page.ID, "one", first); err != nil {
		t.Fatal(err)
	}

	// Re-running recognition produces a fresh tokenisation with fresh offsets.
	// Leaving the old rows would leave spans pointing into a string that no
	// longer exists.
	second := []Token{
		{Text: "two", Original: "two", Confidence: 0.8, Tier: "amber", Start: 0, End: 3},
		{Text: "words", Original: "words", Confidence: 0.7, Tier: "red", Start: 4, End: 9},
	}
	if err := s.SavePageResult(ctx, page.ID, "two words", second); err != nil {
		t.Fatal(err)
	}

	tokens, err := s.ListTokens(ctx, page.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2 {
		t.Fatalf("want 2 tokens after replacement, got %d", len(tokens))
	}
	if tokens[0].Text != "two" || tokens[1].Tier != "red" {
		t.Errorf("tokens not replaced correctly: %+v", tokens)
	}

	loaded, _ := s.GetPage(ctx, page.ID)
	if loaded.Text != "two words" || loaded.Status != StatusDone {
		t.Errorf("page not updated: %+v", loaded)
	}
}

func TestTokensRoundTripEveryField(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.png")
	page, _ := s.CreatePage(ctx, document.ID, 0, "/tmp/page.png")

	want := Token{
		Text: "matrix", Original: "rnatrix", Confidence: 0.61, Tier: "amber",
		Reason: "rescored", Suggestion: "matrix", Start: 4, End: 10,
		X0: 1.5, Y0: 2.5, X1: 3.5, Y1: 4.5, LineIndex: 2, ParaIndex: 1, Struck: true,
	}
	if err := s.SavePageResult(ctx, page.ID, "the matrix", []Token{want}); err != nil {
		t.Fatal(err)
	}

	got, err := s.ListTokens(ctx, page.ID)
	if err != nil {
		t.Fatal(err)
	}
	want.Index = 0
	if got[0] != want {
		t.Errorf("round trip lost data:\n got %+v\nwant %+v", got[0], want)
	}
}

func TestPagesAreOrderedByIndex(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.pdf")
	s.CreatePage(ctx, document.ID, 2, "/tmp/2.png")
	s.CreatePage(ctx, document.ID, 0, "/tmp/0.png")
	s.CreatePage(ctx, document.ID, 1, "/tmp/1.png")

	pages, err := s.ListPages(ctx, document.ID)
	if err != nil {
		t.Fatal(err)
	}
	for i, page := range pages {
		if page.Index != i {
			t.Fatalf("page %d out of order: %+v", i, pages)
		}
	}
}

func TestCorrectionsAreRecorded(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.png")
	page, _ := s.CreatePage(ctx, document.ID, 0, "/tmp/page.png")

	if err := s.AddCorrection(ctx, page.ID, "the rnatrix", "the matrix"); err != nil {
		t.Fatal(err)
	}
	count, err := s.CountCorrections(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Errorf("want 1 correction, got %d", count)
	}
}

func TestDeletingADocumentCascades(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.png")
	page, _ := s.CreatePage(ctx, document.ID, 0, "/tmp/page.png")
	s.SavePageResult(ctx, page.ID, "text", []Token{{Text: "text", Original: "text", Tier: "none"}})

	if _, err := s.DB().ExecContext(ctx, `DELETE FROM documents WHERE id = ?`, document.ID); err != nil {
		t.Fatal(err)
	}

	tokens, err := s.ListTokens(ctx, page.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 0 {
		t.Errorf("tokens outlived their document: %+v", tokens)
	}
}

func TestClaimJobHandsOutEachJobOnce(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.pdf")

	const jobs = 20
	for i := 0; i < jobs; i++ {
		page, _ := s.CreatePage(ctx, document.ID, i, "/tmp/page.png")
		if _, err := s.EnqueueJob(ctx, document.ID, page.ID, KindPage); err != nil {
			t.Fatal(err)
		}
	}

	// The property the worker pool depends on: a job handed out twice means a
	// page recognised twice, which on a paid API is a bill as well as a bug.
	var mu sync.Mutex
	seen := map[int64]int{}
	var wg sync.WaitGroup

	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				job, err := s.ClaimJob(ctx)
				if err != nil || job == nil {
					return
				}
				mu.Lock()
				seen[job.ID]++
				mu.Unlock()
				s.FinishJob(ctx, job.ID, JobDone, "")
			}
		}()
	}
	wg.Wait()

	if len(seen) != jobs {
		t.Errorf("want %d jobs claimed, got %d", jobs, len(seen))
	}
	for id, count := range seen {
		if count != 1 {
			t.Errorf("job %d was claimed %d times", id, count)
		}
	}
}

func TestClaimJobReturnsNilWhenEmpty(t *testing.T) {
	job, err := newStore(t).ClaimJob(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if job != nil {
		t.Errorf("want nil on an empty queue, got %+v", job)
	}
}

func TestRequeueRunningRecoversFromACrash(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.png")
	page, _ := s.CreatePage(ctx, document.ID, 0, "/tmp/page.png")
	s.EnqueueJob(ctx, document.ID, page.ID, KindPage)

	claimed, err := s.ClaimJob(ctx)
	if err != nil || claimed == nil {
		t.Fatalf("could not claim: %v %v", claimed, err)
	}
	// Process dies here, leaving the job marked running and nothing scheduled
	// to finish it.

	requeued, err := s.RequeueRunning(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if requeued != 1 {
		t.Fatalf("want 1 job requeued, got %d", requeued)
	}

	again, err := s.ClaimJob(ctx)
	if err != nil || again == nil {
		t.Fatalf("requeued job was not claimable: %v %v", again, err)
	}
	if again.Attempts != 2 {
		t.Errorf("want the retry counted, got attempts=%d", again.Attempts)
	}
}

func TestPendingJobsCountsQueuedAndRunning(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	document, _ := s.CreateDocument(ctx, "Notes", "notes.pdf")
	page, _ := s.CreatePage(ctx, document.ID, 0, "/tmp/page.png")
	s.EnqueueJob(ctx, document.ID, page.ID, KindPage)
	s.EnqueueJob(ctx, document.ID, page.ID, KindPage)

	claimed, _ := s.ClaimJob(ctx)

	pending, err := s.PendingJobs(ctx, document.ID)
	if err != nil {
		t.Fatal(err)
	}
	if pending != 2 {
		t.Errorf("want both the queued and the running job counted, got %d", pending)
	}

	s.FinishJob(ctx, claimed.ID, JobDone, "")
	if pending, _ = s.PendingJobs(ctx, document.ID); pending != 1 {
		t.Errorf("want 1 pending after one finished, got %d", pending)
	}
}
