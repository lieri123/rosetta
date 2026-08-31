package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lieri123/rosetta/service/internal/config"
	"github.com/lieri123/rosetta/service/internal/events"
	"github.com/lieri123/rosetta/service/internal/pipeline"
	"github.com/lieri123/rosetta/service/internal/preprocess"
	"github.com/lieri123/rosetta/service/internal/queue"
	"github.com/lieri123/rosetta/service/internal/recognize"
	"github.com/lieri123/rosetta/service/internal/rescore"
	"github.com/lieri123/rosetta/service/internal/store"
)

type harness struct {
	server *httptest.Server
	store  *store.Store
	cancel context.CancelFunc
	pool   *queue.Pool
}

// newHarness wires up the real service against a temporary database, the mock
// provider and no rescorer, so the test exercises the actual pipeline rather
// than a stand-in for it.
func newHarness(t *testing.T, rescorerURL string) *harness {
	t.Helper()

	dir := t.TempDir()
	cfg := config.Default()
	cfg.DataDir = dir
	cfg.DatabasePath = filepath.Join(dir, "test.db")
	cfg.StaticDir = ""
	cfg.PreprocessBin = filepath.Join(dir, "no-such-binary")
	cfg.Workers = 2
	if rescorerURL != "" {
		cfg.RescorerURL = rescorerURL
	}

	database, err := store.Open(cfg.DatabasePath)
	if err != nil {
		t.Fatal(err)
	}

	broker := events.NewBroker()
	var rescorer *rescore.Client
	if rescorerURL != "" {
		rescorer = rescore.New(rescorerURL)
	}

	processor := &pipeline.Processor{
		Store:      database,
		Preprocess: preprocess.NewRunner(cfg.PreprocessBin),
		Provider:   recognize.NewMock(),
		Rescorer:   rescorer,
		Broker:     broker,
		DataDir:    cfg.DataDir,
	}
	pool := &queue.Pool{
		Store: database, Handler: processor.ProcessPage, Broker: broker,
		Workers: cfg.Workers, IdlePoll: 20 * time.Millisecond,
	}

	ctx, cancel := context.WithCancel(context.Background())
	pool.Start(ctx)

	api := &Server{
		Config: cfg, Store: database, Broker: broker, Pool: pool,
		Rescorer: rescorer, Provider: "mock",
	}
	server := httptest.NewServer(api.Routes())

	h := &harness{server: server, store: database, cancel: cancel, pool: pool}
	t.Cleanup(func() {
		server.Close()
		cancel()
		pool.Wait()
		database.Close()
	})
	return h
}

func (h *harness) upload(t *testing.T, filename string, content []byte) int64 {
	t.Helper()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		t.Fatal(err)
	}
	part.Write(content)
	writer.WriteField("title", "Test notes")
	writer.Close()

	response, err := http.Post(h.server.URL+"/api/documents", writer.FormDataContentType(), &body)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusAccepted {
		raw, _ := io.ReadAll(response.Body)
		t.Fatalf("upload returned %d: %s", response.StatusCode, raw)
	}

	var decoded struct {
		ID    int64 `json:"id"`
		Pages int   `json:"pages"`
	}
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatal(err)
	}
	return decoded.ID
}

func (h *harness) waitForDone(t *testing.T, documentID int64) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		document, err := h.store.GetDocument(context.Background(), documentID)
		if err != nil {
			t.Fatal(err)
		}
		if document.Status == store.StatusDone {
			return
		}
		if document.Status == store.StatusFailed {
			t.Fatalf("document failed: %s", document.Error)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("document did not finish within the deadline")
}

func (h *harness) getJSON(t *testing.T, path string, out any) {
	t.Helper()
	response, err := http.Get(h.server.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(response.Body)
		t.Fatalf("GET %s returned %d: %s", path, response.StatusCode, raw)
	}
	if err := json.NewDecoder(response.Body).Decode(out); err != nil {
		t.Fatal(err)
	}
}

type pageResponse struct {
	Page   store.Page    `json:"page"`
	Tokens []store.Token `json:"tokens"`
}

func TestUploadRunsThePipelineEndToEnd(t *testing.T) {
	h := newHarness(t, "")
	documentID := h.upload(t, "notes.png", []byte("pretend this is a photo of a page"))
	h.waitForDone(t, documentID)

	var document struct {
		Document store.Document `json:"document"`
		Pages    []store.Page   `json:"pages"`
	}
	h.getJSON(t, fmt.Sprintf("/api/documents/%d", documentID), &document)

	if len(document.Pages) != 1 {
		t.Fatalf("want 1 page, got %d", len(document.Pages))
	}
	if document.Pages[0].Text == "" {
		t.Fatal("page has no text")
	}

	var page pageResponse
	h.getJSON(t, fmt.Sprintf("/api/pages/%d", document.Pages[0].ID), &page)
	if len(page.Tokens) == 0 {
		t.Fatal("page has no tokens")
	}

	// The contract the editor relies on: every token's offsets index exactly
	// that token in the page text. If this drifts, every underline lands on
	// the wrong word.
	for _, token := range page.Tokens {
		if token.Struck {
			continue
		}
		if token.Start < 0 || token.End > len(page.Page.Text) {
			t.Fatalf("token %q offsets %d..%d outside text of length %d",
				token.Text, token.Start, token.End, len(page.Page.Text))
		}
		if got := page.Page.Text[token.Start:token.End]; got != token.Text {
			t.Errorf("offsets give %q, want %q", got, token.Text)
		}
	}
}

func TestTiersAreAssignedWithoutTheRescorer(t *testing.T) {
	// With the rescorer unreachable the service must still produce a usable
	// page: red for unreadable tokens, and no amber, because nothing is
	// modelling context and inventing an amber tier would misrepresent what
	// the underline means.
	h := newHarness(t, "")
	documentID := h.upload(t, "notes.png", []byte("a page with some doubtful words"))
	h.waitForDone(t, documentID)

	var document struct {
		Pages []store.Page `json:"pages"`
	}
	h.getJSON(t, fmt.Sprintf("/api/documents/%d", documentID), &document)

	var page pageResponse
	h.getJSON(t, fmt.Sprintf("/api/pages/%d", document.Pages[0].ID), &page)

	counts := map[string]int{}
	for _, token := range page.Tokens {
		counts[token.Tier]++
	}
	if counts["red"] == 0 {
		t.Error("want some low-confidence tokens flagged red")
	}
	if counts["amber"] != 0 {
		t.Errorf("want no amber without a language model, got %d", counts["amber"])
	}
	if counts["none"] == 0 {
		t.Error("want most tokens unflagged")
	}
}

func TestCorrectionIsRecordedAndForwarded(t *testing.T) {
	var learned []map[string]any
	rescorer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Write([]byte(`{"status":"ok"}`))
		case "/rescore":
			// Echo the tokens back unchanged so the pipeline completes.
			var request struct {
				Tokens []rescore.Token `json:"tokens"`
			}
			json.NewDecoder(r.Body).Decode(&request)
			decoded := make([]map[string]any, len(request.Tokens))
			for i, token := range request.Tokens {
				decoded[i] = map[string]any{
					"index": i, "text": token.Text, "original": token.Text,
					"confidence": token.Confidence, "tier": "none", "reason": "",
				}
			}
			json.NewEncoder(w).Encode(map[string]any{"tokens": decoded})
		case "/learn":
			var request struct {
				Pairs []map[string]any `json:"pairs"`
			}
			json.NewDecoder(r.Body).Decode(&request)
			learned = append(learned, request.Pairs...)
			w.Write([]byte(`{"pairs":1}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer rescorer.Close()

	h := newHarness(t, rescorer.URL)
	documentID := h.upload(t, "notes.png", []byte("a page to correct"))
	h.waitForDone(t, documentID)

	var document struct {
		Pages []store.Page `json:"pages"`
	}
	h.getJSON(t, fmt.Sprintf("/api/documents/%d", documentID), &document)
	pageID := document.Pages[0].ID

	var page pageResponse
	h.getJSON(t, fmt.Sprintf("/api/pages/%d", pageID), &page)

	// Edit the first line only.
	lines := strings.Split(page.Page.Text, "\n")
	lines[0] = "the corrected first line"
	corrected := strings.Join(lines, "\n")

	body, _ := json.Marshal(map[string]string{"text": corrected})
	response, err := http.Post(
		fmt.Sprintf("%s/api/pages/%d/corrections", h.server.URL, pageID),
		"application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	var result struct {
		Learned int    `json:"learned"`
		Pairs   int    `json:"pairs"`
		Warning string `json:"warning"`
	}
	json.NewDecoder(response.Body).Decode(&result)

	if result.Warning != "" {
		t.Errorf("unexpected warning: %s", result.Warning)
	}
	if result.Pairs != 1 {
		t.Errorf("want exactly the changed line sent, got %d pairs", result.Pairs)
	}
	if len(learned) != 1 {
		t.Fatalf("want 1 pair forwarded to the rescorer, got %d", len(learned))
	}
	if learned[0]["corrected"] != "the corrected first line" {
		t.Errorf("wrong pair forwarded: %+v", learned[0])
	}

	count, _ := h.store.CountCorrections(context.Background())
	if count != 1 {
		t.Errorf("want the correction stored, got %d", count)
	}

	updated, _ := h.store.GetPage(context.Background(), pageID)
	if updated.Text != corrected {
		t.Error("the edited text was not saved")
	}
}

func TestCorrectionSurvivesAnUnreachableRescorer(t *testing.T) {
	// The correction is the valuable artefact and is saved first. A rescorer
	// that is down must not make the user retype.
	h := newHarness(t, "http://127.0.0.1:1")
	documentID := h.upload(t, "notes.png", []byte("a page"))
	h.waitForDone(t, documentID)

	var document struct {
		Pages []store.Page `json:"pages"`
	}
	h.getJSON(t, fmt.Sprintf("/api/documents/%d", documentID), &document)
	pageID := document.Pages[0].ID

	page, _ := h.store.GetPage(context.Background(), pageID)
	lines := strings.Split(page.Text, "\n")
	lines[0] = "an edit made while the rescorer is down"

	body, _ := json.Marshal(map[string]string{"text": strings.Join(lines, "\n")})
	response, err := http.Post(
		fmt.Sprintf("%s/api/pages/%d/corrections", h.server.URL, pageID),
		"application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		t.Fatalf("want 200 even with the rescorer down, got %d", response.StatusCode)
	}
	var result struct {
		Warning string `json:"warning"`
	}
	json.NewDecoder(response.Body).Decode(&result)
	if result.Warning == "" {
		t.Error("want a warning explaining the rescorer did not accept it")
	}

	count, _ := h.store.CountCorrections(context.Background())
	if count != 1 {
		t.Errorf("the correction was lost: %d stored", count)
	}
}

func TestNoChangeIsNotACorrection(t *testing.T) {
	h := newHarness(t, "")
	documentID := h.upload(t, "notes.png", []byte("a page"))
	h.waitForDone(t, documentID)

	var document struct {
		Pages []store.Page `json:"pages"`
	}
	h.getJSON(t, fmt.Sprintf("/api/documents/%d", documentID), &document)
	page, _ := h.store.GetPage(context.Background(), document.Pages[0].ID)

	body, _ := json.Marshal(map[string]string{"text": page.Text})
	response, _ := http.Post(
		fmt.Sprintf("%s/api/pages/%d/corrections", h.server.URL, page.ID),
		"application/json", bytes.NewReader(body))
	defer response.Body.Close()

	count, _ := h.store.CountCorrections(context.Background())
	if count != 0 {
		t.Errorf("saving unchanged text recorded %d corrections", count)
	}
}

func TestEventStreamReportsProgress(t *testing.T) {
	h := newHarness(t, "")

	// Subscribe before uploading, or the work may finish first.
	document, err := h.store.CreateDocument(context.Background(), "streamed", "notes.png")
	if err != nil {
		t.Fatal(err)
	}

	request, _ := http.NewRequest(http.MethodGet,
		fmt.Sprintf("%s/api/documents/%d/events", h.server.URL, document.ID), nil)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if got := response.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("want an SSE content type, got %q", got)
	}

	page, _ := h.store.CreatePage(context.Background(), document.ID, 0, writeTempImage(t))
	h.store.EnqueueJob(context.Background(), document.ID, page.ID, store.KindPage)
	h.pool.Notify()

	seen := map[string]bool{}
	scanner := bufio.NewScanner(response.Body)
	deadline := time.AfterFunc(10*time.Second, func() { response.Body.Close() })
	defer deadline.Stop()

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "event: ") {
			continue
		}
		seen[strings.TrimPrefix(line, "event: ")] = true
		if seen["done"] {
			break
		}
	}

	if !seen["snapshot"] {
		t.Error("want an initial snapshot event")
	}
	if !seen["page"] {
		t.Error("want a per-page event")
	}
	if !seen["done"] {
		t.Error("want a completion event")
	}
}

func writeTempImage(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "page.png")
	if err := os.WriteFile(path, []byte("page bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestUnknownDocumentIsA404(t *testing.T) {
	h := newHarness(t, "")
	response, err := http.Get(h.server.URL + "/api/documents/4242")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("want 404, got %d", response.StatusCode)
	}
}

func TestInvalidIDIsA400(t *testing.T) {
	h := newHarness(t, "")
	response, err := http.Get(h.server.URL + "/api/pages/not-a-number")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusBadRequest {
		t.Errorf("want 400, got %d", response.StatusCode)
	}
}

func TestUploadRejectsUnsupportedTypes(t *testing.T) {
	h := newHarness(t, "")

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, _ := writer.CreateFormFile("file", "notes.docx")
	part.Write([]byte("not an image"))
	writer.Close()

	response, err := http.Post(h.server.URL+"/api/documents", writer.FormDataContentType(), &body)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusBadRequest {
		t.Errorf("want 400 for an unsupported type, got %d", response.StatusCode)
	}
}

func TestUploadRejectsAnEmptyFile(t *testing.T) {
	h := newHarness(t, "")

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	writer.CreateFormFile("file", "notes.png")
	writer.Close()

	response, err := http.Post(h.server.URL+"/api/documents", writer.FormDataContentType(), &body)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusBadRequest {
		t.Errorf("want 400 for an empty file, got %d", response.StatusCode)
	}
}

func TestDocumentTextEndpointReturnsPlainText(t *testing.T) {
	h := newHarness(t, "")
	documentID := h.upload(t, "notes.png", []byte("a page of notes"))
	h.waitForDone(t, documentID)

	response, err := http.Get(fmt.Sprintf("%s/api/documents/%d/text", h.server.URL, documentID))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if got := response.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/plain") {
		t.Errorf("want text/plain, got %q", got)
	}
	raw, _ := io.ReadAll(response.Body)
	if len(bytes.TrimSpace(raw)) == 0 {
		t.Error("want the document's text, got nothing")
	}
}

func TestChangedLinesSendsOnlyWhatChanged(t *testing.T) {
	before := "the first line\nthe second line\nthe third line"
	after := "the first line\nthe corrected second line\nthe third line"

	pairs := ChangedLines(before, after)
	if len(pairs) != 1 {
		t.Fatalf("want 1 pair, got %d: %+v", len(pairs), pairs)
	}
	if pairs[0].Predicted != "the second line" {
		t.Errorf("wrong line paired: %+v", pairs[0])
	}
}

func TestChangedLinesFallsBackWhenLineCountsDiffer(t *testing.T) {
	// Once lines have been added or removed the pairing is no longer
	// positional, and pairing by index would align unrelated lines and teach
	// the confusion matrix nonsense.
	pairs := ChangedLines("one\ntwo", "one\ntwo\nthree")
	if len(pairs) != 1 {
		t.Fatalf("want a single whole-text pair, got %d", len(pairs))
	}
	if !strings.Contains(pairs[0].Corrected, "three") {
		t.Errorf("want the whole text, got %+v", pairs[0])
	}
}

func TestChangedLinesIgnoresBlankLines(t *testing.T) {
	pairs := ChangedLines("one\n\nthree", "one\nsomething\nthree")
	if len(pairs) != 0 {
		t.Errorf("filling in a blank line is not error evidence: %+v", pairs)
	}
}

func TestHealthReportsProviderAndRescorer(t *testing.T) {
	h := newHarness(t, "")
	var health struct {
		Status   string `json:"status"`
		Provider string `json:"provider"`
		Rescorer struct {
			Reachable bool `json:"reachable"`
		} `json:"rescorer"`
	}
	h.getJSON(t, "/api/healthz", &health)
	if health.Status != "ok" || health.Provider != "mock" {
		t.Errorf("unexpected health: %+v", health)
	}
}
