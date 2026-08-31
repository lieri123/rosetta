// Package api is the HTTP surface.
//
// Routing is net/http's own pattern matching -- method plus path, with typed
// path values -- which covers everything here without a third-party router.
// The endpoints fall into three groups: uploading documents, reading pages
// (text plus the confidence spans that go with it), and sending corrections
// back, which is the loop that makes the whole system adapt.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/lieri123/rosetta/service/internal/config"
	"github.com/lieri123/rosetta/service/internal/events"
	"github.com/lieri123/rosetta/service/internal/pdf"
	"github.com/lieri123/rosetta/service/internal/queue"
	"github.com/lieri123/rosetta/service/internal/rescore"
	"github.com/lieri123/rosetta/service/internal/store"
)

type Server struct {
	Config   config.Config
	Store    *store.Store
	Broker   *events.Broker
	Pool     *queue.Pool
	Rescorer *rescore.Client
	Provider string
	Logger   *log.Logger
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/healthz", s.handleHealth)
	mux.HandleFunc("POST /api/documents", s.handleUpload)
	mux.HandleFunc("GET /api/documents", s.handleListDocuments)
	mux.HandleFunc("GET /api/documents/{id}", s.handleGetDocument)
	mux.HandleFunc("GET /api/documents/{id}/events", s.handleEvents)
	mux.HandleFunc("GET /api/documents/{id}/text", s.handleDocumentText)
	mux.HandleFunc("GET /api/pages/{id}", s.handleGetPage)
	mux.HandleFunc("GET /api/pages/{id}/image", s.handlePageImage)
	mux.HandleFunc("POST /api/pages/{id}/corrections", s.handleCorrection)

	if s.Config.StaticDir != "" {
		mux.Handle("/", s.staticHandler())
	}

	return logRequests(s.Logger, mux)
}

// ----------------------------------------------------------------------

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	corrections, _ := s.Store.CountCorrections(r.Context())
	rescorerUp := s.Rescorer != nil && s.Rescorer.Healthy(r.Context())

	writeJSON(w, http.StatusOK, map[string]any{
		"status":      "ok",
		"provider":    s.Provider,
		"rescorer":    map[string]any{"url": s.Config.RescorerURL, "reachable": rescorerUp},
		"corrections": corrections,
		"workers":     s.Config.Workers,
	})
}

func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.Config.MaxUploadBytes)
	if err := r.ParseMultipartForm(16 << 20); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("reading upload: %w", err))
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("expected a file field named 'file': %w", err))
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(data) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("the uploaded file is empty"))
		return
	}

	title := strings.TrimSpace(r.FormValue("title"))
	if title == "" {
		title = header.Filename
	}

	ctx := r.Context()
	document, err := s.Store.CreateDocument(ctx, title, header.Filename)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	pagePaths, err := s.materialisePages(ctx, document.ID, header.Filename, data)
	if err != nil {
		s.Store.SetDocumentStatus(ctx, document.ID, store.StatusFailed, err.Error())
		writeError(w, http.StatusBadRequest, err)
		return
	}

	for index, path := range pagePaths {
		page, err := s.Store.CreatePage(ctx, document.ID, index, path)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		if _, err := s.Store.EnqueueJob(ctx, document.ID, page.ID, store.KindPage); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
	}

	if err := s.Store.SetDocumentPageCount(ctx, document.ID, len(pagePaths)); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := s.Store.SetDocumentStatus(ctx, document.ID, store.StatusRunning, ""); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	s.Broker.Publish(events.Event{
		Type: "queued", DocumentID: document.ID, Total: len(pagePaths),
		Message: fmt.Sprintf("%d page(s) queued", len(pagePaths)),
	})
	if s.Pool != nil {
		s.Pool.Notify()
	}

	// 202: the pages are accepted and queued, not finished. The client follows
	// the event stream from here.
	writeJSON(w, http.StatusAccepted, map[string]any{
		"id":     document.ID,
		"title":  title,
		"pages":  len(pagePaths),
		"status": store.StatusRunning,
	})
}

// materialisePages writes the upload to disk, splitting a PDF into pages.
func (s *Server) materialisePages(ctx context.Context, documentID int64, filename string, data []byte) ([]string, error) {
	dir := s.Config.PageDir(documentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	if pdf.IsPDF(data) {
		pdfPath := filepath.Join(dir, "source.pdf")
		if err := os.WriteFile(pdfPath, data, 0o644); err != nil {
			return nil, err
		}
		return pdf.Split(ctx, pdfPath, dir, 300)
	}

	extension := strings.ToLower(filepath.Ext(filename))
	switch extension {
	case ".png", ".jpg", ".jpeg", ".bmp", ".tga":
	case "":
		extension = ".png"
	default:
		return nil, fmt.Errorf("unsupported file type %q: upload a PNG, JPEG or PDF", extension)
	}

	path := filepath.Join(dir, "0-source"+extension)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return nil, err
	}
	return []string{path}, nil
}

func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	documents, err := s.Store.ListDocuments(r.Context(), 200)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": documents})
}

func (s *Server) handleGetDocument(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	document, err := s.Store.GetDocument(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, errors.New("no such document"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	pages, err := s.Store.ListPages(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"document": document, "pages": pages})
}

// handleDocumentText returns the whole document as plain text, which is the
// point of the exercise: something to paste somewhere else.
func (s *Server) handleDocumentText(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	pages, err := s.Store.ListPages(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	var builder strings.Builder
	for i, page := range pages {
		if i > 0 {
			builder.WriteString("\n\n")
		}
		builder.WriteString(page.Text)
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	io.WriteString(w, builder.String())
}

func (s *Server) handleGetPage(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	page, err := s.Store.GetPage(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, errors.New("no such page"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	tokens, err := s.Store.ListTokens(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	// Text and spans in one response, on purpose: they are only meaningful
	// together, and fetching them separately would let the editor decorate one
	// version of the text with the offsets of another.
	writeJSON(w, http.StatusOK, map[string]any{"page": page, "tokens": tokens})
}

func (s *Server) handlePageImage(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	page, err := s.Store.GetPage(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, errors.New("no such page"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	path := page.CleanPath
	if r.URL.Query().Get("original") == "1" || path == "" {
		path = page.SourcePath
	}
	if path == "" {
		writeError(w, http.StatusNotFound, errors.New("no image for this page"))
		return
	}
	http.ServeFile(w, r, path)
}

type correctionRequest struct {
	Text string `json:"text"`
}

// handleCorrection is the feedback loop.
//
// The edited text arrives, the difference against what the recogniser produced
// becomes training evidence, and that evidence goes to the rescorer, which
// folds it into the confusion matrix, the lexicon and the language model at
// once. This is the endpoint that makes the system get better at reading one
// particular person's handwriting.
func (s *Server) handleCorrection(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	var request correctionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}

	ctx := r.Context()
	page, err := s.Store.GetPage(ctx, id)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, errors.New("no such page"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	if request.Text == page.Text {
		writeJSON(w, http.StatusOK, map[string]any{"learned": 0, "message": "no change"})
		return
	}

	pairs := ChangedLines(page.Text, request.Text)
	for _, pair := range pairs {
		if err := s.Store.AddCorrection(ctx, id, pair.Predicted, pair.Corrected); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
	}
	if err := s.Store.SetPageText(ctx, id, request.Text); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	learned := 0
	warning := ""
	if s.Rescorer != nil && len(pairs) > 0 {
		if err := s.Rescorer.Learn(ctx, pairs, id); err != nil {
			// The correction is already saved, so this is recoverable: the
			// models can be rebuilt from the corrections table. Say so rather
			// than failing the request and making the user retype.
			warning = fmt.Sprintf("saved, but the rescorer did not accept it: %v", err)
			s.logf("learn failed for page %d: %v", id, err)
		} else {
			learned = len(pairs)
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"learned": learned,
		"pairs":   len(pairs),
		"warning": warning,
	})
}

// ChangedLines pairs up the lines that actually differ.
//
// Sending the whole page as one (predicted, corrected) pair would work, but it
// makes for poor evidence: one long alignment across unrelated edits invents
// substitutions between characters that were never confused for each other.
// Line by line keeps each alignment local. When the line count changes the
// pairing is no longer positional and the whole text is sent instead, which is
// noisier but never wrong.
func ChangedLines(before, after string) []rescore.Pair {
	beforeLines := strings.Split(before, "\n")
	afterLines := strings.Split(after, "\n")

	if len(beforeLines) != len(afterLines) {
		if strings.TrimSpace(before) == "" {
			return nil
		}
		return []rescore.Pair{{Predicted: before, Corrected: after}}
	}

	pairs := make([]rescore.Pair, 0, 4)
	for i := range beforeLines {
		if beforeLines[i] == afterLines[i] {
			continue
		}
		if strings.TrimSpace(beforeLines[i]) == "" || strings.TrimSpace(afterLines[i]) == "" {
			continue
		}
		pairs = append(pairs, rescore.Pair{
			Predicted: beforeLines[i], Corrected: afterLines[i],
		})
	}
	return pairs
}

// ----------------------------------------------------------------------
// Server-sent events
// ----------------------------------------------------------------------

// handleEvents streams progress for one document.
//
// SSE rather than websockets: the traffic is one-directional and low volume,
// browsers reconnect on their own, and it is plain HTTP all the way down --
// no upgrade handshake, no second protocol to proxy.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, errors.New("streaming unsupported"))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Buffering proxies are the usual reason a progress stream appears to hang
	// until it finishes.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	stream, cancel := s.Broker.Subscribe(id)
	defer cancel()

	document, err := s.Store.GetDocument(r.Context(), id)
	if err == nil {
		writeEvent(w, flusher, events.Event{
			Type: "snapshot", DocumentID: id, Total: document.PageCount,
			Message: document.Status,
		})
	}

	// Keep-alive comments stop intermediaries from closing an idle stream
	// while a slow page is being recognised.
	ticker := time.NewTicker(20 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case event, open := <-stream:
			if !open {
				return
			}
			writeEvent(w, flusher, event)
			if event.Type == "done" || event.Type == "failed" {
				// Nothing more will happen for this document; let the client
				// close rather than holding the connection open forever.
				if event.Type == "done" {
					return
				}
			}
		case <-ticker.C:
			io.WriteString(w, ": keep-alive\n\n")
			flusher.Flush()
		}
	}
}

func writeEvent(w http.ResponseWriter, flusher http.Flusher, event events.Event) {
	payload, err := json.Marshal(event)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, payload)
	flusher.Flush()
}

// ----------------------------------------------------------------------

func (s *Server) staticHandler() http.Handler {
	files := http.FileServer(http.Dir(s.Config.StaticDir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(s.Config.StaticDir, filepath.Clean(r.URL.Path))
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			// Single page app: unknown paths are routes, not missing files.
			index := filepath.Join(s.Config.StaticDir, "index.html")
			if _, err := os.Stat(index); err == nil {
				http.ServeFile(w, r, index)
				return
			}
			http.Error(w, "the web UI is not built: run `make web`", http.StatusNotFound)
			return
		}
		files.ServeHTTP(w, r)
	})
}

func (s *Server) logf(format string, args ...any) {
	if s.Logger != nil {
		s.Logger.Printf(format, args...)
	}
}

func pathID(r *http.Request) (int64, error) {
	raw := r.PathValue("id")
	id, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || id <= 0 {
		return 0, fmt.Errorf("invalid id %q", raw)
	}
	return id, nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]any{"error": err.Error()})
}

func logRequests(logger *log.Logger, next http.Handler) http.Handler {
	if logger == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(recorder, r)
		if strings.HasPrefix(r.URL.Path, "/api/") {
			logger.Printf("%s %s %d %s", r.Method, r.URL.Path, recorder.status, time.Since(start).Round(time.Millisecond))
		}
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// Flush forwards to the underlying writer so SSE keeps working through the
// logging wrapper. Without it the progress stream buffers until the handler
// returns, which for a long document is exactly never.
func (r *statusRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}
