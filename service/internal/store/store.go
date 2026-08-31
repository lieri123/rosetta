// Package store is the SQLite persistence layer.
//
// One connection, serialised writes, WAL journalling. SQLite allows exactly
// one writer at a time whatever the pool says, so capping the pool at one
// connection turns what would be intermittent SQLITE_BUSY errors under the
// worker pool into ordinary queueing. At this scale the write path is nowhere
// near the bottleneck -- recognition takes a network round trip per page --
// and predictable beats clever.
package store

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite" // pure Go driver: no cgo, so the service cross-compiles
)

//go:embed schema.sql
var schema string

var ErrNotFound = errors.New("not found")

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("creating %s: %w", dir, err)
		}
	}

	// WAL lets the Python rescorer read the same file while we write to it.
	dsn := path + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(ON)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	db.SetMaxOpenConns(1)

	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("applying schema: %w", err)
	}

	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

// DB exposes the handle for tests that need to poke at raw rows.
func (s *Store) DB() *sql.DB { return s.db }

func now() float64 { return float64(time.Now().UnixNano()) / 1e9 }

// ----------------------------------------------------------------------
// Documents
// ----------------------------------------------------------------------

type Document struct {
	ID         int64   `json:"id"`
	Title      string  `json:"title"`
	SourceName string  `json:"source_name"`
	Status     string  `json:"status"`
	Error      string  `json:"error,omitempty"`
	PageCount  int     `json:"page_count"`
	CreatedAt  float64 `json:"created_at"`
	UpdatedAt  float64 `json:"updated_at"`
}

const (
	StatusPending = "pending"
	StatusRunning = "running"
	StatusDone    = "done"
	StatusFailed  = "failed"
)

func (s *Store) CreateDocument(ctx context.Context, title, sourceName string) (*Document, error) {
	timestamp := now()
	result, err := s.db.ExecContext(ctx,
		`INSERT INTO documents (title, source_name, status, page_count, created_at, updated_at)
		 VALUES (?, ?, ?, 0, ?, ?)`,
		title, sourceName, StatusPending, timestamp, timestamp)
	if err != nil {
		return nil, err
	}
	id, err := result.LastInsertId()
	if err != nil {
		return nil, err
	}
	return &Document{
		ID: id, Title: title, SourceName: sourceName, Status: StatusPending,
		CreatedAt: timestamp, UpdatedAt: timestamp,
	}, nil
}

func (s *Store) GetDocument(ctx context.Context, id int64) (*Document, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, title, source_name, status, error, page_count, created_at, updated_at
		 FROM documents WHERE id = ?`, id)
	return scanDocument(row)
}

func (s *Store) ListDocuments(ctx context.Context, limit int) ([]Document, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, title, source_name, status, error, page_count, created_at, updated_at
		 FROM documents ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	documents := []Document{}
	for rows.Next() {
		var d Document
		if err := rows.Scan(&d.ID, &d.Title, &d.SourceName, &d.Status, &d.Error,
			&d.PageCount, &d.CreatedAt, &d.UpdatedAt); err != nil {
			return nil, err
		}
		documents = append(documents, d)
	}
	return documents, rows.Err()
}

func (s *Store) SetDocumentStatus(ctx context.Context, id int64, status, errText string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE id = ?`,
		status, errText, now(), id)
	return err
}

func (s *Store) SetDocumentPageCount(ctx context.Context, id int64, count int) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE documents SET page_count = ?, updated_at = ? WHERE id = ?`, count, now(), id)
	return err
}

func scanDocument(row *sql.Row) (*Document, error) {
	var d Document
	err := row.Scan(&d.ID, &d.Title, &d.SourceName, &d.Status, &d.Error,
		&d.PageCount, &d.CreatedAt, &d.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &d, nil
}

// ----------------------------------------------------------------------
// Pages
// ----------------------------------------------------------------------

type Page struct {
	ID         int64   `json:"id"`
	DocumentID int64   `json:"document_id"`
	Index      int     `json:"index"`
	SourcePath string  `json:"-"`
	CleanPath  string  `json:"-"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Status     string  `json:"status"`
	Error      string  `json:"error,omitempty"`
	Text       string  `json:"text"`
	SkewDeg    float64 `json:"skew_deg"`
	Rectified  bool    `json:"rectified"`
}

func (s *Store) CreatePage(ctx context.Context, documentID int64, index int, sourcePath string) (*Page, error) {
	result, err := s.db.ExecContext(ctx,
		`INSERT INTO pages (document_id, page_index, source_path, status) VALUES (?, ?, ?, ?)`,
		documentID, index, sourcePath, StatusPending)
	if err != nil {
		return nil, err
	}
	id, err := result.LastInsertId()
	if err != nil {
		return nil, err
	}
	return &Page{ID: id, DocumentID: documentID, Index: index,
		SourcePath: sourcePath, Status: StatusPending}, nil
}

const pageColumns = `id, document_id, page_index, source_path, clean_path, width, height,
	status, error, text, skew_deg, rectified`

func (s *Store) GetPage(ctx context.Context, id int64) (*Page, error) {
	row := s.db.QueryRowContext(ctx, `SELECT `+pageColumns+` FROM pages WHERE id = ?`, id)
	var p Page
	var rectified int
	err := row.Scan(&p.ID, &p.DocumentID, &p.Index, &p.SourcePath, &p.CleanPath,
		&p.Width, &p.Height, &p.Status, &p.Error, &p.Text, &p.SkewDeg, &rectified)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	p.Rectified = rectified != 0
	return &p, nil
}

func (s *Store) ListPages(ctx context.Context, documentID int64) ([]Page, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT `+pageColumns+` FROM pages WHERE document_id = ? ORDER BY page_index`, documentID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	pages := []Page{}
	for rows.Next() {
		var p Page
		var rectified int
		if err := rows.Scan(&p.ID, &p.DocumentID, &p.Index, &p.SourcePath, &p.CleanPath,
			&p.Width, &p.Height, &p.Status, &p.Error, &p.Text, &p.SkewDeg, &rectified); err != nil {
			return nil, err
		}
		p.Rectified = rectified != 0
		pages = append(pages, p)
	}
	return pages, rows.Err()
}

func (s *Store) SetPagePreprocessed(ctx context.Context, id int64, cleanPath string, width, height int, skew float64, rectified bool) error {
	flag := 0
	if rectified {
		flag = 1
	}
	_, err := s.db.ExecContext(ctx,
		`UPDATE pages SET clean_path = ?, width = ?, height = ?, skew_deg = ?, rectified = ?
		 WHERE id = ?`, cleanPath, width, height, skew, flag, id)
	return err
}

func (s *Store) SetPageStatus(ctx context.Context, id int64, status, errText string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE pages SET status = ?, error = ? WHERE id = ?`, status, errText, id)
	return err
}

// SavePageResult replaces a page's text and tokens as one transaction.
//
// Replace rather than merge: re-running recognition on a page produces a fresh
// tokenisation with fresh offsets, and half-old half-new tokens would point
// into a string that no longer exists.
func (s *Store) SavePageResult(ctx context.Context, pageID int64, text string, tokens []Token) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, `DELETE FROM tokens WHERE page_id = ?`, pageID); err != nil {
		return err
	}

	statement, err := tx.PrepareContext(ctx,
		`INSERT INTO tokens (page_id, idx, text, original, confidence, tier, reason, suggestion,
		 start_offset, end_offset, x0, y0, x1, y1, line_index, para_index, struck)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer statement.Close()

	for i, token := range tokens {
		struck := 0
		if token.Struck {
			struck = 1
		}
		if _, err := statement.ExecContext(ctx, pageID, i, token.Text, token.Original,
			token.Confidence, token.Tier, token.Reason, token.Suggestion,
			token.Start, token.End, token.X0, token.Y0, token.X1, token.Y1,
			token.LineIndex, token.ParaIndex, struck); err != nil {
			return err
		}
	}

	if _, err := tx.ExecContext(ctx,
		`UPDATE pages SET text = ?, status = ?, error = '' WHERE id = ?`,
		text, StatusDone, pageID); err != nil {
		return err
	}

	return tx.Commit()
}

// ----------------------------------------------------------------------
// Tokens
// ----------------------------------------------------------------------

type Token struct {
	Index      int     `json:"index"`
	Text       string  `json:"text"`
	Original   string  `json:"original"`
	Confidence float64 `json:"confidence"`
	Tier       string  `json:"tier"`
	Reason     string  `json:"reason,omitempty"`
	Suggestion string  `json:"suggestion,omitempty"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	X0         float64 `json:"x0"`
	Y0         float64 `json:"y0"`
	X1         float64 `json:"x1"`
	Y1         float64 `json:"y1"`
	LineIndex  int     `json:"line"`
	ParaIndex  int     `json:"paragraph"`
	Struck     bool    `json:"struck,omitempty"`
}

func (s *Store) ListTokens(ctx context.Context, pageID int64) ([]Token, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT idx, text, original, confidence, tier, reason, suggestion, start_offset,
		 end_offset, x0, y0, x1, y1, line_index, para_index, struck
		 FROM tokens WHERE page_id = ? ORDER BY idx`, pageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	tokens := []Token{}
	for rows.Next() {
		var t Token
		var struck int
		if err := rows.Scan(&t.Index, &t.Text, &t.Original, &t.Confidence, &t.Tier,
			&t.Reason, &t.Suggestion, &t.Start, &t.End, &t.X0, &t.Y0, &t.X1, &t.Y1,
			&t.LineIndex, &t.ParaIndex, &struck); err != nil {
			return nil, err
		}
		t.Struck = struck != 0
		tokens = append(tokens, t)
	}
	return tokens, rows.Err()
}

// ----------------------------------------------------------------------
// Corrections
// ----------------------------------------------------------------------

func (s *Store) AddCorrection(ctx context.Context, pageID int64, predicted, corrected string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO corrections (page_id, predicted, corrected, created_at) VALUES (?, ?, ?, ?)`,
		pageID, predicted, corrected, now())
	return err
}

func (s *Store) CountCorrections(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM corrections`).Scan(&count)
	return count, err
}

// SetPageText records a user edit without touching the tokens.
//
// The tokens still describe the recogniser's reading and the offsets that went
// with it; once a person has edited the text those offsets are stale, and the
// page is marked as such rather than pretending the underlines still line up.
func (s *Store) SetPageText(ctx context.Context, pageID int64, text string) error {
	_, err := s.db.ExecContext(ctx, `UPDATE pages SET text = ? WHERE id = ?`, text, pageID)
	return err
}
