package store

import (
	"context"
	"database/sql"
	"errors"
)

const (
	JobQueued  = "queued"
	JobRunning = "running"
	JobDone    = "done"
	JobFailed  = "failed"

	KindPage = "page" // preprocess, recognise and rescore one page
)

type Job struct {
	ID         int64
	DocumentID int64
	PageID     int64
	Kind       string
	State      string
	Attempts   int
	Error      string
}

func (s *Store) EnqueueJob(ctx context.Context, documentID, pageID int64, kind string) (int64, error) {
	timestamp := now()
	result, err := s.db.ExecContext(ctx,
		`INSERT INTO jobs (document_id, page_id, kind, state, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		documentID, pageID, kind, JobQueued, timestamp, timestamp)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

// ClaimJob takes the oldest queued job and marks it running.
//
// The UPDATE ... WHERE state = 'queued' is the claim: two workers racing for
// the same row means one of them updates zero rows and moves on, so no job is
// ever handed out twice. That matters more than it looks, because the pool is
// the only thing between a duplicated claim and a page recognised (and paid
// for) twice.
func (s *Store) ClaimJob(ctx context.Context) (*Job, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	var job Job
	var pageID sql.NullInt64
	err = tx.QueryRowContext(ctx,
		`SELECT id, document_id, page_id, kind, state, attempts, error
		 FROM jobs WHERE state = ? ORDER BY id LIMIT 1`, JobQueued).
		Scan(&job.ID, &job.DocumentID, &pageID, &job.Kind, &job.State, &job.Attempts, &job.Error)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	job.PageID = pageID.Int64

	result, err := tx.ExecContext(ctx,
		`UPDATE jobs SET state = ?, attempts = attempts + 1, updated_at = ?
		 WHERE id = ? AND state = ?`, JobRunning, now(), job.ID, JobQueued)
	if err != nil {
		return nil, err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, nil // someone else claimed it between the select and the update
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}

	job.State = JobRunning
	job.Attempts++
	return &job, nil
}

func (s *Store) FinishJob(ctx context.Context, id int64, state, errText string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET state = ?, error = ?, updated_at = ? WHERE id = ?`,
		state, errText, now(), id)
	return err
}

// RequeueRunning puts jobs that were in flight when the process died back on
// the queue. Called once at startup; without it a crash mid-page leaves a
// document stuck at "running" forever with nothing scheduled to finish it.
func (s *Store) RequeueRunning(ctx context.Context) (int, error) {
	result, err := s.db.ExecContext(ctx,
		`UPDATE jobs SET state = ?, updated_at = ? WHERE state = ?`,
		JobQueued, now(), JobRunning)
	if err != nil {
		return 0, err
	}
	affected, err := result.RowsAffected()
	return int(affected), err
}

func (s *Store) CountJobs(ctx context.Context, documentID int64, state string) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM jobs WHERE document_id = ? AND state = ?`, documentID, state).
		Scan(&count)
	return count, err
}

func (s *Store) PendingJobs(ctx context.Context, documentID int64) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM jobs WHERE document_id = ? AND state IN (?, ?)`,
		documentID, JobQueued, JobRunning).Scan(&count)
	return count, err
}
