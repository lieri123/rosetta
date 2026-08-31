-- One SQLite file holds everything: documents, pages, tokens, corrections and
-- the job queue, alongside the tables the Python rescorer learns into. At the
-- scale this is built for -- one person's notes over years -- Postgres would be
-- ceremony, and a single file is one thing to back up.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    source_name TEXT    NOT NULL,
    status      TEXT    NOT NULL,            -- pending | running | done | failed
    error       TEXT    NOT NULL DEFAULT '',
    page_count  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index  INTEGER NOT NULL,
    source_path TEXT    NOT NULL,
    clean_path  TEXT    NOT NULL DEFAULT '',
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL,
    error       TEXT    NOT NULL DEFAULT '',
    text        TEXT    NOT NULL DEFAULT '',
    skew_deg    REAL    NOT NULL DEFAULT 0,
    rectified   INTEGER NOT NULL DEFAULT 0,
    UNIQUE (document_id, page_index)
);

-- One row per recognised word, carrying both where it sits on the page and
-- where it sits in the assembled text. The character offsets are what the
-- editor decorates; keeping them here means the service is the single source
-- of truth for how text and underlines line up, and the browser never has to
-- re-tokenise and hope it agrees.
CREATE TABLE IF NOT EXISTS tokens (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id      INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    original     TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    tier         TEXT    NOT NULL,           -- none | amber | red
    reason       TEXT    NOT NULL DEFAULT '',
    suggestion   TEXT    NOT NULL DEFAULT '',
    start_offset INTEGER NOT NULL,
    end_offset   INTEGER NOT NULL,
    x0           REAL    NOT NULL DEFAULT 0,
    y0           REAL    NOT NULL DEFAULT 0,
    x1           REAL    NOT NULL DEFAULT 0,
    y1           REAL    NOT NULL DEFAULT 0,
    line_index   INTEGER NOT NULL DEFAULT 0,
    para_index   INTEGER NOT NULL DEFAULT 0,
    struck       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS tokens_by_page ON tokens (page_id, idx);

CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id    INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    predicted  TEXT    NOT NULL,
    corrected  TEXT    NOT NULL,
    created_at REAL    NOT NULL
);

-- The queue lives in the database rather than only in memory so that a crash
-- mid-page is recoverable: anything left running when the process died is
-- requeued at startup instead of being silently lost.
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id     INTEGER,
    kind        TEXT    NOT NULL,
    state       TEXT    NOT NULL,            -- queued | running | done | failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs (state, id);
