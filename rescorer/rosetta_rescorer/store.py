"""SQLite persistence for the learned models.

The whole system shares one database file. The Go service owns documents,
pages, tokens and corrections; this module owns the four tables the rescorer
learns into. Splitting the file would mean two things to back up and a
consistency question at every correction; splitting the schema inside one file
costs nothing and keeps ownership obvious.

Models are written by replacing their tables wholesale inside a transaction.
At personal scale that is a few thousand rows and a few milliseconds, and it
avoids a class of bug that incremental upserts invite: a partially applied
update leaving counts that no sequence of corrections could have produced.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .confusion import ConfusionModel
from .lexicon import Lexicon
from .lm import LanguageModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS rescorer_confusion (
    kind  TEXT NOT NULL,           -- sub | del | ins | span
    a     TEXT NOT NULL,
    b     TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (kind, a, b)
);

CREATE TABLE IF NOT EXISTS rescorer_lexicon (
    word  TEXT PRIMARY KEY,
    count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rescorer_ngrams (
    n       INTEGER NOT NULL,
    context TEXT NOT NULL,
    word    TEXT NOT NULL,
    count   INTEGER NOT NULL,
    PRIMARY KEY (n, context, word)
);

CREATE TABLE IF NOT EXISTS rescorer_char_ngrams (
    context TEXT NOT NULL,
    ch      TEXT NOT NULL,
    count   INTEGER NOT NULL,
    PRIMARY KEY (context, ch)
);

-- Every correction ever seen, kept verbatim. The models above are derived
-- state and can be rebuilt from this; that is worth the disk it costs.
CREATE TABLE IF NOT EXISTS rescorer_corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted  TEXT NOT NULL,
    corrected  TEXT NOT NULL,
    page_id    INTEGER,
    created_at REAL NOT NULL
);
"""


@dataclass
class Models:
    confusion: ConfusionModel
    lexicon: Lexicon
    language_model: LanguageModel


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False)
        # WAL so the Go service can read while we write. Both processes touch
        # the same file; neither should ever block the other on a read.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    # ------------------------------------------------------------------

    def load(self) -> Models:
        cursor = self._connection.cursor()

        confusion_rows = cursor.execute(
            "SELECT kind, a, b, count FROM rescorer_confusion"
        ).fetchall()
        lexicon_rows = cursor.execute("SELECT word, count FROM rescorer_lexicon").fetchall()
        ngram_rows = cursor.execute(
            "SELECT n, context, word, count FROM rescorer_ngrams"
        ).fetchall()
        char_rows = cursor.execute(
            "SELECT context, ch, count FROM rescorer_char_ngrams"
        ).fetchall()

        return Models(
            confusion=ConfusionModel.from_rows(confusion_rows),
            lexicon=Lexicon.from_rows(lexicon_rows),
            language_model=LanguageModel.from_rows(ngram_rows, char_rows),
        )

    def save(self, models: Models) -> None:
        word_rows, char_rows = models.language_model.to_rows()
        with self._connection:
            cursor = self._connection.cursor()
            cursor.execute("DELETE FROM rescorer_confusion")
            cursor.executemany(
                "INSERT INTO rescorer_confusion (kind, a, b, count) VALUES (?, ?, ?, ?)",
                models.confusion.to_rows(),
            )
            cursor.execute("DELETE FROM rescorer_lexicon")
            cursor.executemany(
                "INSERT INTO rescorer_lexicon (word, count) VALUES (?, ?)",
                models.lexicon.to_rows(),
            )
            cursor.execute("DELETE FROM rescorer_ngrams")
            cursor.executemany(
                "INSERT INTO rescorer_ngrams (n, context, word, count) VALUES (?, ?, ?, ?)",
                word_rows,
            )
            cursor.execute("DELETE FROM rescorer_char_ngrams")
            cursor.executemany(
                "INSERT INTO rescorer_char_ngrams (context, ch, count) VALUES (?, ?, ?)",
                char_rows,
            )

    # ------------------------------------------------------------------

    def log_correction(self, predicted: str, corrected: str, page_id: Optional[int] = None) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO rescorer_corrections (predicted, corrected, page_id, created_at)"
                " VALUES (?, ?, ?, ?)",
                (predicted, corrected, page_id, time.time()),
            )

    def corrections(self, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        query = "SELECT predicted, corrected FROM rescorer_corrections ORDER BY id"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        return self._connection.execute(query).fetchall()

    def correction_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM rescorer_corrections"
        ).fetchone()
        return int(row[0]) if row else 0
