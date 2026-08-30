"""The personal lexicon: the words you actually use.

A general dictionary is the wrong prior for private notes. It does not contain
your colleagues' names, your project codenames, the notation you invented last
week, or the abbreviations you use only in your own margins -- and it does
contain a hundred thousand words you will never write, every one of them a
chance for the rescorer to "correct" something that was already right.

So the lexicon is built from corrected text: words are only trusted once they
have survived your own review. A system word list can be loaded as a weak
background, but personal counts always dominate.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .align import bounded_distance

# Words, including internal apostrophes and hyphens: "don't" and "noisy-channel"
# are single tokens, and splitting them would teach the model nonsense.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

# Weight given to a word that only appears in a background word list. Low
# enough that a single appearance in your own corrected writing outweighs it.
BACKGROUND_WEIGHT = 1


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def match_case(source: str, target: str) -> str:
    """Give `target` the capitalisation pattern of `source`.

    Candidates are generated and compared in lower case so that "Modern" and
    "modern" share their counts, but the substitution written back into the
    document has to look like what it replaced.
    """
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


@dataclass
class Lexicon:
    counts: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        # Length buckets: candidate lookup only ever asks about words within a
        # couple of edits, and length differs by at most that much. Scanning
        # three buckets instead of the whole lexicon is the difference between
        # a snappy page and a visible pause.
        self._by_length: Dict[int, Set[str]] = defaultdict(set)
        for word in self.counts:
            self._by_length[len(word)].add(word)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_word(self, word: str, count: int = 1) -> None:
        word = word.lower()
        if not word:
            return
        self.counts[word] += count
        self._by_length[len(word)].add(word)

    def add_text(self, text: str, count: int = 1) -> int:
        words = tokenize(text)
        for word in words:
            self.add_word(word, count)
        return len(words)

    def add_background(self, words: Iterable[str]) -> int:
        added = 0
        for word in words:
            word = word.strip().lower()
            if word and word.isalpha():
                self.add_word(word, BACKGROUND_WEIGHT)
                added += 1
        return added

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.counts

    def __len__(self) -> int:
        return len(self.counts)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def count(self, word: str) -> int:
        return self.counts.get(word.lower(), 0)

    def candidates(
        self, token: str, max_edits: int = 2, limit: int = 64
    ) -> List[Tuple[str, int]]:
        """Lexicon entries within `max_edits` of `token`, commonest first.

        A linear scan over the plausible length buckets. At personal-corpus
        scale -- tens of thousands of words -- that is a few milliseconds, and
        it keeps the index a plain dictionary. If the lexicon ever grew by two
        orders of magnitude this is where a deletion-neighbourhood index would
        go.
        """
        needle = token.lower()
        if not needle:
            return []

        # Short tokens are mostly function words and abbreviations; allowing two
        # edits on a three-letter token makes half the lexicon a candidate.
        if len(needle) <= 4:
            max_edits = min(max_edits, 1)

        found: List[Tuple[str, int]] = []
        for length in range(len(needle) - max_edits, len(needle) + max_edits + 1):
            for word in self._by_length.get(length, ()):
                if word == needle:
                    continue
                if bounded_distance(needle, word, max_edits) <= max_edits:
                    found.append((word, self.counts[word]))

        found.sort(key=lambda item: item[1], reverse=True)
        return found[:limit]

    def most_common(self, limit: int = 20) -> List[Tuple[str, int]]:
        return self.counts.most_common(limit)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def to_rows(self) -> List[Tuple[str, int]]:
        return list(self.counts.items())

    @classmethod
    def from_rows(cls, rows: Iterable[Tuple[str, int]]) -> "Lexicon":
        lexicon = cls()
        for word, count in rows:
            lexicon.add_word(word, count)
        return lexicon


def load_word_list(path: str, limit: Optional[int] = None) -> List[str]:
    """Read a newline-delimited word list, e.g. /usr/share/dict/words."""
    words: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            word = line.strip()
            if word:
                words.append(word)
            if limit is not None and len(words) >= limit:
                break
    return words
