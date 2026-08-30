"""The language prior: P(true text), learned from your own writing.

Two models in one, because they cover for each other's blind spot:

* An interpolated word trigram. It knows that "the meeting" is likelier than
  "the meetlng", and -- the part a spell checker cannot do -- that "form" is
  wrong in "fill in the form field" only if you never write that. Context is
  what catches plausible-looking wrong words, which are exactly the ones a
  confidence score alone will not flag.

* A character n-gram over the same text. This is what stops the word model
  from being tyrannical. A word model alone assigns probability zero to
  anything it has never seen, so the rescorer would rewrite every new name and
  every piece of jargon into the nearest familiar word. The character model
  says "this is not a word I know, but it is spelled the way this person
  spells things", which is enough to leave it alone.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .lexicon import tokenize

BOUNDARY = "\x02"  # sentence start/end marker, never a real character

# Interpolation weights, highest order first, then the character model. The
# character model's share is deliberately small: it is a fallback for unseen
# words, not a competitor to real context.
LAMBDA_TRIGRAM = 0.55
LAMBDA_BIGRAM = 0.28
LAMBDA_UNIGRAM = 0.13
LAMBDA_CHAR = 0.04

CHAR_ORDER = 4
CHAR_SMOOTHING = 0.1
UNIGRAM_SMOOTHING = 0.5


@dataclass
class LanguageModel:
    unigrams: Counter = field(default_factory=Counter)
    bigrams: Counter = field(default_factory=Counter)
    trigrams: Counter = field(default_factory=Counter)
    char_ngrams: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._rebuild_contexts()

    def _rebuild_contexts(self) -> None:
        self._bigram_context: Counter = Counter()
        for (previous, _word), count in self.bigrams.items():
            self._bigram_context[previous] += count

        self._trigram_context: Counter = Counter()
        for (first, second, _word), count in self.trigrams.items():
            self._trigram_context[(first, second)] += count

        self._char_context: Counter = Counter()
        for (context, _char), count in self.char_ngrams.items():
            self._char_context[context] += count

        self._total_words = int(sum(self.unigrams.values()))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def add_text(self, text: str) -> int:
        """Fold a piece of corrected text into the model."""
        added = 0
        for line in text.splitlines():
            words = [word.lower() for word in tokenize(line)]
            if not words:
                continue
            added += len(words)

            padded = [BOUNDARY, BOUNDARY] + words + [BOUNDARY]
            for index in range(2, len(padded)):
                word = padded[index]
                self.unigrams[word] += 1
                self.bigrams[(padded[index - 1], word)] += 1
                self.trigrams[(padded[index - 2], padded[index - 1], word)] += 1

            for word in words:
                self._add_word_chars(word)

        self._rebuild_contexts()
        return added

    def _add_word_chars(self, word: str) -> None:
        padded = BOUNDARY * (CHAR_ORDER - 1) + word + BOUNDARY
        for index in range(CHAR_ORDER - 1, len(padded)):
            context = padded[index - CHAR_ORDER + 1:index]
            self.char_ngrams[(context, padded[index])] += 1

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:
        return max(len(self.unigrams), 1)

    @property
    def total_words(self) -> int:
        return self._total_words

    def char_log_prob(self, word: str) -> float:
        """log P(word) under the character model.

        Not a proper distribution over the whole string space -- it does not
        normalise across lengths -- but it is consistently computed for every
        candidate, and it is only ever used to compare candidates against each
        other.
        """
        word = word.lower()
        if not word:
            return math.log(1e-12)

        padded = BOUNDARY * (CHAR_ORDER - 1) + word + BOUNDARY
        total = 0.0
        alphabet = max(len({char for _context, char in self.char_ngrams}), 32)

        for index in range(CHAR_ORDER - 1, len(padded)):
            context = padded[index - CHAR_ORDER + 1:index]
            char = padded[index]
            numerator = self.char_ngrams.get((context, char), 0) + CHAR_SMOOTHING
            denominator = self._char_context.get(context, 0) + CHAR_SMOOTHING * alphabet
            total += math.log(numerator) - math.log(denominator)

        # Long words accumulate more negative log-probability simply for being
        # long, which would bias the rescorer towards shorter candidates. A
        # per-character average, rescaled to a typical word, removes the bias.
        return total / max(len(word), 1) * 5.0

    def log_prob(self, word: str, context: Sequence[str] = ()) -> float:
        """log P(word | context) under the interpolated model."""
        word = word.lower()
        history = [item.lower() for item in context][-2:]
        while len(history) < 2:
            history.insert(0, BOUNDARY)
        first, second = history[0], history[1]

        unigram = (self.unigrams.get(word, 0) + UNIGRAM_SMOOTHING) / (
            self._total_words + UNIGRAM_SMOOTHING * self.vocabulary_size
        )

        bigram_context = self._bigram_context.get(second, 0)
        bigram = self.bigrams.get((second, word), 0) / bigram_context if bigram_context else 0.0

        trigram_context = self._trigram_context.get((first, second), 0)
        trigram = (
            self.trigrams.get((first, second, word), 0) / trigram_context
            if trigram_context
            else 0.0
        )

        char = math.exp(self.char_log_prob(word))

        mixed = (
            LAMBDA_TRIGRAM * trigram
            + LAMBDA_BIGRAM * bigram
            + LAMBDA_UNIGRAM * unigram
            + LAMBDA_CHAR * char
        )
        return math.log(max(mixed, 1e-300))

    def surprisal(self, word: str, context: Sequence[str] = ()) -> float:
        """How surprising this word is here, in bits."""
        return -self.log_prob(word, context) / math.log(2.0)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def to_rows(self) -> Tuple[List[Tuple[int, str, str, int]], List[Tuple[str, str, int]]]:
        word_rows: List[Tuple[int, str, str, int]] = []
        word_rows += [(1, "", word, count) for word, count in self.unigrams.items()]
        word_rows += [(2, previous, word, count) for (previous, word), count in self.bigrams.items()]
        word_rows += [
            (3, f"{first}\x1f{second}", word, count)
            for (first, second, word), count in self.trigrams.items()
        ]
        char_rows = [(context, char, count) for (context, char), count in self.char_ngrams.items()]
        return word_rows, char_rows

    @classmethod
    def from_rows(
        cls,
        word_rows: Iterable[Tuple[int, str, str, int]],
        char_rows: Iterable[Tuple[str, str, int]],
    ) -> "LanguageModel":
        model = cls()
        for order, context, word, count in word_rows:
            if order == 1:
                model.unigrams[word] += count
            elif order == 2:
                model.bigrams[(context, word)] += count
            elif order == 3:
                first, _, second = context.partition("\x1f")
                model.trigrams[(first, second, word)] += count
        for context, char, count in char_rows:
            model.char_ngrams[(context, char)] += count
        model._rebuild_contexts()
        return model
