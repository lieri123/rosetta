"""The error model: P(observed | true), fitted from your own corrections.

Every correction in the UI yields an aligned pair, and every aligned pair
yields character-level evidence about how this recogniser mangles this
handwriting. Accumulated, that is a confusion matrix, and it is personal: it
will say that your `a` reads as `o` fourteen percent of the time, that your
`rn` collapses into `m`, that your `5` and your `S` are indistinguishable.

Two views of the same evidence are kept, because they answer different
questions:

* Character-level counts drive the channel score. Given a candidate string,
  how likely is it that the recogniser would have produced what we actually
  saw? That is a Viterbi alignment over substitution, deletion and insertion
  log-probabilities.
* Span-level counts drive candidate generation. `rn` -> `m` is not a character
  substitution and cannot be represented as one; recording the whole error
  region lets us propose `modern` when we see `rnodern`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .align import DELETE, INSERT, MATCH, SUBSTITUTE, EditOp, align

# Add-k smoothing constant. Small: with a personal corpus the counts are
# sparse and heavy smoothing would flatten exactly the structure we want.
SMOOTHING = 0.05

# Pseudo-count on the diagonal, applied before any correction has been made.
# Without it a freshly installed system has a uniform channel -- every string
# equally likely to have produced every other -- and the language prior would
# happily rewrite correct text into commoner words. With it, an untrained
# rescorer behaves like plain edit distance and gets out of the way, and real
# counts overwhelm it after a few dozen corrections.
DIAGONAL_PRIOR = 25.0

# Prior belief about how often the recogniser invents a character out of
# nothing. Insertions cannot be conditioned on a true character (there is
# none), so they get a global rate refined by observation.
INSERTION_PRIOR_RATE = 0.02

MAX_SPAN = 4  # longest error region recorded as a span confusion


@dataclass
class ConfusionModel:
    """Counts plus the probabilities derived from them."""

    # (true_char, observed_char) -> count, including correctly read characters
    substitutions: Counter = field(default_factory=Counter)
    # true_char -> times the recogniser dropped it entirely
    deletions: Counter = field(default_factory=Counter)
    # observed_char -> times the recogniser produced it from nothing
    insertions: Counter = field(default_factory=Counter)
    # (observed_span, true_span) -> count, for multi-character confusions
    spans: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._true_totals: Counter = Counter()
        self._rebuild_totals()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def observe(self, predicted: str, corrected: str) -> List[EditOp]:
        """Fold one correction into the model. Returns the alignment used."""
        ops = align(predicted, corrected)
        for op in ops:
            if op.kind in (MATCH, SUBSTITUTE):
                self.substitutions[(op.target, op.source)] += 1
            elif op.kind == DELETE:
                self.deletions[op.target] += 1
            elif op.kind == INSERT:
                self.insertions[op.source] += 1

        for observed_span, true_span in self._error_spans(ops):
            self.spans[(observed_span, true_span)] += 1

        self._rebuild_totals()
        return ops

    def observe_many(self, pairs: Iterable[Tuple[str, str]]) -> int:
        count = 0
        for predicted, corrected in pairs:
            self.observe(predicted, corrected)
            count += 1
        return count

    @staticmethod
    def _error_spans(ops: Sequence[EditOp]) -> List[Tuple[str, str]]:
        """Group the alignment into maximal runs of consecutive errors.

        A run is the unit that actually explains a multi-character confusion:
        aligning `rnodern` to `modern` gives an insert and a substitution side
        by side, and only together do they say `rn` was read where `m` was
        written.
        """
        spans: List[Tuple[str, str]] = []
        observed: List[str] = []
        truth: List[str] = []

        for op in ops:
            if op.is_error:
                observed.append(op.source)
                truth.append(op.target)
                continue
            if observed or truth:
                spans.append(("".join(observed), "".join(truth)))
                observed, truth = [], []

        if observed or truth:
            spans.append(("".join(observed), "".join(truth)))

        return [
            (o, t)
            for o, t in spans
            if (o or t) and len(o) <= MAX_SPAN and len(t) <= MAX_SPAN
        ]

    def _rebuild_totals(self) -> None:
        totals: Counter = Counter()
        for (true_char, _observed), count in self.substitutions.items():
            totals[true_char] += count
        for true_char, count in self.deletions.items():
            totals[true_char] += count
        self._true_totals = totals

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------

    @property
    def alphabet(self) -> Set[str]:
        chars = {t for t, _ in self.substitutions}
        chars |= {o for _, o in self.substitutions}
        chars |= set(self.deletions)
        chars |= set(self.insertions)
        return chars

    @property
    def alphabet_size(self) -> int:
        # A floor keeps smoothing sane on an empty model.
        return max(len(self.alphabet), 64)

    @property
    def correction_count(self) -> int:
        return int(sum(self.spans.values()))

    def _denominator(self, true_char: str) -> float:
        vocab = self.alphabet_size
        return (
            self._true_totals.get(true_char, 0)
            + DIAGONAL_PRIOR
            + SMOOTHING * (vocab + 1)
        )

    def log_substitution(self, true_char: str, observed_char: str) -> float:
        """log P(the recogniser output `observed_char` | `true_char` written)."""
        count = self.substitutions.get((true_char, observed_char), 0) + SMOOTHING
        if true_char == observed_char:
            count += DIAGONAL_PRIOR
        return math.log(count) - math.log(self._denominator(true_char))

    def log_deletion(self, true_char: str) -> float:
        """log P(the recogniser dropped `true_char`)."""
        count = self.deletions.get(true_char, 0) + SMOOTHING
        return math.log(count) - math.log(self._denominator(true_char))

    def log_insertion(self, observed_char: str) -> float:
        """log P(the recogniser invented `observed_char`).

        There is no true character to condition on, so this is a global
        insertion rate multiplied by the distribution over what tends to get
        inserted. Approximate, but insertions are rare enough that the
        approximation costs little and the alternative (an epsilon symbol
        threaded through the alignment) buys nothing here.
        """
        total_insertions = sum(self.insertions.values())
        total_observed = sum(self.substitutions.values()) + total_insertions
        vocab = self.alphabet_size

        rate = (total_insertions + INSERTION_PRIOR_RATE * max(total_observed, 1)) / (
            max(total_observed, 1) * (1.0 + INSERTION_PRIOR_RATE)
        )
        rate = min(max(rate, 1e-4), 0.5)

        share = (self.insertions.get(observed_char, 0) + SMOOTHING) / (
            total_insertions + SMOOTHING * vocab
        )
        return math.log(rate) + math.log(share)

    def log_channel(self, observed: str, candidate: str) -> float:
        """log P(observed | candidate), maximised over alignments.

        Viterbi rather than a sum over all alignments: the best alignment
        carries almost all the mass for strings this short, and the max keeps
        the score interpretable as "the single most likely way the recogniser
        could have produced this".
        """
        n, m = len(observed), len(candidate)
        neg_inf = float("-inf")

        previous = [0.0] + [neg_inf] * 0
        # dp[j] over candidate prefix for the current observed prefix
        dp = [0.0] * (m + 1)
        for j in range(1, m + 1):
            dp[j] = dp[j - 1] + self.log_deletion(candidate[j - 1])

        for i in range(1, n + 1):
            current = [dp[0] + self.log_insertion(observed[i - 1])] + [neg_inf] * m
            for j in range(1, m + 1):
                best = dp[j - 1] + self.log_substitution(candidate[j - 1], observed[i - 1])
                via_insert = dp[j] + self.log_insertion(observed[i - 1])
                if via_insert > best:
                    best = via_insert
                via_delete = current[j - 1] + self.log_deletion(candidate[j - 1])
                if via_delete > best:
                    best = via_delete
                current[j] = best
            dp = current

        del previous
        return dp[m]

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def likely_sources(self, observed_char: str, limit: int = 4) -> List[Tuple[str, float]]:
        """True characters that plausibly produced `observed_char`.

        This is the matrix read backwards, which is the direction candidate
        generation needs: we have the output and want the inputs worth trying.
        """
        scored = []
        for (true_char, obs), count in self.substitutions.items():
            if obs != observed_char or true_char == observed_char or count == 0:
                continue
            scored.append((true_char, self.log_substitution(true_char, observed_char)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def inverse_variants(self, token: str, max_variants: int = 48) -> List[str]:
        """Strings that could have been written given `token` was observed.

        Driven entirely by what has actually been confused before, so it stays
        small and relevant. This is the path that recovers words no lexicon
        can help with -- names, jargon, notation you invented last week.
        """
        variants: Set[str] = set()

        # Single-character reversals, from the matrix read backwards.
        for index, char in enumerate(token):
            for source, _score in self.likely_sources(char):
                variants.add(token[:index] + source + token[index + 1:])
                if len(variants) >= max_variants:
                    return sorted(variants)

        # Whole-span reversals: `m` back to `rn`, `5` back to `S`, and any
        # other collapse or split the corrections have shown.
        span_by_observed: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for (observed_span, true_span), count in self.spans.items():
            if observed_span:
                span_by_observed[observed_span].append((true_span, count))

        for observed_span, replacements in span_by_observed.items():
            start = token.find(observed_span)
            while start != -1:
                for true_span, _count in sorted(
                    replacements, key=lambda item: item[1], reverse=True
                )[:3]:
                    variants.add(
                        token[:start] + true_span + token[start + len(observed_span):]
                    )
                    if len(variants) >= max_variants:
                        return sorted(variants)
                start = token.find(observed_span, start + 1)

        variants.discard(token)
        return sorted(variants)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def top_confusions(self, limit: int = 20) -> List[Tuple[str, str, int, float]]:
        """The most frequent genuine errors, worst first.

        Sorted by count rather than rate: a character misread once out of one
        occurrence has a rate of 1.0 and tells you nothing.
        """
        rows = []
        for (true_char, observed_char), count in self.substitutions.items():
            if true_char == observed_char:
                continue
            total = self._true_totals.get(true_char, 0)
            rate = count / total if total else 0.0
            rows.append((true_char, observed_char, count, rate))
        rows.sort(key=lambda row: (row[2], row[3]), reverse=True)
        return rows[:limit]

    def to_rows(self) -> List[Tuple[str, str, str, int]]:
        """Flatten to (kind, a, b, count) for storage."""
        rows: List[Tuple[str, str, str, int]] = []
        rows += [("sub", t, o, c) for (t, o), c in self.substitutions.items()]
        rows += [("del", t, "", c) for t, c in self.deletions.items()]
        rows += [("ins", "", o, c) for o, c in self.insertions.items()]
        rows += [("span", o, t, c) for (o, t), c in self.spans.items()]
        return rows

    @classmethod
    def from_rows(cls, rows: Iterable[Tuple[str, str, str, int]]) -> "ConfusionModel":
        model = cls()
        for kind, a, b, count in rows:
            if kind == "sub":
                model.substitutions[(a, b)] += count
            elif kind == "del":
                model.deletions[a] += count
            elif kind == "ins":
                model.insertions[b] += count
            elif kind == "span":
                model.spans[(a, b)] += count
        model._rebuild_totals()
        return model
