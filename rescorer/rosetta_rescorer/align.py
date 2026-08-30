"""Levenshtein alignment.

Every correction made in the UI is a pair (predicted, corrected). The pair on
its own says only "this was wrong"; aligned character by character it says
which character became which, which is the raw material for the confusion
matrix. This module does that alignment and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

MATCH = "match"
SUBSTITUTE = "sub"
DELETE = "del"      # a character in the truth that the recogniser dropped
INSERT = "ins"      # a character the recogniser invented


@dataclass(frozen=True)
class EditOp:
    """One aligned position.

    `source` is the recogniser's output and `target` is the truth, so a
    substitution reads "the recogniser saw `source` where `target` was
    written". Deletions carry an empty `source`, insertions an empty `target`.
    """

    kind: str
    source: str
    target: str
    source_index: int
    target_index: int

    @property
    def is_error(self) -> bool:
        return self.kind != MATCH


def distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance with unit costs.

    Two rows rather than the full matrix: the caller that only wants a number
    (candidate filtering, CER accumulation) runs this over every lexicon entry,
    and the full matrix is only needed when a traceback is wanted.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution or match
            )
        previous = current
    return previous[-1]


def bounded_distance(a: Sequence[str], b: Sequence[str], limit: int) -> int:
    """Levenshtein distance, giving up once it is certain to exceed `limit`.

    Candidate generation asks "is this lexicon word within two edits" thousands
    of times per page, and the answer is almost always no. Bailing out on the
    length difference and on a whole row exceeding the limit turns most of
    those questions into a few comparisons.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
            row_min = min(row_min, current[j])
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def align(source: str, target: str) -> List[EditOp]:
    """Align two strings, returning one EditOp per aligned position.

    Ties are broken towards substitution over an insert/delete pair, which
    keeps the confusion counts interpretable: a one-for-one character swap is
    recorded as a swap rather than as an unrelated drop and invention.
    """
    n, m = len(source), len(target)
    # cost[i][j] = distance between source[:i] and target[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (source[i - 1] != target[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )

    ops: List[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = cost[i - 1][j - 1] + (source[i - 1] != target[j - 1])
            if cost[i][j] == sub_cost:
                kind = MATCH if source[i - 1] == target[j - 1] else SUBSTITUTE
                ops.append(EditOp(kind, source[i - 1], target[j - 1], i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            # The recogniser produced a character that is not in the truth.
            ops.append(EditOp(INSERT, source[i - 1], "", i - 1, j))
            i -= 1
            continue
        # The truth has a character the recogniser did not produce.
        ops.append(EditOp(DELETE, "", target[j - 1], i, j - 1))
        j -= 1

    ops.reverse()
    return ops


def error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate of `hypothesis` against `reference`."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return distance(hypothesis, reference) / len(reference)


def corpus_error_rate(pairs: Iterable[Tuple[str, str]]) -> float:
    """CER over a corpus, pooled rather than averaged per line.

    Averaging per-line rates over-weights short lines: a one-character mistake
    in a two-character line would count as much as ten mistakes in a full
    paragraph. Pooling the edits and the reference length is the standard
    definition and the one the eval harness reports.
    """
    edits = 0
    length = 0
    for reference, hypothesis in pairs:
        edits += distance(hypothesis, reference)
        length += len(reference)
    return edits / length if length else 0.0


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER, computed by aligning token sequences rather than characters."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return distance(hyp_words, ref_words) / len(ref_words)
