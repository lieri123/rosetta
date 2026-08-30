"""Evaluation harness: does the rescorer actually beat the raw API?

Two questions, and they are not the same question:

1. Error rate. Does decoding lower CER and WER against the raw recognition
   output? This is the claim that the noisy-channel model is worth having.
2. Flag quality. Do the underline tiers point at the tokens that are actually
   wrong? A scheme that underlines everything has perfect recall and is
   useless; one that underlines nothing has perfect precision and is worse.
   This is the claim that the tiers are worth looking at.

Both need reference text. Real pages come with it once you have corrected
them, and `load_pairs` reads that. Before you have a corrected corpus, the
`synthesize` path corrupts clean text through a known error model so the
harness has something to chew on -- useful for development and for sweeping
thresholds, but it measures the decoder against an error model rather than
against a recogniser, and any number produced that way is labelled synthetic
wherever it is reported.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .align import corpus_error_rate, distance, word_error_rate
from .engine import Engine
from .lexicon import tokenize
from .rescore import TIER_AMBER, TIER_NONE, TIER_RED, ScoredToken, Thresholds, Token


@dataclass
class Sample:
    """One line of text: what was written, and what the recogniser said."""

    truth: List[str]
    observed: List[Token]

    @property
    def truth_text(self) -> str:
        return " ".join(self.truth)

    @property
    def observed_text(self) -> str:
        return " ".join(token.text for token in self.observed)


@dataclass
class FlagQuality:
    """How well the underline tiers identify genuinely wrong tokens."""

    flagged_errors: int = 0
    flagged_correct: int = 0
    missed_errors: int = 0
    clean_correct: int = 0

    @property
    def precision(self) -> float:
        flagged = self.flagged_errors + self.flagged_correct
        return self.flagged_errors / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        errors = self.flagged_errors + self.missed_errors
        return self.flagged_errors / errors if errors else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalResult:
    samples: int = 0
    tokens: int = 0
    baseline_cer: float = 0.0
    rescored_cer: float = 0.0
    baseline_wer: float = 0.0
    rescored_wer: float = 0.0
    substitutions: int = 0
    substitutions_helped: int = 0
    substitutions_hurt: int = 0
    tier_counts: Dict[str, int] = field(default_factory=dict)
    flags: FlagQuality = field(default_factory=FlagQuality)
    synthetic: bool = False

    @property
    def cer_reduction(self) -> float:
        if self.baseline_cer <= 0:
            return 0.0
        return (self.baseline_cer - self.rescored_cer) / self.baseline_cer

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["flags"] = {
            "precision": round(self.flags.precision, 4),
            "recall": round(self.flags.recall, 4),
            "f1": round(self.flags.f1, 4),
            "flagged_errors": self.flags.flagged_errors,
            "flagged_correct": self.flags.flagged_correct,
            "missed_errors": self.flags.missed_errors,
            "clean_correct": self.flags.clean_correct,
        }
        data["cer_reduction"] = round(self.cer_reduction, 4)
        for key in ("baseline_cer", "rescored_cer", "baseline_wer", "rescored_wer"):
            data[key] = round(data[key], 5)
        return data

    def format_table(self) -> str:
        label = " (synthetic error model)" if self.synthetic else ""
        lines = [
            f"samples {self.samples}, tokens {self.tokens}{label}",
            "",
            f"{'metric':<24}{'baseline':>12}{'rescored':>12}{'change':>12}",
            f"{'-' * 60}",
            _row("character error rate", self.baseline_cer, self.rescored_cer),
            _row("word error rate", self.baseline_wer, self.rescored_wer),
            "",
            f"substitutions made      {self.substitutions:>6}"
            f"  (helped {self.substitutions_helped}, hurt {self.substitutions_hurt})",
            "",
            "underline tiers",
            f"  red    {self.tier_counts.get(TIER_RED, 0):>6}",
            f"  amber  {self.tier_counts.get(TIER_AMBER, 0):>6}",
            f"  none   {self.tier_counts.get(TIER_NONE, 0):>6}",
            "",
            f"flag precision {self.flags.precision:.3f}  recall {self.flags.recall:.3f}"
            f"  f1 {self.flags.f1:.3f}",
        ]
        return "\n".join(lines)


def format_adaptation(before: "EvalResult", after: "EvalResult", corrections: int) -> str:
    """Baseline, cold rescorer, and adapted rescorer, in one table.

    Three columns because there are three distinct claims, and conflating them
    would overstate the result: the API on its own, the decoder before it has
    seen any of your corrections, and the decoder after it has.
    """
    lines = [
        f"held-out samples {after.samples}, tokens {after.tokens}"
        + (" (synthetic error model)" if after.synthetic else ""),
        f"adapted on {corrections} corrections",
        "",
        f"{'metric':<24}{'baseline':>11}{'cold':>11}{'adapted':>11}{'vs baseline':>13}",
        "-" * 70,
    ]
    for name, base, cold, adapted in (
        ("character error rate", before.baseline_cer, before.rescored_cer, after.rescored_cer),
        ("word error rate", before.baseline_wer, before.rescored_wer, after.rescored_wer),
    ):
        delta = adapted - base
        verdict = "better" if delta < 0 else ("worse" if delta > 0 else "same")
        lines.append(
            f"{name:<24}{base:>11.4f}{cold:>11.4f}{adapted:>11.4f}{delta:>+12.4f} {verdict}"
        )

    lines += [
        "",
        f"substitutions {after.substitutions} "
        f"(helped {after.substitutions_helped}, hurt {after.substitutions_hurt})",
        "",
        "underline tiers   "
        f"red {after.tier_counts.get(TIER_RED, 0)}  "
        f"amber {after.tier_counts.get(TIER_AMBER, 0)}  "
        f"none {after.tier_counts.get(TIER_NONE, 0)}",
        f"flag precision {after.flags.precision:.3f}  "
        f"recall {after.flags.recall:.3f}  f1 {after.flags.f1:.3f}",
    ]
    return "\n".join(lines)


def _row(name: str, before: float, after: float) -> str:
    change = after - before
    arrow = "better" if change < 0 else ("worse" if change > 0 else "same")
    return f"{name:<24}{before:>12.4f}{after:>12.4f}{change:>+11.4f} {arrow}"


# ----------------------------------------------------------------------
# Running an evaluation
# ----------------------------------------------------------------------


def evaluate(engine: Engine, samples: Sequence[Sample], synthetic: bool = False) -> EvalResult:
    result = EvalResult(samples=len(samples), synthetic=synthetic)
    baseline_pairs: List[Tuple[str, str]] = []
    rescored_pairs: List[Tuple[str, str]] = []
    baseline_wer_total = 0.0
    rescored_wer_total = 0.0

    for sample in samples:
        scored = engine.rescore(sample.observed)
        rescored_text = " ".join(token.text for token in scored)

        baseline_pairs.append((sample.truth_text, sample.observed_text))
        rescored_pairs.append((sample.truth_text, rescored_text))
        baseline_wer_total += word_error_rate(sample.truth_text, sample.observed_text)
        rescored_wer_total += word_error_rate(sample.truth_text, rescored_text)

        _accumulate_token_stats(result, sample, scored)

    result.baseline_cer = corpus_error_rate(baseline_pairs)
    result.rescored_cer = corpus_error_rate(rescored_pairs)
    result.baseline_wer = baseline_wer_total / len(samples) if samples else 0.0
    result.rescored_wer = rescored_wer_total / len(samples) if samples else 0.0
    return result


def _accumulate_token_stats(
    result: EvalResult, sample: Sample, scored: Sequence[ScoredToken]
) -> None:
    for index, token in enumerate(scored):
        result.tokens += 1
        result.tier_counts[token.tier] = result.tier_counts.get(token.tier, 0) + 1

        # Token-level truth is only available when the recogniser did not merge
        # or split words. Where the counts differ we skip the comparison rather
        # than align badly and report a number we do not believe.
        if len(sample.truth) != len(sample.observed):
            continue
        truth = sample.truth[index]
        was_wrong = sample.observed[index].text.lower() != truth.lower()
        now_wrong = token.text.lower() != truth.lower()

        if token.substituted:
            result.substitutions += 1
            if was_wrong and not now_wrong:
                result.substitutions_helped += 1
            elif not was_wrong and now_wrong:
                result.substitutions_hurt += 1

        # A flag is judged against the state the reader sees: whether the token
        # that ends up on screen is wrong.
        flagged = token.tier != TIER_NONE
        if now_wrong and flagged:
            result.flags.flagged_errors += 1
        elif now_wrong:
            result.flags.missed_errors += 1
        elif flagged:
            result.flags.flagged_correct += 1
        else:
            result.flags.clean_correct += 1


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------

# A plausible starting error model for handwriting: shape collisions, not
# random noise. Real ones are fitted from corrections; this is only a stand-in
# for development.
DEFAULT_CONFUSIONS: Dict[str, List[str]] = {
    "a": ["o", "u"],
    "o": ["a", "e"],
    "e": ["c", "o"],
    "c": ["e"],
    "i": ["l", "e"],
    "l": ["i", "t"],
    "t": ["l", "f"],
    "n": ["r", "h"],
    "m": ["rn"],
    "rn": ["m"],
    "s": ["5"],
    "5": ["s", "S"],
    "S": ["5"],
    "u": ["v", "a"],
    "v": ["u"],
    "g": ["q", "y"],
    "h": ["b", "n"],
    "0": ["o", "O"],
    "1": ["l", "I"],
}


def synthesize(
    text: str,
    error_rate: float = 0.08,
    seed: int = 11,
    confusions: Optional[Dict[str, List[str]]] = None,
) -> List[Sample]:
    """Corrupt clean text through a character error model.

    Confidence is drawn to mirror the property that motivates the amber tier:
    most corrupted tokens come back with visibly lower confidence, but a
    deliberate minority come back looking perfectly confident. Those are the
    plausible-looking wrong words, and a system that only reads the confidence
    score cannot see them at all.
    """
    rng = random.Random(seed)
    table = confusions or DEFAULT_CONFUSIONS
    samples: List[Sample] = []

    for line in text.splitlines():
        words = tokenize(line)
        if not words:
            continue

        observed: List[Token] = []
        for word in words:
            corrupted = _corrupt_word(word, rng, table, error_rate)
            if corrupted == word:
                confidence = rng.uniform(0.88, 0.99)
            elif rng.random() < 0.25:
                # Confidently wrong: the recogniser found a clean reading, it
                # just was not the one on the page.
                confidence = rng.uniform(0.87, 0.97)
            else:
                confidence = rng.uniform(0.30, 0.80)
            observed.append(Token(text=corrupted, confidence=confidence))

        samples.append(Sample(truth=words, observed=observed))

    return samples


def _corrupt_word(
    word: str, rng: random.Random, table: Dict[str, List[str]], error_rate: float
) -> str:
    out: List[str] = []
    index = 0
    while index < len(word):
        # Two-character sources first, so `rn` -> `m` can happen at all.
        pair = word[index:index + 2]
        if pair in table and rng.random() < error_rate:
            out.append(rng.choice(table[pair]))
            index += 2
            continue

        char = word[index]
        if char in table and rng.random() < error_rate:
            out.append(rng.choice(table[char]))
        else:
            out.append(char)
        index += 1
    return "".join(out)


def split_samples(
    samples: Sequence[Sample], train_fraction: float = 0.5, seed: int = 3
) -> Tuple[List[Sample], List[Sample]]:
    """Split into a set to learn from and a set to measure on.

    Held out, and shuffled first so the split does not fall along whatever
    order the source text happened to be in. Measuring on lines the confusion
    matrix was fitted to would report the model's memory, not its adaptation.
    """
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * train_fraction)
    train = [samples[i] for i in indices[:cut]]
    test = [samples[i] for i in indices[cut:]]
    return train, test


def correction_pairs(samples: Sequence[Sample]) -> List[Tuple[str, str]]:
    """Turn samples into the (predicted, corrected) pairs the UI would produce.

    Whole lines, not aligned token pairs: that is what a person actually hands
    back when they fix a line in the editor, and the alignment that extracts
    character-level evidence from it is the model's job, not the caller's.
    """
    return [
        (sample.observed_text, sample.truth_text)
        for sample in samples
        if sample.observed_text != sample.truth_text
    ]


def load_pairs(path: str) -> List[Sample]:
    """Read evaluation data as JSON lines.

    Each line: {"truth": "...", "observed": [{"text": "...", "confidence": 0.9}]}
    or the shorthand {"truth": "...", "observed": "..."} when no per-token
    confidence is available, in which case every token is treated as certain --
    a deliberately unflattering assumption for the rescorer.
    """
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            truth = tokenize(record["truth"])
            raw = record["observed"]
            if isinstance(raw, str):
                observed = [Token(text=word, confidence=1.0) for word in tokenize(raw)]
            else:
                observed = []
                for item in raw:
                    confidence = float(item.get("confidence", 1.0))
                    observed.append(
                        Token(
                            text=item["text"],
                            confidence=confidence,
                            alternatives=Token.parse_alternatives(
                                item.get("alternatives", []), confidence
                            ),
                        )
                    )
            samples.append(Sample(truth=truth, observed=observed))
    return samples


# ----------------------------------------------------------------------
# Threshold sweeps
# ----------------------------------------------------------------------


def sweep_substitution_margin(
    engine: Engine, samples: Sequence[Sample], values: Sequence[float]
) -> List[Tuple[float, float]]:
    """CER as a function of how decisive a substitution has to be.

    The shape of this curve is the argument for having a margin at all: at zero
    the decoder rewrites on any preference and CER goes up, and far to the
    right it never rewrites and CER returns to baseline. The useful setting is
    the trough, and it is not at either end.
    """
    original = engine.thresholds
    curve: List[Tuple[float, float]] = []
    try:
        for value in values:
            engine.thresholds = Thresholds(
                low_confidence=original.low_confidence,
                settled_confidence=original.settled_confidence,
                surprisal_bits=original.surprisal_bits,
                close_call_nats=original.close_call_nats,
                substitute_nats=value,
                prior_weight=original.prior_weight,
                max_candidates=original.max_candidates,
            )
            curve.append((value, evaluate(engine, samples).rescored_cer))
    finally:
        engine.thresholds = original
    return curve


def plot_curve(curve: Sequence[Tuple[float, float]], path: str, baseline: Optional[float] = None) -> bool:
    """Write the CER curve to a PNG. Returns False if matplotlib is absent.

    Plotting is the one place a dependency is allowed to be optional: the
    numbers are the deliverable and the picture is a convenience.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    xs = [point[0] for point in curve]
    ys = [point[1] for point in curve]

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(xs, ys, marker="o", linewidth=2, label="rescored")
    if baseline is not None:
        axes.axhline(baseline, linestyle="--", linewidth=1.5, color="#b04a3a", label="baseline")
    axes.set_xlabel("substitution margin (nats)")
    axes.set_ylabel("character error rate")
    axes.set_title("CER against substitution margin")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True
