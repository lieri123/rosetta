"""The decoder: pick the most likely true text, and say how sure we are.

For each token the recogniser gives us, we assemble candidates for what was
actually written, score each one as

    log P(observed | candidate)  +  alpha * log P(candidate | context)

and take the best. The first term comes from the confusion model, the second
from the language model. Substitution only happens when the winner beats what
the recogniser said by a decisive margin -- a rescorer that rewrites on a
hairline is worse than no rescorer, because it replaces errors you can see
with errors you cannot.

The same machinery produces the underline tiers, which is the point: the
decision of what to flag falls out of the decoding rather than being a second,
separately-tuned heuristic.

    red    low recognition confidence -- probably garbage
    amber  decent confidence, but the language model finds the token
           improbable here, or the rescorer had a close call between two
           candidates, or it substituted something
    none   high confidence and unsurprising

The amber tier is the one that earns its keep. Recognition confidence alone
cannot flag a plausible-looking wrong word, because the recogniser is
confident about it; only the context model notices that it does not belong.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .confusion import ConfusionModel
from .lexicon import Lexicon, match_case
from .lm import LanguageModel

TIER_NONE = "none"
TIER_AMBER = "amber"
TIER_RED = "red"

# Tokens that are not words: punctuation, bare numbers, symbols. The language
# prior has nothing useful to say about them and the lexicon should not try to
# "correct" them, so they pass through on recognition confidence alone.
NON_WORD_RE = re.compile(r"^[^A-Za-z]*$")


@dataclass
class Thresholds:
    """Every tuneable in one place, so the eval harness can sweep them."""

    # Below this recognition confidence the token is red whatever else we think.
    low_confidence: float = 0.55
    # At or above this, the recogniser is trusted unless the language model
    # actively objects.
    settled_confidence: float = 0.86
    # Above this surprisal (in bits) a token is improbable enough in context to
    # be worth a second look, even at high recognition confidence.
    surprisal_bits: float = 15.0
    # If the top two candidates are within this many nats, the rescorer had a
    # close call and should say so rather than silently picking one.
    close_call_nats: float = 1.5
    # A candidate must beat the recogniser's own output by this much before it
    # replaces it. Swept by the eval harness: on synthetic corruptions of the
    # sample corpus, substitutions helped 14 times and hurt zero across three
    # seeds, and the CER curve rises monotonically from here -- 3.0, the
    # instinctive setting, left two thirds of the good corrections unmade. Low
    # but not zero: the margin still has to mean something, and this is fitted
    # against a synthetic error model, so re-run the sweep once real
    # corrections exist.
    substitute_nats: float = 1.0
    # Weight on the language prior relative to the channel model.
    prior_weight: float = 1.0
    # Candidates considered per token, after ranking by prior plausibility.
    max_candidates: int = 40


@dataclass
class Alternative:
    """A runner-up reading the provider itself considered."""

    text: str
    confidence: float = 0.0


@dataclass
class Token:
    """One token as the recognition provider handed it over."""

    text: str
    confidence: float = 1.0
    # Some providers (Azure Read, and Vision at symbol level) offer runners-up.
    # These are qualitatively different from candidates we generate: they came
    # from the ink, scored by a model that saw the pixels, where ours are
    # inferred from a confusion matrix that never did. They are scored
    # accordingly -- see _score_candidates.
    alternatives: List[Alternative] = field(default_factory=list)

    @staticmethod
    def parse_alternatives(raw: Iterable, observed_confidence: float) -> List["Alternative"]:
        """Accept either bare strings or {text, confidence} objects.

        When a provider names an alternative without scoring it, the sensible
        reading is that the probability mass it did not give the top candidate
        is spread over the ones it did name.
        """
        items = list(raw)
        if not items:
            return []

        residual = max(1.0 - observed_confidence, 0.0) / len(items)
        alternatives: List[Alternative] = []
        for item in items:
            if isinstance(item, str):
                alternatives.append(Alternative(text=item, confidence=residual))
            else:
                alternatives.append(
                    Alternative(
                        text=str(item.get("text", "")),
                        confidence=float(item.get("confidence", residual)),
                    )
                )
        return [alternative for alternative in alternatives if alternative.text]


@dataclass
class Candidate:
    text: str
    channel: float
    prior: float

    @property
    def score(self) -> float:
        return self.channel + self.prior


@dataclass
class ScoredToken:
    index: int
    text: str            # what we now believe was written
    original: str        # what the recogniser said
    confidence: float
    tier: str
    reason: str
    surprisal: float
    margin: float
    substituted: bool
    suggestion: Optional[str] = None
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "text": self.text,
            "original": self.original,
            "confidence": round(self.confidence, 4),
            "tier": self.tier,
            "reason": self.reason,
            "surprisal": round(self.surprisal, 2),
            "margin": round(self.margin, 3),
            "substituted": self.substituted,
            "suggestion": self.suggestion,
            "alternatives": [
                {"text": text, "score": round(score, 3)} for text, score in self.alternatives
            ],
        }


class Rescorer:
    def __init__(
        self,
        confusion: ConfusionModel,
        lexicon: Lexicon,
        language_model: LanguageModel,
        thresholds: Optional[Thresholds] = None,
    ) -> None:
        self.confusion = confusion
        self.lexicon = lexicon
        self.lm = language_model
        self.thresholds = thresholds or Thresholds()

    # ------------------------------------------------------------------

    def rescore(self, tokens: Sequence[Token]) -> List[ScoredToken]:
        """Decode a token sequence left to right.

        Greedy, conditioning on tokens already decoded rather than searching
        the whole sequence jointly. A full Viterbi pass over the sentence would
        be the textbook answer and is a contained change -- the per-token
        candidate sets are already built -- but greedy decoding gets the large
        majority of the benefit, because the evidence that fixes a token is
        almost always its immediate left context plus its own pixels.
        """
        results: List[ScoredToken] = []
        history: List[str] = []

        for index, token in enumerate(tokens):
            scored = self._rescore_one(index, token, history)
            results.append(scored)
            history.append(scored.text)

        return results

    # ------------------------------------------------------------------

    def _rescore_one(self, index: int, token: Token, history: Sequence[str]) -> ScoredToken:
        observed = token.text
        confidence = _clamp(token.confidence)
        context = list(history[-2:])

        if not observed or NON_WORD_RE.match(observed):
            # Punctuation and digits: nothing to rescore, but low confidence is
            # still worth flagging.
            tier = TIER_RED if confidence < self.thresholds.low_confidence else TIER_NONE
            return ScoredToken(
                index=index,
                text=observed,
                original=observed,
                confidence=confidence,
                tier=tier,
                reason="low recognition confidence" if tier == TIER_RED else "",
                surprisal=0.0,
                margin=0.0,
                substituted=False,
            )

        candidates = self._score_candidates(
            observed, confidence, token.alternatives, context
        )
        # The observed string is always in the running; it is what the pixels
        # actually said.
        observed_score = next(
            candidate.score for candidate in candidates if candidate.text == observed.lower()
        )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        margin = best.score - runner_up.score if runner_up else float("inf")

        surprisal = self.lm.surprisal(observed, context)
        gain = best.score - observed_score
        is_new = best.text != observed.lower()

        substituted = bool(is_new and gain >= self.thresholds.substitute_nats)
        text = match_case(observed, best.text) if substituted else observed
        suggestion = (
            match_case(observed, best.text) if is_new and not substituted else None
        )

        tier, reason = self._classify(
            confidence=confidence,
            surprisal=surprisal,
            margin=margin,
            substituted=substituted,
            observed=observed,
            best=best,
            is_new=is_new,
        )

        return ScoredToken(
            index=index,
            text=text,
            original=observed,
            confidence=confidence,
            tier=tier,
            reason=reason,
            surprisal=surprisal,
            margin=margin if margin != float("inf") else 0.0,
            substituted=substituted,
            suggestion=suggestion,
            alternatives=[
                (match_case(observed, candidate.text), candidate.score)
                for candidate in candidates[:4]
            ],
        )

    def _classify(
        self,
        confidence: float,
        surprisal: float,
        margin: float,
        substituted: bool,
        observed: str,
        best: Candidate,
        is_new: bool,
    ) -> Tuple[str, str]:
        limits = self.thresholds

        # Red first: whatever the language model thinks, if the recogniser could
        # barely read the ink then the token is a guess.
        if confidence < limits.low_confidence:
            return TIER_RED, "low recognition confidence"

        if substituted:
            return TIER_AMBER, f"rescored from {observed!r}"

        if surprisal > limits.surprisal_bits:
            return TIER_AMBER, "improbable in context"

        if is_new and margin < limits.close_call_nats:
            return TIER_AMBER, f"close call with {best.text!r}"

        # Middling confidence with nothing else against it: the recogniser is
        # not sure, and neither are we, but there is no better story to tell.
        if confidence < limits.settled_confidence and surprisal > limits.surprisal_bits * 0.6:
            return TIER_AMBER, "unsettled recognition and weak context support"

        return TIER_NONE, ""

    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        observed: str,
        confidence: float,
        alternatives: Sequence[Alternative],
        context: Sequence[str],
    ) -> List[Candidate]:
        """Score every candidate on one comparable scale.

        Two sources of evidence about how well a candidate explains the ink,
        and they are not measured in the same units:

        * For a reading the provider proposed, its own confidence is a direct
          estimate of P(ink | reading) -- from a model that actually saw the
          pixels. Running it through our confusion matrix instead would charge
          it an edit-distance penalty for differing from another reading of the
          same ink, which is not a real cost and buries good candidates.
        * For a candidate we invented, the confusion model is all we have.

        The two are made comparable by anchoring: our channel is shifted so
        that the observed token scores identically under both. That leaves one
        scale, with each candidate priced by the best evidence available for
        it.
        """
        by_provider = {
            alternative.text.lower(): alternative.confidence
            for alternative in alternatives
            if alternative.text
        }
        pool = self._candidate_pool(observed, alternatives)

        needle = observed.lower()
        provider_anchor = math.log(max(confidence, 1e-6))
        model_anchor = self.confusion.log_channel(needle, needle)
        offset = provider_anchor - model_anchor

        scored: List[Candidate] = []
        for text in pool:
            if text == needle:
                channel = provider_anchor
            elif text in by_provider:
                channel = math.log(max(by_provider[text], 1e-6))
            else:
                channel = self.confusion.log_channel(needle, text) + offset
            prior = self.thresholds.prior_weight * self.lm.log_prob(text, context)
            scored.append(Candidate(text=text, channel=channel, prior=prior))
        return scored

    def _candidate_pool(self, observed: str, alternatives: Sequence[Alternative]) -> List[str]:
        needle = observed.lower()
        pool = {needle}
        pool.update(
            alternative.text.lower() for alternative in alternatives if alternative.text
        )
        pool.update(word for word, _count in self.lexicon.candidates(needle))
        pool.update(variant.lower() for variant in self.confusion.inverse_variants(needle))

        if len(pool) <= self.thresholds.max_candidates:
            return sorted(pool)

        # Too many: keep the observed string, whatever the provider offered, and
        # the most frequent of the rest. Frequency is a cheap stand-in for the
        # full score, which we have not computed yet.
        keep = {needle}
        keep.update(
            alternative.text.lower() for alternative in alternatives if alternative.text
        )
        ranked = sorted(
            pool - keep,
            key=lambda word: (self.lexicon.count(word), self.lm.unigrams.get(word, 0)),
            reverse=True,
        )
        keep.update(ranked[: max(self.thresholds.max_candidates - len(keep), 0)])
        return sorted(keep)


def _clamp(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return min(max(value, 0.0), 1.0)


def apply_tokens(tokens: Sequence[ScoredToken]) -> str:
    """Join decoded tokens back into text.

    Naive spacing: the service layer holds the bounding boxes and does the real
    line and paragraph assembly. This exists for the eval harness and the CLI.
    """
    parts: List[str] = []
    for token in tokens:
        if parts and not NON_WORD_RE.match(token.text):
            parts.append(" ")
        elif parts and token.text and token.text[0] not in ".,;:!?)]}'\"":
            parts.append(" ")
        parts.append(token.text)
    return "".join(parts)
