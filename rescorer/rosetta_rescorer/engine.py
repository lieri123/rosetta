"""Ties the models, the store and the decoder together.

Both entry points -- the HTTP service the Go layer calls and the command line
used for batch work -- go through this, so learning and decoding cannot drift
apart between them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .confusion import ConfusionModel
from .lexicon import Lexicon
from .lm import LanguageModel
from .rescore import Rescorer, ScoredToken, Thresholds, Token
from .store import Models, Store


@dataclass
class LearnResult:
    pairs: int
    substitutions: int
    lexicon_size: int
    corrections_total: int


class Engine:
    def __init__(
        self,
        store: Optional[Store] = None,
        models: Optional[Models] = None,
        thresholds: Optional[Thresholds] = None,
    ) -> None:
        self.store = store
        self.models = models or (
            store.load()
            if store
            else Models(ConfusionModel(), Lexicon(), LanguageModel())
        )
        self.thresholds = thresholds or Thresholds()
        # One writer at a time: /learn mutates the models in place while
        # /rescore reads them, and the worker pool on the Go side calls both
        # concurrently.
        self._lock = threading.Lock()

    @property
    def rescorer(self) -> Rescorer:
        return Rescorer(
            self.models.confusion,
            self.models.lexicon,
            self.models.language_model,
            self.thresholds,
        )

    # ------------------------------------------------------------------

    def rescore(self, tokens: Sequence[Token]) -> List[ScoredToken]:
        with self._lock:
            return self.rescorer.rescore(tokens)

    def learn(
        self, pairs: Sequence[Tuple[str, str]], page_id: Optional[int] = None
    ) -> LearnResult:
        """Fold corrections into all three models and persist.

        A correction teaches three things at once, which is what makes the
        feedback loop cheap: how the recogniser errs (confusion), which words
        this person uses (lexicon), and how they string words together (LM).
        """
        with self._lock:
            substitutions = 0
            for predicted, corrected in pairs:
                ops = self.models.confusion.observe(predicted, corrected)
                substitutions += sum(1 for op in ops if op.is_error)
                # Only the corrected side trains the prior. Training on
                # recogniser output would teach the model to expect its own
                # mistakes.
                self.models.lexicon.add_text(corrected)
                self.models.language_model.add_text(corrected)
                if self.store:
                    self.store.log_correction(predicted, corrected, page_id)

            if self.store:
                self.store.save(self.models)

            return LearnResult(
                pairs=len(pairs),
                substitutions=substitutions,
                lexicon_size=len(self.models.lexicon),
                corrections_total=self.store.correction_count() if self.store else len(pairs),
            )

    def ingest_text(self, text: str) -> int:
        """Train the prior on known-good text without any error evidence.

        Used to seed the models from existing writing before any corrections
        exist, which is what stops a new install from being useless.
        """
        with self._lock:
            words = self.models.language_model.add_text(text)
            self.models.lexicon.add_text(text)
            if self.store:
                self.store.save(self.models)
            return words

    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        confusion = self.models.confusion
        return {
            "lexicon_words": len(self.models.lexicon),
            "lexicon_tokens": self.models.lexicon.total,
            "lm_vocabulary": self.models.language_model.vocabulary_size,
            "lm_tokens": self.models.language_model.total_words,
            "confusion_pairs": len(confusion.substitutions),
            "confusion_spans": len(confusion.spans),
            "corrections": self.store.correction_count() if self.store else 0,
            "top_confusions": [
                {"written": written, "read_as": read_as, "count": count, "rate": round(rate, 4)}
                for written, read_as, count, rate in confusion.top_confusions(15)
            ],
            "thresholds": {
                "low_confidence": self.thresholds.low_confidence,
                "settled_confidence": self.thresholds.settled_confidence,
                "surprisal_bits": self.thresholds.surprisal_bits,
                "close_call_nats": self.thresholds.close_call_nats,
                "substitute_nats": self.thresholds.substitute_nats,
            },
        }
