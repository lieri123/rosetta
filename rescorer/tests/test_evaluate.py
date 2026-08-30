import unittest
from pathlib import Path

from rosetta_rescorer.engine import Engine
from rosetta_rescorer.evaluate import (
    Sample,
    correction_pairs,
    evaluate,
    split_samples,
    sweep_substitution_margin,
    synthesize,
)
from rosetta_rescorer.rescore import Token

CORPUS = (Path(__file__).resolve().parents[1] / "data" / "sample-notes.txt").read_text()


class TestSynthesis(unittest.TestCase):
    def test_corruption_is_deterministic_for_a_seed(self):
        first = synthesize(CORPUS, seed=5)
        second = synthesize(CORPUS, seed=5)
        self.assertEqual(
            [token.text for sample in first for token in sample.observed],
            [token.text for sample in second for token in sample.observed],
        )

    def test_a_higher_error_rate_corrupts_more(self):
        light = synthesize(CORPUS, error_rate=0.02, seed=5)
        heavy = synthesize(CORPUS, error_rate=0.30, seed=5)
        self.assertLess(_wrong(light), _wrong(heavy))

    def test_some_corrupted_tokens_look_confident(self):
        # The property that makes the amber tier necessary: if every error came
        # with low confidence, a threshold on confidence alone would be enough
        # and none of this would be worth building.
        samples = synthesize(CORPUS, error_rate=0.15, seed=5)
        confident_errors = sum(
            1
            for sample in samples
            for truth, token in zip(sample.truth, sample.observed)
            if token.text != truth and token.confidence > 0.85
        )
        self.assertGreater(confident_errors, 0)


def _wrong(samples):
    return sum(
        1
        for sample in samples
        for truth, token in zip(sample.truth, sample.observed)
        if token.text != truth
    )


class TestSplit(unittest.TestCase):
    def test_split_is_disjoint_and_complete(self):
        samples = synthesize(CORPUS, seed=5)
        train, test = split_samples(samples, 0.6, seed=1)
        self.assertEqual(len(train) + len(test), len(samples))
        self.assertTrue(train and test)
        train_ids = {id(sample) for sample in train}
        self.assertFalse(train_ids & {id(sample) for sample in test})

    def test_correction_pairs_skip_lines_with_no_errors(self):
        clean = Sample(truth=["the", "matrix"], observed=[Token("the", 1.0), Token("matrix", 1.0)])
        dirty = Sample(truth=["the", "matrix"], observed=[Token("the", 1.0), Token("rnatrix", 0.5)])
        self.assertEqual(correction_pairs([clean, dirty]), [("the rnatrix", "the matrix")])


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.samples = synthesize(CORPUS, error_rate=0.08, seed=11)
        self.train, self.test = split_samples(self.samples, 0.7, seed=11)

    def _adapted_engine(self):
        engine = Engine()
        for sample in self.train:
            engine.ingest_text(sample.truth_text)
        engine.learn(correction_pairs(self.train))
        return engine

    def test_adaptation_lowers_the_error_rate(self):
        # The headline claim, measured on held-out lines.
        engine = self._adapted_engine()
        result = evaluate(engine, self.test, synthetic=True)
        self.assertLess(result.rescored_cer, result.baseline_cer)
        self.assertLess(result.rescored_wer, result.baseline_wer)

    def test_substitutions_do_more_good_than_harm(self):
        engine = self._adapted_engine()
        result = evaluate(engine, self.test, synthetic=True)
        self.assertGreater(result.substitutions, 0)
        self.assertGreater(result.substitutions_helped, result.substitutions_hurt)

    def test_an_untrained_engine_does_not_make_things_worse(self):
        # The floor that matters: a fresh install must never degrade the API's
        # own output.
        result = evaluate(Engine(), self.test, synthetic=True)
        self.assertLessEqual(result.rescored_cer, result.baseline_cer + 1e-9)

    def test_flags_are_better_than_flagging_everything(self):
        engine = self._adapted_engine()
        result = evaluate(engine, self.test, synthetic=True)
        flagged = result.tier_counts.get("red", 0) + result.tier_counts.get("amber", 0)
        self.assertLess(flagged, result.tokens * 0.5)
        self.assertGreater(result.flags.precision, 0.5)
        self.assertGreater(result.flags.recall, 0.4)

    def test_reported_numbers_are_self_consistent(self):
        engine = self._adapted_engine()
        result = evaluate(engine, self.test, synthetic=True)
        counted = sum(result.tier_counts.values())
        self.assertEqual(counted, result.tokens)
        data = result.to_dict()
        self.assertTrue(data["synthetic"])
        self.assertAlmostEqual(
            data["cer_reduction"],
            round((result.baseline_cer - result.rescored_cer) / result.baseline_cer, 4),
        )

    def test_margin_sweep_returns_to_baseline_at_the_top(self):
        # A margin nothing can clear means no substitutions, which must land
        # exactly on the raw recogniser's error rate.
        engine = self._adapted_engine()
        curve = sweep_substitution_margin(engine, self.test, [1.0, 1000.0])
        baseline = evaluate(engine, self.test, synthetic=True).baseline_cer
        self.assertAlmostEqual(curve[-1][1], baseline, places=9)
        self.assertLess(curve[0][1], curve[-1][1])

    def test_sweep_restores_the_original_thresholds(self):
        engine = self._adapted_engine()
        before = engine.thresholds.substitute_nats
        sweep_substitution_margin(engine, self.test, [0.0, 5.0])
        self.assertEqual(engine.thresholds.substitute_nats, before)


if __name__ == "__main__":
    unittest.main()
