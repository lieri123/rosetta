import unittest

from rosetta_rescorer.confusion import ConfusionModel
from rosetta_rescorer.lexicon import Lexicon
from rosetta_rescorer.lm import LanguageModel
from rosetta_rescorer.rescore import (
    TIER_AMBER,
    TIER_NONE,
    TIER_RED,
    Alternative,
    Rescorer,
    Thresholds,
    Token,
)

CORPUS = """the confusion matrix shows which characters collapse into each other
the meeting notes are in the shared folder
we discussed the noisy channel model at the meeting
the shared folder has the meeting notes from last week
the matrix is personal and shows my own failure modes
"""


def build(trained=True, **threshold_overrides):
    lm = LanguageModel()
    lm.add_text(CORPUS)
    lexicon = Lexicon()
    lexicon.add_text(CORPUS)
    confusion = ConfusionModel()
    if trained:
        for _ in range(20):
            confusion.observe("rnatrix", "matrix")
            confusion.observe("rnodel", "model")
    return Rescorer(confusion, lexicon, lm, Thresholds(**threshold_overrides))


class TestTiers(unittest.TestCase):
    def test_confident_and_expected_is_left_alone(self):
        scored = build().rescore([Token("the", 0.99), Token("meeting", 0.97)])
        self.assertTrue(all(token.tier == TIER_NONE for token in scored))
        self.assertTrue(all(token.reason == "" for token in scored))

    def test_low_confidence_is_red(self):
        scored = build().rescore([Token("the", 0.99), Token("meeting", 0.20)])
        self.assertEqual(scored[1].tier, TIER_RED)
        self.assertIn("confidence", scored[1].reason)

    def test_confidently_wrong_word_is_amber(self):
        # The case that motivates the tier: recognition confidence is high, so
        # a confidence-only scheme sees nothing at all, but the word does not
        # belong in this context. Asserted differentially -- an expected word
        # and an unexpected one, same confidence, opposite outcomes -- because
        # the absolute bit threshold is calibrated for a real corpus and this
        # fixture is five lines long.
        rescorer = build(surprisal_bits=6.0)
        expected = rescorer.rescore(
            [Token("the", 0.99), Token("shared", 0.98), Token("folder", 0.96)]
        )
        unexpected = rescorer.rescore(
            [Token("the", 0.99), Token("shared", 0.98), Token("zebra", 0.96)]
        )
        self.assertEqual(expected[2].tier, TIER_NONE)
        self.assertEqual(unexpected[2].tier, TIER_AMBER)
        self.assertEqual(unexpected[2].reason, "improbable in context")

    def test_red_takes_priority_over_amber(self):
        scored = build().rescore([Token("zebra", 0.10)])
        self.assertEqual(scored[0].tier, TIER_RED)

    def test_punctuation_and_numbers_pass_through(self):
        scored = build().rescore([Token(".", 0.99), Token("42", 0.98), Token("--", 0.95)])
        self.assertTrue(all(token.tier == TIER_NONE for token in scored))
        self.assertTrue(all(not token.substituted for token in scored))

    def test_low_confidence_punctuation_is_still_flagged(self):
        scored = build().rescore([Token(".", 0.10)])
        self.assertEqual(scored[0].tier, TIER_RED)

    def test_every_token_gets_a_tier(self):
        scored = build().rescore([Token(word, 0.9) for word in "the shared folder".split()])
        self.assertEqual(len(scored), 3)
        self.assertTrue(all(token.tier in (TIER_NONE, TIER_AMBER, TIER_RED) for token in scored))


class TestSubstitution(unittest.TestCase):
    def test_learned_confusion_is_corrected(self):
        scored = build().rescore([Token("the", 0.99), Token("rnatrix", 0.62)])
        self.assertTrue(scored[1].substituted)
        self.assertEqual(scored[1].text, "matrix")
        self.assertEqual(scored[1].original, "rnatrix")

    def test_substitution_preserves_capitalisation(self):
        scored = build().rescore([Token("the", 0.99), Token("Rnatrix", 0.62)])
        self.assertEqual(scored[1].text, "Matrix")

    def test_an_untrained_rescorer_leaves_text_alone(self):
        # Before it has evidence the decoder must get out of the way rather
        # than rewriting text towards whatever is commonest.
        scored = build(trained=False).rescore(
            [Token("the", 0.99), Token("kubernetes", 0.75), Token("rnatrix", 0.75)]
        )
        self.assertFalse(any(token.substituted for token in scored))

    def test_a_high_margin_suppresses_substitution(self):
        scored = build(substitute_nats=1000.0).rescore(
            [Token("the", 0.99), Token("rnatrix", 0.62)]
        )
        self.assertFalse(scored[1].substituted)
        # It still offers the suggestion rather than hiding what it thinks.
        self.assertEqual(scored[1].suggestion, "matrix")

    def test_substituted_tokens_are_flagged_not_silent(self):
        scored = build().rescore([Token("the", 0.99), Token("rnatrix", 0.62)])
        self.assertNotEqual(scored[1].tier, TIER_NONE)

    def test_unknown_words_are_not_forced_into_the_lexicon(self):
        # Jargon and names must survive. A rescorer that "corrects" them is
        # worse than no rescorer.
        scored = build().rescore([Token("the", 0.99), Token("kubernetes", 0.93)])
        self.assertEqual(scored[1].text, "kubernetes")
        self.assertFalse(scored[1].substituted)

    def test_provider_alternatives_are_scored_on_the_provider_scale(self):
        # A reading the provider itself proposed came from the same ink, so it
        # must not be charged an edit-distance penalty for differing from the
        # top reading. Priced correctly it becomes the leading candidate; the
        # margin between two readings of the same ink is genuinely narrow, so
        # the right outcome is to surface it and flag the token rather than
        # rewrite silently.
        token = Token(
            "rnodei", 0.55, alternatives=Token.parse_alternatives(["model"], 0.55)
        )
        scored = build().rescore([Token("the", 0.99), Token("model", 0.99), token])
        self.assertEqual(scored[2].suggestion, "model")
        self.assertNotEqual(scored[2].tier, TIER_NONE)

    def test_a_confident_alternative_wins_outright(self):
        # When the provider is clear that its second reading is nearly as good
        # as its first, and the language model agrees, the decoder should take
        # it.
        token = Token(
            "rnodei",
            0.40,
            alternatives=[Alternative(text="model", confidence=0.38)],
        )
        scored = build().rescore([Token("the", 0.99), token])
        self.assertEqual(scored[1].text, "model")
        self.assertTrue(scored[1].substituted)


class TestOutput(unittest.TestCase):
    def test_serialisation_keeps_the_fields_the_editor_needs(self):
        scored = build().rescore([Token("rnatrix", 0.62)])
        data = scored[0].to_dict()
        for key in ("index", "text", "original", "confidence", "tier", "reason", "suggestion"):
            self.assertIn(key, data)

    def test_indices_are_positional(self):
        scored = build().rescore([Token(word, 0.9) for word in "a b c".split()])
        self.assertEqual([token.index for token in scored], [0, 1, 2])

    def test_context_comes_from_decoded_text(self):
        # The third token is scored against the corrected form of the second,
        # not the recogniser's version of it, so one fix helps the next token.
        scored = build().rescore(
            [Token("the", 0.99), Token("rnatrix", 0.60), Token("shows", 0.80)]
        )
        self.assertEqual(scored[1].text, "matrix")
        self.assertEqual(scored[2].text, "shows")
        self.assertEqual(scored[2].tier, TIER_NONE)


if __name__ == "__main__":
    unittest.main()
