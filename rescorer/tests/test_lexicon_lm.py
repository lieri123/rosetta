import unittest

from rosetta_rescorer.lexicon import Lexicon, match_case, tokenize
from rosetta_rescorer.lm import LanguageModel

CORPUS = """the confusion matrix shows which characters collapse into each other
the meeting notes are in the shared folder
we discussed the noisy channel model at the meeting
the shared folder has the meeting notes from last week
"""


class TestTokenize(unittest.TestCase):
    def test_keeps_apostrophes_and_hyphens_inside_words(self):
        self.assertEqual(
            tokenize("don't split noisy-channel"), ["don't", "split", "noisy-channel"]
        )

    def test_drops_punctuation(self):
        self.assertEqual(tokenize("hello, world!"), ["hello", "world"])

    def test_keeps_alphanumeric_tokens(self):
        self.assertEqual(tokenize("page 42 of v2"), ["page", "42", "of", "v2"])


class TestMatchCase(unittest.TestCase):
    def test_restores_the_shape_of_what_was_replaced(self):
        self.assertEqual(match_case("Modern", "modern"), "Modern")
        self.assertEqual(match_case("MODERN", "modern"), "MODERN")
        self.assertEqual(match_case("modern", "modern"), "modern")

    def test_single_capital_is_not_treated_as_all_caps(self):
        self.assertEqual(match_case("I", "i"), "I")


class TestLexicon(unittest.TestCase):
    def setUp(self):
        self.lexicon = Lexicon()
        self.lexicon.add_text(CORPUS)

    def test_counts_repeated_words(self):
        self.assertEqual(self.lexicon.count("meeting"), 3)
        self.assertIn("folder", self.lexicon)

    def test_is_case_insensitive(self):
        self.assertIn("MEETING", self.lexicon)

    def test_candidates_finds_near_misses_ordered_by_frequency(self):
        words = [word for word, _count in self.lexicon.candidates("meetlng")]
        self.assertIn("meeting", words)

    def test_candidates_excludes_the_token_itself(self):
        words = [word for word, _count in self.lexicon.candidates("meeting")]
        self.assertNotIn("meeting", words)

    def test_short_tokens_get_a_tighter_budget(self):
        # Two edits on a three-letter word would make most of the lexicon a
        # candidate and turn every short word into a coin flip.
        self.assertEqual(
            [word for word, _ in self.lexicon.candidates("the", max_edits=2)],
            [word for word, _ in self.lexicon.candidates("the", max_edits=1)],
        )

    def test_background_words_are_outweighed_by_personal_use(self):
        self.lexicon.add_background(["folder", "flounder"])
        self.assertGreater(self.lexicon.count("folder"), self.lexicon.count("flounder"))

    def test_round_trip_through_rows(self):
        restored = Lexicon.from_rows(self.lexicon.to_rows())
        self.assertEqual(restored.counts, self.lexicon.counts)


class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        self.lm = LanguageModel()
        self.lm.add_text(CORPUS)

    def test_context_changes_the_estimate(self):
        # "meeting" is common after "the" in this corpus and never appears
        # after "noisy". A unigram model could not tell the difference, and
        # telling the difference is the whole reason for the trigram.
        self.assertLess(
            self.lm.surprisal("meeting", ["the"]),
            self.lm.surprisal("meeting", ["noisy", "channel"]),
        )

    def test_seen_words_beat_misspellings(self):
        self.assertLess(self.lm.surprisal("meeting", ["the"]), self.lm.surprisal("meetlng", ["the"]))

    def test_unseen_words_keep_a_finite_score(self):
        # Zero probability for anything unseen would make the rescorer rewrite
        # every new name and every piece of jargon into a familiar word.
        surprisal = self.lm.surprisal("kubernetes", ["the"])
        self.assertTrue(0 < surprisal < float("inf"))

    def test_probabilities_are_log_scale_and_negative(self):
        self.assertLess(self.lm.log_prob("meeting", ["the"]), 0.0)

    def test_char_model_prefers_plausible_spellings(self):
        # Both are unseen; one is spelled like the corpus and one is not.
        self.assertGreater(self.lm.char_log_prob("shored"), self.lm.char_log_prob("xzqwj"))

    def test_char_score_is_not_dominated_by_length(self):
        # Without per-character normalisation a long candidate could never win,
        # and the rescorer would develop a permanent bias towards short words.
        short = self.lm.char_log_prob("the")
        long = self.lm.char_log_prob("thethethethe")
        self.assertLess(abs(short - long), 12.0)

    def test_round_trip_through_rows(self):
        word_rows, char_rows = self.lm.to_rows()
        restored = LanguageModel.from_rows(word_rows, char_rows)
        self.assertAlmostEqual(
            restored.log_prob("meeting", ["the"]), self.lm.log_prob("meeting", ["the"])
        )
        self.assertEqual(restored.total_words, self.lm.total_words)


if __name__ == "__main__":
    unittest.main()
