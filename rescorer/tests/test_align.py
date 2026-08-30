import unittest

from rosetta_rescorer.align import (
    DELETE,
    INSERT,
    MATCH,
    SUBSTITUTE,
    align,
    bounded_distance,
    corpus_error_rate,
    distance,
    error_rate,
    word_error_rate,
)


class TestDistance(unittest.TestCase):
    def test_basic_cases(self):
        self.assertEqual(distance("", ""), 0)
        self.assertEqual(distance("abc", "abc"), 0)
        self.assertEqual(distance("abc", ""), 3)
        self.assertEqual(distance("", "abc"), 3)
        self.assertEqual(distance("kitten", "sitting"), 3)

    def test_bounded_matches_unbounded_when_under_limit(self):
        for a, b in [("modern", "rnodern"), ("stack", "5tack"), ("a", "b"), ("", "xy")]:
            exact = distance(a, b)
            self.assertEqual(bounded_distance(a, b, exact), exact)

    def test_bounded_gives_up_past_the_limit(self):
        # The contract is only "greater than the limit", not the true distance:
        # the whole point is that it stops computing once the answer is known
        # to be useless to the caller.
        self.assertGreater(bounded_distance("handwriting", "xyz", 2), 2)

    def test_bounded_rejects_on_length_alone(self):
        self.assertGreater(bounded_distance("a", "aaaaaaa", 2), 2)


class TestAlign(unittest.TestCase):
    def test_identical_strings_are_all_matches(self):
        ops = align("notes", "notes")
        self.assertTrue(all(op.kind == MATCH for op in ops))
        self.assertEqual(len(ops), 5)

    def test_substitution_is_preferred_over_insert_delete_pair(self):
        # 'a' read as 'o' must be recorded as one substitution, not as a
        # deletion plus an unrelated insertion, or the confusion matrix learns
        # nothing usable.
        ops = [op for op in align("hond", "hand") if op.is_error]
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].kind, SUBSTITUTE)
        self.assertEqual((ops[0].source, ops[0].target), ("o", "a"))

    def test_collapse_is_captured_as_insert_plus_substitute(self):
        ops = [op for op in align("rnodern", "modern") if op.is_error]
        self.assertEqual({op.kind for op in ops}, {INSERT, SUBSTITUTE})

    def test_missing_character_is_a_deletion(self):
        ops = [op for op in align("hadwriting", "handwriting") if op.is_error]
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].kind, DELETE)
        self.assertEqual(ops[0].target, "n")

    def test_alignment_reconstructs_both_strings(self):
        source, target = "rnatrix5", "matrixS"
        ops = align(source, target)
        self.assertEqual("".join(op.source for op in ops), source)
        self.assertEqual("".join(op.target for op in ops), target)


class TestErrorRates(unittest.TestCase):
    def test_error_rate_is_edits_over_reference_length(self):
        self.assertAlmostEqual(error_rate("abcd", "abcd"), 0.0)
        self.assertAlmostEqual(error_rate("abcd", "abXd"), 0.25)

    def test_corpus_rate_pools_rather_than_averaging(self):
        # One bad short line and one clean long line. Averaging per line would
        # report 0.25; pooling reports the two edits against all 22 reference
        # characters, which is the honest number.
        pairs = [("ab", "xy"), ("a" * 20, "a" * 20)]
        self.assertAlmostEqual(corpus_error_rate(pairs), 2 / 22)

    def test_word_error_rate_aligns_tokens(self):
        self.assertAlmostEqual(word_error_rate("the quick fox", "the quick fox"), 0.0)
        self.assertAlmostEqual(word_error_rate("the quick fox", "the quirk fox"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(error_rate("", ""), 0.0)
        self.assertEqual(error_rate("", "junk"), 1.0)


if __name__ == "__main__":
    unittest.main()
