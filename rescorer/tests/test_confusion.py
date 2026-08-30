import unittest

from rosetta_rescorer.confusion import ConfusionModel


class TestFitting(unittest.TestCase):
    def test_counts_a_simple_substitution(self):
        model = ConfusionModel()
        model.observe("hond", "hand")
        self.assertEqual(model.substitutions[("a", "o")], 1)
        self.assertEqual(model.substitutions[("h", "h")], 1)

    def test_records_multi_character_collapse_as_a_span(self):
        model = ConfusionModel()
        for _ in range(5):
            model.observe("rnodern", "modern")
        self.assertEqual(model.spans[("rn", "m")], 5)

    def test_top_confusions_reports_real_errors_only(self):
        model = ConfusionModel()
        for _ in range(4):
            model.observe("5tack", "Stack")
        rows = model.top_confusions()
        self.assertTrue(rows)
        written, read_as, count, _rate = rows[0]
        self.assertEqual((written, read_as), ("S", "5"))
        self.assertEqual(count, 4)
        self.assertFalse(any(w == r for w, r, _c, _rt in rows))


class TestChannel(unittest.TestCase):
    def test_identity_is_the_most_likely_reading(self):
        model = ConfusionModel()
        self.assertGreater(
            model.log_channel("notes", "notes"), model.log_channel("notes", "nodes")
        )

    def test_untrained_model_behaves_like_edit_distance(self):
        # Before any corrections there is no evidence that one confusion is
        # likelier than another, and the channel must not pretend otherwise:
        # two candidates the same distance away should score the same.
        model = ConfusionModel()
        one = model.log_channel("hand", "band")
        two = model.log_channel("hand", "land")
        self.assertAlmostEqual(one, two, places=9)
        self.assertGreater(one, model.log_channel("hand", "bane"))

    def test_learned_confusions_shift_the_channel(self):
        model = ConfusionModel()
        for _ in range(30):
            model.observe("hond", "hand")
        # Having seen a -> o thirty times, reading 'o' as a written 'a' is now
        # far likelier than reading it as a written 'e', which has never
        # happened.
        self.assertGreater(
            model.log_substitution("a", "o"), model.log_substitution("e", "o")
        )

    def test_learning_narrows_the_penalty_on_the_true_reading(self):
        # The channel on its own always prefers the identity reading, and
        # should: P(observed | observed) is what the diagonal means. What
        # adaptation changes is the size of the penalty on the true string --
        # it has to fall far enough for the language prior to be able to
        # overturn it, which is exactly how a correction becomes reachable.
        cold = ConfusionModel()
        gap_before = cold.log_channel("rnatrix", "rnatrix") - cold.log_channel("rnatrix", "matrix")

        warm = ConfusionModel()
        for _ in range(20):
            warm.observe("rnatrix", "matrix")
        gap_after = warm.log_channel("rnatrix", "rnatrix") - warm.log_channel("rnatrix", "matrix")

        self.assertGreater(gap_before, 0.0)
        self.assertLess(gap_after, gap_before / 2)


class TestCandidateGeneration(unittest.TestCase):
    def test_inverse_variants_undo_a_learned_span(self):
        model = ConfusionModel()
        for _ in range(6):
            model.observe("rnodern", "modern")
        self.assertIn("modern", model.inverse_variants("rnodern"))

    def test_inverse_variants_undo_a_learned_character(self):
        model = ConfusionModel()
        for _ in range(6):
            model.observe("5tack", "Stack")
        self.assertIn("System", model.inverse_variants("5ystem"))

    def test_untrained_model_proposes_nothing(self):
        # Nothing has been confused yet, so there is nothing to propose. This
        # is what keeps a fresh install from inventing corrections.
        self.assertEqual(ConfusionModel().inverse_variants("anything"), [])

    def test_variant_generation_is_bounded(self):
        model = ConfusionModel()
        for _ in range(3):
            model.observe("aeiou", "eioua")
        self.assertLessEqual(len(model.inverse_variants("aeiouaeiou", max_variants=10)), 10)


class TestPersistence(unittest.TestCase):
    def test_round_trip_through_rows(self):
        model = ConfusionModel()
        model.observe("rnodern", "modern")
        model.observe("5tack", "Stack")

        restored = ConfusionModel.from_rows(model.to_rows())
        self.assertEqual(restored.substitutions, model.substitutions)
        self.assertEqual(restored.spans, model.spans)
        self.assertEqual(restored.insertions, model.insertions)
        self.assertAlmostEqual(
            restored.log_channel("rnodern", "modern"),
            model.log_channel("rnodern", "modern"),
        )


if __name__ == "__main__":
    unittest.main()
