import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from rosetta_rescorer.engine import Engine
from rosetta_rescorer.server import make_server
from rosetta_rescorer.store import Store


class TestStore(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(self.path)
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.unlink(self.path + suffix)

    def test_starts_empty(self):
        models = self.store.load()
        self.assertEqual(len(models.lexicon), 0)
        self.assertEqual(self.store.correction_count(), 0)

    def test_models_survive_a_reload(self):
        engine = Engine(store=self.store)
        engine.ingest_text("the confusion matrix shows my own failure modes")
        engine.learn([("the rnatrix", "the matrix")])

        reopened = Store(self.path)
        try:
            models = reopened.load()
            self.assertIn("matrix", models.lexicon)
            self.assertEqual(models.confusion.spans[("rn", "m")], 1)
            self.assertGreater(models.language_model.total_words, 0)
            self.assertAlmostEqual(
                models.language_model.log_prob("matrix", ["the"]),
                engine.models.language_model.log_prob("matrix", ["the"]),
            )
        finally:
            reopened.close()

    def test_corrections_are_kept_verbatim(self):
        # The models are derived state; the corrections are the record, and
        # must be replayable into a rebuilt model.
        engine = Engine(store=self.store)
        engine.learn([("teh rnatrix", "the matrix"), ("5tack", "Stack")])
        self.assertEqual(self.store.correction_count(), 2)
        self.assertIn(("5tack", "Stack"), self.store.corrections())

    def test_saving_twice_does_not_double_the_counts(self):
        engine = Engine(store=self.store)
        engine.ingest_text("the matrix")
        first = self.store.load().lexicon.count("matrix")
        self.store.save(engine.models)
        self.assertEqual(self.store.load().lexicon.count("matrix"), first)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.engine.ingest_text(
            "the confusion matrix shows which characters collapse into each other\n"
            "the matrix is personal and shows my own failure modes\n"
        )
        cls.server = make_server(cls.engine, host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return response.status, json.loads(response.read())

    def _post(self, path, payload):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())

    def test_healthz(self):
        status, body = self._get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_rescore_returns_a_tier_for_every_token(self):
        status, body = self._post(
            "/rescore",
            {"tokens": [{"text": "the", "confidence": 0.99},
                        {"text": "matrix", "confidence": 0.30}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["tokens"]), 2)
        self.assertEqual(body["tokens"][1]["tier"], "red")
        self.assertIn("text", body)

    def test_rescore_accepts_alternatives_in_both_shapes(self):
        for alternatives in (["model"], [{"text": "model", "confidence": 0.3}]):
            status, body = self._post(
                "/rescore",
                {"tokens": [{"text": "rnodel", "confidence": 0.5,
                             "alternatives": alternatives}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(body["tokens"]), 1)

    def test_learn_updates_the_models(self):
        before = len(self.engine.models.lexicon)
        status, body = self._post(
            "/learn", {"pairs": [{"predicted": "the rnatrix", "corrected": "the matrix"}]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["pairs"], 1)
        self.assertGreaterEqual(len(self.engine.models.lexicon), before)

    def test_stats_reports_what_was_learned(self):
        status, body = self._get("/stats")
        self.assertEqual(status, 200)
        for key in ("lexicon_words", "confusion_pairs", "top_confusions", "thresholds"):
            self.assertIn(key, body)

    def test_unknown_route_is_a_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get("/nope")
        self.assertEqual(caught.exception.code, 404)

    def test_malformed_body_is_a_400_not_a_500(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/rescore",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)

    def test_empty_token_list_is_accepted(self):
        status, body = self._post("/rescore", {"tokens": []})
        self.assertEqual(status, 200)
        self.assertEqual(body["tokens"], [])


if __name__ == "__main__":
    unittest.main()
