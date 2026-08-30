"""HTTP front end for the rescorer.

Deliberately the standard library and nothing else. The service has four
endpoints, is only ever called from the Go process on the same host, and
handles one request per correction or per page -- a framework would be pure
overhead, and the dependency would have to be installed everywhere this runs.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple

from .engine import Engine
from .rescore import Token

LOGGER = logging.getLogger("rosetta.rescorer")

MAX_BODY_BYTES = 8 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    engine: Engine  # injected by make_server

    server_version = "rosetta-rescorer/0.1"

    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/stats":
            self._send_json(200, self.engine.stats())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        routes: Dict[str, Callable[[Dict[str, Any]], Tuple[int, Dict[str, Any]]]] = {
            "/rescore": self._rescore,
            "/learn": self._learn,
            "/ingest": self._ingest,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._send_json(404, {"error": "not found"})
            return

        try:
            payload = self._read_json()
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return

        try:
            status, body = handler(payload)
        except (KeyError, TypeError, ValueError) as error:
            # Malformed input is the caller's problem and should not look like
            # an outage; anything else is genuinely ours and is left to raise.
            self._send_json(400, {"error": f"{type(error).__name__}: {error}"})
            return

        self._send_json(status, body)

    # ------------------------------------------------------------------

    def _rescore(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        raw_tokens = payload.get("tokens") or []
        tokens = []
        for item in raw_tokens:
            confidence = float(item.get("confidence", 1.0))
            tokens.append(
                Token(
                    text=str(item.get("text", "")),
                    confidence=confidence,
                    alternatives=Token.parse_alternatives(
                        item.get("alternatives", []), confidence
                    ),
                )
            )
        scored = self.engine.rescore(tokens)
        return 200, {
            "tokens": [token.to_dict() for token in scored],
            "text": " ".join(token.text for token in scored),
        }

    def _learn(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        pairs = [
            (str(item["predicted"]), str(item["corrected"]))
            for item in payload.get("pairs", [])
        ]
        page_id = payload.get("page_id")
        result = self.engine.learn(pairs, page_id=int(page_id) if page_id else None)
        return 200, {
            "pairs": result.pairs,
            "edits_observed": result.substitutions,
            "lexicon_size": result.lexicon_size,
            "corrections_total": result.corrections_total,
        }

    def _ingest(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        text = str(payload.get("text", ""))
        words = self.engine.ingest_text(text)
        return 200, {"words": words, "lexicon_size": len(self.engine.models.lexicon)}

    # ------------------------------------------------------------------

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.info("%s %s", self.address_string(), format % args)


def make_server(engine: Engine, host: str = "127.0.0.1", port: int = 8801) -> ThreadingHTTPServer:
    handler = type("RescorerHandler", (_Handler,), {"engine": engine})
    return ThreadingHTTPServer((host, port), handler)


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8801) -> None:
    server = make_server(engine, host, port)
    LOGGER.info("rescorer listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
