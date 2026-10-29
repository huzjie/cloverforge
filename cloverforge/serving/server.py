# -*- coding: utf-8 -*-
"""A minimal stdlib HTTP server exposing the dLLM over JSON."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .api import handle_generate, handle_decision, handle_models

_CFG = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "name": "cloverforge"})
        elif self.path == "/v1/models":
            self._send(200, handle_models())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            body = {}
        if self.path == "/v1/generate":
            self._send(200, handle_generate(_CFG, body))
        elif self.path == "/v1/decision":
            self._send(200, handle_decision(_CFG, body))
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # silence default request logging


def serve(host, port, cfg):
    global _CFG
    _CFG = cfg
    server = HTTPServer((host, port), Handler)
    print("[serve] cloverforge listening on http://%s:%d" % (host, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
