"""Serve the existing Atlas in review mode with synchronous JSON persistence.

Local-only HTTP endpoint. Review controls are part of knowledge.html/code.html,
not a standalone review application. A browser key in the fragment is never
returned by GET /api/review/state or stored in public static assets.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import quote, urlsplit

from harness.core import agent_flow as engine
from harness.core import agent_proposal_review as proposal

ROOT = engine.ROOT / "harness/visualization"
MAX_BODY = 8192
STATIC = {
    "/" : None,
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/core.html": ("core.html", "text/html; charset=utf-8"),
    "/knowledge.html": ("knowledge.html", "text/html; charset=utf-8"),
    "/code.html": ("code.html", "text/html; charset=utf-8"),
    "/environment.html": ("environment.html", "text/html; charset=utf-8"),
    "/context.html": ("context.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/data.js": ("data.js", "text/javascript; charset=utf-8"),
}


def create_server(plan_id: str, port: int = 0) -> ThreadingHTTPServer:
    engine.load(plan_id)
    secret = secrets.token_urlsafe(32)
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self):
            host, bound = self.server.server_address
            return self.headers.get("Host") == f"{host}:{bound}"

        def send(self, status, data=b"", mime="application/json; charset=utf-8",
                 *, location=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; img-src 'self'; frame-ancestors 'none'")
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.allowed_host():
                return self.send(403, b'{"error":"Untrusted host"}')
            path = urlsplit(self.path).path
            if path == "/":
                url = "/knowledge.html?review=" + quote(plan_id) + "#key=" + secret
                return self.send(302, b"", location=url)
            if path == "/api/review/state":
                try:
                    with lock:
                        body = proposal.snapshot(engine.load(plan_id))
                    return self.send(200, json.dumps(body, ensure_ascii=False).encode())
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    return self.send(409, json.dumps({"error": str(exc)}).encode())
            if path in STATIC:
                filename, mime = STATIC[path]
                try:
                    return self.send(200, (ROOT / filename).read_bytes(), mime)
                except OSError:
                    return self.send(404, b'{"error":"Atlas asset missing"}')
            return self.send(404, b'{"error":"Not found"}')

        def do_POST(self):
            if not self.allowed_host():
                return self.send(403, b'{"error":"Untrusted host"}')
            if urlsplit(self.path).path != "/api/review/decision":
                return self.send(404, b'{"error":"Not found"}')
            expected_origin = "http://" + self.headers.get("Host", "")
            if (self.headers.get("Origin") != expected_origin or
                self.headers.get("X-Atlas-Review-Key") != secret or
                self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                return self.send(403, b'{"error":"Human review origin or key rejected"}')
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= MAX_BODY:
                    return self.send(413, b'{"error":"Invalid review request size"}')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise engine.FlowError("Expected a review object")
                with lock:
                    plan = engine.load(plan_id)
                    if type(data.get("revision")) is not int or data["revision"] != plan["revision"]:
                        raise engine.FlowError("Stale review; reload Atlas before continuing")
                    body = proposal.decide(plan, data.get("key"), data.get("decision"),
                                           actor="Human via local Atlas")
                return self.send(200, json.dumps(body, ensure_ascii=False).encode())
            except (ValueError, KeyError, OSError, TypeError) as exc:
                return self.send(409, json.dumps({"error": str(exc)}).encode())

        def log_message(self, format, *args):
            # Never log review keys, artifacts or human decisions.
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.review_key = secret
    return server


def serve(plan_id: str, port: int = 8766) -> None:
    server = create_server(plan_id, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/knowledge.html?review={quote(plan_id)}#key={server.review_key}"
    print("Atlas human review: " + url, flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
