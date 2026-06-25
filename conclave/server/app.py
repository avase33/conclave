"""A tiny stdlib HTTP server for viewing traces.

No web framework — just ``http.server``. Serves the rendered dashboard for a
trace dict at ``/`` and the raw JSON at ``/trace.json``.
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .dashboard import render_dashboard


def make_handler(trace: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    html = render_dashboard(trace).encode("utf-8")
    raw = json.dumps(trace, ensure_ascii=False, indent=2).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.rstrip("/") in ("", "/index.html"):
                self._send(html, "text/html; charset=utf-8")
            elif self.path.startswith("/trace.json"):
                self._send(raw, "application/json; charset=utf-8")
            else:
                self.send_error(404)

        def log_message(self, *args: Any) -> None:  # silence default logging
            pass

    return Handler


def serve_trace(trace: dict[str, Any], host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = HTTPServer((host, port), make_handler(trace))
    url = f"http://{host}:{port}/"
    print(f"Conclave dashboard serving at {url} (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nStopped.")
    finally:
        server.server_close()
