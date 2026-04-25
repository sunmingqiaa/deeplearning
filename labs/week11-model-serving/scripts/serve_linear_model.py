"""Week 11: minimal HTTP inference service using the standard library.

Run:
    python labs/week11-model-serving/scripts/serve_linear_model.py --port 8000

Then open:
    http://127.0.0.1:8000/predict?x=2.5
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class Handler(BaseHTTPRequestHandler):
    w = 2.7
    b = -0.8

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"use /predict?x=1.0"}')
            return
        params = parse_qs(parsed.query)
        try:
            x = float(params.get("x", [""])[0])
        except ValueError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"x must be a number"}')
            return
        payload = {"x": x, "prediction": self.w * x + self.b}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--w", type=float, default=2.7)
    parser.add_argument("--b", type=float, default=-0.8)
    args = parser.parse_args()
    Handler.w = args.w
    Handler.b = args.b
    server = HTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}/predict?x=2.5")
    server.serve_forever()


if __name__ == "__main__":
    main()
