#!/usr/bin/env python3
"""Reproducibly capture a Claude Code request template → data/cc_request_template.json.

The fidelity-replay pipeline needs the version-matched Claude Code system prompt + tool
definitions + beta headers to reconstruct faithful API requests. The SWE-chat dataset (and
some session logs) don't carry system/tools, so we supply them from a real captured CC
request. This script makes that capture reproducible instead of a private one-off:

  1. starts a local recording HTTP server,
  2. points the `claude` CLI at it via ANTHROPIC_BASE_URL and fires one trivial prompt,
  3. writes the captured POST body + headers to data/cc_request_template.json (tracked).

Usage:  python3 scripts/capture_cc_template.py   (requires the `claude` CLI installed)
The committed template is a snapshot; re-run this to refresh it against a newer CC version.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cc_request_template.json")
PORT = 18923
captured = {}

CANNED = {"id": "msg_capture", "type": "message", "role": "assistant",
          "model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "hi"}],
          "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 1}}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not captured:
            captured["path"] = self.path
            captured["headers"] = dict(self.headers)
            captured["body"] = json.loads(body)
        # minimal streaming response so the CLI exits cleanly
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for name, ev in [
            ("message_start", {"type": "message_start", "message": {**CANNED, "content": []}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "hi"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"})]:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(ev)}\n\n".encode())

    def log_message(self, *a):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{PORT}",
           "ANTHROPIC_API_KEY": "dummy"}
    subprocess.run(["claude", "-p", "hi", "--model", "claude-sonnet-4-6", "--max-turns", "1"],
                   env=env, capture_output=True, timeout=60)
    srv.shutdown()
    if not captured:
        sys.exit("no request captured — is the `claude` CLI installed and on PATH?")
    with open(OUT, "w") as f:
        json.dump(captured, f)
    b = captured["body"]
    print(f"wrote {os.path.relpath(OUT)}: {len(b['tools'])} tools, model {b['model']}")


if __name__ == "__main__":
    main()
