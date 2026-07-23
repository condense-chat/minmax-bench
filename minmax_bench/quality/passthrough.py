"""Passthrough control proxy — vanilla routed through a local proxy that changes NOTHING.

Why this exists: Claude Code composes a measurably different request whenever
``ANTHROPIC_BASE_URL`` is non-default (~8-9k extra tokens per request in our runs) — a
flat confound shared by EVERY proxy arm (condense, headroom). The ``vanilla-proxy`` arm
runs the vanilla agent through this do-nothing forwarder, so:

  - vanilla-proxy vs vanilla  = the proxy-WIRING effect alone (the confound, isolated);
  - a proxy arm vs vanilla-proxy = the arm's own effect with the confound subtracted.

Bytes are forwarded verbatim in both directions (headers minus hop-by-hop, body
streamed chunk-by-chunk so SSE latency is preserved). Standard library plus a
certifi fallback for interpreters with no OpenSSL CA paths (see minmax_bench.tls).
"""
import http.server
import socketserver
import threading
import urllib.error
import urllib.request

from minmax_bench.tls import ssl_context

UPSTREAM = "https://api.anthropic.com"

# hop-by-hop headers must not be forwarded (RFC 9110 §7.6.1); content-length is
# recomputed by urllib, accept-encoding is pinned to identity so compressed bytes
# aren't re-chunked into something the client didn't negotiate
_SKIP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
         "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
         "accept-encoding"}


class _Forwarder(http.server.BaseHTTPRequestHandler):
    # HTTP/1.0 + Connection: close => the client reads to EOF; no chunked re-encoding
    # needed for streamed (SSE) upstream responses
    protocol_version = "HTTP/1.0"
    upstream = UPSTREAM

    def log_message(self, *args):  # quiet: one line per request would swamp the run log
        pass

    def _forward(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _SKIP}
        headers["accept-encoding"] = "identity"
        req = urllib.request.Request(self.upstream + self.path, data=body,
                                     headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=600, context=ssl_context())
        except urllib.error.HTTPError as e:
            resp = e  # an upstream 4xx/5xx is still a response — forward it verbatim
        except OSError as e:
            self.send_error(502, explain=str(e)[:200])
            return
        with resp:
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in _SKIP:
                    self.send_header(k, v)
            self.send_header("connection", "close")
            self.end_headers()
            while True:
                # read1 = whatever bytes are available NOW — an SSE event is relayed the
                # moment it arrives instead of buffering to a fixed block size
                chunk = resp.read1(65536)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return  # client went away mid-stream; nothing to salvage

    do_GET = do_POST = do_PUT = do_DELETE = _forward


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True  # in-flight relays must not block shutdown


class PassthroughProxy:
    """In-process passthrough server. Context manager; ``port`` on the instance is the
    bound port (useful with port=0 = OS-assigned)."""

    def __init__(self, port=0, upstream=UPSTREAM):
        self._want_port, self.upstream, self.port = port, upstream, None
        self._srv = self._thread = None

    def start(self):
        handler = type("H", (_Forwarder,), {"upstream": self.upstream})
        self._srv = _Server(("0.0.0.0", self._want_port), handler)  # noqa: S104 — the
        # docker container reaches the host via host.docker.internal, so bind all ifaces
        self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None

    __enter__ = start

    def __exit__(self, *exc):
        self.stop()
