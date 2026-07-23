"""SSL context for the raw urllib call sites.

The anthropic/openai/httpx clients bundle certifi, but ``urllib`` verifies
against OpenSSL's compiled-in CA paths — which don't exist in uv's
python-build-standalone interpreters, so verification fails unless the OS
exports ``SSL_CERT_FILE`` (NixOS shells often don't). Fall back to certifi
only when the default context loaded no CAs, so an OS-provided trust store
(including corporate/self-signed additions) still wins when present.
"""
import functools
import ssl


@functools.lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"] == 0:
        import certifi
        ctx.load_verify_locations(certifi.where())
    return ctx
