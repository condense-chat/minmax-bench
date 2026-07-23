from .base import Executor, Measurement, output_tokens_for
from .noop import NoopExecutor
from .proxy import ProxyExecutor
from .rewrite import CaptureRewriteExecutor, RewriteExecutor, RewriteInvokeExecutor

__all__ = [
    "Executor",
    "Measurement",
    "output_tokens_for",
    "NoopExecutor",
    "ProxyExecutor",
    "RewriteExecutor",
    "CaptureRewriteExecutor",
    "RewriteInvokeExecutor",
    "make_executor",
]


def make_executor(kind: str, **kwargs) -> Executor:
    if kind == "proxy":
        return ProxyExecutor(**kwargs)
    if kind == "rewrite":
        return RewriteExecutor(**kwargs)
    if kind == "rewrite_capture":
        return CaptureRewriteExecutor(**kwargs)
    if kind == "rewrite_invoke":
        return RewriteInvokeExecutor(**kwargs)
    if kind == "noop":
        return NoopExecutor(**kwargs)
    raise ValueError(f"unknown executor kind {kind!r}")
