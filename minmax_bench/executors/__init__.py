from .base import Executor, Measurement, output_tokens_for
from .noop import NoopExecutor
from .proxy import ProxyExecutor

__all__ = [
    "Executor",
    "Measurement",
    "output_tokens_for",
    "NoopExecutor",
    "ProxyExecutor",
    "make_executor",
]


def make_executor(kind: str, **kwargs) -> Executor:
    if kind == "proxy":
        return ProxyExecutor(**kwargs)
    if kind == "noop":
        return NoopExecutor(**kwargs)
    raise ValueError(f"unknown executor kind {kind!r}")
