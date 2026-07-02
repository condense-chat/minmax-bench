"""Auto-provision the local tools a run needs.

``run`` can set up its own tooling instead of making the user prepare it by hand.
For each tool the user opts into (in the wizard's setup step, or ``--setup`` in
headless mode) we install it if missing, then bring it up:

* **dense**    — install the CLI if absent, then confirm creds exist; if not, hint
  (or run) ``dense login`` so condense creds land in ``~/.config/dense``.
* **headroom** — install the CLI if absent, then start ``headroom proxy`` as a
  managed subprocess, wait until reachable, and stop it when the run ends.

Install commands come from settings (``*_install_cmd``), whose defaults are each
project's own documented install line. Everything is best-effort: whatever can't
be provisioned falls back to what preflight reports (headroom skipped, condense skipped).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import httpx
from rich.console import Console
from rich.prompt import Confirm

from .config import get_settings
from .strategies import tool_for


def tools_for(strategies: list[str]) -> list[str]:
    """The provisionable tools the given strategies imply (deduped, in order),
    derived from each matrix entry's runner (headroom/dense)."""
    out: list[str] = []
    for st in strategies:
        tool = tool_for(st)
        if tool and tool not in out:
            out.append(tool)
    return out


def _reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        httpx.get(url, timeout=timeout)  # any HTTP reply proves something listens
        return True
    except Exception:
        return False


def _host_port(url: str) -> tuple[str, int]:
    u = urlparse(url)
    return (u.hostname or "127.0.0.1"), (u.port or 80)


def _run(cmd: str, console: Console, timeout: int = 900) -> bool:
    """Run a documented install command (shell so `curl … | sh` works)."""
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout, env=os.environ.copy())
        return True
    except (subprocess.SubprocessError, OSError) as e:
        console.print(f"[red]✗[/] `{cmd}` failed: {e}")
        return False


def ensure_dense(console: Console) -> None:
    """Install the dense CLI if missing; ensure (or hint) login for condense creds."""
    s = get_settings()
    if not shutil.which("dense"):
        console.print(f"[yellow]•[/] dense not found; installing — [dim]{s.dense_install_cmd}[/]")
        if not _run(s.dense_install_cmd, console, timeout=300) or not shutil.which("dense"):
            console.print("[red]✗[/] dense unavailable; condense will be skipped")
            return
        console.print(f"[green]✓[/] installed dense ({shutil.which('dense')})")
    else:
        console.print("[green]✓[/] dense CLI present")

    try:
        from .dense import load_profile
        prof = load_profile(s.condense_profile or None)
        logged_in = bool(getattr(prof, "auth_token", None) and getattr(prof, "user_id", None))
    except Exception:
        logged_in = False
    if logged_in:
        console.print("[green]✓[/] dense logged in (condense creds present)")
        return
    console.print(
        "[yellow]•[/] dense not logged in — run [bold]dense login[/] "
        "to populate ~/.config/dense"
    )
    if console.is_terminal and Confirm.ask(
        "[cyan]run `dense login` now?[/]", default=False, console=console
    ):
        with contextlib.suppress(Exception):
            subprocess.run(["dense", "login"], env=os.environ.copy())



def start_headroom(console: Console) -> subprocess.Popen | None:
    """Install headroom if missing, start `headroom proxy`, wait until reachable."""
    s = get_settings()
    base = s.headroom_base_url
    if _reachable(base):
        console.print(f"[green]✓[/] headroom already running at {base}")
        return None
    if not shutil.which("headroom"):
        console.print(
            f"[yellow]•[/] headroom not found; installing — [dim]{s.headroom_install_cmd}[/]"
        )
        if not _run(s.headroom_install_cmd, console, timeout=600):
            console.print("[red]✗[/] headroom unavailable; it will be skipped")
            return None
    exe = shutil.which("headroom")
    if not exe:
        console.print("[red]✗[/] headroom not on PATH after install; it will be skipped")
        return None

    host, port = _host_port(base)
    log = tempfile.NamedTemporaryFile(prefix="headroom-", suffix=".log", delete=False)
    # No global --mode: the headroom/headroom-kompress strategies pick their mode
    # per request via the x-headroom-mode header.
    proc = subprocess.Popen(
        [exe, "proxy", "--host", host, "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            console.print(f"[red]✗[/] headroom exited on startup; see {log.name}")
            return None
        if _reachable(base):
            console.print(f"[green]✓[/] started headroom proxy at {base} (mode=token)")
            return proc
        time.sleep(0.3)
    console.print(f"[red]✗[/] headroom did not come up in time; see {log.name}")
    _terminate(proc)
    return None


def _terminate(proc: subprocess.Popen) -> None:
    with contextlib.suppress(Exception):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@contextlib.contextmanager
def provision(setup: list[str], console: Console | None = None):
    """Install/start the requested tools; tear down anything we started on exit.

    ``setup`` is a subset of {"dense", "headroom"}.
    """
    console = console or Console()
    procs: list[subprocess.Popen] = []
    if setup:
        console.rule("[bold]local setup")
    if "dense" in setup:
        ensure_dense(console)
    if "headroom" in setup:
        proc = start_headroom(console)
        if proc is not None:
            procs.append(proc)
    try:
        yield
    finally:
        for p in procs:
            _terminate(p)
        if procs:
            console.print("[dim]stopped auto-started proxies[/]")
