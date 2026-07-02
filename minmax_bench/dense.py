"""Read condense creds from the local ``dense`` CLI config.

The ``dense`` client stores per-environment creds under ``~/.config/dense``:

    ~/.config/dense/token           prod auth token  (-> x-condense-auth-token)
    ~/.config/dense/user            prod user id     (-> x-condense-user-id)
    ~/.config/dense/target          active profile name (empty -> prod)
    ~/.config/dense/<name>/token    profile auth token
    ~/.config/dense/<name>/user     profile user id
    ~/.config/dense/<name>/profile.toml   { api_url, name }

``prod`` is the zero-config default baked into the binary (no profile.toml), so we
carry its api_url here. Other profiles read api_url from their
profile.toml. This lets the condense strategy authenticate exactly as a real
``dense`` invocation would, with no keys in ``.env``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DENSE_HOME = Path.home() / ".config" / "dense"

# prod is baked into the dense binary (no profile.toml on disk).
PROD_API_URL = "https://api.condense.chat"


@dataclass(frozen=True)
class DenseProfile:
    name: str
    api_url: str
    auth_token: str | None
    user_id: str | None


def _read(path: Path) -> str | None:
    try:
        v = path.read_text().strip()
        return v or None
    except OSError:
        return None


def active_profile_name(home: Path = DENSE_HOME) -> str:
    """The profile the ``target`` pointer selects; empty/absent means prod."""
    return _read(home / "target") or "prod"


def load_profile(name: str | None = None, home: Path = DENSE_HOME) -> DenseProfile:
    """Resolve a dense profile's api_url + creds. ``None`` follows ``target``."""
    name = name or active_profile_name(home)
    if name == "prod":
        return DenseProfile(
            name="prod",
            api_url=PROD_API_URL,
            auth_token=_read(home / "token"),
            user_id=_read(home / "user"),
        )
    pdir = home / name
    api_url = PROD_API_URL
    toml_path = pdir / "profile.toml"
    if toml_path.exists():
        try:
            api_url = tomllib.loads(toml_path.read_text()).get("api_url", api_url)
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return DenseProfile(
        name=name,
        api_url=api_url,
        auth_token=_read(pdir / "token"),
        user_id=_read(pdir / "user"),
    )
