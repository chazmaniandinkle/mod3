"""Single source of truth for the mod3 runtime version.

Reads from ``importlib.metadata`` (works for both editable ``pip install -e .``
and packaged installs).  Falls back to reading ``pyproject.toml`` directly so
that ``python server.py`` from a bare checkout without ``pip install`` also
returns the correct value rather than a hardcoded literal.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _read_version() -> str:
    # Preferred path: package metadata installed by pip / uv.
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("mod3")
    except Exception:
        pass

    # Fallback: parse pyproject.toml from the same directory as this file.
    _here = Path(__file__).parent
    _pyproject = _here / "pyproject.toml"
    if _pyproject.exists():
        # tomllib is stdlib in Python 3.11+; tomli is the back-port.
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomllib  # type: ignore[no-redef]
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]

        with _pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        ver = data.get("project", {}).get("version")
        if ver:
            return ver

    # Last resort — should never be reached in a well-formed checkout.
    return "unknown"


__version__: str = _read_version()
