"""Locate bundled resources in source, PyInstaller, and wheel installations."""

import sys
from pathlib import Path


def resource_path(name: str) -> Path:
    source = Path(__file__).resolve().parent.parent / name
    if source.exists():
        return source
    return Path(sys.prefix) / "share" / "oghdeii" / name
