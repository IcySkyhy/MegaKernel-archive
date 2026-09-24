"""Pytest path setup for local module imports."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MEGAKERNEL_DIR = REPO_ROOT / "csrc" / "megakernel"

for path in (REPO_ROOT, MEGAKERNEL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
