"""Ensure backend package imports work when running scripts from tests/."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent

for root in (BACKEND_ROOT, TESTS_ROOT):
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
