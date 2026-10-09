"""Shared PAL audit fixtures. Run safe and fast in separate Python processes."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the complete PAL package, including PixWindow and OpenGLWindow APIs.
# The repository root must be importable (e.g. PYTHONPATH=<repo parent>).
import PythonAutomataLibrary as pal  # noqa: E402

MODE = os.environ.get("PAL_TEST_MODE", "safe").lower()
if MODE not in ("safe", "fast"):
    raise ValueError("PAL_TEST_MODE must be 'safe' or 'fast'")
if MODE == "fast":
    pal.FastMode()


@pytest.fixture(scope="session")
def api():
    return pal


@pytest.fixture(scope="session")
def safe_mode():
    return MODE == "safe"
