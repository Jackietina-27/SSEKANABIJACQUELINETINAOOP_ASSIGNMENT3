"""Path setup and factory fixtures.

Matplotlib runs headless and every weather test replays recorded JSON, so the
suite needs neither a display nor a network.
"""
import json
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.weather.providers import RECORDINGS, Recorder     # noqa: E402


@pytest.fixture(scope="session")
def recordings_dir() -> Path:
    assert RECORDINGS.exists(), f"no recordings at {RECORDINGS}"
    return RECORDINGS


@pytest.fixture
def replay() -> Recorder:
    """A caller that serves saved replies instead of making requests."""
    return Recorder()


@pytest.fixture
def body(recordings_dir):
    """Return the decoded body of one recording, by stem."""
    def read(stem: str):
        path = recordings_dir / f"{stem}.json"
        assert path.exists(), f"no recording named {stem}"
        return json.loads(path.read_text(encoding="utf-8"))["body"]
    return read
