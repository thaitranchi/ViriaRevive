"""Shared utility functions for ViriaRevive."""

import time
from pathlib import Path


def fmt_time(seconds: int | float) -> str:
    """Format seconds to H:MM:SS or M:SS."""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def wait_for_file_unlock(path: Path, timeout: float = 5.0, interval: float = 0.5) -> bool:
    """Wait up to `timeout` seconds for a file to become readable.

    On Windows, newly written files can be temporarily locked by the OS
    (antivirus, indexing). Returns True if the file became readable, False
    if the timeout expired.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(path, 'rb'):
                return True
        except OSError:
            time.sleep(interval)
    return False
