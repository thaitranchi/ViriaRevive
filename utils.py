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


def auto_clip_count(vid_duration: float, clip_duration: int) -> int:
    """Smart auto: scale clips based on video length."""
    vid_mins = vid_duration / 60
    if vid_mins < 5:
        n = max(2, min(3, int(vid_mins / 1.5)))
    elif vid_mins < 15:
        n = max(3, int(vid_mins / 3))
    elif vid_mins < 30:
        n = max(5, int(vid_mins / 3.5))
    elif vid_mins < 60:
        n = max(8, int(vid_mins / 3.5))
    elif vid_mins < 120:
        n = max(15, min(30, int(vid_mins / 4)))
    else:
        n = max(25, min(50, int(vid_mins / 4)))
    if clip_duration < 20:
        n = int(n * 1.3)
    elif clip_duration > 60:
        n = max(2, int(n * 0.7))
    return max(2, min(50, n))
