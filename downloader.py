import yt_dlp
from pathlib import Path
import logging
from config import DOWNLOADS_DIR
from utils import wait_for_file_unlock

logger = logging.getLogger(__name__)

# Prefer H.264 (avc1) which every ffmpeg supports.
# Fallback chain avoids AV1/VP9 codec issues on Windows.
_FORMAT = (
    "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[acodec^=mp4a]/"
    "bestvideo[vcodec^=avc1][height<=1080]+bestaudio/"
    "bestvideo[height<=1080]+bestaudio/"
    "best"
)


def download_video(url: str, output_dir: Path = DOWNLOADS_DIR) -> Path | None:
    """Download a YouTube video and return the file path, or None on failure."""
    output_dir.mkdir(exist_ok=True)

    ydl_opts = {
        "format": _FORMAT,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "restrictfilenames": True,
        "quiet": False,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            path = Path(filename)
            if path.exists():
                # Wait briefly for the OS to release the file handle
                wait_for_file_unlock(path, timeout=2.0)
            return path
    except yt_dlp.DownloadError as e:
        logger.error(f"yt-dlp download failed for {url}: {e}")
        return None
    except Exception:
        logger.exception(f"Unexpected download error for {url}")
        return None
