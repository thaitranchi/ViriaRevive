#!/usr/bin/env python3
"""ViriaRevive Desktop App – debug mode (shows console for logs).
For no-console launch, double-click app.pyw instead."""

import logging
import sys
import webview
from pathlib import Path
from api_bridge import ApiBridge
from tray import TrayManager

logger = logging.getLogger(__name__)


def _get_base_dir():
    """Get the base directory — works for both dev and PyInstaller frozen builds."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


_force_closing = False


def main():
    global _force_closing
    start_minimized = "--minimized" in sys.argv or "--startup" in sys.argv

    api = ApiBridge()
    gui_dir = _get_base_dir() / "gui"
    gui_html = gui_dir / "index.html"

    if not gui_html.exists():
        # Fallback: serve an inline error page so the user sees something useful
        fallback = f"""
<!DOCTYPE html><html><body style="background:#0a0a0f;color:#eee;font-family:sans-serif;
padding:40px;text-align:center"><h1>ViriaRevive</h1>
<p style="color:#f88">GUI files not found.</p>
<p>Expected: <code>{gui_html}</code></p>
<p>Try reinstalling or building the app.</p></body></html>"""
        _url = f"data:text/html,{fallback.replace(chr(10),'').replace(chr(13),'')}"
    else:
        _url = str(gui_html)

    window = webview.create_window(
        title="ViriaRevive",
        url=_url,
        js_api=api,
        width=1100,
        height=750,
        min_size=(900, 600),
        resizable=True,
        background_color="#0a0a0f",
        minimized=start_minimized,
    )

    api._window = window

    # System tray — minimize to tray instead of closing
    tray = TrayManager(window, on_quit_callback=lambda: _force_quit(window, tray))

    def on_loaded():
        tray.start()
        if start_minimized:
            tray.on_minimize()

    def on_minimized():
        tray.on_minimize()

    def on_closing():
        # If force-quit was triggered, allow the close
        if _force_closing:
            return True
        # Otherwise minimize to tray
        tray.on_minimize()
        return False

    window.events.loaded += on_loaded
    window.events.minimized += on_minimized
    window.events.closing += on_closing

    webview.start(debug=True)


def _force_quit(window, tray):
    """Force-quit: stop tray, destroy window, and exit."""
    global _force_closing
    _force_closing = True
    tray.stop()
    try:
        window.destroy()
    except Exception as e:
        logger.debug("Force quit destroy error: %s", e)
    sys.exit(0)


if __name__ == "__main__":
    main()
