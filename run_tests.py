"""Comprehensive test suite for ViriaRevive core logic.

Run with: python run_tests.py [-v]
"""

import sys
import os
import json
import unittest
import importlib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def _has_module(name):
    """Check if a module is available without importing it."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


class TestUtils(unittest.TestCase):
    """Tests for shared utils."""

    def test_fmt_time_hms(self):
        from utils import fmt_time
        self.assertEqual(fmt_time(3661), "1:01:01")

    def test_fmt_time_ms(self):
        from utils import fmt_time
        self.assertEqual(fmt_time(65), "1:05")

    def test_fmt_time_zero(self):
        from utils import fmt_time
        self.assertEqual(fmt_time(0), "0:00")

    def test_wait_for_file_unlock_missing(self):
        from utils import wait_for_file_unlock
        result = wait_for_file_unlock(Path("nonexistent_file_xyz"), timeout=0.1)
        self.assertFalse(result)

    def test_wait_for_file_unlock_exists(self):
        from utils import wait_for_file_unlock
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            p = Path(f.name)
        try:
            result = wait_for_file_unlock(p, timeout=1.0)
            self.assertTrue(result)
        finally:
            p.unlink(missing_ok=True)


class TestSubtitleLogic(unittest.TestCase):
    """Tests for subtitle generation and sanitization."""

    def setUp(self):
        import subtitler
        self.subtitler = subtitler

    def test_sanitize_removes_brackets(self):
        words = [{"text": "[Music]", "start": 0.0, "end": 0.5}]
        result = self.subtitler._sanitize_word_times(words)
        self.assertEqual(len(result), 0)

    def test_sanitize_fixes_overlap(self):
        words = [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.4, "end": 1.0},
        ]
        result = self.subtitler._sanitize_word_times(words)
        self.assertGreaterEqual(result[1]["start"], result[0]["end"])

    def test_sanitize_fixes_zero_duration(self):
        words = [{"text": "Hello", "start": 1.0, "end": 1.0}]
        result = self.subtitler._sanitize_word_times(words)
        self.assertGreater(result[0]["end"], result[0]["start"])

    def test_sanitize_cleans_punctuation(self):
        words = [{"text": "Hello?!", "start": 0.0, "end": 0.5}]
        result = self.subtitler._sanitize_word_times(words)
        self.assertEqual(result[0]["text"], "Hello")

    def test_sanitize_keeps_apostrophe(self):
        words = [{"text": "don't", "start": 0.0, "end": 0.5}]
        result = self.subtitler._sanitize_word_times(words)
        self.assertEqual(result[0]["text"], "don't")

    def test_group_phrases_defaults(self):
        words = [
            {"text": "A", "start": 0.0, "end": 0.3},
            {"text": "B", "start": 0.4, "end": 0.7},
            {"text": "C", "start": 0.8, "end": 1.1},
        ]
        phrases = self.subtitler._group_phrases(words, max_words=4, max_dur=2.5, max_gap=0.8)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(len(phrases[0]["words"]), 3)

    def test_group_phrases_max_words(self):
        words = [
            {"text": "A", "start": 0.0, "end": 0.2},
            {"text": "B", "start": 0.3, "end": 0.5},
            {"text": "C", "start": 0.6, "end": 0.8},
            {"text": "D", "start": 0.9, "end": 1.1},
        ]
        phrases = self.subtitler._group_phrases(words, max_words=2)
        self.assertEqual(len(phrases), 2)

    def test_ass_time_format(self):
        result = self.subtitler._ass_time(3661.25)
        self.assertEqual(result, "1:01:01.25")

    def test_ass_header_margin_v(self):
        header = self.subtitler._ass_header(1080, 1920, dict(self.subtitler.STYLES["tiktok"]), margin_v=100)
        self.assertIn("PlayResX: 1080", header)
        self.assertIn("PlayResY: 1920", header)
        self.assertIn("100", header)

    def test_generate_subtitles_none_words(self):
        result = self.subtitler.generate_subtitles([], Path("dummy.ass"))
        self.assertIsNone(result)

    def test_get_available_styles(self):
        styles = self.subtitler.get_available_styles()
        self.assertGreaterEqual(len(styles), 4)
        self.assertTrue(any(s["id"] == "tiktok" for s in styles))


class TestClipperLogic(unittest.TestCase):
    """Tests for FFmpeg expression builder and crop logic."""

    def setUp(self):
        import clipper
        self.clipper = clipper

    def test_lerp_expr_three_keyframes(self):
        times = [0, 2, 4]
        values = [10, 20, 30]
        expr = self.clipper._build_lerp_expr(times, values)
        expected = "if(lt(t\\,2.000)\\,10\\,if(lt(t\\,4.000)\\,20\\,30))"
        self.assertEqual(expr, expected)

    def test_lerp_expr_single_keyframe(self):
        expr = self.clipper._build_lerp_expr([0], [42])
        self.assertEqual(expr, "42")

    def test_lerp_expr_constant(self):
        expr = self.clipper._build_lerp_expr([0, 2, 4], [5, 5, 5])
        self.assertEqual(expr, "5")

    def test_chain_vf_none(self):
        result = self.clipper._chain_vf(None, "scale=1080:1920", None)
        self.assertEqual(result, "scale=1080:1920")

    def test_chain_vf_all_none(self):
        result = self.clipper._chain_vf(None, None)
        self.assertEqual(result, "")


class TestHwaccelLogic(unittest.TestCase):
    """Tests for hardware acceleration fallback."""

    def setUp(self):
        import hwaccel
        hwaccel.reset_probe_cache()
        self.hwaccel = hwaccel

    def test_swap_hwenc_to_cpu(self):
        cmd = [
            "ffmpeg", "-y", "-ss", "10",
            "-hwaccel", "auto",
            "-i", "in.mp4", "-t", "30",
            "-vf", "scale=1080:1920",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc:v", "vbr",
            "-cq:v", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "out.mp4",
        ]
        rewritten = self.hwaccel._swap_cmd_encode_to_cpu(cmd, "medium", "23")
        self.assertIn("libx264", rewritten)
        self.assertNotIn("-hwaccel", rewritten)
        self.assertNotIn("h264_nvenc", rewritten)
        self.assertNotIn("-cq:v", rewritten)
        self.assertIn("-crf", rewritten)

    def test_swap_hwdec_cuvid(self):
        cmd = [
            "ffmpeg", "-y",
            "-c:v", "h264_cuvid",
            "-i", "in.mp4",
            "-c:v", "libx264",
            "out.mp4",
        ]
        rewritten = self.hwaccel._swap_cmd_encode_to_cpu(cmd, "medium", "23")
        self.assertNotIn("h264_cuvid", rewritten)

    def test_resolve_yolo_device_cpu_fallback(self):
        device = self.hwaccel.resolve_yolo_device("cpu")
        self.assertEqual(device, "cpu")

    def test_video_encode_args_cpu(self):
        args = self.hwaccel.video_encode_args(preset="medium", crf="23", force_cpu=True)
        self.assertIn("libx264", args)
        self.assertIn("-crf", args)

    def test_crf_clamping(self):
        args = self.hwaccel.video_encode_args(preset="ultrafast", crf="999", force_cpu=True)
        self.assertIn("51", args)  # max CRF is 51


class TestDetectorLogic(unittest.TestCase):
    """Tests for detector helper functions."""

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_fallback_moments_short_video(self):
        import detector
        moments = detector._fallback_moments(total_seconds=5, num_clips=3, clip_duration=30, min_gap=15)
        self.assertEqual(len(moments), 0)

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_fallback_moments_long_video(self):
        import detector
        moments = detector._fallback_moments(total_seconds=300, num_clips=3, clip_duration=30, min_gap=15)
        self.assertEqual(len(moments), 3)
        starts = [m["start"] for m in moments]
        self.assertEqual(starts, sorted(starts))

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_fallback_moments_edge_single(self):
        import detector
        moments = detector._fallback_moments(total_seconds=30, num_clips=1, clip_duration=30, min_gap=15)
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["duration"], 30)


class TestCropperConstants(unittest.TestCase):
    """Tests that cropper constants are consistent."""

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_constants_import(self):
        import cropper
        self.assertIsNotNone(cropper.YOLO_CONF_BATCH)
        self.assertIsNotNone(cropper.YOLO_CONF_RETRY)
        self.assertIsNotNone(cropper.YOLO_CONF_SINGLE)
        self.assertGreater(cropper.YOLO_CONF_SINGLE, cropper.YOLO_CONF_RETRY)

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_dimensions(self):
        import cropper
        w, h = cropper._get_dimensions(Path("nonexistent_file.mp4"))
        self.assertEqual(w, 0)
        self.assertEqual(h, 0)


class TestMainLogic(unittest.TestCase):
    """Tests for main.py entry point logic."""

    @unittest.skipIf(not _has_module("yt_dlp"), "yt-dlp not installed")
    def test_process_returns_list_on_fail(self):
        import main
        result = main.process("invalid://url", num_clips=1, clip_duration=10)
        self.assertIsInstance(result, list)


class TestConfig(unittest.TestCase):
    """Tests for configuration."""

    def test_base_dir_exists(self):
        import config
        self.assertTrue(config.BASE_DIR.exists())

    def test_dirs_created(self):
        import config
        self.assertTrue(config.CLIPS_DIR.exists())
        self.assertTrue(config.SUBTITLES_DIR.exists())


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
