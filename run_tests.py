"""Comprehensive test suite for ViriaRevive core logic.

Run with: python run_tests.py [-v]
"""

import sys
import os
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
        self.assertGreaterEqual(len(styles), 6)
        self.assertTrue(any(s["id"] == "tiktok" for s in styles))
        self.assertTrue(any(s["id"] == "game" for s in styles))


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

    def test_effects_includes_gaming_presets(self):
        effects = self.clipper.get_effects_list()
        effect_ids = [e["id"] for e in effects]
        self.assertIn("streamer", effect_ids)
        self.assertIn("hdr", effect_ids)


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
        self.assertIsNotNone(cropper.YOLO_CONF)
        self.assertIsNotNone(cropper.YOLO_CONF_SINGLE)
        self.assertGreater(cropper.YOLO_CONF_SINGLE, cropper.YOLO_CONF)

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

    def test_auto_clip_count_short(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=120, clip_duration=25)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 2)

    def test_auto_clip_count_medium(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=600, clip_duration=25)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 3)

    def test_auto_clip_count_long(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=3600, clip_duration=25)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 5)

    def test_auto_clip_count_very_long(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=7200, clip_duration=25)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 15)

    def test_auto_clip_count_clamps_min(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=10, clip_duration=25)
        self.assertGreaterEqual(n, 2)

    def test_auto_clip_count_clamps_max(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=86400, clip_duration=25)
        self.assertLessEqual(n, 50)

    def test_auto_clip_count_short_clip_duration(self):
        from utils import auto_clip_count
        n = auto_clip_count(vid_duration=600, clip_duration=15)
        n_default = auto_clip_count(vid_duration=600, clip_duration=25)
        self.assertGreaterEqual(n, n_default)


class TestPipelineCache(unittest.TestCase):
    """Tests for pipeline cache."""

    def setUp(self):
        import pipeline_cache
        # Use a unique stem to avoid cross-test contamination
        self.cache = pipeline_cache.PipelineCache("_test_stem_" + str(id(self)))

    def tearDown(self):
        self.cache.clear_state()
        self.cache.cleanup_all_wavs()

    def test_state_roundtrip(self):
        state = self.cache.load_state()
        self.assertIsNotNone(state)
        self.assertEqual(state.stem, self.cache.stem)

    def test_save_and_load_state(self):
        from pipeline_cache import PipelineState
        state = PipelineState(stem=self.cache.stem, num_clips=5, step_downloaded=True)
        self.cache.save_state(state)
        loaded = self.cache.load_state()
        self.assertEqual(loaded.num_clips, 5)
        self.assertTrue(loaded.step_downloaded)

    def test_mark_and_check_clip_done(self):
        self.assertFalse(self.cache.is_clip_done(1))
        self.cache.mark_clip_done(1)
        self.assertTrue(self.cache.is_clip_done(1))

    def test_done_clips(self):
        self.cache.mark_clip_done(1)
        self.cache.mark_clip_done(3)
        done = self.cache.done_clips()
        self.assertIn(1, done)
        self.assertIn(3, done)
        self.assertNotIn(2, done)

    def test_resume_step_download(self):
        from pipeline_cache import PipelineState
        state = PipelineState()
        self.assertEqual(state.resume_step, "download")

    def test_resume_step_clips(self):
        from pipeline_cache import PipelineState
        state = PipelineState(step_downloaded=True, step_detected=True, step_reranked=True, num_clips=5)
        self.assertEqual(state.resume_step, "clips")

    def test_resume_step_done(self):
        from pipeline_cache import PipelineState
        state = PipelineState(step_downloaded=True, step_detected=True, step_reranked=True,
                              num_clips=3, clips_completed=[1, 2, 3])
        self.assertEqual(state.resume_step, "done")

    def test_all_clips_done(self):
        type("S", (), {"num_clips": 3, "clips_completed": [1, 2, 3]})()
        from pipeline_cache import PipelineState
        ps = PipelineState(num_clips=3, clips_completed=[1, 2, 3])
        self.assertTrue(ps.all_clips_done)
        ps2 = PipelineState(num_clips=3, clips_completed=[1, 2])
        self.assertFalse(ps2.all_clips_done)

    def test_set_and_get_moments(self):
        moments = [{"start": 10, "end": 35}]
        self.cache.set_moments(moments)
        loaded = self.cache.get_moments()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["start"], 10)

    def test_transcript_cache(self):
        words = [{"text": "hello", "start": 0.0, "end": 0.5}]
        self.cache.set_transcript(1, words, "hello")
        loaded = self.cache.get_transcript(1)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded[0]["text"], "hello")

    def test_set_and_get_crop_params(self):
        crop = (540, 960, 100, 0)
        self.cache.set_crop_params(1, crop)
        loaded = self.cache.get_crop_params(1)
        self.assertEqual(loaded, crop)


class TestSentenceBoundary(unittest.TestCase):
    """Tests for sentence boundary detection."""

    def setUp(self):
        from transcriber import find_sentence_boundary
        self.find_sentence_boundary = find_sentence_boundary

    def test_no_words_returns_none(self):
        result = self.find_sentence_boundary([], clip_duration=30.0)
        self.assertIsNone(result)

    def test_period_boundary(self):
        words = [
            {"text": "This", "start": 0.0, "end": 0.3},
            {"text": "is", "start": 0.4, "end": 0.6},
            {"text": "amazing.", "start": 0.7, "end": 1.0},
            {"text": "Next", "start": 1.1, "end": 1.3},
        ]
        result = self.find_sentence_boundary(words, clip_duration=30.0, min_keep=0.0, max_extend=5.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.3, delta=0.1)  # 1.0 + 0.3 pad

    def test_exclamation_boundary(self):
        words = [
            {"text": "Wow", "start": 0.0, "end": 0.3},
            {"text": "that", "start": 0.4, "end": 0.6},
            {"text": "was", "start": 0.7, "end": 0.9},
            {"text": "insane!", "start": 1.0, "end": 1.3},
        ]
        result = self.find_sentence_boundary(words, clip_duration=30.0, min_keep=0.0, max_extend=5.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.6, delta=0.1)

    def test_no_boundary_returns_none(self):
        words = [
            {"text": "Hello", "start": 0.0, "end": 0.3},
            {"text": "world", "start": 0.3, "end": 0.6},
        ]
        result = self.find_sentence_boundary(words, clip_duration=30.0, min_keep=0.0, max_extend=5.0)
        self.assertIsNone(result)

    def test_respects_min_keep(self):
        words = [
            {"text": "A.", "start": 0.0, "end": 0.2},
            {"text": "B.", "start": 0.3, "end": 0.5},
            {"text": "C.", "start": 0.6, "end": 0.8},
            {"text": "D.", "start": 10.0, "end": 10.3},
        ]
        # min_keep=0.8 → don't cut before 24s
        result = self.find_sentence_boundary(words, clip_duration=30.0, min_keep=0.8, max_extend=5.0)
        self.assertIsNone(result)


class TestConfig(unittest.TestCase):
    """Tests for configuration."""

    def test_base_dir_exists(self):
        import config
        self.assertTrue(config.BASE_DIR.exists())

    def test_dirs_created(self):
        import config
        self.assertTrue(config.CLIPS_DIR.exists())
        self.assertTrue(config.SUBTITLES_DIR.exists())

    def test_vision_constants(self):
        import config
        self.assertIsInstance(config.VISION_ENABLED, bool)
        self.assertTrue(config.VISION_MODEL)
        self.assertGreater(config.VISION_FRAMES_PER_MOMENT, 0)
        self.assertTrue(config.TITLE_MODEL)
        self.assertTrue(config.TRANSLATE_MODEL)


class TestVectorLayer(unittest.TestCase):
    """Tests for the pgvector column typing, DDL, and Python similarity path."""

    def setUp(self):
        from sqlalchemy.schema import CreateTable
        self.CreateTable = CreateTable
        import database.vector as vector
        self.vector = vector
        self.vector.reset_backend_cache()

    def tearDown(self):
        self.vector.reset_backend_cache()

    def test_default_column_type_is_vector(self):
        import database.models as models

        col_type = models.ClipEmbedding.__table__.c.embedding.type
        self.assertEqual(str(col_type).upper(), f"VECTOR({self.vector.VECTOR_DIM})")

    def test_array_fallback_renders_float_array(self):
        # Without the extension the column must render as float[] so DDL
        # matches a plain Postgres server.
        from sqlalchemy.dialects import postgresql
        import database.models as models

        ddl = str(
            self.CreateTable(models.ClipEmbedding.__table__).compile(
                dialect=postgresql.dialect()
            )
        ).upper()
        self.vector.force_array_type(True)
        try:
            fallback = str(
                self.CreateTable(models.ClipEmbedding.__table__).compile(
                    dialect=postgresql.dialect()
                )
            ).upper()
            self.assertIn("FLOAT[]", fallback)
            self.assertNotIn("VECTOR(", fallback)
        finally:
            self.vector.force_array_type(False)
        self.assertIn("VECTOR(1024)", ddl)

    def test_ddl_follows_force_array_flag(self):
        # The whole point of the TypeDecorator: the flag can flip *after* the
        # model class exists and the rendered DDL still follows it.
        from sqlalchemy import create_engine
        from sqlalchemy.dialects import postgresql
        import database.models as models

        dialect = postgresql.dialect()
        table = models.ClipEmbedding.__table__

        self.assertIn("VECTOR(1024)", str(self.CreateTable(table).compile(dialect=dialect)).upper())

        self.vector.force_array_type(True)
        try:
            self.assertIn(
                "FLOAT[]", str(self.CreateTable(table).compile(dialect=dialect)).upper()
            )
        finally:
            self.vector.force_array_type(False)

    def test_savepoint_keeps_transaction_usable_after_ddl_failure(self):
        # A denied CREATE EXTENSION aborts the PostgreSQL transaction; without a
        # savepoint every later statement in the caller's block would fail.
        class Conn:
            def __init__(self):
                self.saved = False
                self.rolled_back = False

            def begin_nested(self):
                conn = self

                class _Ctx:
                    def __enter__(_self):
                        conn.saved = True

                    def __exit__(_self, exc_type, *a):
                        if exc_type is not None:
                            conn.rolled_back = True
                        return False

                return _Ctx()

            def execute(self, stmt, *a, **k):
                raise RuntimeError("permission denied to create extension")

        conn = Conn()
        self.assertFalse(self.vector.ensure_vector_extension(conn))
        self.assertTrue(conn.saved)
        self.assertTrue(conn.rolled_back)

    def test_savepoint_is_optional(self):
        # Connections without savepoint support still run the statement.
        class Minimal:
            def __init__(self):
                self.executed = []

            def execute(self, stmt, *a, **k):
                self.executed.append(stmt)
                return _FakeResult([(1,)])

        conn = Minimal()
        self.assertTrue(self.vector.ensure_vector_extension(conn))
        self.assertTrue(conn.executed)

    def test_embedding_column_is_vector_false_without_table(self):
        class NoTable:
            def execute(self, stmt, *a, **k):
                return _FakeResult([])

        self.assertFalse(self.vector.embedding_column_is_vector(NoTable()))
        self.assertFalse(self.vector.table_exists(NoTable()))

    def test_alembic_revision_reads_version(self):
        class Stamped:
            def __init__(self, row):
                self.row = row

            def execute(self, stmt, *a, **k):
                return _FakeResult(self.row)

        self.assertEqual(self.vector.alembic_revision(Stamped([("b7e2f1a4c9d0",)])), "b7e2f1a4c9d0")
        self.assertIsNone(self.vector.alembic_revision(Stamped([])))
        self.assertIsNone(self.vector.alembic_revision(_DeadConn()))

    def test_use_sql_cosine_tracks_flag(self):
        self.assertTrue(self.vector.use_sql_cosine())
        self.vector.force_array_type(True)
        try:
            self.assertFalse(self.vector.use_sql_cosine())
        finally:
            self.vector.force_array_type(False)

    def test_index_is_not_declared_on_model(self):
        # Declared at import time it would freeze the wrong column type; init_db
        # and the Alembic migration create it instead.
        import database.models as models

        self.assertEqual(list(models.ClipEmbedding.__table__.indexes), [])

    def test_hnsw_ddl_uses_cosine_ops(self):
        ddl = self.vector.HNSW_INDEX_SQL
        self.assertIn("USING hnsw", ddl)
        self.assertIn("vector_cosine_ops", ddl)
        self.assertIn("ef_construction", ddl)
        self.assertIn("IF NOT EXISTS", ddl)

    def test_ivfflat_ddl_present_as_fallback(self):
        ddl = self.vector.IVFFLAT_INDEX_SQL
        self.assertIn("USING ivfflat", ddl)
        self.assertIn("vector_cosine_ops", ddl)

    def test_ef_search_sql_is_parameterised(self):
        sql = self.vector.hnsw_ef_search_sql(150)
        self.assertIn("150", sql)
        self.assertNotIn("%d", sql)

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(self.vector.cosine_similarity(v, v), 1.0, places=6)

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_cosine_similarity_orthogonal(self):
        self.assertAlmostEqual(
            self.vector.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6
        )

    @unittest.skipIf(not _has_module("numpy"), "numpy not installed")
    def test_cosine_similarity_unnormalized(self):
        # Scale must not matter — that is the point of cosine over L2.
        a = [1.0, 2.0, 3.0]
        b = [10.0, 20.0, 30.0]
        self.assertAlmostEqual(self.vector.cosine_similarity(a, b), 1.0, places=6)

    def test_cosine_similarity_zero_vector_is_zero(self):
        self.assertEqual(self.vector.cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_cosine_similarity_mismatched_length_is_zero(self):
        self.assertEqual(self.vector.cosine_similarity([1.0, 2.0], [1.0]), 0.0)

    def test_cosine_similarity_none_inputs(self):
        self.assertEqual(self.vector.cosine_similarity(None, [1.0]), 0.0)
        self.assertEqual(self.vector.cosine_similarity([1.0], None), 0.0)

    def test_create_ann_index_falls_back_to_ivfflat(self):
        # hnsw needs pgvector >= 0.5. On an older server we must still get an
        # index rather than silently running sequential scans.
        class OldPgvectorConn:
            def __init__(self):
                self.executed = []

            def execute(self, stmt, *a, **k):
                sql = str(stmt)
                self.executed.append(sql)
                if "hnsw" in sql.lower():
                    raise RuntimeError("hnsw not supported")
                return _FakeResult([])

        conn = OldPgvectorConn()
        self.assertTrue(self.vector.create_ann_indexes(conn))
        joined = " ".join(conn.executed)
        self.assertIn("hnsw", joined)
        self.assertIn("ivfflat", joined)

    def test_create_ann_index_reports_failure_when_both_fail(self):
        class DeadConn:
            def execute(self, stmt, *a, **k):
                raise RuntimeError("no indexes available")

        self.assertFalse(self.vector.create_ann_indexes(DeadConn()))

    def test_create_ann_index_uses_savepoint_per_attempt(self):
        class Conn:
            def __init__(self):
                self.saved = 0
                self.rolled_back = 0

            def begin_nested(self):
                conn = self

                class _Ctx:
                    def __enter__(_self):
                        conn.saved += 1

                    def __exit__(_self, exc_type, *a):
                        if exc_type is not None:
                            conn.rolled_back += 1
                        return False

                return _Ctx()

            def execute(self, stmt, *a, **k):
                if "hnsw" in str(stmt).lower():
                    raise RuntimeError("hnsw not supported")
                return _FakeResult([])

        conn = Conn()
        self.assertTrue(self.vector.create_ann_indexes(conn))
        self.assertEqual(conn.saved, 2)
        self.assertEqual(conn.rolled_back, 1)

    def test_has_ann_index_detects_hnsw(self):
        class HasHnsw:
            def execute(self, stmt, *a, **k):
                return _FakeResult([(1,)])

        self.assertTrue(self.vector.has_ann_index(HasHnsw()))

    def test_has_ann_index_false_when_absent(self):
        class NoIndex:
            def execute(self, stmt, *a, **k):
                return _FakeResult([])

        self.assertFalse(self.vector.has_ann_index(NoIndex()))

    def test_ensure_extension_returns_false_when_blocked(self):
        class BlockedConn:
            def execute(self, stmt, *a, **k):
                raise RuntimeError("permission denied")

        self.assertFalse(self.vector.ensure_vector_extension(BlockedConn()))

    def test_probe_returns_none_when_db_unreachable(self):
        class DeadConn:
            def execute(self, *a, **k):
                raise OSError("connection refused")

        self.assertIsNone(self.vector.probe_vector_support(DeadConn()))

    def test_probe_detects_missing_extension(self):
        class NoExtConn:
            def execute(self, stmt, *a, **k):
                return _FakeResult([])

        self.assertFalse(self.vector.probe_vector_support(NoExtConn()))

    def test_probe_detects_present_extension(self):
        class HasExtConn:
            def execute(self, stmt, *a, **k):
                return _FakeResult([(1,)])

        self.assertTrue(self.vector.probe_vector_support(HasExtConn()))

    def test_vector_backend_defaults_false_without_db(self):
        # No server on this machine → the flag must be False, not True or None.
        self.assertFalse(self.vector.vector_backend())

    def test_cosine_distance_sql_matches_column_type(self):
        # SQL cosine distance is used exactly when the column is vector(N);
        # otherwise the search falls back to the Python scan.
        import database.repository as repo

        query = [0.0] * self.vector.VECTOR_DIM
        self.assertIsNotNone(repo._cosine_distance_sql(query))
        self.vector.force_array_type(True)
        try:
            self.assertIsNone(repo._cosine_distance_sql(query))
        finally:
            self.vector.force_array_type(False)

    def test_find_similar_clips_empty_query_returns_nothing(self):
        # Guard against a zero-length query reaching Postgres or the scan.
        from database.repository import find_similar_clips

        self.assertEqual(find_similar_clips(None, [], limit=5), [])


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0][0] if self._rows else None


class _DeadConn:
    """Connection double that raises on every statement."""

    def execute(self, *a, **k):
        raise OSError("connection refused")

    def exec_driver_sql(self, *a, **k):
        raise OSError("connection refused")


class TestVectorSql(unittest.TestCase):
    """Tests for the SQL the vector layer emits. No database required."""

    def test_cosine_distance_expr_uses_cosine_operator(self):
        from sqlalchemy.dialects import postgresql

        import database.repository as repo
        from database.models import ClipEmbedding

        if not self._sql_cosine_available():
            self.skipTest("pgvector package not installed")

        expr = repo._cosine_distance_sql([0.0] * 1024)
        self.assertIsNotNone(expr)
        sql = str(expr.compile(dialect=postgresql.dialect()))
        self.assertIn("<=>", sql)

    def test_find_similar_clips_orders_by_distance(self):
        from sqlalchemy import select
        from sqlalchemy.dialects import postgresql

        import database.repository as repo
        from database.models import ClipEmbedding

        if not self._sql_cosine_available():
            self.skipTest("pgvector package not installed")

        d = repo._cosine_distance_sql([0.0] * 1024)
        stmt = select(ClipEmbedding).order_by(d).limit(3)
        sql = str(stmt.compile(dialect=postgresql.dialect())).upper()
        self.assertIn("ORDER BY CLIP_EMBEDDINGS.EMBEDDING <=>", sql)
        self.assertIn("LIMIT", sql)

    def test_with_stats_joins_uploads_in_one_query(self):
        from sqlalchemy import select
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.orm import joinedload

        from database.models import Clip, ClipEmbedding

        opts = [joinedload(ClipEmbedding.clip).joinedload(Clip.upload)]
        sql = str(
            select(ClipEmbedding).options(*opts).compile(dialect=postgresql.dialect())
        ).lower()
        self.assertIn("left outer join clip_uploads", sql)

    def test_upsert_clip_upload_accepts_stats(self):
        import inspect

        from database.repository import upsert_clip_upload

        params = inspect.signature(upsert_clip_upload).parameters
        for name in ("view_count", "like_count", "comment_count"):
            self.assertIn(name, params)

    @staticmethod
    def _sql_cosine_available():
        from database.vector import use_sql_cosine
        return use_sql_cosine()


class TestRecommenderScoring(unittest.TestCase):
    """Tests for the pure scoring helpers in recommender.py."""

    def setUp(self):
        import recommender
        self.rec = recommender

    def test_engagement_rate_positive(self):
        row = {"views": 1000, "likes": 50, "comments": 20}
        self.assertAlmostEqual(self.rec._engagement_rate(row), 0.07, places=6)

    def test_engagement_rate_zero_views(self):
        self.assertEqual(self.rec._engagement_rate({"views": 0, "likes": 5}), 0.0)

    def test_engagement_rate_clamped(self):
        # Impossible engagement (more likes than views) must clamp, not blow up.
        row = {"views": 10, "likes": 999, "comments": 999}
        self.assertEqual(self.rec._engagement_rate(row), 1.0)

    def test_engagement_rate_missing_keys(self):
        self.assertEqual(self.rec._engagement_rate({}), 0.0)

    def test_score_candidates_sorts_by_similarity_then_engagement(self):
        cands = [
            {"clip_id": 1, "similarity": 0.5, "views": 100, "likes": 1, "comments": 0},
            {"clip_id": 2, "similarity": 0.9, "views": 100, "likes": 1, "comments": 0},
            {"clip_id": 3, "similarity": 0.9, "views": 100, "likes": 50, "comments": 10},
        ]
        ranked = self.rec._score_candidates(cands)
        self.assertEqual([r["clip_id"] for r in ranked], [3, 2, 1])

    def test_score_candidates_adds_engagement_field(self):
        ranked = self.rec._score_candidates([
            {"clip_id": 1, "similarity": 0.5, "views": 100, "likes": 10, "comments": 0},
        ])
        self.assertAlmostEqual(ranked[0]["engagement"], 0.1, places=6)

    def test_score_candidates_empty(self):
        self.assertEqual(self.rec._score_candidates([]), [])

    def test_recommend_clip_params_defaults_without_history(self):
        # No DB reachable → falls back to the documented defaults.
        result = self.rec.recommend_clip_params("some transcript")
        self.assertEqual(result["clip_duration"], 25)
        self.assertEqual(result["subtitle_style"], "tiktok")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["source_clips"], 0)

    def test_recommend_clip_params_blank_transcript(self):
        result = self.rec.recommend_clip_params("   ")
        self.assertEqual(result["source_clips"], 0)

    def test_recommend_title_suggestions_empty_without_db(self):
        self.assertEqual(self.rec.recommend_title_suggestions("hello"), [])

    def test_recommend_content_unknown_channel(self):
        self.assertEqual(self.rec.recommend_content_for_channel("UC_nonexistent"), [])

    def test_predictor_no_db_returns_neutral(self):
        # With no reachable database the predictor must not invent stats, and
        # must hand back a neutral multiplier rather than a bogus boost.
        from predictor import ViralPredictor

        p = ViralPredictor()
        p.refresh_stats()
        self.assertEqual(p._averages, {})
        self.assertEqual(p.predict_boost({"score": 0.9, "duration": 25}), 1.0)
        self.assertEqual(p.adjust_moments([{"score": 0.8}])[0]["score"], 0.8)

    def test_min_views_threshold_is_positive(self):
        # Guards against a regression that would let brand-new uploads (0 views)
        # dominate the recommendations.
        self.assertGreater(self.rec.MIN_VIEWS_FOR_SIGNAL, 0)


class TestEmbedderGuards(unittest.TestCase):
    """Tests for embedder input guards (no model load)."""

    def test_blank_text_returns_none(self):
        import embedder
        self.assertIsNone(embedder.generate_text_embedding(""))
        self.assertIsNone(embedder.generate_text_embedding("   \n"))

    def test_dimension_constant_is_positive(self):
        from database.vector import VECTOR_DIM
        self.assertEqual(VECTOR_DIM, 1024)


class TestVisionAnalysis(unittest.TestCase):
    """Tests for the Qwen3-VL vision integration (offline / mocked)."""

    def test_model_exists_namespaced_tag(self):
        from ollama_client import model_exists
        import ollama_client
        # Control list_models so the test is deterministic regardless of Ollama state.
        original = ollama_client.list_models
        ollama_client.list_models = lambda *a, **k: [
            "qcwind/qwen3-8b-instruct-Q4-K-M:latest",
            "qwen3-vl:4b-instruct",
        ]
        try:
            # Fully-qualified and bare (suffix-stripped) forms both match.
            self.assertTrue(model_exists("qcwind/qwen3-8b-instruct-Q4-K-M:latest"))
            self.assertTrue(model_exists("qcwind/qwen3-8b-instruct-Q4-K-M"))
            # A different family/model does not match.
            self.assertFalse(model_exists("differentfamily/model:latest"))
        finally:
            ollama_client.list_models = original

    def test_vision_score_candidate_rejects_ui_screen(self):
        # UI screens must be hard-rejected without any network call.
        import ollama_detector
        result = ollama_detector.vision_score_candidate(
            transcript="menu open",
            moment={"start": 0, "end": 25, "score": 0.5},
            clip_duration=25,
            vision_meta={"is_ui_screen": True, "highlight_score": 0.95,
                         "ocr_text": "", "scene": "main menu", "action": "",
                         "reason": "loading screen", "objects": []},
            model="qwen3-vl:4b-instruct",
        )
        self.assertIsNotNone(result)
        self.assertLessEqual(result["viral_score"], 0.15)
        self.assertIn("UI", result["reason"])

    def test_vision_score_candidate_passes_through(self):
        # Non-UI candidate routes to the LLM; we mock generate_json to avoid network.
        import ollama_detector
        original = ollama_detector.generate_json
        ollama_detector.generate_json = lambda *a, **k: {
            "viral_score": 0.8, "reason": "boss kill",
            "better_start_offset": 0, "better_end_offset": 25,
        }
        try:
            result = ollama_detector.vision_score_candidate(
                transcript="we killed the boss",
                moment={"start": 0, "end": 25, "score": 0.5},
                clip_duration=25,
                vision_meta={"is_ui_screen": False, "highlight_score": 0.9,
                             "ocr_text": "Victory", "scene": "boss defeated",
                             "action": "final blow", "reason": "epic", "objects": ["boss"]},
                model="qwen3-vl:4b-instruct",
            )
        finally:
            ollama_detector.generate_json = original
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["viral_score"], 0.8)

    def test_analyze_moment_frames_normalizes(self):
        import vision_analyzer
        # Patch frame encoding + the network call so the test stays offline.
        original_b64 = vision_analyzer._frames_to_b64
        original_gen = vision_analyzer.generate_vision
        vision_analyzer._frames_to_b64 = lambda frames: ["Zm9v"]
        vision_analyzer.generate_vision = lambda *a, **k: {
            "highlight_score": 1.7,  # out of range → clamped
            "ocr_text": "Victory",
            "scene": "boss room",
            "objects": ["sword", "boss"],
            "is_ui_screen": False,
            "action": "attack",
            "reason": "great moment",
        }
        try:
            meta = vision_analyzer.analyze_moment_frames(
                [__file__, __file__], model="qwen3-vl:4b-instruct", timeout=5,
            )
        finally:
            vision_analyzer._frames_to_b64 = original_b64
            vision_analyzer.generate_vision = original_gen
        self.assertIsNotNone(meta)
        self.assertEqual(meta["highlight_score"], 1.0)  # clamped to [0,1]
        self.assertEqual(meta["ocr_text"], "Victory")
        self.assertFalse(meta["is_ui_screen"])

    def test_analyze_moment_frames_ui_clamped(self):
        import vision_analyzer
        original_b64 = vision_analyzer._frames_to_b64
        original_gen = vision_analyzer.generate_vision
        vision_analyzer._frames_to_b64 = lambda frames: ["Zm9v"]
        vision_analyzer.generate_vision = lambda *a, **k: {
            "highlight_score": 0.9, "is_ui_screen": True,
            "ocr_text": "", "scene": "menu", "objects": [],
            "action": "", "reason": "loading",
        }
        try:
            meta = vision_analyzer.analyze_moment_frames(
                [__file__], model="qwen3-vl:4b-instruct", timeout=5,
            )
        finally:
            vision_analyzer._frames_to_b64 = original_b64
            vision_analyzer.generate_vision = original_gen
        self.assertIsNotNone(meta)
        self.assertLessEqual(meta["highlight_score"], 0.15)

    def test_title_generator_vision_enrichment(self):
        import title_generator
        captured = {}
        orig_generate = title_generator.generate
        orig_ensure = title_generator.ensure_model
        def fake_generate(prompt, model=title_generator.DEFAULT_MODEL, timeout=30, options=None):
            captured["prompt"] = prompt
            return "Victory Clutch OP"
        title_generator.generate = fake_generate
        title_generator.ensure_model = lambda *a, **k: True
        try:
            title = title_generator.generate_title(
                "we won the match", model="qwen3:8b-instruct",
                vision_meta={"ocr_text": "Victory", "scene": "boss defeated",
                             "action": "final blow", "objects": ["boss"]},
            )
        finally:
            title_generator.generate = orig_generate
            title_generator.ensure_model = orig_ensure
        self.assertEqual(title, "Victory Clutch OP")
        self.assertIn("Victory", captured["prompt"])
        self.assertIn("boss defeated", captured["prompt"])

    def test_title_generator_batch_vision_contexts(self):
        import title_generator
        seen = []
        orig_generate = title_generator.generate
        orig_ensure = title_generator.ensure_model
        def fake_generate(prompt, model=title_generator.DEFAULT_MODEL, timeout=30, options=None):
            seen.append(prompt)
            return "Title"
        title_generator.generate = fake_generate
        title_generator.ensure_model = lambda *a, **k: True
        try:
            titles = title_generator.generate_titles_batch(
                ["a", "b"], model="qwen3:8b-instruct",
                vision_contexts=[{"ocr_text": "GG"}, None],
            )
        finally:
            title_generator.generate = orig_generate
            title_generator.ensure_model = orig_ensure
        self.assertEqual(titles, ["Title", "Title"])
        self.assertIn("GG", seen[0])
        self.assertNotIn("on-screen text", seen[1])


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
