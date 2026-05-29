import sys
import os
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    print("[*] Testing environment...")
    import hwaccel
    import shutil
    
    ffmpeg = shutil.which("ffmpeg")
    print(f"  - FFmpeg: {ffmpeg or 'FAILED'}")
    
    prof = hwaccel.probe_ffmpeg()
    print(f"  - Encoder: {prof.active_encoder_label}")
    print(f"  - HW Accel: {prof.active_hwaccel or 'None'}")
    
    try:
        import torch
        print(f"  - Torch CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("  - Torch: NOT INSTALLED")

def test_subtitle_logic():
    print("[*] Testing Subtitle Logic...")
    import subtitler
    words = [
        {"text": "Hello!", "start": 0.0, "end": 0.5},
        {"text": "test.", "start": 0.4, "end": 1.0}, # Overlap
        {"text": "[Music]", "start": 1.1, "end": 2.0}, # Junk
    ]
    sanitized = subtitler._sanitize_word_times(words)
    
    # Check cleaning
    texts = [w["text"] for w in sanitized]
    if "HELLO" not in [t.upper() for t in texts]:
        print(f"  [!] Cleaning failed: {texts}")
    if any("[" in t for t in texts):
        print(f"  [!] Metadata removal failed: {texts}")
        
    # Check overlap fix
    if sanitized[1]["start"] < sanitized[0]["end"]:
        print("  [!] Overlap fix failed")
    else:
        print("  [OK] Sanitization & Overlap logic")

def test_clipper_logic():
    print("[*] Testing Clipper Logic...")
    import clipper
    times = [0, 2, 4]
    values = [10, 20, 30]
    expr = clipper._build_lerp_expr(times, values)
    expected = "if(lt(t\\,2.000)\\,10\\,if(lt(t\\,4.000)\\,20\\,30))"
    if expr == expected:
        print("  [OK] FFmpeg expression builder")
    else:
        print(f"  [!] Builder failed. Got: {expr}")

def test_hwaccel_fallback_rewrite():
    print("[*] Testing FFmpeg fallback rewrite...")
    import hwaccel
    cmd = [
        "ffmpeg", "-y", "-ss", "10",
        "-hwaccel", "auto",
        "-i", "in.mp4", "-t", "30",
        "-vf", "scale=1080:1920",
        "-c:v", "h264_v4l2m2m",
        "-preset", "fast",
        "-qp", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "out.mp4",
    ]
    rewritten = hwaccel._swap_cmd_encode_to_cpu(cmd, "fast", "18")
    bad_tokens = {"-hwaccel", "h264_v4l2m2m", "-qp"}
    if "libx264" in rewritten and not any(token in rewritten for token in bad_tokens):
        print("  [OK] Hardware encode fallback rewrites cleanly")
    else:
        print(f"  [!] Fallback rewrite failed: {rewritten}")

def test_api_state():
    print("[*] Testing API Bridge state...")
    from api_bridge import ApiBridge
    bridge = ApiBridge()
    settings = bridge.get_settings()
    if isinstance(settings, dict) and "num_clips" in settings:
        print("  [OK] Settings retrieval")
    else:
        print("  [!] Settings retrieval failed")

if __name__ == "__main__":
    print("=== ViriaRevive Logic Test Suite ===\n")
    test_environment()
    test_subtitle_logic()
    test_clipper_logic()
    test_hwaccel_fallback_rewrite()
    test_api_state()
    print("\nTests completed.")
