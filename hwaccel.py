"""FFmpeg hardware acceleration — probe encoders/decoders and build ffmpeg args."""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass

from subprocess_utils import run as _run

# User-facing encoder keys → ffmpeg codec names
_ENCODER_MAP = {
    "auto": None,
    "cpu": "libx264",
    "nvenc": "h264_nvenc",
    "qsv": "h264_qsv",
    "amf": "h264_amf",
}

_AUTO_PRIORITY = ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")

# libx264 preset → hardware preset mapping
_X264_TO_NVENC = {
    "ultrafast": "p1",
    "superfast": "p2",
    "veryfast": "p3",
    "faster": "p4",
    "fast": "p5",
    "medium": "p6",
    "slow": "p7",
}

_X264_TO_QSV = {
    "ultrafast": "veryfast",
    "superfast": "veryfast",
    "veryfast": "veryfast",
    "faster": "faster",
    "fast": "fast",
    "medium": "medium",
    "slow": "slow",
}

_X264_TO_AMF = {
    "ultrafast": "speed",
    "superfast": "speed",
    "veryfast": "speed",
    "faster": "balanced",
    "fast": "balanced",
    "medium": "quality",
    "slow": "quality",
}

# Windows decode preference
_WIN_HWACCEL = ("cuda", "d3d11va", "dxva2")
_UNIX_HWACCEL = ("cuda", "videotoolbox", "vaapi")


@dataclass
class HwProfile:
    encoders: frozenset[str]
    hwaccels: frozenset[str]
    active_encoder: str
    active_encoder_label: str
    active_hwaccel: str | None


_profile: HwProfile | None = None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg_list(flag: str) -> str:
    if not _ffmpeg_available():
        return ""
    try:
        r = _run(
            ["ffmpeg", "-hide_banner", flag],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
        return r.stdout or ""
    except Exception:
        return ""


def _parse_encoders(output: str) -> frozenset[str]:
    found = set()
    for line in output.splitlines():
        # V..... h264_nvenc           NVIDIA NVENC ...
        m = re.match(r"\s*V[.F.SX]+\s+(\S+)", line)
        if m:
            found.add(m.group(1))
    return frozenset(found)


def _parse_hwaccels(output: str) -> frozenset[str]:
    found = set()
    for line in output.splitlines():
        line = line.strip()
        if line and not line.startswith("Hardware"):
            found.add(line.split()[0])
    return frozenset(found)


def _encoder_label(codec: str) -> str:
    labels = {
        "h264_nvenc": "NVIDIA NVENC",
        "h264_qsv": "Intel Quick Sync",
        "h264_amf": "AMD AMF",
        "libx264": "CPU (libx264)",
    }
    return labels.get(codec, codec)


def _resolve_encoder(codec: str | None, available: frozenset[str]) -> str:
    if codec == "libx264" or codec == "cpu":
        return "libx264"
    if codec and codec in available:
        return codec
    for c in _AUTO_PRIORITY:
        if c in available:
            return c
    return "libx264"


def probe_ffmpeg(encoder_pref: str = "auto") -> HwProfile:
    """Probe ffmpeg once per process; return cached HwProfile."""
    global _profile
    if _profile is not None and encoder_pref in ("auto", None):
        return _profile

    enc_out = _run_ffmpeg_list("-encoders")
    hw_out = _run_ffmpeg_list("-hwaccels")
    encoders = _parse_encoders(enc_out)
    hwaccels = _parse_hwaccels(hw_out)

    key = (encoder_pref or "auto").lower()
    requested = _ENCODER_MAP.get(key)
    if key == "cpu":
        requested = "libx264"

    active = _resolve_encoder(requested, encoders)

    hwaccel = None
    candidates = _WIN_HWACCEL if sys.platform == "win32" else _UNIX_HWACCEL
    for h in candidates:
        if h in hwaccels:
            hwaccel = h
            break

    profile = HwProfile(
        encoders=encoders,
        hwaccels=hwaccels,
        active_encoder=active,
        active_encoder_label=_encoder_label(active),
        active_hwaccel=hwaccel,
    )
    if encoder_pref in ("auto", None):
        _profile = profile
    return profile


def reset_probe_cache():
    """Clear cached probe (for tests)."""
    global _profile
    _profile = None


def get_hardware_summary(
    encoder_pref: str = "auto",
    yolo_device: str = "auto",
    whisper_device: str = "auto",
) -> str:
    """One-line summary for startup logging."""
    prof = probe_ffmpeg(encoder_pref)
    decode = prof.active_hwaccel or "software"
    yolo = resolve_yolo_device(yolo_device)
    whisper_dev, whisper_compute = resolve_whisper_device(whisper_device)
    return (
        f"encoder={prof.active_encoder}, decode={decode}, "
        f"yolo={yolo}, whisper={whisper_dev}/{whisper_compute}"
    )


def video_encode_args(
    preset: str = "ultrafast",
    crf: str = "23",
    encoder: str = "auto",
    force_cpu: bool = False,
) -> list[str]:
    """Build ffmpeg video encode arguments for the chosen encoder."""
    if force_cpu:
        codec = "libx264"
    else:
        prof = probe_ffmpeg(encoder)
        key = (encoder or "auto").lower()
        if key == "cpu":
            codec = "libx264"
        elif key in _ENCODER_MAP and _ENCODER_MAP[key]:
            codec = _ENCODER_MAP[key] if _ENCODER_MAP[key] in prof.encoders else prof.active_encoder
        else:
            codec = prof.active_encoder

    cq = str(max(18, min(32, int(crf))))
    x264_preset = preset if preset in _X264_TO_NVENC else "ultrafast"

    if codec == "h264_nvenc":
        nv_preset = _X264_TO_NVENC.get(x264_preset, "p4")
        return [
            "-c:v", "h264_nvenc",
            "-preset", nv_preset,
            "-rc", "vbr",
            "-cq", cq,
            "-pix_fmt", "yuv420p",
        ]
    if codec == "h264_qsv":
        qsv_preset = _X264_TO_QSV.get(x264_preset, "veryfast")
        return [
            "-c:v", "h264_qsv",
            "-preset", qsv_preset,
            "-global_quality", cq,
            "-pix_fmt", "yuv420p",
        ]
    if codec == "h264_amf":
        amf_preset = _X264_TO_AMF.get(x264_preset, "balanced")
        return [
            "-c:v", "h264_amf",
            "-quality", amf_preset,
            "-rc", "vbr_latency",
            "-qp_i", cq,
            "-qp_p", cq,
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264",
        "-preset", x264_preset,
        "-crf", cq,
        "-pix_fmt", "yuv420p",
    ]


def input_hwaccel_args(decoder: str = "auto") -> list[str]:
    """Build ffmpeg hardware decode args (before -i). Empty if unavailable."""
    if decoder == "none" or decoder == "cpu":
        return []
    prof = probe_ffmpeg()
    if not prof.active_hwaccel:
        return []
    return ["-hwaccel", prof.active_hwaccel]


_HW_CODECS = frozenset({"h264_nvenc", "h264_qsv", "h264_amf"})


def _swap_cmd_encode_to_cpu(cmd: list[str], preset: str, crf: str) -> list[str]:
    """Replace hardware -c:v block in an ffmpeg command with libx264 args."""
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if (
            i + 1 < len(cmd)
            and cmd[i] == "-c:v"
            and cmd[i + 1] in _HW_CODECS
        ):
            out.extend(video_encode_args(preset, crf, force_cpu=True))
            i += 2
            while i < len(cmd):
                if cmd[i] == "-pix_fmt":
                    i += 2
                    continue
                if cmd[i] in ("-c:a", "-map", "-movflags", "-f", "-t", "-ss"):
                    break
                if cmd[i].startswith("-") and i + 1 < len(cmd):
                    i += 2
                    continue
                break
            continue
        out.append(cmd[i])
        i += 1
    return out


def run_ffmpeg_with_encode_fallback(
    cmd: list[str],
    run_fn,
    preset: str = "ultrafast",
    crf: str = "23",
):
    """Run ffmpeg; on failure, retry once with libx264 if a hardware encoder was used."""
    r = run_fn(cmd)
    if r.returncode == 0:
        return r
    if not any(c in cmd for c in _HW_CODECS):
        return r
    print("[!] Hardware encode failed, retrying with libx264...")
    fallback = _swap_cmd_encode_to_cpu(cmd, preset, crf)
    return run_fn(fallback)


# ── YOLO / Whisper device helpers (used by cropper & transcriber) ───────────


def resolve_yolo_device(pref: str = "auto") -> str:
    """Return 'cuda:0' or 'cpu' for ultralytics."""
    pref = (pref or "auto").lower()
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        
        if pref != "cpu" and cuda_avail:
            return "cuda:0"
            
        if pref == "cuda" and not cuda_avail:
            print("[!] YOLO: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
    except ImportError:
        if pref == "cuda":
            print("[!] YOLO: CUDA requested but 'torch' is not installed.")

    return "cpu"


def resolve_whisper_device(pref: str = "auto") -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper."""
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu", "int8"

    try:
        import torch
        if pref == "cuda" or pref == "auto":
            if torch.cuda.is_available():
                return "cuda", "float16"
        if pref in ("auto", "mps") and getattr(torch.backends, "mps", None):
            if torch.backends.mps.is_available():
                return "cpu", "int8"  # faster-whisper lacks stable MPS — stay CPU
    except ImportError:
        pass
    return "cpu", "int8"


def log_hardware_startup(
    encoder_pref: str = "auto",
    yolo_device: str = "auto",
    whisper_device: str = "auto",
):
    """Print hardware profile once at startup."""
    summary = get_hardware_summary(encoder_pref, yolo_device, whisper_device)
    prof = probe_ffmpeg(encoder_pref)
    print(f"[+] Hardware: {summary}")
    print(f"    Video encoder: {prof.active_encoder_label}")
