"""FFmpeg hardware acceleration — probe encoders/decoders and build ffmpeg args."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

from subprocess_utils import CancelledError

logger = logging.getLogger(__name__)



def _run_ffmpeg_direct(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run ffmpeg subprocess directly, bypassing the global cancel flag.

    Unlike :func:`subprocess_utils.run`, this does **not** check
    ``_cancel_flag``, so probe operations are never interrupted by a
    prior cancellation request.
    """
    kwargs.setdefault(
        "creationflags",
        subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return subprocess.run(args, **kwargs)

# ── GPU discovery ─────────────────────────────────────────────────────────


def list_cuda_devices() -> list[dict]:
    """Enumerate available CUDA GPUs with name, total_mib, free_mib.

    Returns empty list if PyTorch is not available or no CUDA devices found.
    Each entry: {index, name, total_mib, free_mib}.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            except Exception:
                free_bytes = 0
                total_bytes = props.total_memory if hasattr(props, 'total_memory') else 0
            devices.append({
                "index": i,
                "name": props.name if hasattr(props, 'name') else f"CUDA GPU {i}",
                "total_mib": total_bytes // (1024 * 1024),
                "free_mib": free_bytes // (1024 * 1024),
            })
        return devices
    except ImportError:
        return []
    except Exception:
        return []


def get_gpu_count() -> int:
    """Return the number of available CUDA GPUs (0 if none / CPU-only)."""
    return len(list_cuda_devices())


def select_least_loaded_gpu(gpu_indices: list[int] | None = None) -> int:
    """Pick the GPU with the most free VRAM.

    If *gpu_indices* is None, checks all available GPUs.
    Returns the device index (int).  Defaults to 0 if detection fails.
    """
    devices = list_cuda_devices()
    if not devices:
        return 0
    candidates = [d for d in devices if gpu_indices is None or d["index"] in gpu_indices]
    if not candidates:
        return 0
    # Pick the one with the most free MiB; tie → lowest index
    best = max(candidates, key=lambda d: (d["free_mib"], -d["index"]))
    return best["index"]


def get_gpu_device_str(gpu_index: int) -> str:
    """Return the torch device string for a GPU index (e.g. ``"cuda:0"``)."""
    if gpu_index < 0:
        return "cpu"
    return f"cuda:{gpu_index}"

# User-facing encoder keys → ffmpeg codec names
_ENCODER_MAP = {
    "auto": None,
    "cpu": "libx264",
    "nvenc": "h264_nvenc",
    "qsv": "h264_qsv",
    "amf": "h264_amf",
    "v4l2m2m": "h264_v4l2m2m", # Added h264_v4l2m2m
    "nvenc_hevc": "hevc_nvenc",
    "qsv_hevc": "hevc_qsv",
    "amf_hevc": "hevc_amf",
    "cpu_hevc": "libx265",
}

_AUTO_PRIORITY = ("h264_nvenc", "h264_qsv", "h264_amf", "h264_v4l2m2m", "libx264") # Added h264_v4l2m2m

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

# New: libx264 preset → h264_v4l2m2m preset mapping (example, adjust as needed)
_X264_TO_V4L2M2M = {
    "ultrafast": "ultrafast",
    "superfast": "superfast",
    "veryfast": "veryfast",
    "faster": "fast",
    "fast": "medium",
    "medium": "slow",
    "slow": "slow",
}

# Windows decode preference
_WIN_HWACCEL = ("cuda", "d3d11va", "dxva2")
_UNIX_HWACCEL = ("cuda", "videotoolbox", "vaapi", "v4l2m2m")


@dataclass
class HwProfile:
    encoders: frozenset[str]
    hwaccels: frozenset[str]
    active_encoder: str
    active_encoder_label: str
    active_hwaccel: str | None


_profile: HwProfile | None = None
_profile_lock = threading.Lock()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg_list(flag: str) -> str:
    if not _ffmpeg_available():
        return ""
    try:
        r = _run_ffmpeg_direct(
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
        m = re.match(r"\s*V\S*\s+(\S+)", line)
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
        "h264_nvenc": "NVIDIA NVENC H.264",
        "h264_qsv": "Intel Quick Sync H.264",
        "h264_amf": "AMD AMF H.264",
        "h264_v4l2m2m": "V4L2 M2M H.264",
        "libx264": "CPU (libx264)",
        "hevc_nvenc": "NVIDIA NVENC HEVC",
        "hevc_qsv": "Intel Quick Sync HEVC",
        "hevc_amf": "AMD AMF HEVC",
        "libx265": "CPU (libx265)",
    }
    return labels.get(codec, codec)


def _hardware_encode_works(codec: str) -> bool:
    if codec in ("libx264", "libx265"):
        return True
    if not _ffmpeg_available():
        return False

    args_by_codec = {
        "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p3", "-rc:v", "vbr", "-cq:v", "23", "-pix_fmt", "yuv420p"],
        "h264_qsv": ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality:v", "23", "-pix_fmt", "nv12"],
        "h264_amf": ["-c:v", "h264_amf", "-quality", "balanced", "-rc:v", "vbr_latency", "-qp_i:v", "23", "-qp_p:v", "23", "-pix_fmt", "yuv420p"],
        "h264_v4l2m2m": ["-c:v", "h264_v4l2m2m", "-qp", "23", "-pix_fmt", "yuv420p"],
        "hevc_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p3", "-rc:v", "vbr", "-cq:v", "23", "-pix_fmt", "yuv420p"],
        "hevc_qsv": ["-c:v", "hevc_qsv", "-preset", "veryfast", "-global_quality:v", "23", "-pix_fmt", "nv12"],
        "hevc_amf": ["-c:v", "hevc_amf", "-quality", "balanced", "-rc:v", "vbr_latency", "-qp_i:v", "23", "-qp_p:v", "23", "-pix_fmt", "yuv420p"],
    }
    enc_args = args_by_codec.get(codec)
    if not enc_args:
        return False

    try:
        r = _run_ffmpeg_direct(
            [
                "ffmpeg", "-hide_banner", "-y",
                "-f", "lavfi", "-i", "testsrc2=duration=0.5:size=640x360:rate=15",
                *enc_args,
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=8,
        )
        if r.returncode != 0:
            logger.info("FFmpeg hardware encoder probe failed for %s: %s", codec, (r.stderr or "")[-300:])
        return r.returncode == 0
    except Exception as exc:
        logger.info("FFmpeg hardware encoder probe failed for %s: %s", codec, exc)
        return False


def _resolve_encoder(codec: str | None, available: frozenset[str], verify: bool = True) -> str:
    if codec == "libx264" or codec == "cpu":
        return "libx264"
    if codec and codec in available and (not verify or _hardware_encode_works(codec)):
        return codec
    if codec and codec in available:
        logger.warning("%s is listed by FFmpeg but failed the encode probe; falling back.", codec)
    for c in _AUTO_PRIORITY:
        if c in available and (not verify or _hardware_encode_works(c)):
            return c
    return "libx264"


def probe_ffmpeg(encoder_pref: str = "auto") -> HwProfile:
    """Probe ffmpeg once per process; return cached HwProfile."""
    global _profile
    with _profile_lock:
        if _profile is not None:
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
    with _profile_lock:
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
    whisper_dev, whisper_compute, whisper_idx = resolve_whisper_device(whisper_device)
    gpu_count = get_gpu_count()
    gpu_str = f", gpus={gpu_count}" if gpu_count > 0 else ""
    return (
        f"encoder={prof.active_encoder}, decode={decode}, "
        f"yolo={yolo}, whisper={whisper_dev}/{whisper_compute}{gpu_str}"
    )


def video_encode_args(
    preset: str = "ultrafast",
    crf: str = "23",
    encoder: str = "auto",
    force_cpu: bool = False,
    gpu_index: int | None = None,
) -> list[str]:
    """Build ffmpeg video encode arguments for the chosen encoder.

    If *gpu_index* is provided (≥ 0), a ``-gpu:v N`` flag is appended for
    NVENC-based encoders so the correct physical GPU is used.
    """
    if force_cpu:
        codec = "libx264"
    else:
        prof = probe_ffmpeg(encoder)
        key = (encoder or "auto").lower()
        if key == "cpu":
            codec = "libx264"
        elif key in _ENCODER_MAP and _ENCODER_MAP[key]:
            requested = _ENCODER_MAP[key]
            codec = requested if requested in prof.encoders else prof.active_encoder
        else:
            codec = prof.active_encoder

    try:
        cq = str(max(0, min(51, int(crf))))
    except (ValueError, TypeError):
        cq = "23"
    x264_preset = preset if preset in _X264_TO_NVENC else "ultrafast"

    def _maybe_gpu_flag(args: list[str]) -> list[str]:
        """Append ``-gpu:v N`` for NVENC when *gpu_index* is set."""
        if gpu_index is not None and gpu_index >= 0:
            return args + ["-gpu:v", str(gpu_index)]
        return args

    if codec == "h264_nvenc":
        nv_preset = _X264_TO_NVENC.get(x264_preset, "p4")
        return _maybe_gpu_flag([
            "-c:v", "h264_nvenc",
            "-preset", nv_preset,
            "-rc:v", "vbr",
            "-cq:v", cq,
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ])
    if codec == "h264_qsv":
        qsv_preset = _X264_TO_QSV.get(x264_preset, "veryfast")
        args = [
            "-c:v", "h264_qsv",
            "-preset", qsv_preset,
            "-global_quality:v", cq,
            "-pix_fmt", "nv12",
        ]
        if gpu_index is not None and gpu_index >= 0:
            args += ["-engine:v", str(gpu_index)]
        return args
    if codec == "h264_amf":
        amf_preset = _X264_TO_AMF.get(x264_preset, "balanced")
        return _maybe_gpu_flag([
            "-c:v", "h264_amf",
            "-quality", amf_preset,
            "-rc:v", "vbr_latency",
            "-qp_i:v", cq,
            "-qp_p:v", cq,
            "-pix_fmt", "yuv420p",
        ])
    if codec == "hevc_nvenc":
        nv_preset = _X264_TO_NVENC.get(x264_preset, "p4")
        return _maybe_gpu_flag([
            "-c:v", "hevc_nvenc",
            "-preset", nv_preset,
            "-rc:v", "vbr",
            "-cq:v", cq,
            "-pix_fmt", "yuv420p",
        ])
    if codec == "hevc_qsv":
        qsv_preset = _X264_TO_QSV.get(x264_preset, "veryfast")
        args = [
            "-c:v", "hevc_qsv",
            "-preset", qsv_preset,
            "-global_quality:v", cq,
            "-pix_fmt", "nv12",
        ]
        if gpu_index is not None and gpu_index >= 0:
            args += ["-engine:v", str(gpu_index)]
        return args
    if codec == "hevc_amf":
        amf_preset = _X264_TO_AMF.get(x264_preset, "balanced")
        return _maybe_gpu_flag([
            "-c:v", "hevc_amf",
            "-quality", amf_preset,
            "-rc:v", "vbr_latency",
            "-qp_i:v", cq,
            "-qp_p:v", cq,
            "-pix_fmt", "yuv420p",
        ])
    if codec == "libx265":
        return [
            "-c:v", "libx265",
            "-preset", x264_preset,
            "-crf", cq,
            "-pix_fmt", "yuv420p",
        ]
    if codec == "h264_v4l2m2m": # New: h264_v4l2m2m encoding arguments
        v4l2m2m_preset = _X264_TO_V4L2M2M.get(x264_preset, "medium")
        return [
            "-c:v", "h264_v4l2m2m",
            "-preset", v4l2m2m_preset,
            "-qp", cq, # Assuming -qp for quality control
            "-pix_fmt", "yuv420p",
            # Add any other specific v4l2m2m options here, e.g., -b:v, -profile:v
        ]
    return [
        "-c:v", "libx264",
        "-preset", x264_preset,
        "-crf", cq,
        "-pix_fmt", "yuv420p",
    ]


def input_hwaccel_args(decoder: str = "auto",
                       gpu_index: int | None = None) -> list[str]:
    """Build ffmpeg hardware decode args (before -i). Empty if unavailable.

    If *gpu_index* is given (≥ 0), a ``-hwaccel_device N`` flag is appended
    so the correct GPU is used for decode.
    """
    if decoder in ("none", "cpu"):
        return []

    # Handle codec-based hardware decoders (e.g., h264_cuvid, h264_qsv)
    if any(suffix in decoder for suffix in ("_cuvid", "_qsv", "_v4l2m2m")):
        args = ["-c:v", decoder]
        if gpu_index is not None and gpu_index >= 0:
            args += ["-hwaccel_device", str(gpu_index)]
        return args

    prof = probe_ffmpeg()
    if decoder != "auto" and decoder in prof.hwaccels:
        args = ["-hwaccel", decoder]
        if gpu_index is not None and gpu_index >= 0:
            args += ["-hwaccel_device", str(gpu_index)]
        return args

    if prof.active_hwaccel:
        args = ["-hwaccel", "auto"]
        if gpu_index is not None and gpu_index >= 0:
            args += ["-hwaccel_device", str(gpu_index)]
        return args
    return []


_HW_CODECS = frozenset({"h264_nvenc", "h264_qsv", "h264_amf", "h264_v4l2m2m", "hevc_nvenc", "hevc_qsv", "hevc_amf"}) # Added HEVC variants

_HW_OPTION_ARITY = {
    "-preset": 1,
    "-rc": 1,
    "-rc:v": 1,
    "-cq": 1,
    "-cq:v": 1,
    "-global_quality": 1,
    "-global_quality:v": 1,
    "-quality": 1,
    "-qp": 1,
    "-qp:v": 1,
    "-qp_i": 1,
    "-qp_i:v": 1,
    "-qp_p": 1,
    "-qp_p:v": 1,
    "-pix_fmt": 1,
    "-profile:v": 1,
    "-level:v": 1,
    "-tier:v": 1,
}

_HW_DECODER_SUFFIXES = ("_cuvid", "_qsv", "_v4l2m2m")


def _swap_cmd_encode_to_cpu(cmd: list[str], preset: str, crf: str) -> list[str]:
    """Replace hardware -c:v block in an ffmpeg command with libx264 args."""
    out: list[str] = []
    i = 0
    seen_input = False
    while i < len(cmd):
        if cmd[i] == "-i":
            seen_input = True
            out.append(cmd[i])
            i += 1
            continue

        # 1. Strip hardware decoder flags (retry should be pure software)
        if cmd[i] == "-hwaccel":
            i += 2
            # Also skip companion --hwaccel_device flag if present
            if i < len(cmd) and cmd[i] == "-hwaccel_device":
                i += 2
            continue

        # 1b. Strip codec-based hardware decoders (before -i) during fallback
        if (not seen_input and i + 1 < len(cmd) and cmd[i] == "-c:v" and
            any(s in cmd[i + 1] for s in _HW_DECODER_SUFFIXES)):
            i += 2
            continue

        # 2. Identify and swap hardware encoder
        if (
            i + 1 < len(cmd)
            and cmd[i] == "-c:v"
            and cmd[i + 1] in _HW_CODECS
        ):
            out.extend(video_encode_args(preset, crf, force_cpu=True))
            i += 2
            # Skip subsequent encoder-specific flags we might have added.
            while i < len(cmd):
                option_arity = _HW_OPTION_ARITY.get(cmd[i])
                if option_arity is not None:
                    i += 1 + option_arity
                    continue
                # Stop if we hit any other flag or the output filename
                if cmd[i].startswith("-") and cmd[i] not in ("-y", "-n"):
                    break
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
    try:
        r = run_fn(cmd)
    except CancelledError:
        raise

    if r.returncode == 0:
        return r

    # Check if the command used HW encode or HW decode
    has_hw = any(c in cmd for c in _HW_CODECS) or "-hwaccel" in cmd
    if not has_hw:
        return r

    # Log why it failed to help with debugging. Avoid index errors on empty stderr.
    lines = r.stderr.strip().splitlines() if r.stderr else []
    err_msg = lines[-1] if lines else "Unknown error (check ffmpeg installation)"
    
    print(f"[!] Hardware acceleration failed: {err_msg}")
    print("    Retrying with pure software path...")

    fallback = _swap_cmd_encode_to_cpu(cmd, preset, crf)
    return run_fn(fallback)

# ── YOLO / Whisper device helpers (used by cropper & transcriber) ───────────


def resolve_yolo_device(pref: str = "auto", gpu_index: int | None = None) -> str:
    """Return a device string like ``"cuda:0"`` or ``"cpu"`` for ultralytics.

    If *gpu_index* is given (e.g. ``1`` for GPU 1), it takes precedence over *pref*.
    """
    if gpu_index is not None and gpu_index >= 0:
        return f"cuda:{gpu_index}"

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


def resolve_whisper_device(pref: str = "auto", gpu_index: int | None = None) -> tuple[str, str, int]:
    """Return (device, compute_type, device_index) for faster-whisper.

    If *gpu_index* is given (e.g. ``1``), the returned *device_index* is set
    accordingly so whisper can be pinned to a specific GPU.
    """
    if gpu_index is not None and gpu_index >= 0:
        return "cuda", "float16", gpu_index

    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu", "int8", 0

    try:
        import torch
        if pref == "cuda" or pref == "auto":
            if torch.cuda.is_available():
                return "cuda", "float16", 0
        if pref in ("auto", "mps") and getattr(torch.backends, "mps", None):
            if torch.backends.mps.is_available():
                return "cpu", "int8", 0  # faster-whisper lacks stable MPS — stay CPU
    except ImportError:
        logger.debug("torch not available, falling back to CPU whisper device")
    return "cpu", "int8", 0


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
