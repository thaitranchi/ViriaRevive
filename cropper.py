"""
Smart 9:16 vertical cropping with YOLO person detection.

Primary: YOLOv8-nano person detector — detects people regardless of angle,
pose, size, or occlusion. Returns full person bounding boxes.

Fallback: YuNet DNN face detector + Haar cascades (for offline environments
where ultralytics isn't available).

Body-aware cropping: the crop is positioned so the person's head is at ~30%
from the top ("rule of thirds"), keeping their full body visible.
"""

import threading
import numpy as np
import cv2
from pathlib import Path

from subprocess_utils import run as _run, is_cancelled, CancelledError
from hwaccel import resolve_yolo_device

import logging

logger = logging.getLogger(__name__)

# OpenCV VideoCapture is not thread-safe; protect with a lock
_cv2_lock = threading.Lock()

# ── Constants ────────────────────────────────────────────────────────────────

YUNET_MODEL = Path(__file__).parent / "models" / "face_detection_yunet.onnx"
YUNET_MODEL_EXISTS = YUNET_MODEL.exists()
if not YUNET_MODEL_EXISTS:
    logger.warning("YuNet face detection model not found at %s — falling back to YOLO/Haar", YUNET_MODEL)
YUNET_CONF = 0.35
YUNET_NMS = 0.3
YOLO_CONF = 0.15
YOLO_CONF_SINGLE = 0.35
WINDOW_SEC = 2.0       # seconds per crop window (legacy, used by static crop)
CHANGE_THRESH = 0.12   # 12% of crop size = minimum position change to trigger a cut
HEAD_RATIO = 0.30      # place head at 30% from top of crop (rule of thirds)
CAMERA_CUT_THRESH = 0.40  # 40% of frame width jump = camera cut
CUT_DELAY_SEC = 0.200     # delay crop switch 200ms after detected cut to avoid empty frames
CUT_HOLD_BEFORE = 0.050   # hold old crop 50ms before the cut too (covers early-switch)

# YOLO caches its model per-device so we only load once per GPU
_yolo_lock = threading.Lock()
_yolo_models: dict[str, object] = {}       # device_str → YOLO model instance
_yolo_devices: dict[str, str] = {}          # device_str → resolved device string
_yolo_checked_devices: set[str] = set()     # set of device strings that have been probed
_yolo_device_pref = "auto"
YOLO_BATCH_SIZE = 32


# ── Public API ───────────────────────────────────────────────────────────────


def get_crop_params(video_path: Path, start: int, end: int,
                    target_ratio: float = 9 / 16, sample_count: int = 20):
    """Return (crop_w, crop_h, crop_x, crop_y) or None if already vertical.

    Static crop — uses median person position across all frames.
    Kept for backward compatibility.
    """
    width, height = _get_dimensions(video_path)
    if width <= 0 or height <= 0:
        return None

    current = width / height
    if abs(current - target_ratio) < 0.05:
        return None

    if current > target_ratio:
        crop_w = int(height * target_ratio)
        crop_w -= crop_w % 2
        crop_h = height

        person_x, head_y = _detect_people(video_path, start, end, width, height, sample_count)
        if person_x is None:
            logger.warning("No person detected; returning None for fallback (Blurry Padding)")
            return None
        crop_x = person_x - crop_w // 2
        crop_x = max(0, min(crop_x, width - crop_w))
        crop_x -= crop_x % 2
        crop_y = 0
    else:
        crop_w = width
        crop_h = int(width / target_ratio)
        crop_h -= crop_h % 2

        _, head_y = _detect_people(video_path, start, end, width, height, sample_count)
        if head_y is None:
            logger.warning("No person detected; returning None for fallback (Blurry Padding)")
            return None
        
        # Rule of thirds: place head at 30% from top
        target_pos = int(crop_h * HEAD_RATIO)
        crop_y = head_y - target_pos # type: ignore
        crop_y = max(0, min(crop_y, height - crop_h))
        crop_y -= crop_y % 2
        crop_x = 0

    logger.info(f"Crop: {crop_w}x{crop_h} at ({crop_x},{crop_y})  from {width}x{height}")
    return crop_w, crop_h, crop_x, crop_y


def get_crop_params_dynamic(video_path: Path, start: int, end: int,
                            target_ratio: float = 9 / 16, sample_count: int = 50,
                            yolo_device: str = "auto", debug_frames: bool = False,
                            gpu_index: int | None = None):
    """Return (crop_w, crop_h, keyframes) or None if already vertical.

    Dynamic crop — tracks the active person per-window with stable cuts.
    keyframes is a list of (time, crop_x, crop_y) tuples.

    *gpu_index* pins YOLO inference to a specific GPU.
    """
    width, height = _get_dimensions(video_path)
    if width <= 0 or height <= 0:
        return None

    current = width / height
    if abs(current - target_ratio) < 0.05:
        return None

    if current > target_ratio:
        crop_w = int(height * target_ratio)
        crop_w -= crop_w % 2
        crop_h = height
        pan_axis = "x"
    else:
        crop_w = width
        crop_h = int(width / target_ratio)
        crop_h -= crop_h % 2
        pan_axis = "y"

    # Get per-frame person tracking data
    detections, scale_x, scale_y = detect_all_persons(
        video_path, start, end, width, height, sample_count, yolo_device=yolo_device,
        debug_frames=debug_frames, gpu_index=gpu_index,
    )

    if len(detections) < 3:
        # Too few detections — use what we have as static fallback
        if detections:
            all_persons = []
            for t, persons in detections:
                best = max(persons, key=lambda p: p[2])
                all_persons.append(best) # type: ignore
            med_x = int(np.median([p[0] for p in all_persons]))
            head_y = int(np.median([p[1] for p in all_persons]))
            logger.warning(f"Few detections ({len(detections)}), using static at x={med_x} head_y={head_y}")
            if pan_axis == "x":
                cx = med_x - crop_w // 2
                cx = max(0, min(cx, width - crop_w))
                cx -= cx % 2
                cy = 0
                if crop_h < height:
                    cy = head_y - int(crop_h * HEAD_RATIO)
                    cy = max(0, min(cy, height - crop_h))
                    cy -= cy % 2
                return crop_w, crop_h, cx, cy
            else:
                cy = head_y - int(crop_h * HEAD_RATIO)
                cy = max(0, min(cy, height - crop_h))
                cy -= cy % 2
                return crop_w, crop_h, 0, cy
        else:
            logger.warning("No person detected; returning None for fallback (Blurry Padding)")
            return None

    duration = max(1, end - start)

    # Refine transition timing with binary search (sub-frame precision)
    detections = _refine_transitions(detections, video_path, start, width, height,
                                     scale_x, scale_y, gpu_index=gpu_index)

    # Select active person per time window and smooth
    active_positions = _select_active_person(detections, duration, width)

    if pan_axis == "x":
        keyframes = _smooth_crop_trajectory(
            active_positions, duration, width, crop_w, axis="x",
            frame_h=height, crop_h=crop_h,
        )
    else:
        keyframes = _smooth_crop_trajectory(
            active_positions, duration, height, crop_h, axis="y",
        )

    if not keyframes:
        logger.warning("Trajectory generation failed; returning None for fallback (Blurry Padding)")
        return None

    # Log detailed crop info for debugging
    first_kf = keyframes[0] # type: ignore
    logger.info(f"Dynamic crop: {crop_w}x{crop_h}, {len(keyframes)} keyframes  from {width}x{height}")
    logger.info(f"    First keyframe: t={first_kf[0]:.1f}s  crop_x={first_kf[1]}  crop_y={first_kf[2]}")
    if len(keyframes) > 1:
        last_kf = keyframes[-1]
        logger.info(f"    Last keyframe:  t={last_kf[0]:.1f}s  crop_x={last_kf[1]}  crop_y={last_kf[2]}")
    return crop_w, crop_h, keyframes


def get_dimensions(video_path: Path) -> tuple[int, int]:
    """Public wrapper – returns (width, height)."""
    return _get_dimensions(video_path)


# ── internals ────────────────────────────────────────────────────────────────


def _get_dimensions(video_path: Path) -> tuple[int, int]:
    try:
        r = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        parts = [p for p in r.stdout.strip().split(",") if p.strip()]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
        return 0, 0
    except Exception:
        return 0, 0


# ── YOLO person detector (primary) ───────────────────────────────────────────


def _get_yolo_model(device_pref: str = "auto",
                    gpu_index: int | None = None):
    """Load YOLOv8-nano for person detection (auto-downloads 6MB model).

    Cached **per device** so we only load once per GPU.
    When *gpu_index* is provided the model is pinned to that specific GPU.

    Returns (model, device_str) or (None, None).
    """
    global _yolo_device_pref
    device = resolve_yolo_device(device_pref, gpu_index=gpu_index)
    with _yolo_lock:
        _yolo_device_pref = device_pref
        if device in _yolo_models:
            return _yolo_models[device], _yolo_devices.get(device, device)
        if device in _yolo_checked_devices:
            return _yolo_models.get(device), _yolo_devices.get(device)
        _yolo_checked_devices.add(device)
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        if device != "cpu":
            model.to(device)
        with _yolo_lock:
            _yolo_models[device] = model
            _yolo_devices[device] = device
        logger.info("YOLO person detector loaded on %s", device)
        return model, device
    except ImportError:
        logger.warning("ultralytics not installed — falling back to face detection")
        return None, None
    except Exception:
        logger.exception("YOLO initialization failed; falling back to face detection")
        return None, None


def _boxes_to_persons(boxes) -> list:
    """Convert YOLO boxes to (head_x, head_y, area, conf, person_h) tuples."""
    detections = []
    if boxes is None:
        return detections
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        head_x = int((x1 + x2) / 2)
        head_y = int(y1 + (y2 - y1) * 0.15)
        w = int(x2 - x1)
        h = int(y2 - y1)
        area = w * h
        confidence = float(box.conf[0])
        detections.append((head_x, head_y, area, confidence, h))
    return detections


def _detect_persons_yolo_batch(frames, model, device, conf=YOLO_CONF_SINGLE):
    """Detect persons in a batch of frames. Returns list of detection lists."""
    if not frames:
        return []
    results = model.predict(
        frames,
        classes=[0],
        conf=conf,
        verbose=False,
        device=device,
    )
    return [_boxes_to_persons(r.boxes) for r in results]


def _detect_persons_yolo(frame, model, device, conf=YOLO_CONF_SINGLE):
    """Detect persons in a single frame."""
    return _detect_persons_yolo_batch([frame], model, device, conf=conf)[0]


# ── YuNet DNN face detector (fallback) ───────────────────────────────────────


def _create_yunet_detector():
    """Create YuNet DNN face detector if model is available."""
    try:
        import cv2
        if not YUNET_MODEL.exists():
            logger.warning(f"YuNet model not found at {YUNET_MODEL}")
            return None # type: ignore
        detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", (320, 320), YUNET_CONF, YUNET_NMS, 5000, # type: ignore
        )
        return detector
    except Exception:
        logger.exception("YuNet initialization failed")
        return None


def _detect_faces_yunet(frame, detector):
    """Detect faces using YuNet DNN. Returns list of (cx, cy, area, conf, face_h)."""
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    results = []
    for f in faces:
        x, y, fw, fh, conf = int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])
        cx = x + fw // 2
        cy = y + fh // 2
        area = fw * fh
        results.append((cx, cy, area, conf, fh))
    return results


# ── Haar cascade fallback ────────────────────────────────────────────────────


def _load_cascades(cv2):
    """Load Haar cascade classifiers."""
    cascades = []
    cascade_names = [
        ("haarcascade_frontalface_default.xml", 1.0),
        ("haarcascade_frontalface_alt2.xml", 0.9),
        ("haarcascade_profileface.xml", 0.7),
    ]
    for name, weight in cascade_names:
        path = cv2.data.haarcascades + name
        c = cv2.CascadeClassifier(path)
        if not c.empty():
            cascades.append((c, weight))
    return cascades


def _detect_faces_haar(frame, cascades, scale=0.5):
    """Detect faces using Haar cascades. Returns list of (cx, cy, area, conf, face_h).

    Early-exit optimization: if the first cascade run (no equalize) finds
    well-confirmed faces, skip the equalized pass and later cascades.
    """
    import cv2
    if frame is None or frame.size == 0:
        return []

    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    all_faces = {}
    merge_dist = 50
    _HIGH_CONF_AREA = 800  # ~3% of a 540x960 frame → confident detection

    for use_eq in [False, True]:
        detect_gray = cv2.equalizeHist(gray) if use_eq else gray
        for cascade, cascade_weight in cascades:
            min_dim = int(30 * scale)
            faces = cascade.detectMultiScale(
                detect_gray, scaleFactor=1.08, minNeighbors=2,
                minSize=(min_dim, min_dim), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (fx, fy, fw, fh) in faces:
                center_x = int((fx + fw / 2) / scale)
                center_y = int((fy + fh / 2) / scale)
                area = (fw * fh) / (scale * scale)
                face_h = int(fh / scale)
                score = cascade_weight * area

                merged = False
                for key in list(all_faces.keys()):
                    ex, ey, ea, es, _ = all_faces[key]
                    if abs(center_x - ex) < merge_dist and abs(center_y - ey) < merge_dist:
                        if score > es:
                            all_faces[key] = (center_x, center_y, area, score, face_h)
                        merged = True
                        break
                if not merged:
                    all_faces[(center_x // merge_dist, center_y // merge_dist)] = (
                        center_x, center_y, area, score, face_h
                    )

            # Early exit: if we already have high-confidence face detections,
            # skip remaining cascade passes for this frame
            if all_faces and max(v[3] for v in all_faces.values()) > _HIGH_CONF_AREA:
                return list(all_faces.values())

    return list(all_faces.values())


# ── Main detection pipeline ──────────────────────────────────────────────────


def _read_frame_safe(cap, msec=None, timeout=5.0):
    """Read a frame from VideoCapture with a timeout to avoid hangs on corrupt video.
    Uses threading lock because cv2.VideoCapture is not thread-safe.
    """
    if msec is not None:
        with _cv2_lock:
            cap.set(cv2.CAP_PROP_POS_MSEC, msec)

    result = [False, None]

    def _read():
        with _cv2_lock:
            result[0], result[1] = cap.read()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Frame read hung — return failure (thread will die with daemon=True)
        print("[!] cv2.VideoCapture.read() timed out — skipping frame")
        return False, None
    return result[0], result[1]


def detect_all_persons(video_path, start, end, width, height, sample_count,
                        yolo_device: str = "auto", debug_frames: bool = False,
                        gpu_index: int | None = None):
    """Track persons across frames for dynamic cropping.

    Primary: YOLO person detection (catches ALL people regardless of pose).
    Fallback: YuNet face + Haar cascade (when YOLO unavailable).

    Returns list of (relative_time, [(head_x, head_y, area, conf, h), ...]) tuples.
    Gap-fills so every sample time has a position.

    IMPORTANT: Detects dimension mismatches between ffprobe and OpenCV
    (e.g. rotation metadata) and rescales coordinates to match ffprobe's
    dimensions, which are what ffmpeg uses for cropping.
    """

    # Try YOLO first (much more reliable)
    yolo, yolo_dev = _get_yolo_model(yolo_device, gpu_index=gpu_index)
    use_yolo = yolo is not None

    # Fallback: face detectors
    yunet = None
    cascades = None
    if not use_yolo: # type: ignore
        yunet = _create_yunet_detector()
        if yunet:
            logger.info("Using YuNet face detector (fallback)")
        else:
            cascades = _load_cascades(cv2)
            if not cascades: # type: ignore
                return []
            print("[i] Using Haar cascade face detector (fallback)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    try:
        duration = max(1, end - start)
        # Dense sampling — at least 4 samples per second for smooth tracking
        effective_samples = max(sample_count, int(duration * 4))
        step = max(0.25, duration / effective_samples)
        sample_times = list(_frange(0, duration, step))
        if not sample_times:
            sample_times = [0]

        scale = 0.5 if width > 800 else 0.75
        detections = []
        detected_frames = 0
        yolo_frames = 0
        face_frames = 0

        # ── Detect dimension mismatch (rotation / codec quirks) ───────────
        # Read one frame to get actual OpenCV dimensions
        ok, test_frame = _read_frame_safe(cap, msec=start * 1000, timeout=10.0)
        scale_x, scale_y = 1.0, 1.0
        if ok and test_frame is not None: # type: ignore
            cv_h, cv_w = test_frame.shape[:2]
            if cv_w != width or cv_h != height:
                # Dimensions differ — need to rescale coordinates
                # OpenCV reads raw coded dimensions; ffprobe may report rotated
                is_swapped = abs(cv_w - height) < 4 and abs(cv_h - width) < 4
                if is_swapped:
                    scale_x = width / cv_h
                    scale_y = height / cv_w
                else:
                    scale_x = width / cv_w
                    scale_y = height / cv_h
                logger.warning(f"Dimension mismatch: ffprobe={width}x{height}, "
                      f"OpenCV={cv_w}x{cv_h} → rescaling coords by {scale_x:.2f}x{scale_y:.2f}")
                # Check if it's a 90°/270° rotation (width/height swapped)
                if abs(cv_w - height) < 4 and abs(cv_h - width) < 4:
                    logger.warning("Detected 90° rotation — swapping coordinate axes")

        debug_saved = False
        last_good_persons = None # for gap-filling frames with no detection
        last_good_time = -10.0  # timestamp of last valid detection

        # ── Process in chunks to save memory ──
        for i in range(0, len(sample_times), YOLO_BATCH_SIZE):
            batch_times = sample_times[i : i + YOLO_BATCH_SIZE]
            batch_frames = []
            
            for t in batch_times:
                if is_cancelled():
                    raise CancelledError("Person detection cancelled")
                ok, frame = _read_frame_safe(cap, msec=(start + t) * 1000)
                if ok and frame is not None:
                    batch_frames.append((t, frame))

            if not batch_frames: continue

            # Run Inference
            batch_results = []
            if use_yolo:
                y_frames = [f for _, f in batch_frames]
                batch_persons = _detect_persons_yolo_batch(y_frames, yolo, yolo_dev, conf=YOLO_CONF)
                for (t, frame), persons in zip(batch_frames, batch_persons):
                    batch_results.append((t, frame, persons))
            else:
                for t, frame in batch_frames:
                    persons = []
                    if yunet:
                        persons = _detect_faces_yunet(frame, yunet)
                    if not persons and cascades:
                        persons = _detect_faces_haar(frame, cascades, scale)
                    batch_results.append((t, frame, persons))

            # Post-process detections
            for t, frame, persons in batch_results:
                if persons and (scale_x != 1.0 or scale_y != 1.0):
                    # Check for 90/270 swap logic
                    cv_h, cv_w = frame.shape[:2]
                    is_swapped = abs(cv_w - height) < 4 and abs(cv_h - width) < 4
                    
                    new_persons = []
                    for hx, hy, a, c, h in persons:
                        if is_swapped:
                            # 90° or 270° rotation: swap x/y and handle scaling
                            # Try both mappings and pick the one within frame bounds
                            candidate1 = (int(hy * scale_x), int(hx * scale_y))
                            candidate2 = (int((cv_w - hy) * scale_x), int((cv_h - hx) * scale_y))
                            # Pick the candidate that places the head in the upper third
                            if candidate1[1] < candidate2[1] or candidate2[1] >= height:
                                new_hx, new_hy = candidate1
                            else:
                                new_hx, new_hy = candidate2
                        else:
                            new_hx, new_hy = int(hx * scale_x), int(hy * scale_y)
                        new_persons.append((new_hx, new_hy, int(a * scale_x * scale_y), c, int(h * scale_y)))
                    persons = new_persons

                if persons:
                    detections.append((t, persons))
                    detected_frames += 1
                    if use_yolo: yolo_frames += 1
                    else: face_frames += 1
                    last_good_persons = persons
                    last_good_time = t
                    # Save debug frame only once for the first detection if debug_frames is True
                    if not debug_saved and use_yolo and debug_frames:
                        _save_debug_frame(frame, persons, width, height, scale_x, scale_y, video_path)
                        debug_saved = True
                elif last_good_persons is not None and t - last_good_time <= 5.0:
                    # Only reuse last detection if within 5 seconds (avoids stale tracking)
                    detections.append((t, last_good_persons))
    finally:
        cap.release()

    total_persons = sum(len(p) for _, p in detections)
    method = "YOLO" if use_yolo else "face detection"
    extras = [f"{yolo_frames} YOLO" if yolo_frames else "", f"{face_frames} face" if face_frames else ""]
    extras = [e for e in extras if e]
    extra = f" ({', '.join(extras)})" if extras else ""

    logger.info(f"Tracking ({method}): {detected_frames}/{len(sample_times)} frames, {total_persons} total detections{extra}")

    if detected_frames == 0:
        logger.warning("No persons detected in any frame -> fallback (Blurry Padding)")
        return [], 1.0, 1.0

    return detections, scale_x, scale_y




def _refine_transitions(detections, video_path, start, width, height,
                        scale_x=1.0, scale_y=1.0,
                        max_iterations=4, min_gap=0.033,
                        gpu_index: int | None = None):
    """Binary-search for exact transition frames between large position jumps.

    When consecutive detections show a person position jump > CAMERA_CUT_THRESH
    of frame width, binary-searches between them to find the exact cut frame
    (within ~16ms / sub-frame precision). Inserts refined detections into the list.

    This eliminates the 250ms timing gap that caused empty frames during transitions.
    """
    if len(detections) < 2:
        return detections

    cut_threshold = width * CAMERA_CUT_THRESH

    # Identify transition pairs: consecutive samples where dominant person jumps
    transitions = []
    for i in range(len(detections) - 1):
        t_a, persons_a = detections[i]
        t_b, persons_b = detections[i + 1]
        if not persons_a or not persons_b:
            continue
        best_a = max(persons_a, key=lambda p: p[2])  # largest by area
        best_b = max(persons_b, key=lambda p: p[2])
        if abs(best_a[0] - best_b[0]) > cut_threshold:
            transitions.append((i, t_a, t_b, best_a[0]))  # store old person's x

    if not transitions:
        return detections

    # Load YOLO model for refinement reads
    yolo, yolo_dev = _get_yolo_model(_yolo_device_pref, gpu_index=gpu_index)
    if yolo is None:
        print("[!] No YOLO model for transition refinement, skipping")
        return detections

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return detections

    try:
        insertions = []  # (insert_after_index, [(t, persons), ...])

        for orig_idx, t_lo, t_hi, old_person_x in transitions:
            refined = []
            lo, hi = t_lo, t_hi

            for iteration in range(max_iterations):
                if hi - lo < min_gap:
                    break

                if is_cancelled():
                    raise CancelledError("Person detection cancelled")

                t_mid = (lo + hi) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + t_mid) * 1000)
                ok, frame = _read_frame_safe(cap, timeout=5.0)
                if not ok or frame is None:
                    break

                persons = _detect_persons_yolo(frame, yolo, yolo_dev, conf=YOLO_CONF)

                # Rescale coordinates if needed
                if persons and (scale_x != 1.0 or scale_y != 1.0):
                    persons = [
                        (int(hx * scale_x), int(hy * scale_y),
                         int(a * scale_x * scale_y), c, int(h * scale_y))
                        for hx, hy, a, c, h in persons
                    ]

                if persons:
                    best_mid = max(persons, key=lambda p: p[2])
                    if abs(best_mid[0] - old_person_x) > cut_threshold:
                        # Mid looks like the new scene → cut is between lo and mid
                        hi = t_mid
                    else:
                        # Mid looks like the old scene → cut is between mid and hi
                        lo = t_mid
                    refined.append((t_mid, persons))
                else:
                    # No detection (motion blur / black frame during cut)
                    # Assume cut is happening here, narrow to upper half
                    hi = t_mid

            if refined:
                insertions.append((orig_idx, refined))
    finally:
        cap.release()

    if not insertions:
        return detections

    # Merge insertions into detections list (in reverse to preserve indices)
    result = list(detections)
    for insert_after, new_entries in reversed(insertions):
        # Sort new entries by time and insert after the original detection
        new_entries.sort(key=lambda e: e[0])
        for entry in reversed(new_entries):
            result.insert(insert_after + 1, entry)

    # Re-sort by time to ensure correct order
    result.sort(key=lambda e: e[0])

    # Remove detections that fall inside the transition gap (cut frames).
    # These are frames near the cut boundary where YOLO might detect a person
    # from the wrong scene, causing the crop to briefly show an empty area.
    # For each transition, suppress detections in a window around the cut
    # point — let the old crop hold through the entire transition.
    for orig_idx, t_lo, t_hi, old_person_x in transitions:
        # The binary search narrowed lo..hi; the cut is near hi (upper bound)
        # Suppress detections from [lo - HOLD_BEFORE, lo + DELAY] that look
        # like the new scene (they'd cause a premature crop switch)
        suppress_start = t_lo - CUT_HOLD_BEFORE
        suppress_end = t_lo + CUT_DELAY_SEC # type: ignore
        result = [
            (t, persons) for t, persons in result
            if not (suppress_start < t < suppress_end and persons and
                    abs(max(persons, key=lambda p: p[2])[0] - old_person_x) > cut_threshold)
        ]

    total_refined = sum(len(entries) for _, entries in insertions)
    logger.info(f"Refined {len(insertions)} transition(s), "
          f"added {total_refined} sub-frame detections")

    return result


def _detect_people(video_path, start, end, width, height, sample_count):
    """Detect persons for static crop. Returns (median_x, head_y) or (None, None)."""
    detections, _, _ = detect_all_persons(video_path, start, end, width, height, sample_count)
    if not detections:
        logger.warning("No persons detected -> fallback (Blurry Padding)")
        return None, None

    frame_detections = []
    for t, persons in detections:
        best = max(persons, key=lambda p: p[2])
        frame_detections.append((best[0], best[1]))
    # type: ignore
    xs = [d[0] for d in frame_detections]
    ys = [d[1] for d in frame_detections]
    result_x = int(np.median(xs))
    result_y = int(np.median(ys))
    print(f"[+] Main subject at x={result_x}, head_y={result_y}  "
          f"(median of {len(frame_detections)} detections)")
    return result_x, result_y


# ── Tracking & smoothing ─────────────────────────────────────────────────────


def _select_active_person(detections, duration, frame_width=1920):
    """Select the most prominent person per frame with camera cut detection.

    Camera cut detection: if the largest person jumps by >40% of frame width,
    we treat it as a camera cut and always pick the largest person (don't try
    to match the old person from the previous scene).

    Returns list of (time, head_x, head_y, person_h).
    """
    if not detections:
        return []

    active = []
    prev_x, prev_y = None, None
    cut_threshold = frame_width * CAMERA_CUT_THRESH

    for t, persons in detections:
        if not persons:
            continue

        if len(persons) == 1:
            best = persons[0]
        else:
            sorted_persons = sorted(persons, key=lambda p: p[2], reverse=True)
            largest = sorted_persons[0]
            largest_area = largest[2]

            # Detect camera cut: largest person jumped far from previous
            is_camera_cut = False
            if prev_x is not None:
                if abs(largest[0] - prev_x) > cut_threshold:
                    is_camera_cut = True

            if is_camera_cut or prev_x is None:
                # Camera cut or first frame: always pick the largest person
                best = largest
            else:
                # Stable shot: prefer proximity among large-enough candidates
                candidates = [p for p in sorted_persons if p[2] > largest_area * 0.6]
                if len(candidates) > 1:
                    best = min(candidates,
                               key=lambda p: abs(p[0] - prev_x) + abs(p[1] - prev_y))
                else:
                    best = largest

        person_h = best[4] if len(best) > 4 else 0
        active.append((t, best[0], best[1], person_h))
        prev_x, prev_y = best[0], best[1]

    return active


def _smooth_crop_trajectory(active_positions, duration, frame_size, crop_size,
                            axis="x", frame_h=0, crop_h=0):
    """Convert per-frame person positions into crop keyframes.

    Per-frame sequential processing with hysteresis:
    - Each frame's detected person position is converted to a crop offset
    - A new keyframe is emitted only when the position changes by more than
      CHANGE_THRESH of the crop size (prevents jitter on stable shots)
    - Frames without detections simply don't appear — the crop holds its
      last known good position (no gap-fill, no phantom positions)

    For Y axis: places head at ~30% from top (rule of thirds).
    For X axis: centers person, and also computes a body-aware Y offset.
    """
    if not active_positions:
        return []

    edge_pad = int(crop_size * 0.20)

    def _to_crop_offset(head_center):
        if axis == "y":
            target_pos = int(crop_size * HEAD_RATIO)
            crop_offset = head_center - target_pos
        else:
            crop_offset = head_center - crop_size // 2

        crop_offset = max(0, min(crop_offset, frame_size - crop_size))

        # Safety clamp: ensure person is not too close to crop edge
        pos_in_crop = head_center - crop_offset
        if pos_in_crop < edge_pad:
            crop_offset = max(0, head_center - edge_pad)
        elif pos_in_crop > crop_size - edge_pad:
            crop_offset = min(frame_size - crop_size, head_center - crop_size + edge_pad)

        crop_offset = max(0, min(crop_offset, frame_size - crop_size))
        crop_offset -= crop_offset % 2
        return crop_offset

    # ── Body-aware Y offset (when panning on X axis) ──
    fixed_y_offset = 0
    if axis == "x" and frame_h > 0 and crop_h > 0 and crop_h < frame_h:
        all_head_ys = [y for t, x, y, *rest in active_positions]
        if all_head_ys:
            median_head_y = int(np.median(all_head_ys))
            target_pos = int(crop_h * HEAD_RATIO)
            fixed_y_offset = median_head_y - target_pos
            fixed_y_offset = max(0, min(fixed_y_offset, frame_h - crop_h))
            fixed_y_offset -= fixed_y_offset % 2
            print(f"[+] Body-aware Y offset: {fixed_y_offset}  (head_y={median_head_y})")

    # ── Per-frame hysteresis ──
    change_threshold = crop_size * CHANGE_THRESH
    # Camera cut = large jump (same threshold as detection)
    camera_cut_threshold = frame_size * CAMERA_CUT_THRESH

    # Extract the relevant axis value per frame
    if axis == "x":
        frame_data = [(t, x) for t, x, y, *rest in active_positions]
    else:
        frame_data = [(t, y) for t, x, y, *rest in active_positions]

    # Initialize with first frame
    first_t, first_val = frame_data[0]
    held_center = first_val
    held_offset = _to_crop_offset(held_center)

    if axis == "x":
        keyframes = [(0, held_offset, fixed_y_offset)]
    else:
        keyframes = [(0, 0, held_offset)]

    # Process remaining frames sequentially
    for t, val in frame_data[1:]:
        if abs(val - held_center) > change_threshold:
            # Detect if this is a camera cut (large jump) vs gradual movement
            is_camera_cut = abs(val - held_center) > camera_cut_threshold

            old_offset = held_offset  # remember old position for hold keyframe
            held_center = val
            held_offset = _to_crop_offset(held_center)

            if is_camera_cut:
                # Camera cut detected — insert explicit "hold old position"
                # keyframe right before the transition, then delay the new
                # position keyframe AFTER the cut. This creates a clean gap:
                #
                #   [old crop holds] ──── cut happens ──── [new crop starts]
                #                    ^hold_before    ^delay_after
                #
                # The old crop stays put through the transition frames,
                # preventing any empty/random frame from showing.

                # 1. Pin the OLD crop position just before the cut
                hold_time = max(0, t - CUT_HOLD_BEFORE)
                if axis == "x":
                    hold_kf = (hold_time, old_offset, fixed_y_offset)
                else:
                    hold_kf = (hold_time, 0, old_offset)
                last_kf = keyframes[-1]
                if hold_kf[0] > last_kf[0]:  # only if it's a new time
                    keyframes.append(hold_kf)

                # 2. Delay the new crop position until after the cut settles
                kf_time = t + CUT_DELAY_SEC
            else:
                kf_time = t

            if axis == "x":
                new_kf = (kf_time, held_offset, fixed_y_offset)
            else:
                new_kf = (kf_time, 0, held_offset)

            # Deduplicate: only append if position actually differs
            last_kf = keyframes[-1]
            if new_kf[1] != last_kf[1] or new_kf[2] != last_kf[2]:
                keyframes.append(new_kf)

    return keyframes


# ── Utility ──────────────────────────────────────────────────────────────────


def _save_debug_frame(frame, persons, ffprobe_w, ffprobe_h, scale_x, scale_y,
                      video_path):
    """Save an annotated debug frame showing YOLO detections + crop region.

    Writes to <video_dir>/crop_debug.jpg so you can visually verify
    what YOLO detected and where the crop would land.
    """
    try:
        import cv2
        if frame is None or frame.size == 0:
            return
        debug = frame.copy()
        cv_h, cv_w = debug.shape[:2]

        # Draw each person detection (in OpenCV coordinates)
        for hx, hy, area, conf, ph in persons:
            # Convert back from ffprobe coords to OpenCV coords for drawing
            draw_hx = int(hx / scale_x) if scale_x != 1.0 else hx
            draw_hy = int(hy / scale_y) if scale_y != 1.0 else hy
            draw_ph = int(ph / scale_y) if scale_y != 1.0 else ph

            # Estimate person bbox from head position + height
            est_w = draw_ph // 3  # rough width estimate
            x1 = max(0, draw_hx - est_w // 2)
            y1 = max(0, draw_hy - int(draw_ph * 0.15))  # head is at 15% from top
            x2 = min(cv_w, draw_hx + est_w // 2)
            y2 = min(cv_h, y1 + draw_ph)

            # Green box = person detection
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Red dot = head position
            cv2.circle(debug, (draw_hx, draw_hy), 8, (0, 0, 255), -1)
            cv2.putText(debug, f"{conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw crop region (9:16 from ffprobe dimensions, mapped to OpenCV)
        target_ratio = 9 / 16
        if ffprobe_w / ffprobe_h > target_ratio:
            crop_w = int(ffprobe_h * target_ratio)
            crop_h = ffprobe_h
            # Use largest person for crop X
            best = max(persons, key=lambda p: p[2])
            crop_x = best[0] - crop_w // 2
            crop_x = max(0, min(crop_x, ffprobe_w - crop_w))
            crop_y = 0
        else:
            crop_w = ffprobe_w
            crop_h = int(ffprobe_w / target_ratio)
            best = max(persons, key=lambda p: p[2])
            crop_y = best[1] - int(crop_h * HEAD_RATIO)
            crop_y = max(0, min(crop_y, ffprobe_h - crop_h))
            crop_x = 0

        # Map crop rect to OpenCV coordinates for drawing
        dcx1 = int(crop_x / scale_x) if scale_x != 1.0 else crop_x
        dcy1 = int(crop_y / scale_y) if scale_y != 1.0 else crop_y
        dcx2 = int((crop_x + crop_w) / scale_x) if scale_x != 1.0 else crop_x + crop_w
        dcy2 = int((crop_y + crop_h) / scale_y) if scale_y != 1.0 else crop_y + crop_h

        # Blue rectangle = crop region
        cv2.rectangle(debug, (dcx1, dcy1), (dcx2, dcy2), (255, 100, 0), 4)

        # Info text
        info = f"ffprobe: {ffprobe_w}x{ffprobe_h}  OpenCV: {cv_w}x{cv_h}  crop: {crop_w}x{crop_h}"
        cv2.putText(debug, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug, f"Persons: {len(persons)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if scale_x != 1.0 or scale_y != 1.0:
            cv2.putText(debug, f"RESCALE: {scale_x:.2f}x{scale_y:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save
        debug_path = Path(video_path).parent / "crop_debug.jpg"
        cv2.imwrite(str(debug_path), debug)
        logger.info(f"Debug frame saved: {debug_path}")
    except Exception:
        logger.exception("Debug frame save failed")


def _frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step
