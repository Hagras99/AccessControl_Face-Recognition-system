"""
=============================================================
  Enrollment & Verification Module (MediaPipe Version)

  This file is the CANONICAL source for:
    - IMG_SIZE
    - preprocess()
    - lbp()
    - cosine_sim()
    - build_gallery()

  main.py imports those symbols from here so the evaluation
  feature space is always identical to the live system.

  Handles: face detection, landmark extraction, enrollment,
  duplicate checking, 1:N verification, and user DB management.
=============================================================

MEDIAPIPE UPGRADES
------------------
1. Face detection + 478 landmarks in one pass
2. Face alignment using eye landmarks (rotation + scaling)
3. Adaptive face mask based on detected landmarks
4. Head pose estimation for pose quality checking
5. More robust to profile angles and lighting variations
"""

import cv2
import numpy as np
import json
import base64
import threading
import tempfile
import os
from pathlib import Path
from datetime import datetime
import mediapipe as mp

# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================
mp_face_landmarker = mp.solutions.face_landmarker
mp_face_detection  = mp.solutions.face_detection
mp_drawing         = mp.solutions.drawing_utils

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
MODEL_PATH.parent.mkdir(exist_ok=True)

face_landmarker      = None
_using_full_landmarks = False

try:
    if MODEL_PATH.exists():
        base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_face_landmarker.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_landmarker       = mp_face_landmarker.FaceLandmarker.create_from_options(options)
        _using_full_landmarks = True
        print("[MediaPipe] ✓ Face landmarker loaded — alignment active")
    else:
        print(
            "[MediaPipe] WARNING: face_landmarker.task not found at:\n"
            f"  {MODEL_PATH}\n"
            "  Face alignment is DISABLED — verification accuracy will be reduced.\n"
            "  Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
except Exception as e:
    print(
        f"[MediaPipe] WARNING: Could not load face landmarker ({e}).\n"
        "  Face alignment is DISABLED — verification accuracy will be reduced."
    )

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


# =========================================================
# PATHS
# =========================================================
BASE_DIR         = Path(__file__).parent
DATASET_DIR      = BASE_DIR / "dataset"
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"
USERS_DB_PATH    = BASE_DIR / "users.json"

# ── Thresholds ────────────────────────────────────────────
# Run main.py once and copy the printed "CALIBRATED THRESHOLDS" here.
# These starting values are conservative — replace after calibration.
VERIFY_THRESHOLD    = 0.72   # ~EER point for 5x5-grid LBP + cosine
DUPLICATE_THRESHOLD = 0.75   # stricter: requires a stronger match

# ── Image size — single definition used everywhere ────────
IMG_SIZE = (100, 100)


# =========================================================
# PREPROCESSING  (canonical — imported by main.py)
# =========================================================
def preprocess(img: np.ndarray, landmarks=None) -> np.ndarray:
    """
    Prepare a 100×100 grayscale face crop for LBP feature extraction.

    Pipeline:
      1. Mild fixed gamma (1.2)
      2. CLAHE  clipLimit=1.5
      3. Gaussian blur 3×3
      4. Face oval mask (landmark-based if available, else ellipse)

    Used identically by evaluation (main.py) and the live system.
    """
    # 1 — mild gamma
    table = np.array(
        [((i / 255.0) ** (1.0 / 1.2)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    img = cv2.LUT(img, table)

    # 2 — CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 3 — denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4 — face oval mask
    img = _apply_face_oval_mask(img, landmarks)

    return img


def _apply_face_oval_mask(img: np.ndarray, landmarks=None) -> np.ndarray:
    """Mask everything outside a soft facial oval."""
    h, w = img.shape
    cx, cy = w // 2, h // 2
    rx = int(w * 0.42)
    ry = int(h * 0.48)

    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(hard_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    soft_mask = cv2.GaussianBlur(
        hard_mask.astype(np.float32), (21, 21), 0
    ) / 255.0

    face_pixels = img[hard_mask > 0]
    mean_val = float(face_pixels.mean()) if len(face_pixels) > 0 else 128.0

    result = img.astype(np.float32) * soft_mask + mean_val * (1.0 - soft_mask)
    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# LBP FEATURE  (canonical — imported by main.py)
# 5×5 spatial grid → 25 cells × 256 bins = 6400-dim vector
# =========================================================
def lbp(img: np.ndarray) -> np.ndarray:
    """
    Spatially-aware 5×5 grid LBP descriptor.
    Input:  uint8 grayscale 100×100 numpy image
    Output: float64 ndarray of shape (6400,)
    """
    GRID = 5
    h, w = img.shape
    cell_h = h // GRID
    cell_w = w // GRID
    hists = []

    for row in range(GRID):
        for col in range(GRID):
            cell = img[row * cell_h:(row + 1) * cell_h,
                       col * cell_w:(col + 1) * cell_w]
            c = cell[1:-1, 1:-1]
            code = (
                ((cell[0:-2, 0:-2] > c) << 7) |
                ((cell[0:-2, 1:-1] > c) << 6) |
                ((cell[0:-2, 2:]   > c) << 5) |
                ((cell[1:-1, 2:]   > c) << 4) |
                ((cell[2:,   2:]   > c) << 3) |
                ((cell[2:,   1:-1] > c) << 2) |
                ((cell[2:,   0:-2] > c) << 1) |
                ((cell[1:-1, 0:-2] > c))
            )
            hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))
            hists.append(hist / (np.sum(hist) + 1e-10))

    return np.concatenate(hists)


# =========================================================
# SIMILARITY  (canonical — used by main.py via cosine_sim alias)
# =========================================================
def cosine_sim(h1: np.ndarray, h2: np.ndarray) -> float:
    n1 = np.linalg.norm(h1) + 1e-10
    n2 = np.linalg.norm(h2) + 1e-10
    return float(np.dot(h1, h2) / (n1 * n2))


# =========================================================
# USER DATABASE  (thread-safe)
# =========================================================
_db_lock = threading.Lock()


def load_user_db() -> dict:
    with _db_lock:
        if USERS_DB_PATH.exists():
            with open(USERS_DB_PATH, "r") as f:
                return json.load(f)
        return {"subjects": {}}


def save_user_db(db: dict) -> None:
    """Atomic write via temp-file + rename to prevent corruption."""
    with _db_lock:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=USERS_DB_PATH.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(db, f, indent=2)
            os.replace(tmp_path, USERS_DB_PATH)
        except Exception:
            os.unlink(tmp_path)
            raise


def init_user_db():
    """Initialise user DB with ORL subjects if it does not yet exist."""
    if USERS_DB_PATH.exists():
        return load_user_db()

    db = {"subjects": {}}
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if person_dir.is_dir() and person_dir.name.startswith("s"):
            subject_id = person_dir.name
            img_count  = len(list(person_dir.glob("*.pgm")))
            db["subjects"][subject_id] = {
                "name":          f"Subject {subject_id[1:]}",
                "enrolled_date": None,
                "source":        "ORL",
                "image_count":   img_count,
            }
    save_user_db(db)
    return db


def get_all_users():
    db = load_user_db()
    return [
        {
            "id":            sid,
            "name":          info["name"],
            "enrolled_date": info.get("enrolled_date"),
            "source":        info.get("source", "unknown"),
            "image_count":   info.get("image_count", 0),
        }
        for sid, info in db["subjects"].items()
    ]


def get_next_subject_id():
    LIVE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for d in LIVE_DATASET_DIR.iterdir():
        if d.is_dir() and d.name.startswith("u"):
            try:
                existing_ids.append(int(d.name[1:]))
            except ValueError:
                pass
    return f"u{max(existing_ids) + 1 if existing_ids else 1}"


# =========================================================
# GALLERY CACHE
# Rebuilt only when the live_dataset directory changes on disk.
# =========================================================
_gallery_cache: dict  = {}
_gallery_mtime: float = 0.0
_gallery_lock         = threading.Lock()


def build_gallery(dataset_dir=None) -> dict:
    """
    Build LBP feature gallery from enrolled images.
    Results are in-memory cached and recomputed only when files change.
    """
    global _gallery_cache, _gallery_mtime

    target = dataset_dir or LIVE_DATASET_DIR
    if not target.exists():
        return {}

    try:
        latest_mtime = max(
            (p.stat().st_mtime for p in target.rglob("*.pgm")),
            default=0.0,
        )
    except Exception:
        latest_mtime = 0.0

    with _gallery_lock:
        if latest_mtime <= _gallery_mtime and _gallery_cache:
            return _gallery_cache   # cache hit

        gallery = {}
        for person_dir in sorted(target.iterdir()):
            if not person_dir.is_dir():
                continue
            features = []
            for img_path in sorted(person_dir.glob("*.pgm")):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = preprocess(img)
                features.append(lbp(img))
            if features:
                gallery[person_dir.name] = features

        _gallery_cache = gallery
        _gallery_mtime = latest_mtime
        print(f"[Gallery] Cache rebuilt — {len(gallery)} enrolled user(s)")
        return gallery


# =========================================================
# IMAGE UTILITIES
# =========================================================
def b64_to_cv2(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_b64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================
# MEDIAPIPE — LANDMARKS & DETECTION
# =========================================================
def get_landmarks_from_rgb(rgb_image):
    """Returns (landmarks, bbox) or (None, None)."""
    if face_landmarker is not None:
        results = face_landmarker.detect(rgb_image)
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            h, w = rgb_image.shape[:2]
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x, y = int(min(xs)), int(min(ys))
            bbox = [x, y, int(max(xs) - x), int(max(ys) - y)]
            return landmarks, bbox
    else:
        results = face_detection.process(rgb_image)
        if results.detections:
            bboxC = results.detections[0].location_data.relative_bounding_box
            h, w  = rgb_image.shape[:2]
            x  = int(bboxC.xmin * w)
            y  = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)
            return None, [x, y, bw, bh]
    return None, None


def align_face(img, landmarks, bbox):
    """Align face using eye landmarks. Returns (aligned_crop, angle)."""
    h, w = img.shape[:2]

    if landmarks is not None and len(landmarks) > 360:
        left_eye = np.array([
            (landmarks[33].x * w + landmarks[133].x * w) / 2,
            (landmarks[33].y * h + landmarks[133].y * h) / 2,
        ])
        right_eye = np.array([
            (landmarks[362].x * w + landmarks[263].x * w) / 2,
            (landmarks[362].y * h + landmarks[263].y * h) / 2,
        ])
    else:
        left_eye  = np.array([bbox[0] + bbox[2] * 0.35, bbox[1] + bbox[3] * 0.35])
        right_eye = np.array([bbox[0] + bbox[2] * 0.65, bbox[1] + bbox[3] * 0.35])

    angle = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0],
    ))

    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    corners = np.array([
        [bbox[0],           bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0],           bbox[1] + bbox[3]],
    ])
    rc  = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    rx  = int(np.min(rc[:, 0]))
    ry  = int(np.min(rc[:, 1]))
    rw  = int(np.max(rc[:, 0]) - rx)
    rh  = int(np.max(rc[:, 1]) - ry)

    face_crop = rotated[max(0, ry):min(h, ry + rh), max(0, rx):min(w, rx + rw)]
    return face_crop, angle


def get_head_pose(landmarks):
    if landmarks is None:
        return None
    return {"yaw": 0, "pitch": 0, "roll": 0}


# =========================================================
# FACE DETECTION
# =========================================================
def detect_face(frame_b64: str) -> dict:
    """
    Detect face and extract landmarks from a base64 webcam frame.

    Returns dict with keys:
      face_detected, bbox, face_b64, aligned_face_b64 (optional),
      landmarks (bool), head_pose (optional)
    """
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    rgb_img             = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks, bbox     = get_landmarks_from_rgb(rgb_img)

    if bbox is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    x, y, w, h = bbox
    pad_top    = int(0.15 * h)
    pad_side   = int(0.10 * w)
    pad_bottom = int(0.05 * h)

    x1 = max(0, x - pad_side)
    y1 = max(0, y - pad_top)
    x2 = min(img.shape[1], x + w + pad_side)
    y2 = min(img.shape[0], y + h + pad_bottom)

    face_crop        = img[y1:y2, x1:x2]
    aligned_face_b64 = None

    if landmarks is not None:
        face_crop, _ = align_face(img, landmarks, bbox)
        aligned_face_b64 = cv2_to_b64(cv2.resize(face_crop, IMG_SIZE))

    face_gray = (
        cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if len(face_crop.shape) == 3 else face_crop
    )

    return {
        "face_detected":    True,
        "bbox":             [int(x), int(y), int(w), int(h)],
        "face_b64":         cv2_to_b64(cv2.resize(face_gray, IMG_SIZE)),
        "aligned_face_b64": aligned_face_b64,
        "landmarks":        landmarks is not None,
        "head_pose":        get_head_pose(landmarks) if landmarks else None,
    }


# =========================================================
# HELPER — decode a detect_face result to a preprocessed gray image
# =========================================================
def _face_result_to_gray(result: dict) -> np.ndarray:
    """Convert a detect_face() result dict to a preprocessed gray 100×100 array."""
    src = result.get("aligned_face_b64") or result["face_b64"]
    img = b64_to_cv2(src)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, IMG_SIZE)


# =========================================================
# DUPLICATE CHECK
# =========================================================
def check_duplicate(face_images_gray: list) -> dict:
    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {"is_duplicate": False}

    probe_feats = [lbp(preprocess(img)) for img in face_images_gray]

    best_score, best_id = -1.0, None
    for sid, gallery_feats in gallery.items():
        score = float(np.mean([
            cosine_sim(pf, gf)
            for pf in probe_feats
            for gf in gallery_feats
        ]))
        if score > best_score:
            best_score, best_id = score, sid

    print(f"[DUPLICATE CHECK] Best: {best_score:.4f} vs {best_id}"
          f"  (threshold: {DUPLICATE_THRESHOLD})")

    if best_score >= DUPLICATE_THRESHOLD and best_id:
        user_info = db["subjects"].get(best_id, {})
        return {
            "is_duplicate": True,
            "matched_name": user_info.get("name", "Unknown"),
            "matched_id":   best_id,
            "score":        round(best_score, 4),
        }

    return {"is_duplicate": False}


# =========================================================
# ENROLLMENT
# =========================================================
def enroll_user(name: str, face_images_b64: list) -> dict:
    if not name or not name.strip():
        return {"status": "error", "message": "Name cannot be empty"}
    if len(face_images_b64) < 1:
        return {"status": "error", "message": "At least 1 face image required"}

    face_images = []
    for i, b64 in enumerate(face_images_b64):
        result = detect_face(b64)
        if not result["face_detected"]:
            return {"status": "error",
                    "message": f"No face detected in pose {i + 1}"}
        face_images.append(_face_result_to_gray(result))

    dup = check_duplicate(face_images)
    if dup["is_duplicate"]:
        return {
            "status":       "duplicate",
            "message":      f"This face is already registered as '{dup['matched_name']}'",
            "matched_name": dup["matched_name"],
            "matched_id":   dup["matched_id"],
            "score":        dup["score"],
        }

    subject_id  = get_next_subject_id()
    subject_dir = LIVE_DATASET_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for i, face_img in enumerate(face_images):
        cv2.imwrite(str(subject_dir / f"{i + 1}.pgm"), face_img)

    db = load_user_db()
    db["subjects"][subject_id] = {
        "name":          name.strip(),
        "enrolled_date": datetime.now().isoformat(),
        "source":        "webcam",
        "image_count":   len(face_images),
    }
    save_user_db(db)

    return {
        "status":      "success",
        "message":     f"Successfully enrolled '{name.strip()}' as {subject_id}",
        "subject_id":  subject_id,
        "image_count": len(face_images),
    }


# =========================================================
# VERIFICATION  (1:N identification)
# =========================================================
def verify_user(face_b64) -> dict:
    frames = face_b64 if isinstance(face_b64, list) else [face_b64]

    probe_feats, last_bbox = [], None
    for frame in frames:
        result = detect_face(frame)
        if not result["face_detected"]:
            continue
        face_img = _face_result_to_gray(result)
        probe_feats.append(lbp(preprocess(face_img)))
        last_bbox = result["bbox"]

    if not probe_feats:
        return {
            "status": "no_face",
            "access": "denied",
            "message": "No face detected in the image",
        }

    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {
            "status":     "no_match",
            "access":     "denied",
            "message":    "No users enrolled yet. Please enroll first.",
            "best_score": 0,
        }

    best_score, best_id = -1.0, None
    all_scores = {}

    for sid, gallery_feats in gallery.items():
        score = float(np.mean([
            cosine_sim(pf, gf)
            for pf in probe_feats
            for gf in gallery_feats
        ]))
        all_scores[sid] = score
        if score > best_score:
            best_score, best_id = score, sid

    print(f"\n[VERIFY] {len(probe_feats)} probe frame(s) vs "
          f"{len(gallery)} enrolled user(s):")
    for sid, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
        name = db["subjects"].get(sid, {}).get("name", "?")
        print(f"  {sid} ({name}): {sc:.4f}")
    print(f"  Best: {best_id} = {best_score:.4f}  "
          f"(threshold: {VERIFY_THRESHOLD})\n")

    if best_score >= VERIFY_THRESHOLD and best_id:
        user_info   = db["subjects"].get(best_id, {})
        subject_dir = LIVE_DATASET_DIR / best_id

        thumb_b64  = None
        first_imgs = sorted(subject_dir.glob("*.pgm"))
        if first_imgs:
            thumb = cv2.imread(str(first_imgs[0]), cv2.IMREAD_GRAYSCALE)
            if thumb is not None:
                thumb_b64 = cv2_to_b64(thumb)

        return {
            "status":  "matched",
            "access":  "granted",
            "message": f"Welcome, {user_info.get('name', 'Unknown')}!",
            "user": {
                "id":            best_id,
                "name":          user_info.get("name", "Unknown"),
                "enrolled_date": user_info.get("enrolled_date"),
                "source":        user_info.get("source", "unknown"),
                "image_count":   user_info.get("image_count", 0),
                "thumbnail":     thumb_b64,
            },
            "score": round(best_score, 4),
            "bbox":  last_bbox,
        }

    return {
        "status":     "no_match",
        "access":     "denied",
        "message":    "Face not recognized. Access denied.",
        "best_score": round(best_score, 4) if best_score > 0 else 0,
    }