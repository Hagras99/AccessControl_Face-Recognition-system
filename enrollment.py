"""
=============================================================
  Enrollment & Verification Module
  Handles: face detection, enrollment (9 poses), duplicate
  checking, 1:N verification, and user DB management.
=============================================================
"""

import cv2
import numpy as np
import json
import base64
import os
from pathlib import Path
from datetime import datetime


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"          # ORL — evaluation only
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"  # webcam enrollments
USERS_DB_PATH = BASE_DIR / "users.json"

# Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Matching thresholds (chi-squared similarity — 0 to 1, higher = more similar)
DUPLICATE_THRESHOLD = 0.92
VERIFY_THRESHOLD = 0.25

IMG_SIZE = (100, 100)


# =========================================================
# USER DATABASE
# =========================================================
def load_user_db():
    """Load user database from JSON file."""
    if USERS_DB_PATH.exists():
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    return {"subjects": {}}


def save_user_db(db):
    """Save user database to JSON file."""
    with open(USERS_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def init_user_db():
    """Initialize user DB with ORL subjects if not exists."""
    if USERS_DB_PATH.exists():
        return load_user_db()

    db = {"subjects": {}}
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if person_dir.is_dir() and person_dir.name.startswith("s"):
            subject_id = person_dir.name
            img_count = len(list(person_dir.glob("*.pgm")))
            db["subjects"][subject_id] = {
                "name": f"Subject {subject_id[1:]}",
                "enrolled_date": None,
                "source": "ORL",
                "image_count": img_count,
            }
    save_user_db(db)
    return db


def get_all_users():
    """Return list of all enrolled users."""
    db = load_user_db()
    users = []
    for sid, info in db["subjects"].items():
        users.append({
            "id": sid,
            "name": info["name"],
            "enrolled_date": info.get("enrolled_date"),
            "source": info.get("source", "unknown"),
            "image_count": info.get("image_count", 0),
        })
    return users


def get_next_subject_id():
    """Return the next available subject ID in live_dataset (e.g., 'u1', 'u2')."""
    LIVE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for d in LIVE_DATASET_DIR.iterdir():
        if d.is_dir() and d.name.startswith("u"):
            try:
                existing_ids.append(int(d.name[1:]))
            except ValueError:
                pass
    next_id = max(existing_ids) + 1 if existing_ids else 1
    return f"u{next_id}"


# =========================================================
# IMAGE UTILITIES
# =========================================================
def b64_to_cv2(b64_string):
    """Convert base64 image string to OpenCV image (BGR)."""
    # Strip data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_b64(img):
    """Convert OpenCV image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


# =========================================================
# LOW-LIGHT ENHANCEMENT
# =========================================================
def enhance_clahe(gray):
    """Apply CLAHE for adaptive contrast enhancement (works great in low light)."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def gamma_correction(img, gamma=2.0):
    """Brighten a dark image using gamma correction."""
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


# =========================================================
# FACE DETECTION (with low-light robustness)
# =========================================================
def _try_detect(gray, scale=1.1, neighbors=5, min_size=(60, 60)):
    """Run Haar cascade on a grayscale image."""
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
    )
    return faces


def detect_face(frame_b64):
    """
    Detect face in a base64-encoded frame with low-light robustness.
    Pipeline: raw → CLAHE → gamma-brightened (fallbacks).
    Returns: {face_detected, bbox, face_b64 (cropped face)}
    """
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Attempt 1: CLAHE-enhanced image (handles most low-light)
    enhanced = enhance_clahe(gray)
    faces = _try_detect(enhanced)

    # Attempt 2: gamma-brightened + CLAHE (very dark scenes)
    if len(faces) == 0:
        bright = gamma_correction(gray, gamma=2.5)
        bright_clahe = enhance_clahe(bright)
        faces = _try_detect(bright_clahe, neighbors=4)

    # Attempt 3: raw image, relaxed params (fallback)
    if len(faces) == 0:
        faces = _try_detect(gray, neighbors=3, min_size=(50, 50))

    if len(faces) == 0:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    # Take the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Generous crop to include hair, forehead, ears, chin
    # Haar cascade box goes roughly eyebrows→chin, so we need
    # extra padding on top (hair) and moderate padding elsewhere.
    pad_top    = int(0.70 * h)   # lots of room for hair
    pad_side   = int(0.30 * w)   # ears + some background
    pad_bottom = int(0.20 * h)   # chin/neck

    x1 = max(0, x - pad_side)
    y1 = max(0, y - pad_top)
    x2 = min(gray.shape[1], x + w + pad_side)
    y2 = min(gray.shape[0], y + h + pad_bottom)

    face_crop = gray[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, IMG_SIZE)

    return {
        "face_detected": True,
        "bbox": [int(x), int(y), int(w), int(h)],
        "face_b64": cv2_to_b64(face_resized),
    }


# =========================================================
# LBP FEATURE EXTRACTION (same as main.py)
# =========================================================
def lbp(img):
    """Compute LBP histogram feature vector."""
    c = img[1:-1, 1:-1]
    code = (
        ((img[0:-2, 0:-2] > c) << 7) |
        ((img[0:-2, 1:-1] > c) << 6) |
        ((img[0:-2, 2:]   > c) << 5) |
        ((img[1:-1, 2:]   > c) << 4) |
        ((img[2:,   2:]   > c) << 3) |
        ((img[2:,   1:-1] > c) << 2) |
        ((img[2:,   0:-2] > c) << 1) |
        ((img[1:-1, 0:-2] > c))
    )
    hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))
    return hist / (np.sum(hist) + 1e-10)


def preprocess(img):
    """CLAHE-based preprocessing (better than plain equalizeHist for webcam)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def chi_squared_sim(h1, h2):
    """
    Chi-squared similarity between two histograms.
    Standard metric for LBP (Ojala et al. 2002).
    Returns value in [0, 1] — higher = more similar.
    """
    dist = 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-10))
    return float(1.0 / (1.0 + dist))


# =========================================================
# BUILD GALLERY (LBP features from a specific directory)
# =========================================================
def build_gallery(dataset_dir=None):
    """
    Build LBP feature gallery from a dataset directory.
    
    Args:
        dataset_dir: Path to scan. Defaults to LIVE_DATASET_DIR.
    Returns: {subject_id: [lbp_vectors], ...}
    """
    target = dataset_dir or LIVE_DATASET_DIR
    if not target.exists():
        return {}
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
    return gallery


# =========================================================
# DUPLICATE CHECK
# =========================================================
def check_duplicate(face_images_gray):
    """
    Check if a face is already enrolled using multiple probe images.
    ONLY checks against webcam-enrolled users, NOT ORL dataset subjects.
    face_images_gray: list of grayscale 100x100 numpy arrays
    Returns: {is_duplicate, matched_name, matched_id, score} or {is_duplicate: False}
    """
    # Only check against live-enrolled users
    gallery = build_gallery(LIVE_DATASET_DIR)
    db = load_user_db()

    # No webcam users enrolled yet — can't be a duplicate
    if not gallery:
        return {"is_duplicate": False}

    probe_feats = [lbp(preprocess(img)) for img in face_images_gray]

    best_score = -1
    best_id = None

    for sid, feats in gallery.items():
        per_probe_max = []
        for pf in probe_feats:
            scores = [chi_squared_sim(pf, gf) for gf in feats]
            per_probe_max.append(max(scores))
        subject_score = float(np.median(per_probe_max))
        if subject_score > best_score:
            best_score = subject_score
            best_id = sid

    if best_score >= DUPLICATE_THRESHOLD and best_id:
        user_info = db["subjects"].get(best_id, {})
        return {
            "is_duplicate": True,
            "matched_name": user_info.get("name", "Unknown"),
            "matched_id": best_id,
            "score": round(best_score, 4),
        }

    return {"is_duplicate": False}


# =========================================================
# ENROLLMENT
# =========================================================
def enroll_user(name, face_images_b64):
    """
    Enroll a new user with 9 face poses.
    
    Args:
        name: User's name
        face_images_b64: List of 9 base64-encoded face images
    
    Returns:
        {status: "success"/"duplicate"/"error", ...}
    """
    if not name or not name.strip():
        return {"status": "error", "message": "Name cannot be empty"}

    if len(face_images_b64) < 1:
        return {"status": "error", "message": "At least 1 face image required"}

    # Decode and detect faces
    face_images = []
    for i, b64 in enumerate(face_images_b64):
        result = detect_face(b64)
        if not result["face_detected"]:
            return {"status": "error", "message": f"No face detected in pose {i + 1}"}
        
        # Decode the cropped face
        face_gray = b64_to_cv2(result["face_b64"])
        if len(face_gray.shape) == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, IMG_SIZE)
        face_images.append(face_gray)

    # Check for duplicate using ALL captured images (more robust)
    dup_check = check_duplicate(face_images)
    if dup_check["is_duplicate"]:
        return {
            "status": "duplicate",
            "message": f"This face is already registered as '{dup_check['matched_name']}'",
            "matched_name": dup_check["matched_name"],
            "matched_id": dup_check["matched_id"],
            "score": dup_check["score"],
        }

    # Create new subject directory in live_dataset
    subject_id = get_next_subject_id()
    subject_dir = LIVE_DATASET_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Save images as PGM
    for i, face_img in enumerate(face_images):
        img_path = subject_dir / f"{i + 1}.pgm"
        cv2.imwrite(str(img_path), face_img)

    # Update user database
    db = load_user_db()
    db["subjects"][subject_id] = {
        "name": name.strip(),
        "enrolled_date": datetime.now().isoformat(),
        "source": "webcam",
        "image_count": len(face_images),
    }
    save_user_db(db)

    return {
        "status": "success",
        "message": f"Successfully enrolled '{name.strip()}' as {subject_id}",
        "subject_id": subject_id,
        "image_count": len(face_images),
    }


# =========================================================
# VERIFICATION (1:N Identification)
# =========================================================
def verify_user(face_b64):
    """
    Verify a face against all enrolled subjects (1:N identification).
    
    Args:
        face_b64: base64-encoded webcam frame
    
    Returns:
        {status, access, user, score, ...}
    """
    # Detect face
    result = detect_face(face_b64)
    if not result["face_detected"]:
        return {
            "status": "no_face",
            "access": "denied",
            "message": "No face detected in the image",
        }

    # Get cropped face
    face_img = b64_to_cv2(result["face_b64"])
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, IMG_SIZE)

    # Extract features
    face_prep = preprocess(face_img)
    probe_feat = lbp(face_prep)

    # Build gallery ONLY from live-enrolled users
    gallery = build_gallery(LIVE_DATASET_DIR)
    db = load_user_db()

    # No webcam users enrolled — can't identify anyone
    if not gallery:
        return {
            "status": "no_match",
            "access": "denied",
            "message": "No users enrolled yet. Please enroll first.",
            "best_score": 0,
        }

    best_score = -1
    best_id = None

    all_scores = {}
    for sid, feats in gallery.items():
        scores = [chi_squared_sim(probe_feat, f) for f in feats]
        # Use max score (best matching image) instead of average
        max_score = float(np.max(scores))
        all_scores[sid] = max_score
        if max_score > best_score:
            best_score = max_score
            best_id = sid

    # Debug: print all scores to help diagnose matching
    print(f"\n[VERIFY] Scores against {len(gallery)} enrolled user(s):")
    for sid, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
        name = db['subjects'].get(sid, {}).get('name', '?')
        print(f"  {sid} ({name}): {sc:.4f}")
    print(f"  Best: {best_id} = {best_score:.4f}  (threshold: {VERIFY_THRESHOLD})\n")

    if best_score >= VERIFY_THRESHOLD and best_id:
        user_info = db["subjects"].get(best_id, {})
        
        # Get first enrolled image as thumbnail
        subject_dir = LIVE_DATASET_DIR / best_id
        thumb_b64 = None
        first_img = sorted(subject_dir.glob("*.pgm"))
        if first_img:
            thumb = cv2.imread(str(first_img[0]), cv2.IMREAD_GRAYSCALE)
            if thumb is not None:
                thumb_b64 = cv2_to_b64(thumb)

        return {
            "status": "matched",
            "access": "granted",
            "message": f"Welcome, {user_info.get('name', 'Unknown')}!",
            "user": {
                "id": best_id,
                "name": user_info.get("name", "Unknown"),
                "enrolled_date": user_info.get("enrolled_date"),
                "source": user_info.get("source", "unknown"),
                "image_count": user_info.get("image_count", 0),
                "thumbnail": thumb_b64,
            },
            "score": round(best_score, 4),
            "bbox": result["bbox"],
        }
    else:
        return {
            "status": "no_match",
            "access": "denied",
            "message": "Face not recognized. Access denied.",
            "best_score": round(best_score, 4) if best_score > 0 else 0,
        }
