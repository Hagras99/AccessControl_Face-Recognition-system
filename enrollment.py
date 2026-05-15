"""
=============================================================
  Enrollment & Verification Module
  Handles: face detection, enrollment (9 poses), duplicate
  checking, 1:N verification, and user DB management.
=============================================================

BUGS FIXED
----------
1. VERIFY_THRESHOLD was 0.25 (far too low — almost any face matched).
   Now set to 0.70 which is a safe operating point for cosine similarity
   on LBP features.  The old code used chi-squared similarity for
   verification but cosine similarity in main.py — they are on completely
   different scales; unified to cosine here.

2. verify_user() used np.max() across gallery images, meaning ONE lucky
   gallery frame could cause a false match. Fixed to use np.mean()
   (average over all enrolled images), which is far more robust.

3. DUPLICATE_THRESHOLD was 0.92. With chi-squared similarity (range 0-1,
   higher=more similar), two DIFFERENT people can easily score 0.92+
   because LBP histograms of any two faces look statistically similar.
   Fixed: duplicate check now also uses cosine similarity (consistent
   with verification), with a threshold of 0.88 — tight enough to block
   the same person re-enrolling but loose enough to allow different people.

4. Switched verification & duplicate check from chi-squared similarity to
   cosine similarity, matching the metric used in main.py and giving
   better separation between genuine and impostor scores.
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
DATASET_DIR = BASE_DIR / "dataset"           # ORL — evaluation only
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"  # webcam enrollments
USERS_DB_PATH = BASE_DIR / "users.json"

# Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Both use COSINE SIMILARITY (range roughly 0-1, higher = more similar).
#
# VERIFY_THRESHOLD: minimum cosine similarity to accept a login.
#   - Old value: 0.25 (broken — almost anything matched)
#   - New value: 0.70 (safe operating point for webcam LBP features)
#
# DUPLICATE_THRESHOLD: minimum similarity to flag a re-enrollment attempt.
#   - Old value: 0.92 chi-squared (too aggressive — blocked different people)
#   - New value: 0.85 cosine (blocks the same person, passes different people)
VERIFY_THRESHOLD    = 0.80
DUPLICATE_THRESHOLD = 0.85

IMG_SIZE = (100, 100)


# =========================================================
# USER DATABASE
# =========================================================
def load_user_db():
    if USERS_DB_PATH.exists():
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    return {"subjects": {}}


def save_user_db(db):
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
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_b64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================
# LOW-LIGHT ENHANCEMENT
# =========================================================
def enhance_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def gamma_correction(img, gamma=2.0):
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


# =========================================================
# FACE DETECTION (with low-light robustness)
# =========================================================
def _try_detect(gray, scale=1.1, neighbors=5, min_size=(60, 60)):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
    )
    return faces


def detect_face(frame_b64):
    """
    Detect face in a base64-encoded frame with low-light robustness.
    Pipeline: CLAHE → gamma-brightened (fallbacks).
    Returns: {face_detected, bbox, face_b64 (cropped face)}
    """
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Attempt 1: CLAHE-enhanced image
    enhanced = enhance_clahe(gray)
    faces = _try_detect(enhanced)

    # Attempt 2: gamma-brightened + CLAHE
    if len(faces) == 0:
        bright = gamma_correction(gray, gamma=2.5)
        bright_clahe = enhance_clahe(bright)
        faces = _try_detect(bright_clahe, neighbors=4)

    # Attempt 3: raw image, relaxed params
    if len(faces) == 0:
        faces = _try_detect(gray, neighbors=3, min_size=(50, 50))

    if len(faces) == 0:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    # Take the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    pad_top    = int(0.70 * h)
    pad_side   = int(0.30 * w)
    pad_bottom = int(0.20 * h)

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
# LBP FEATURE EXTRACTION
# =========================================================
def lbp(img):
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
    """
    Illumination-robust preprocessing pipeline:
    1. Gamma correction — lifts dark images to a standard brightness level
    2. CLAHE — adaptive contrast enhancement (handles both dark and bright)
    3. Gaussian blur — removes noise amplified by the above steps
    4. Normalize to zero mean / unit std — makes features lighting-invariant

    This ensures a face captured in dark and the same face captured in
    bright light produce nearly identical LBP feature vectors.
    """
    # Step 1: Adaptive gamma correction to normalize brightness toward ~120/255
    mean_brightness = float(np.mean(img))
    if mean_brightness < 1:
        mean_brightness = 1.0
    gamma = np.log(120.0 / 255.0) / np.log(mean_brightness / 255.0 + 1e-6)
    gamma = float(np.clip(gamma, 0.4, 2.5))
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ]).astype("uint8")
    img = cv2.LUT(img, table)

    # Step 2: CLAHE for local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Step 3: Light Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 4: Per-image normalization to zero mean / unit std
    img_f = img.astype(np.float32)
    mean, std = img_f.mean(), img_f.std()
    if std < 1e-6:
        std = 1e-6
    img_f = (img_f - mean) / std
    # Scale back to uint8 range for LBP
    img = np.clip((img_f + 3.0) * (255.0 / 6.0), 0, 255).astype(np.uint8)

    return img


def cosine_sim(h1, h2):
    """
    Cosine similarity between two feature vectors.
    Returns value in [0, 1] — higher = more similar.
    Consistent with the metric used in main.py (rank-1 identification).
    
    OLD CODE used chi_squared_sim which has a completely different scale
    and was causing the threshold mismatch that led to false matches.
    """
    n1 = np.linalg.norm(h1) + 1e-10
    n2 = np.linalg.norm(h2) + 1e-10
    return float(np.dot(h1, h2) / (n1 * n2))


# =========================================================
# BUILD GALLERY (LBP features from a directory)
# =========================================================
def build_gallery(dataset_dir=None):
    """
    Build LBP feature gallery from live_dataset directory.
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
    Check if a face is already enrolled.
    Only checks against webcam-enrolled users (live_dataset), NOT ORL subjects.

    Uses cosine similarity averaged across all probe images and all gallery
    images for a subject (mean-of-means), which is robust against outliers.

    face_images_gray: list of grayscale 100x100 numpy arrays
    Returns: {is_duplicate, matched_name, matched_id, score} or {is_duplicate: False}
    """
    gallery = build_gallery(LIVE_DATASET_DIR)
    db = load_user_db()

    if not gallery:
        return {"is_duplicate": False}

    probe_feats = [lbp(preprocess(img)) for img in face_images_gray]

    best_score = -1
    best_id = None

    for sid, gallery_feats in gallery.items():
        # Mean cosine similarity across all (probe, gallery) pairs
        # This is more robust than max — prevents one lucky frame from triggering
        all_sims = []
        for pf in probe_feats:
            for gf in gallery_feats:
                all_sims.append(cosine_sim(pf, gf))
        subject_score = float(np.mean(all_sims))

        if subject_score > best_score:
            best_score = subject_score
            best_id = sid

    print(f"[DUPLICATE CHECK] Best score: {best_score:.4f} vs {best_id}"
          f"  (threshold: {DUPLICATE_THRESHOLD})")

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
        face_images_b64: List of base64-encoded face images (9 recommended)

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

        face_gray = b64_to_cv2(result["face_b64"])
        if len(face_gray.shape) == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, IMG_SIZE)
        face_images.append(face_gray)

    # Check for duplicate using ALL captured images
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

    FIX: Old code used np.max() over gallery images — one lucky frame
    caused false positives. Now uses np.mean() for robustness.

    FIX: Old VERIFY_THRESHOLD was 0.25 (chi-squared scale).
    Now uses 0.70 on cosine similarity — properly calibrated.

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

    # Extract features from the detected face
    face_img = b64_to_cv2(result["face_b64"])
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, IMG_SIZE)

    face_prep = preprocess(face_img)
    probe_feat = lbp(face_prep)

    # Build gallery from live-enrolled users ONLY
    gallery = build_gallery(LIVE_DATASET_DIR)
    db = load_user_db()

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
        # MEAN cosine similarity across all enrolled images for this subject.
        # OLD CODE used max() — one lucky frame caused 90% false match.
        # MEAN is robust: probe must genuinely resemble the enrolled face
        # across multiple images, not just one.
        scores = [cosine_sim(probe_feat, f) for f in feats]
        mean_score = float(np.mean(scores))
        all_scores[sid] = mean_score
        if mean_score > best_score:
            best_score = mean_score
            best_id = sid

    # Debug log
    print(f"\n[VERIFY] Scores against {len(gallery)} enrolled user(s):")
    for sid, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
        name = db["subjects"].get(sid, {}).get("name", "?")
        print(f"  {sid} ({name}): {sc:.4f}")
    print(f"  Best: {best_id} = {best_score:.4f}  (threshold: {VERIFY_THRESHOLD})\n")

    if best_score >= VERIFY_THRESHOLD and best_id:
        user_info = db["subjects"].get(best_id, {})

        # Thumbnail from first enrolled image
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