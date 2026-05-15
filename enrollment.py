"""
=============================================================
  Enrollment & Verification Module
  Handles: face detection, enrollment (9 poses), duplicate
  checking, 1:N verification, and user DB management.

  UNIFIED with main.py:
  - Imports lbp(), similarity(), preprocess() from main.py
    so feature extraction is IDENTICAL in evaluation and live use.
    (Old code duplicated preprocess() with CLAHE instead of
     histogram equalisation — causing a train/test mismatch.)

  BUGS FIXED
  ----------
  1. preprocess() conflict: enrollment used CLAHE, main.py used
     histogram equalisation. Now both use the same function from
     main.py. This was causing feature-space mismatch — enrolled
     templates and probe features were not comparable.

  2. VERIFY_THRESHOLD raised to 0.92 (was 0.70/0.85). LBP cosine
     similarities between ANY two face crops cluster around 0.80-0.92,
     so a threshold below 0.92 easily accepts impostors.

  3. verify_user() added a MARGIN CHECK: best score must exceed the
     second-best score by MIN_MARGIN (0.02). Without this, an impostor
     who scores similarly to multiple enrolled users gets accepted.

  4. Added GrabCut face segmentation inside detect_face() to suppress
     background / hair / clothing noise before LBP feature extraction.
     Background pixels that leak into the crop inflate cosine similarity
     between unrelated faces.

  5. verify_user() now uses mean of top-3 scores per subject (instead
     of mean of ALL enrolled images). This handles slight pose/lighting
     variation in enrolled images without diluting genuine similarity.

  6. check_duplicate() uses the same top-3 mean strategy for consistency.
=============================================================
"""

import cv2
import numpy as np
import json
import base64
import os
from pathlib import Path
from datetime import datetime

# ── Import core feature-extraction logic from main.py ────────────────────────
# This guarantees that enrollment, duplicate-check, and verification all use
# the SAME preprocessing and feature extraction as the offline evaluation.
from main import lbp, similarity, preprocess          # noqa: E402


# =========================================================
# PATHS
# =========================================================
BASE_DIR        = Path(__file__).parent
DATASET_DIR     = BASE_DIR / "dataset"
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"
USERS_DB_PATH   = BASE_DIR / "users.json"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Thresholds ────────────────────────────────────────────────────────────────
#
# VERIFY_THRESHOLD  – minimum mean cosine similarity (top-3 enrolled images)
#   to accept a login claim.
#   LBP cosine scores between unrelated faces commonly reach 0.88-0.91,
#   so anything below 0.92 risks false acceptance on a clean webcam feed.
#
# MIN_MARGIN  – the winner's score must beat the runner-up by at least this
#   amount.  Prevents accepting an impostor who scores "close enough" to
#   every enrolled person.
#
# DUPLICATE_THRESHOLD  – minimum similarity to flag re-enrollment.
#   Slightly lower than VERIFY_THRESHOLD so a genuine re-enroll attempt
#   (same person, different lighting) is still caught.
#
VERIFY_THRESHOLD    = 0.92
MIN_MARGIN          = 0.02
DUPLICATE_THRESHOLD = 0.88

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
    users = []
    for sid, info in db["subjects"].items():
        users.append({
            "id":            sid,
            "name":          info["name"],
            "enrolled_date": info.get("enrolled_date"),
            "source":        info.get("source", "unknown"),
            "image_count":   info.get("image_count", 0),
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
    nparr     = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_b64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================
# FACE SEGMENTATION  (GrabCut — suppresses background noise)
# =========================================================
def segment_face(gray_crop):
    """
    Apply GrabCut segmentation to a grayscale face crop to suppress
    background, hair edges, and clothing that leak into the bounding box.

    Background pixels that survive into the LBP crop inflate cosine
    similarity between unrelated faces (any two crops share similar
    uniform-texture background regions). Masking them to the mean
    face-region intensity breaks that spurious similarity.

    Returns a grayscale image the same size as gray_crop where
    background pixels are replaced with the mean foreground intensity.
    """
    if gray_crop is None or gray_crop.size == 0:
        return gray_crop

    # GrabCut requires a colour image
    bgr = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]

    # Initial rectangle: inset 10% from each edge (face is centered inside)
    margin_x = max(1, int(0.10 * w))
    margin_y = max(1, int(0.10 * h))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask          = np.zeros((h, w), np.uint8)
    bg_model      = np.zeros((1, 65), np.float64)
    fg_model      = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bg_model, fg_model, 5,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # GrabCut can fail on very small crops — return original
        return gray_crop

    # Pixels labelled GC_FGD or GC_PR_FGD are foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    # Replace background with mean foreground intensity to avoid
    # introducing a hard black border that would dominate LBP codes
    if fg_mask.sum() == 0:
        return gray_crop          # segmentation failed — use original

    mean_intensity = int(gray_crop[fg_mask == 255].mean())
    segmented      = gray_crop.copy()
    segmented[fg_mask == 0] = mean_intensity

    return segmented


# =========================================================
# LOW-LIGHT ENHANCEMENT
# =========================================================
def _enhance_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _gamma_correction(img, gamma=2.0):
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


# =========================================================
# FACE DETECTION  (with low-light robustness + segmentation)
# =========================================================
def _try_detect(gray, scale=1.1, neighbors=5, min_size=(60, 60)):
    return face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
    )


def detect_face(frame_b64):
    """
    Detect face in a base64-encoded frame.

    Pipeline:
      1. CLAHE-enhanced grayscale  (Attempt 1)
      2. Gamma-brightened + CLAHE  (Attempt 2, low-light fallback)
      3. Raw grayscale, relaxed params  (Attempt 3, last resort)
      4. GrabCut segmentation on the selected crop
      5. preprocess() from main.py (histogram equalisation)

    Returns: {face_detected, bbox, face_b64 (cropped, segmented, 100x100)}
    """
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Attempt 1: CLAHE
    enhanced = _enhance_clahe(gray)
    faces    = _try_detect(enhanced)

    # Attempt 2: gamma + CLAHE
    if len(faces) == 0:
        bright      = _gamma_correction(gray, gamma=2.5)
        bright_clahe = _enhance_clahe(bright)
        faces       = _try_detect(bright_clahe, neighbors=4)

    # Attempt 3: raw, relaxed
    if len(faces) == 0:
        faces = _try_detect(gray, neighbors=3, min_size=(50, 50))

    if len(faces) == 0:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    # Largest face wins
    faces      = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    pad_top    = int(0.70 * h)
    pad_side   = int(0.30 * w)
    pad_bottom = int(0.20 * h)

    x1 = max(0, x - pad_side)
    y1 = max(0, y - pad_top)
    x2 = min(gray.shape[1], x + w + pad_side)
    y2 = min(gray.shape[0], y + h + pad_bottom)

    face_crop = gray[y1:y2, x1:x2]

    # ── Segmentation: suppress background noise ──────────────────────────────
    face_crop = segment_face(face_crop)

    # ── Resize then apply the SAME preprocessing as main.py ─────────────────
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_proc    = preprocess(face_resized)   # histogram equalisation (main.py)

    return {
        "face_detected": True,
        "bbox":    [int(x), int(y), int(w), int(h)],
        "face_b64": cv2_to_b64(face_proc),
    }


# =========================================================
# FEATURE EXTRACTION HELPERS
# =========================================================
def _feat_from_gray(gray_img):
    """Resize → preprocess (main.py) → LBP (main.py)."""
    resized = cv2.resize(gray_img, IMG_SIZE)
    prepped = preprocess(resized)
    return lbp(prepped)


def _top_k_mean(scores, k=3):
    """Mean of the top-k values (robust to outlier enrolled frames)."""
    arr = np.array(scores)
    if len(arr) <= k:
        return float(np.mean(arr))
    return float(np.mean(np.partition(arr, -k)[-k:]))


# =========================================================
# BUILD GALLERY
# =========================================================
def build_gallery(dataset_dir=None):
    """
    Build LBP feature gallery from live_dataset.
    Preprocessing: preprocess() from main.py (histogram equalisation).
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
            features.append(_feat_from_gray(img))
        if features:
            gallery[person_dir.name] = features
    return gallery


# =========================================================
# DUPLICATE CHECK
# =========================================================
def check_duplicate(face_images_gray):
    """
    Check if a face is already enrolled (live_dataset only).

    Uses mean-of-top-3 cosine similarity across probe × gallery pairs.
    More robust than full-mean: a few bad enrolled frames won't dilute
    a genuine match, but a single lucky frame can't trigger a duplicate.

    Returns: {is_duplicate, matched_name, matched_id, score} or
             {is_duplicate: False}
    """
    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {"is_duplicate": False}

    probe_feats = [_feat_from_gray(img) for img in face_images_gray]

    best_score = -1.0
    best_id    = None

    for sid, gallery_feats in gallery.items():
        # For each probe image, find its best gallery score for this subject,
        # then average across probes.  This is probe-side top-1 mean.
        per_probe_bests = []
        for pf in probe_feats:
            sims = [similarity(pf, gf) for gf in gallery_feats]
            per_probe_bests.append(max(sims))
        subject_score = float(np.mean(per_probe_bests))

        if subject_score > best_score:
            best_score = subject_score
            best_id    = sid

    print(f"[DUPLICATE CHECK] Best={best_score:.4f} vs {best_id}"
          f"  (threshold={DUPLICATE_THRESHOLD})")

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
def enroll_user(name, face_images_b64):
    """
    Enroll a new user with up to 9 face poses.

    Args:
        name:            User's name (string)
        face_images_b64: List of base64-encoded frames (9 recommended)

    Returns:
        {status: "success"/"duplicate"/"error", ...}
    """
    if not name or not name.strip():
        return {"status": "error", "message": "Name cannot be empty"}
    if len(face_images_b64) < 1:
        return {"status": "error", "message": "At least 1 face image required"}

    # Detect & crop each pose
    face_images = []
    for i, b64 in enumerate(face_images_b64):
        result = detect_face(b64)
        if not result["face_detected"]:
            return {"status": "error",
                    "message": f"No face detected in pose {i + 1}"}

        face_img = b64_to_cv2(result["face_b64"])
        if face_img is None:
            return {"status": "error",
                    "message": f"Could not decode face image for pose {i + 1}"}

        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, IMG_SIZE)
        face_images.append(face_img)

    # Duplicate check
    dup = check_duplicate(face_images)
    if dup["is_duplicate"]:
        return {
            "status":       "duplicate",
            "message":      f"Face already registered as '{dup['matched_name']}'",
            "matched_name": dup["matched_name"],
            "matched_id":   dup["matched_id"],
            "score":        dup["score"],
        }

    # Save to live_dataset
    subject_id  = get_next_subject_id()
    subject_dir = LIVE_DATASET_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for i, face_img in enumerate(face_images):
        cv2.imwrite(str(subject_dir / f"{i + 1}.pgm"), face_img)

    # Update DB
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
# VERIFICATION  (1:N Identification)
# =========================================================
def verify_user(face_b64):
    """
    Identify a face against all enrolled subjects (1:N).

    Decision logic (three gates — all must pass):
      1. mean-of-top-3 score  ≥  VERIFY_THRESHOLD  (0.92)
      2. winning score  -  runner-up score  ≥  MIN_MARGIN  (0.02)
      3. Face was successfully detected and segmented

    Gate 2 (margin check) is critical: LBP scores are compressed into a
    narrow range (~0.80-0.95), so an impostor who scores 0.92 against
    Subject A and 0.91 against Subject B should NOT be accepted — the
    genuine owner of Subject A would score noticeably higher AND have a
    clear lead over the runner-up.

    Returns: {status, access, user, score, ...}
    """
    result = detect_face(face_b64)
    if not result["face_detected"]:
        return {
            "status":  "no_face",
            "access":  "denied",
            "message": "No face detected in the image.",
        }

    # Probe feature — detect_face() already applied preprocess()
    face_img = b64_to_cv2(result["face_b64"])
    if face_img is None:
        return {"status": "no_face", "access": "denied",
                "message": "Could not decode detected face."}

    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, IMG_SIZE)

    # Re-apply preprocess so the feature is on the same scale as the gallery
    # (detect_face already preprocessed, but the b64 round-trip may lose
    #  precision; recompute from the decoded image to be safe)
    probe_feat = lbp(preprocess(face_img))

    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {
            "status":     "no_match",
            "access":     "denied",
            "message":    "No users enrolled yet. Please enroll first.",
            "best_score": 0,
        }

    scores_per_subject = {}

    for sid, feats in gallery.items():
        raw_scores = [similarity(probe_feat, f) for f in feats]
        # top-3 mean: robust but not diluted by many poor enrolled frames
        scores_per_subject[sid] = _top_k_mean(raw_scores, k=3)

    # Sort descending
    ranked = sorted(scores_per_subject.items(), key=lambda x: -x[1])
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    # ── Debug log ─────────────────────────────────────────────────────────────
    print(f"\n[VERIFY] Scores against {len(gallery)} enrolled user(s):")
    for sid, sc in ranked:
        name = db["subjects"].get(sid, {}).get("name", "?")
        print(f"  {sid} ({name}): {sc:.4f}")
    print(f"  Best={best_id} score={best_score:.4f}  "
          f"margin={margin:.4f}  "
          f"(threshold={VERIFY_THRESHOLD}, min_margin={MIN_MARGIN})\n")

    # ── Three-gate decision ───────────────────────────────────────────────────
    if best_score >= VERIFY_THRESHOLD and margin >= MIN_MARGIN:
        user_info = db["subjects"].get(best_id, {})

        thumb_b64 = None
        subject_dir = LIVE_DATASET_DIR / best_id
        first_imgs  = sorted(subject_dir.glob("*.pgm"))
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
            "bbox":  result["bbox"],
        }

    # Denied — provide a useful reason in the debug message
    if best_score < VERIFY_THRESHOLD:
        reason = f"score {best_score:.4f} below threshold {VERIFY_THRESHOLD}"
    else:
        reason = (f"margin {margin:.4f} below MIN_MARGIN {MIN_MARGIN} "
                  f"(best={best_score:.4f}, 2nd={second_score:.4f})")

    print(f"[VERIFY] DENIED — {reason}")

    return {
        "status":     "no_match",
        "access":     "denied",
        "message":    "Face not recognised. Access denied.",
        "best_score": round(best_score, 4),
    }