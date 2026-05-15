"""
=============================================================
  Enrollment & Verification Module
  Handles: face detection, enrollment (9 poses), duplicate
  checking, 1:N verification, and user DB management.

  Imports lbp(), similarity(), preprocess() from main.py.
  All feature extraction is therefore IDENTICAL between the
  offline evaluation pipeline and live webcam use.

  ROOT CAUSE FIX (in main.py, consumed here)
  -------------------------------------------
  Global LBP histogram → spatial grid LBP (7×7 cells × 256 bins
  = 12 544-dim descriptor). Two different people now score 0.60-0.80
  on cosine similarity; the same person scores 0.85-0.97.
  The distributions are well-separated, making thresholding reliable.

  THRESHOLD CHANGE
  ----------------
  Old global-LBP range: all scores 0.88-0.96 → impossible to separate
  New spatial-LBP range:
    Impostor: ~0.60-0.80
    Genuine:  ~0.85-0.97
  VERIFY_THRESHOLD = 0.82  sits cleanly in the gap.
  DUPLICATE_THRESHOLD = 0.80  (slightly lower to catch re-enrolls).
  MIN_MARGIN = 0.04  (winner must beat runner-up; guards against
                      cases where an impostor is close to two subjects).
=============================================================
"""

import cv2
import numpy as np
import json
import base64
from pathlib import Path
from datetime import datetime

# ── All feature-extraction logic comes from main.py ──────────────────────────
from main import lbp, similarity, preprocess, IMG_SIZE


# =========================================================
# PATHS
# =========================================================
BASE_DIR         = Path(__file__).parent
DATASET_DIR      = BASE_DIR / "dataset"
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"
USERS_DB_PATH    = BASE_DIR / "users.json"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Calibrated for spatial 7x7 LBP cosine similarity:
#   Impostor scores cluster ~0.60-0.80
#   Genuine  scores cluster ~0.85-0.97
# The gap between 0.80 and 0.85 is where the threshold lives.
VERIFY_THRESHOLD    = 0.82   # minimum to accept a login
MIN_MARGIN          = 0.04   # winner must beat runner-up by this much
DUPLICATE_THRESHOLD = 0.80   # minimum to flag re-enrollment


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
    if USERS_DB_PATH.exists():
        return load_user_db()
    db = {"subjects": {}}
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if person_dir.is_dir() and person_dir.name.startswith("s"):
            sid = person_dir.name
            db["subjects"][sid] = {
                "name":          f"Subject {sid[1:]}",
                "enrolled_date": None,
                "source":        "ORL",
                "image_count":   len(list(person_dir.glob("*.pgm"))),
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
    existing = []
    for d in LIVE_DATASET_DIR.iterdir():
        if d.is_dir() and d.name.startswith("u"):
            try:
                existing.append(int(d.name[1:]))
            except ValueError:
                pass
    return f"u{max(existing) + 1 if existing else 1}"


# =========================================================
# IMAGE UTILITIES
# =========================================================
def b64_to_cv2(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    nparr = np.frombuffer(base64.b64decode(b64_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


# =========================================================
# FACE SEGMENTATION  (GrabCut)
# =========================================================
def segment_face(gray_crop):
    """
    Suppress background / hair / clothing noise via GrabCut.

    Without segmentation, background pixels that leak into the bounding
    box produce similar uniform-region LBP codes for EVERY person, which
    artificially inflates inter-person similarity.  Replacing background
    pixels with the mean foreground intensity removes that shared noise.
    """
    if gray_crop is None or gray_crop.size == 0:
        return gray_crop

    bgr  = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]

    mx = max(1, int(0.10 * w))
    my = max(1, int(0.10 * h))
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask     = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bg_model, fg_model, 5,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return gray_crop

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    if fg_mask.sum() == 0:
        return gray_crop

    mean_intensity          = int(gray_crop[fg_mask == 255].mean())
    segmented               = gray_crop.copy()
    segmented[fg_mask == 0] = mean_intensity
    return segmented


# =========================================================
# LOW-LIGHT HELPERS
# =========================================================
def _clahe(gray):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def _gamma(img, gamma=2.5):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


# =========================================================
# FACE DETECTION
# =========================================================
def _try_detect(gray, scale=1.1, neighbors=5, min_size=(60, 60)):
    return face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
    )


def detect_face(frame_b64):
    """
    Detect, segment, and preprocess a face from a base64 frame.

    Pipeline:
      CLAHE detect → gamma+CLAHE detect → raw detect (fallbacks)
      → GrabCut segmentation → resize to IMG_SIZE
      → preprocess() [histogram equalisation, from main.py]

    The returned face_b64 image is ready for lbp() without further
    preprocessing — identical to how gallery images are built.
    """
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = _try_detect(_clahe(gray))
    if len(faces) == 0:
        faces = _try_detect(_clahe(_gamma(gray)), neighbors=4)
    if len(faces) == 0:
        faces = _try_detect(gray, neighbors=3, min_size=(50, 50))
    if len(faces) == 0:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    x1 = max(0, x - int(0.30 * w))
    y1 = max(0, y - int(0.70 * h))
    x2 = min(gray.shape[1], x + w + int(0.30 * w))
    y2 = min(gray.shape[0], y + h + int(0.20 * h))

    crop     = gray[y1:y2, x1:x2]
    crop     = segment_face(crop)                    # suppress background
    resized  = cv2.resize(crop, IMG_SIZE)
    prepared = preprocess(resized)                   # histogram equalisation (main.py)

    return {
        "face_detected": True,
        "bbox":    [int(x), int(y), int(w), int(h)],
        "face_b64": cv2_to_b64(prepared),
    }


# =========================================================
# FEATURE HELPER
# =========================================================
def _extract_feat(gray_img):
    """Resize → preprocess → spatial LBP  (all via main.py)."""
    return lbp(preprocess(cv2.resize(gray_img, IMG_SIZE)))


def _top_k_mean(scores_list, k=3):
    """Mean of the top-k scores — robust to a few bad enrolled frames."""
    arr = np.array(scores_list)
    if len(arr) <= k:
        return float(np.mean(arr))
    return float(np.mean(np.partition(arr, -k)[-k:]))


# =========================================================
# BUILD GALLERY
# =========================================================
def build_gallery(dataset_dir=None):
    """
    Load every enrolled .pgm image from live_dataset and compute its
    spatial LBP feature vector (using main.py's lbp + preprocess).
    Returns {subject_id: [feature_vectors]}.
    """
    target = dataset_dir or LIVE_DATASET_DIR
    if not target.exists():
        return {}
    gallery = {}
    for person_dir in sorted(target.iterdir()):
        if not person_dir.is_dir():
            continue
        feats = []
        for img_path in sorted(person_dir.glob("*.pgm")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                feats.append(_extract_feat(img))
        if feats:
            gallery[person_dir.name] = feats
    return gallery


# =========================================================
# DUPLICATE CHECK
# =========================================================
def check_duplicate(face_images_gray):
    """
    Check whether a face being enrolled already exists in live_dataset.

    Strategy: for each probe image, find its maximum similarity against
    any gallery image for a subject (probe-side top-1), then average
    those maxima across all probe images.  This is strict but fair:
    - A genuine re-enroll (same person) will have consistently high
      per-probe maxima -> high mean -> caught.
    - A different person will have low per-probe maxima even if one
      gallery frame happened to score high -> mean stays low -> allowed.
    """
    gallery = build_gallery(LIVE_DATASET_DIR)
    if not gallery:
        return {"is_duplicate": False}

    db          = load_user_db()
    probe_feats = [_extract_feat(img) for img in face_images_gray]

    best_score, best_id = -1.0, None

    for sid, gfeats in gallery.items():
        per_probe = [max(similarity(pf, gf) for gf in gfeats)
                     for pf in probe_feats]
        score = float(np.mean(per_probe))
        if score > best_score:
            best_score, best_id = score, sid

    print(f"[DUPLICATE CHECK] Best={best_score:.4f} vs {best_id}"
          f"  (threshold={DUPLICATE_THRESHOLD})")

    if best_score >= DUPLICATE_THRESHOLD and best_id:
        info = db["subjects"].get(best_id, {})
        return {
            "is_duplicate": True,
            "matched_name": info.get("name", "Unknown"),
            "matched_id":   best_id,
            "score":        round(best_score, 4),
        }
    return {"is_duplicate": False}


# =========================================================
# ENROLLMENT
# =========================================================
def enroll_user(name, face_images_b64):
    if not name or not name.strip():
        return {"status": "error", "message": "Name cannot be empty"}
    if not face_images_b64:
        return {"status": "error", "message": "At least 1 face image required"}

    face_images = []
    for i, b64 in enumerate(face_images_b64):
        res = detect_face(b64)
        if not res["face_detected"]:
            return {"status": "error",
                    "message": f"No face detected in pose {i + 1}"}
        img = b64_to_cv2(res["face_b64"])
        if img is None:
            return {"status": "error",
                    "message": f"Could not decode face for pose {i + 1}"}
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_images.append(cv2.resize(img, IMG_SIZE))

    dup = check_duplicate(face_images)
    if dup["is_duplicate"]:
        return {
            "status":       "duplicate",
            "message":      f"Face already registered as '{dup['matched_name']}'",
            "matched_name": dup["matched_name"],
            "matched_id":   dup["matched_id"],
            "score":        dup["score"],
        }

    sid     = get_next_subject_id()
    sub_dir = LIVE_DATASET_DIR / sid
    sub_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(face_images):
        cv2.imwrite(str(sub_dir / f"{i + 1}.pgm"), img)

    db = load_user_db()
    db["subjects"][sid] = {
        "name":          name.strip(),
        "enrolled_date": datetime.now().isoformat(),
        "source":        "webcam",
        "image_count":   len(face_images),
    }
    save_user_db(db)

    return {
        "status":      "success",
        "message":     f"Enrolled '{name.strip()}' as {sid}",
        "subject_id":  sid,
        "image_count": len(face_images),
    }


# =========================================================
# VERIFICATION  (1:N Identification)
# =========================================================
def verify_user(face_b64):
    """
    Identify a webcam frame against all enrolled subjects.

    Three-gate decision:
      1. top-3 mean cosine similarity  >= VERIFY_THRESHOLD (0.82)
      2. winner score - runner-up score >= MIN_MARGIN      (0.04)
      3. Face successfully detected and segmented

    With spatial LBP the genuine/impostor distributions are well-separated
    (impostor ~0.60-0.80, genuine ~0.85-0.97), so gate 1 alone handles
    most cases.  Gate 2 is a safety net for the rare impostor who happens
    to score in the genuine range for one enrolled subject — the genuine
    owner would still have a clear margin over the runner-up.
    """
    res = detect_face(face_b64)
    if not res["face_detected"]:
        return {"status": "no_face", "access": "denied",
                "message": "No face detected in the image."}

    face_img = b64_to_cv2(res["face_b64"])
    if face_img is None:
        return {"status": "no_face", "access": "denied",
                "message": "Could not decode detected face."}
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    probe_feat = lbp(preprocess(cv2.resize(face_img, IMG_SIZE)))

    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {"status": "no_match", "access": "denied",
                "message": "No users enrolled yet.", "best_score": 0}

    scores = {}
    for sid, feats in gallery.items():
        raw = [similarity(probe_feat, f) for f in feats]
        scores[sid] = _top_k_mean(raw, k=3)

    ranked                  = sorted(scores.items(), key=lambda x: -x[1])
    best_id, best_score     = ranked[0]
    second_score            = ranked[1][1] if len(ranked) > 1 else 0.0
    margin                  = best_score - second_score

    # Debug log
    print(f"\n[VERIFY] {len(gallery)} enrolled subject(s):")
    for sid, sc in ranked:
        name = db["subjects"].get(sid, {}).get("name", "?")
        flag = " <- BEST" if sid == best_id else ""
        print(f"  {sid} ({name}): {sc:.4f}{flag}")
    print(f"  threshold={VERIFY_THRESHOLD}  margin={margin:.4f}"
          f"  (min={MIN_MARGIN})\n")

    if best_score >= VERIFY_THRESHOLD and margin >= MIN_MARGIN:
        info       = db["subjects"].get(best_id, {})
        thumb_b64  = None
        first_imgs = sorted((LIVE_DATASET_DIR / best_id).glob("*.pgm"))
        if first_imgs:
            thumb = cv2.imread(str(first_imgs[0]), cv2.IMREAD_GRAYSCALE)
            if thumb is not None:
                thumb_b64 = cv2_to_b64(thumb)

        return {
            "status":  "matched",
            "access":  "granted",
            "message": f"Welcome, {info.get('name', 'Unknown')}!",
            "user": {
                "id":            best_id,
                "name":          info.get("name", "Unknown"),
                "enrolled_date": info.get("enrolled_date"),
                "source":        info.get("source", "unknown"),
                "image_count":   info.get("image_count", 0),
                "thumbnail":     thumb_b64,
            },
            "score": round(best_score, 4),
            "bbox":  res["bbox"],
        }

    if best_score < VERIFY_THRESHOLD:
        reason = f"score {best_score:.4f} < threshold {VERIFY_THRESHOLD}"
    else:
        reason = (f"margin {margin:.4f} < MIN_MARGIN {MIN_MARGIN} "
                  f"(best={best_score:.4f}, 2nd={second_score:.4f})")
    print(f"[VERIFY] DENIED — {reason}")

    return {
        "status":     "no_match",
        "access":     "denied",
        "message":    "Face not recognised. Access denied.",
        "best_score": round(best_score, 4),
    }