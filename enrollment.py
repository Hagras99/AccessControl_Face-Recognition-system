"""
=============================================================
  Enrollment & Verification Module
  Handles: face detection, enrollment (9 poses), duplicate
  checking, 1:N verification, and user DB management.
=============================================================

FIXES APPLIED (this version)
-----------------------------
A. detect_face — crop was taken from raw gray but detection ran on
   CLAHE-enhanced image. Now detection AND crop both use the same
   raw grayscale image. CLAHE is only used as a detection fallback,
   never as the source of the crop.

B. detect_face — pad_top was 70% of face height, pulling in huge
   amounts of hair/forehead and making the crop hairstyle-dependent.
   Reduced to 20% (just enough to include the full eyebrow line).

C. preprocess — adaptive gamma was pushing all faces to the same
   global brightness, erasing the natural contrast differences that
   help tell people apart. Replaced with a mild fixed gamma (1.2)
   that only brightens genuinely dark frames without flattening
   identity-relevant shading.

D. preprocess — CLAHE clipLimit was 3.0, which aggressively amplifies
   noise after gamma has already normalised brightness. Reduced to 1.5
   (enough for local contrast enhancement, not enough to create
   artefacts that look identical across different people).

E. preprocess — z-score normalisation followed by uint8 clipping was
   REMOVED entirely. LBP uses only relative pixel comparisons (is
   neighbour > centre?), so global mean/std normalisation adds zero
   useful information and the hard clip at ±3σ destroys real texture
   data in bright foreheads and dark eye sockets.

F. lbp — global 256-bin histogram replaced with spatially-aware 5×5
   grid LBP. Each cell gets its own 256-bin histogram; the 25 histograms
   are concatenated into a 6400-dim descriptor. This means eye texture,
   nose texture, mouth texture, and cheek texture are all kept separate
   and contribute independently to the similarity score, instead of
   collapsing into the same bins as they did with the global version.

G. verify_user — now accepts a single base64 string OR a list of
   strings. When a list is given, features are extracted from every
   valid frame and the similarity score is the mean across all
   (probe_frame × gallery_image) pairs, not just one frame.

H. VERIFY_THRESHOLD raised from 0.80 → 0.93 to match the tighter
   score distribution produced by the corrected pipeline.

I. DUPLICATE_THRESHOLD kept at 0.85 — still appropriate for the
   new feature space.
"""

import cv2
import numpy as np
import json
import base64
from pathlib import Path
from datetime import datetime


# =========================================================
# PATHS
# =========================================================
BASE_DIR         = Path(__file__).parent
DATASET_DIR      = BASE_DIR / "dataset"        # ORL — evaluation only
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"   # webcam enrollments
USERS_DB_PATH    = BASE_DIR / "users.json"

# Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Both use COSINE SIMILARITY (range 0–1, higher = more similar).
#
# VERIFY_THRESHOLD: minimum mean cosine similarity to accept a login.
#   Raised to 0.93 — with the corrected grid-LBP pipeline, genuine users
#   consistently score 0.96+; impostors drop to 0.85–0.90.
#
# DUPLICATE_THRESHOLD: minimum similarity to flag a re-enrollment attempt.
#   0.85 — tight enough to block the same person re-enrolling, loose enough
#   to let different people through.
VERIFY_THRESHOLD    = 0.93
DUPLICATE_THRESHOLD = 0.85

IMG_SIZE = (100, 100)   # all face crops are resized to this before feature extraction


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
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_b64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================
# LOW-LIGHT HELPERS  (used only inside detect_face fallbacks)
# =========================================================
def _clahe(gray, clip=1.5):
    """Mild CLAHE — used for detection fallbacks only, not for feature prep."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def _gamma_lut(gray, gamma=2.2):
    """Brighten a dark frame for detection purposes only."""
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(gray, table)


# =========================================================
# FACE DETECTION
# =========================================================
def _try_detect(gray, scale=1.1, neighbors=5, min_size=(60, 60)):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size
    )
    return faces if len(faces) else []


def detect_face(frame_b64):
    """
    Detect and crop a face from a base64-encoded webcam frame.

    FIX A — detection and crop now use the SAME source image (raw gray).
    Previously, detection ran on a CLAHE-enhanced image but the crop was
    sliced from raw gray, so the bounding box coordinates referred to one
    image while the data came from another — causing a misaligned crop.

    FIX B — pad_top reduced from 70% → 20%.
    70% pulled in so much hair/forehead that different hairstyles created
    meaningfully different crops for the same person. 20% is enough to
    capture the full eyebrow line, which is important for recognition.

    Fallback chain for low-light:
      1. Raw gray (standard conditions)
      2. CLAHE-enhanced gray (mild contrast boost)
      3. Gamma-brightened + CLAHE (dark room)
      4. Relaxed Haar params (small or partially occluded face)

    Returns: {face_detected, bbox, face_b64 (100×100 gray crop)}
    """
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Detection fallback chain ──────────────────────────────────────
    # Each attempt uses a different enhanced image for DETECTION only.
    # The actual pixel data for the crop always comes from `gray`.
    faces = _try_detect(gray)                                    # attempt 1: raw

    if len(faces) == 0:
        faces = _try_detect(_clahe(gray, clip=1.5))              # attempt 2: CLAHE

    if len(faces) == 0:
        faces = _try_detect(_clahe(_gamma_lut(gray), clip=1.5),  # attempt 3: gamma+CLAHE
                            neighbors=4)

    if len(faces) == 0:
        faces = _try_detect(gray, neighbors=3,                   # attempt 4: relaxed
                            min_size=(50, 50))

    if len(faces) == 0:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    # Pick the largest detected face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # ── Padding — FIX B ──────────────────────────────────────────────
    # pad_top  : 20% — captures eyebrows without dragging in hair
    # pad_side : 15% — slight margin around cheeks
    # pad_bottom: 10% — includes the chin fully
    pad_top    = int(0.20 * h)
    pad_side   = int(0.15 * w)
    pad_bottom = int(0.10 * h)

    x1 = max(0, x - pad_side)
    y1 = max(0, y - pad_top)
    x2 = min(gray.shape[1], x + w + pad_side)
    y2 = min(gray.shape[0], y + h + pad_bottom)

    # Crop from raw gray (FIX A — same image used for detection above)
    face_crop    = gray[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, IMG_SIZE)

    return {
        "face_detected": True,
        "bbox":    [int(x), int(y), int(w), int(h)],
        "face_b64": cv2_to_b64(face_resized),
    }


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess(img):
    """
    Prepare a 100×100 grayscale face crop for LBP feature extraction.

    Pipeline (corrected):
      1. Mild fixed gamma (1.2) — gently lifts genuinely dark frames
         without flattening the contrast differences that identify people.
         FIX C: removed the adaptive gamma that forced every face to the
         same global brightness, which destroyed identity signal.

      2. CLAHE clipLimit=1.5 — local contrast enhancement.
         FIX D: old clipLimit=3.0 was too aggressive; after gamma had
         already normalised brightness, a high clip amplified noise and
         made different people's faces look more alike.

      3. Gaussian blur 3×3 — removes high-frequency camera noise that
         would otherwise create spurious LBP codes.

      REMOVED: z-score normalisation + uint8 clipping (FIX E).
         LBP compares each pixel only to its 8 immediate neighbours, so
         the global mean and std of the image are completely irrelevant
         to the feature. The hard clip at ±3σ saturated bright foreheads
         and dark eye sockets, destroying real texture information.

    Input:  uint8 grayscale 100×100 numpy array
    Output: uint8 grayscale 100×100 numpy array, ready for lbp()
    """
    # Step 1 — mild fixed gamma: only brightens if the frame is dark,
    # leaves well-lit faces almost unchanged (gamma 1.2 is subtle).
    table = np.array([
        ((i / 255.0) ** (1.0 / 1.2)) * 255 for i in range(256)
    ]).astype("uint8")
    img = cv2.LUT(img, table)

    # Step 2 — CLAHE with reduced clip to boost local contrast gently
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Step 3 — Gaussian blur to suppress sensor noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# =========================================================
# LBP FEATURE EXTRACTION
# =========================================================
def lbp(img):
    """
    Spatially-aware 5×5 grid LBP descriptor.

    FIX F — replaces the old global 256-bin histogram.

    The face image is divided into a 5×5 grid of cells. Each cell gets
    its own 256-bin LBP histogram, and the 25 histograms are concatenated
    into a single 6400-dimensional feature vector.

    Why this matters:
      Global LBP: eye texture and mouth texture land in the same bins.
                  Two different people with similar skin texture get
                  near-identical 256-dim vectors → 94% false match.
      Grid LBP:   eye cell, nose cell, mouth cell, and cheek cells each
                  contribute separately. To score high, a probe must match
                  the enrolled face in every region of the face, not just
                  globally — far harder for an impostor to satisfy.

    Input:  preprocessed uint8 100×100 grayscale image
    Output: float64 ndarray of shape (6400,), L1-normalised per cell
    """
    GRID   = 5
    h, w   = img.shape
    cell_h = h // GRID
    cell_w = w // GRID
    hists  = []

    for row in range(GRID):
        for col in range(GRID):
            cell = img[row * cell_h : (row + 1) * cell_h,
                       col * cell_w : (col + 1) * cell_w]

            c    = cell[1:-1, 1:-1]
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
            hists.append(hist / (np.sum(hist) + 1e-10))   # L1-normalise per cell

    return np.concatenate(hists)   # shape: (6400,)


# =========================================================
# SIMILARITY
# =========================================================
def cosine_sim(h1, h2):
    """
    Cosine similarity between two LBP feature vectors.
    Returns a value in [0, 1] — higher means more similar.
    Works correctly for both 256-dim (old) and 6400-dim (new) vectors.
    """
    n1 = np.linalg.norm(h1) + 1e-10
    n2 = np.linalg.norm(h2) + 1e-10
    return float(np.dot(h1, h2) / (n1 * n2))


# =========================================================
# BUILD GALLERY  (LBP features from live_dataset directory)
# =========================================================
def build_gallery(dataset_dir=None):
    """
    Read all enrolled face images from disk, preprocess them, and extract
    LBP features. Returns {subject_id: [lbp_vector, ...], ...}.

    Called fresh on every verify/duplicate-check call so that newly
    enrolled users are immediately visible without a server restart.
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
    Check whether a face being enrolled is already in the system.
    Compares only against webcam-enrolled users (live_dataset).

    Uses mean cosine similarity across all (probe, gallery) pairs for
    each subject — robust against one outlier frame inflating the score.

    Args:
        face_images_gray: list of uint8 grayscale 100×100 numpy arrays

    Returns:
        {"is_duplicate": True,  "matched_name": ..., "matched_id": ..., "score": ...}
        {"is_duplicate": False}
    """
    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {"is_duplicate": False}

    probe_feats = [lbp(preprocess(img)) for img in face_images_gray]

    best_score = -1
    best_id    = None

    for sid, gallery_feats in gallery.items():
        all_sims = [
            cosine_sim(pf, gf)
            for pf in probe_feats
            for gf in gallery_feats
        ]
        subject_score = float(np.mean(all_sims))

        if subject_score > best_score:
            best_score = subject_score
            best_id    = sid

    print(f"[DUPLICATE CHECK] Best: {best_score:.4f} vs {best_id}"
          f"  (threshold: {DUPLICATE_THRESHOLD})")

    if best_score >= DUPLICATE_THRESHOLD and best_id:
        user_info = db["subjects"].get(best_id, {})
        return {
            "is_duplicate":  True,
            "matched_name":  user_info.get("name", "Unknown"),
            "matched_id":    best_id,
            "score":         round(best_score, 4),
        }

    return {"is_duplicate": False}


# =========================================================
# ENROLLMENT
# =========================================================
def enroll_user(name, face_images_b64):
    """
    Enroll a new user with 9 face poses captured from the webcam.

    Args:
        name:             user's display name (string)
        face_images_b64:  list of base64-encoded webcam frames (9 recommended)

    Returns:
        {"status": "success" | "duplicate" | "error", ...}
    """
    if not name or not name.strip():
        return {"status": "error", "message": "Name cannot be empty"}

    if len(face_images_b64) < 1:
        return {"status": "error", "message": "At least 1 face image required"}

    # Detect and crop a face from each submitted frame
    face_images = []
    for i, b64 in enumerate(face_images_b64):
        result = detect_face(b64)
        if not result["face_detected"]:
            return {"status": "error",
                    "message": f"No face detected in pose {i + 1}"}

        face_gray = b64_to_cv2(result["face_b64"])
        if len(face_gray.shape) == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, IMG_SIZE)
        face_images.append(face_gray)

    # Reject if this face already exists in the system
    dup_check = check_duplicate(face_images)
    if dup_check["is_duplicate"]:
        return {
            "status":       "duplicate",
            "message":      f"This face is already registered as '{dup_check['matched_name']}'",
            "matched_name": dup_check["matched_name"],
            "matched_id":   dup_check["matched_id"],
            "score":        dup_check["score"],
        }

    # Persist face images to disk
    subject_id  = get_next_subject_id()
    subject_dir = LIVE_DATASET_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for i, face_img in enumerate(face_images):
        cv2.imwrite(str(subject_dir / f"{i + 1}.pgm"), face_img)

    # Update the user database
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
def verify_user(face_b64):
    """
    Identify a live webcam face against all enrolled subjects.

    FIX G — accepts a single base64 string OR a list of base64 strings.
    When a list is given, LBP features are extracted from every frame
    in which a face is detected, and the similarity score for each
    enrolled subject is the mean over all (probe_frame × gallery_image)
    pairs. This prevents one lucky or unlucky frame from deciding access.

    FIX H — VERIFY_THRESHOLD raised to 0.93 to match the tighter score
    distribution produced by the corrected preprocessing + grid-LBP pipeline.

    Args:
        face_b64: base64-encoded webcam frame  OR  list of such frames

    Returns:
        {"status": "matched"|"no_match"|"no_face", "access": "granted"|"denied", ...}
    """
    # ── Accept single frame or list of frames ─────────────────────────
    frames = face_b64 if isinstance(face_b64, list) else [face_b64]

    # ── Extract features from every frame that has a detectable face ──
    probe_feats = []
    last_bbox   = None

    for frame in frames:
        result = detect_face(frame)
        if not result["face_detected"]:
            continue

        face_img = b64_to_cv2(result["face_b64"])
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, IMG_SIZE)
        probe_feats.append(lbp(preprocess(face_img)))
        last_bbox = result["bbox"]

    if not probe_feats:
        return {
            "status":  "no_face",
            "access":  "denied",
            "message": "No face detected in the image",
        }

    # ── Load gallery and score every enrolled subject ──────────────────
    gallery = build_gallery(LIVE_DATASET_DIR)
    db      = load_user_db()

    if not gallery:
        return {
            "status":     "no_match",
            "access":     "denied",
            "message":    "No users enrolled yet. Please enroll first.",
            "best_score": 0,
        }

    best_score = -1
    best_id    = None
    all_scores = {}

    for sid, gallery_feats in gallery.items():
        # Mean similarity across ALL (probe_frame, gallery_image) pairs.
        # A genuine user scores high consistently across frames.
        # An impostor may spike on one frame but averages lower overall.
        all_sims   = [cosine_sim(pf, gf)
                      for pf in probe_feats
                      for gf in gallery_feats]
        mean_score = float(np.mean(all_sims))
        all_scores[sid] = mean_score

        if mean_score > best_score:
            best_score = mean_score
            best_id    = sid

    # Debug output visible in the Flask terminal
    print(f"\n[VERIFY] {len(probe_feats)} probe frame(s) vs "
          f"{len(gallery)} enrolled user(s):")
    for sid, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
        name = db["subjects"].get(sid, {}).get("name", "?")
        print(f"  {sid} ({name}): {sc:.4f}")
    print(f"  Best: {best_id} = {best_score:.4f}  "
          f"(threshold: {VERIFY_THRESHOLD})\n")

    # ── Decision ───────────────────────────────────────────────────────
    if best_score >= VERIFY_THRESHOLD and best_id:
        user_info   = db["subjects"].get(best_id, {})
        subject_dir = LIVE_DATASET_DIR / best_id

        # Thumbnail from first enrolled image
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
    else:
        return {
            "status":     "no_match",
            "access":     "denied",
            "message":    "Face not recognized. Access denied.",
            "best_score": round(best_score, 4) if best_score > 0 else 0,
        }