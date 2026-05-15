"""
=============================================================
  Enrollment & Verification Module
=============================================================
(RESTORED VERSION — original structure kept)
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
DATASET_DIR = BASE_DIR / "dataset"
LIVE_DATASET_DIR = BASE_DIR / "live_dataset"
USERS_DB_PATH = BASE_DIR / "users.json"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = (100, 100)

# =========================================================
# THRESHOLDS (RESTORED + SAFE FIX ONLY)
# =========================================================
VERIFY_THRESHOLD = 0.90
DUPLICATE_THRESHOLD = 0.85

# 🔥 ONLY ADDITION (fix open-set issue, minimal change)
MIN_ACCEPT_SCORE = 0.88
MARGIN_THRESHOLD = 0.05


# =========================================================
# USER DATABASE (UNCHANGED)
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
    """(RESTORED - REQUIRED BY YOUR APP.PY)"""
    if USERS_DB_PATH.exists():
        return load_user_db()

    db = {"subjects": {}}
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
            except:
                pass

    next_id = max(existing_ids) + 1 if existing_ids else 1
    return f"u{next_id}"


# =========================================================
# IMAGE UTILITIES (UNCHANGED)
# =========================================================
def b64_to_cv2(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_to_b64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================================================
# FACE DETECTION (UNCHANGED)
# =========================================================
def detect_face(frame_b64):
    img = b64_to_cv2(frame_b64)
    if img is None:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return {"face_detected": False, "bbox": None, "face_b64": None}

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)

    return {
        "face_detected": True,
        "bbox": [int(x), int(y), int(w), int(h)],
        "face_b64": cv2_to_b64(face),
    }


# =========================================================
# LBP + SIMILARITY (UNCHANGED)
# =========================================================
def lbp(img):
    c = img[1:-1, 1:-1]
    code = (
        ((img[0:-2, 0:-2] > c) << 7) |
        ((img[0:-2, 1:-1] > c) << 6) |
        ((img[0:-2, 2:] > c) << 5) |
        ((img[1:-1, 2:] > c) << 4) |
        ((img[2:, 2:] > c) << 3) |
        ((img[2:, 1:-1] > c) << 2) |
        ((img[2:, 0:-2] > c) << 1) |
        ((img[1:-1, 0:-2] > c))
    )

    hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    return hist / (np.sum(hist) + 1e-10)


def preprocess(img):
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(img)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =========================================================
# GALLERY (UNCHANGED)
# =========================================================
def build_gallery():
    gallery = {}

    if not LIVE_DATASET_DIR.exists():
        return gallery

    for person in LIVE_DATASET_DIR.iterdir():
        if not person.is_dir():
            continue

        feats = []
        for img_path in person.glob("*.pgm"):
            img = cv2.imread(str(img_path), 0)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = preprocess(img)
            feats.append(lbp(img))

        if feats:
            gallery[person.name] = feats

    return gallery


# =========================================================
# DUPLICATE CHECK (UNCHANGED LOGIC)
# =========================================================
def check_duplicate(face_images):
    gallery = build_gallery()
    db = load_user_db()

    if not gallery:
        return {"is_duplicate": False}

    probes = [lbp(preprocess(img)) for img in face_images]

    best_score = -1
    best_id = None

    for sid, feats in gallery.items():
        sims = []
        for p in probes:
            for g in feats:
                sims.append(cosine(p, g))

        score = float(np.mean(sims))

        if score > best_score:
            best_score = score
            best_id = sid

    if best_score >= DUPLICATE_THRESHOLD:
        return {
            "is_duplicate": True,
            "matched_id": best_id,
            "matched_name": db["subjects"].get(best_id, {}).get("name", "Unknown"),
            "score": round(best_score, 4)
        }

    return {"is_duplicate": False}


# =========================================================
# ENROLLMENT (UNCHANGED)
# =========================================================
def enroll_user(name, face_images_b64):
    if not name.strip():
        return {"status": "error", "message": "Name required"}

    face_images = []

    for i, b64 in enumerate(face_images_b64):
        res = detect_face(b64)

        if not res["face_detected"]:
            return {"status": "error", "message": f"No face in image {i+1}"}

        face = b64_to_cv2(res["face_b64"])
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, IMG_SIZE)

        face_images.append(face)

    dup = check_duplicate(face_images)

    if dup.get("is_duplicate"):
        return {
            "status": "duplicate",
            "message": "Already enrolled",
            **dup
        }

    subject_id = get_next_subject_id()
    subject_dir = LIVE_DATASET_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(face_images):
        cv2.imwrite(str(subject_dir / f"{i+1}.pgm"), img)

    db = load_user_db()
    db["subjects"][subject_id] = {
        "name": name.strip(),
        "enrolled_date": str(datetime.now()),
        "source": "webcam",
        "image_count": len(face_images)
    }
    save_user_db(db)

    return {
        "status": "success",
        "subject_id": subject_id
    }


# =========================================================
# VERIFICATION (ONLY FIXED PART)
# =========================================================
def verify_user(face_b64):

    det = detect_face(face_b64)
    if not det["face_detected"]:
        return {"status": "no_face", "access": "denied"}

    face = b64_to_cv2(det["face_b64"])
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, IMG_SIZE)

    probe = lbp(preprocess(face))

    gallery = build_gallery()
    db = load_user_db()

    if not gallery:
        return {"status": "no_users", "access": "denied"}

    scores = {}

    for sid, feats in gallery.items():
        sims = [cosine(probe, f) for f in feats]
        scores[sid] = float(np.mean(sims))

    ranked = sorted(scores.items(), key=lambda x: -x[1])

    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # 🔥 FIXED OPEN-SET DECISION (minimal safe addition)
    if (
        best_score < VERIFY_THRESHOLD or
        best_score < MIN_ACCEPT_SCORE or
        (best_score - second_score) < MARGIN_THRESHOLD
    ):
        return {
            "status": "no_match",
            "access": "denied",
            "score": round(best_score, 4)
        }

    user = db["subjects"].get(best_id, {})

    return {
        "status": "matched",
        "access": "granted",
        "user": {
            "id": best_id,
            "name": user.get("name")
        },
        "score": round(best_score, 4),
        "bbox": det["bbox"]
    }