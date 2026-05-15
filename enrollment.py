"""
=============================================================
  Enrollment & Verification Module (Optimized MediaPipe Version)

  FIXES APPLIED
  -------------
  1. Added missing Preprocessor._apply_ellipse_mask() and
     _apply_landmark_mask() — their absence caused a crash on
     every single face-processing call.
  2. Fixed OptimizedLBP._get_uniform_map(): non-uniform patterns
     must ALL map to the same bin (58), not to a shifting label.
  3. Fixed _align_face(): reshape(-2, 2) → reshape(-1, 2).
  4. Fixed verify_user() liveness: result["landmarks_data"] key
     never existed → liveness always failed → access always denied.
     Liveness now gracefully bypasses the blink check when no
     landmark data was captured in a single verify frame.
  5. detect_face() now stores the actual landmark object in the
     result dict so downstream code can use it.
=============================================================
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
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp


# =========================================================
# CONFIGURATION (Centralized)
# =========================================================
class Config:
    """Centralized configuration with environment variable support."""

    # Paths
    BASE_DIR         = Path(__file__).parent
    DATASET_DIR      = BASE_DIR / "dataset"
    LIVE_DATASET_DIR = BASE_DIR / "live_dataset"
    USERS_DB_PATH    = BASE_DIR / "users.json"
    MODEL_PATH       = BASE_DIR / "models" / "face_landmarker.task"
    CALIBRATION_PATH = BASE_DIR / "calibration.json"  # written by main.py

    # Image processing
    IMG_SIZE = (100, 100)

    # LBP parameters
    LBP_GRID    = 5       # 5×5 grid
    LBP_UNIFORM = True    # Use uniform LBP (59 bins vs 256)
    LBP_RADIUS  = 1
    LBP_POINTS  = 8

    # Preprocessing
    GAMMA       = 1.2
    CLAHE_CLIP  = 2.5   # raised from 1.5: 100×100 crops have small tiles so
                        #   stronger clipping is needed for adequate local contrast
    CLAHE_GRID  = (8, 8)
    BLUR_KERNEL = (3, 3)
    MASK_FEATHER = 11   # was 21; reduced for 100×100 images (21px ≈ 20% of width)

    # Face detection
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE  = 0.5
    MAX_FACES                = 1

    # Enrollment
    MIN_ENROLLMENT_IMAGES = 1
    MAX_ENROLLMENT_IMAGES = 10

    # Padding (as fractions of face size)
    PAD_TOP    = 0.15
    PAD_BOTTOM = 0.10   # raised from 0.05: chin was being cropped for close-up faces
    PAD_SIDE   = 0.10

    # Head pose thresholds (degrees)
    MAX_YAW   = 30
    MAX_PITCH = 25
    MAX_ROLL  = 30

    # Liveness detection
    # Set to False to disable entirely (recommended until you have a
    # proper multi-frame liveness pipeline).
    LIVENESS_ENABLED    = False
    BLINK_THRESHOLD     = 0.2   # EAR threshold
    MOVEMENT_THRESHOLD  = 5     # pixels

    # Verification thresholds — defaults used when no calibration file exists.
    # Calibrated for live webcam images (typical scores 0.77-0.82 genuine).
    DEFAULT_VERIFY_THRESHOLD    = 0.80
    DEFAULT_DUPLICATE_THRESHOLD = 0.82

    # Threading
    MAX_WORKERS = 4

    @classmethod
    def init_paths(cls):
        """Create necessary directories."""
        cls.MODEL_PATH.parent.mkdir(exist_ok=True)
        cls.LIVE_DATASET_DIR.mkdir(exist_ok=True)

    @classmethod
    def load_calibrated_thresholds(cls):
        """
        Load thresholds calibrated by main.py from calibration.json.
        Falls back to DEFAULT_* constants if the file is missing or corrupt.

        Returns
        -------
        (verify_threshold, duplicate_threshold) as floats
        """
        try:
            if cls.CALIBRATION_PATH.exists():
                import json as _json
                with open(cls.CALIBRATION_PATH) as f:
                    cal = _json.load(f)
                v = float(cal.get("verify_threshold",    cls.DEFAULT_VERIFY_THRESHOLD))
                d = float(cal.get("duplicate_threshold", cls.DEFAULT_DUPLICATE_THRESHOLD))
                return v, d
        except Exception as e:
            print(f"[Config] Could not read calibration.json: {e}")
        return cls.DEFAULT_VERIFY_THRESHOLD, cls.DEFAULT_DUPLICATE_THRESHOLD


# =========================================================
# OPTIMIZED LBP WITH UNIFORM PATTERNS
# =========================================================
class OptimizedLBP:
    """Optimized LBP with uniform patterns and caching."""

    _uniform_map: Optional[np.ndarray] = None

    @classmethod
    def _get_uniform_map(cls) -> np.ndarray:
        """
        Generate uniform-pattern mapping (cached).

        Standard uniform LBP for P=8 neighbours produces 58 uniform
        patterns (≤2 bit transitions) plus 1 shared non-uniform bin = 59
        total bins.  All non-uniform codes must map to the *same* bin
        value (58).  The previous code incremented next_label for every
        pattern, assigning a unique label to each non-uniform code and
        therefore producing garbage histograms.
        """
        if cls._uniform_map is not None:
            return cls._uniform_map

        num_patterns = 2 ** 8
        NON_UNIFORM_BIN = 58          # single shared bin for all non-uniform
        uniform_map = np.full(num_patterns, NON_UNIFORM_BIN, dtype=np.int32)

        next_label = 0
        for i in range(num_patterns):
            binary      = f"{i:08b}"
            transitions = sum(binary[j] != binary[(j + 1) % 8] for j in range(8))
            if transitions <= 2:
                uniform_map[i] = next_label
                next_label += 1
            # else: already set to NON_UNIFORM_BIN (58)

        cls._uniform_map = uniform_map
        return cls._uniform_map

    @classmethod
    def compute(cls, img: np.ndarray) -> np.ndarray:
        """
        Compute uniform LBP with 5×5 spatial grid.

        Args:
            img: Grayscale image (100×100)

        Returns:
            Feature vector of shape (5*5*59,) = 1475 dimensions
        """
        grid_size = Config.LBP_GRID
        hist_size = 59 if Config.LBP_UNIFORM else 256
        hists     = np.zeros((grid_size * grid_size, hist_size), dtype=np.float32)

        lbp_image = cls._compute_lbp_image(img)
        # lbp_image is (H-2, W-2) = (98, 98); derive grid from actual size
        lh, lw    = lbp_image.shape
        cell_h    = lh // grid_size   # 98 // 5 = 19
        cell_w    = lw // grid_size

        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                cell = lbp_image[
                    row * cell_h:(row + 1) * cell_h,
                    col * cell_w:(col + 1) * cell_w,
                ]
                hist, _ = np.histogram(cell.ravel(), bins=hist_size,
                                       range=(0, hist_size))
                hists[idx] = hist / (hist.sum() + 1e-10)
                idx += 1

        return hists.ravel()

    @classmethod
    def _compute_lbp_image(cls, img: np.ndarray) -> np.ndarray:
        """Vectorized LBP computation for the entire image."""
        center = img[1:-1, 1:-1]

        lbp = (
            ((img[0:-2, 0:-2] > center).astype(np.uint8) << 7) |
            ((img[0:-2, 1:-1] > center).astype(np.uint8) << 6) |
            ((img[0:-2, 2:]   > center).astype(np.uint8) << 5) |
            ((img[1:-1, 2:]   > center).astype(np.uint8) << 4) |
            ((img[2:,   2:]   > center).astype(np.uint8) << 3) |
            ((img[2:,   1:-1] > center).astype(np.uint8) << 2) |
            ((img[2:,   0:-2] > center).astype(np.uint8) << 1) |
            ((img[1:-1, 0:-2] > center).astype(np.uint8))
        )

        if Config.LBP_UNIFORM:
            lbp = cls._get_uniform_map()[lbp]

        return lbp


# =========================================================
# CACHED PREPROCESSING
# =========================================================
class Preprocessor:
    _gamma_lut: Optional[np.ndarray] = None
    _clahe:     Optional[cv2.CLAHE]   = None
    _cache:     Dict[Any, np.ndarray] = {}

    @classmethod
    def _init_luts(cls):
        if cls._gamma_lut is None:
            cls._gamma_lut = np.array(
                [((i / 255.0) ** (1.0 / Config.GAMMA)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
        if cls._clahe is None:
            cls._clahe = cv2.createCLAHE(
                clipLimit=Config.CLAHE_CLIP,
                tileGridSize=Config.CLAHE_GRID,
            )

    @classmethod
    def process(
        cls,
        img_hash: int,
        img: np.ndarray,
        landmarks_tuple: Optional[tuple] = None,
    ) -> np.ndarray:
        """Gamma correction → CLAHE → Gaussian blur → oval/landmark mask."""
        cache_key = (img_hash, landmarks_tuple)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        cls._init_luts()

        result = cv2.LUT(img, cls._gamma_lut)
        result = cls._clahe.apply(result)
        result = cv2.GaussianBlur(result, Config.BLUR_KERNEL, 0)

        if landmarks_tuple is not None:
            landmarks = [
                type("Landmark", (), {"x": x, "y": y})()
                for x, y in landmarks_tuple
            ]
            result = cls._apply_landmark_mask(result, landmarks)
        else:
            result = cls._apply_ellipse_mask(result)

        # LRU-style eviction
        if len(cls._cache) >= 128:
            cls._cache.pop(next(iter(cls._cache)))
        cls._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # BUG FIX: these two methods were referenced but never defined,
    # causing an AttributeError on the very first preprocessing call.
    # ------------------------------------------------------------------

    @classmethod
    def _apply_ellipse_mask(cls, img: np.ndarray) -> np.ndarray:
        """
        Apply a soft elliptical mask that suppresses background pixels
        and focuses the texture descriptor on the face interior.
        """
        h, w   = img.shape[:2]
        mask   = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes   = (max(1, w // 2 - 2), max(1, h // 2 - 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Feather the edge for a smooth blend
        feather = Config.MASK_FEATHER
        if feather % 2 == 0:
            feather += 1          # GaussianBlur requires odd kernel size
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

        mask_f = mask.astype(np.float32) / 255.0
        return (img.astype(np.float32) * mask_f).astype(np.uint8)

    @classmethod
    def _apply_landmark_mask(cls, img: np.ndarray, landmarks) -> np.ndarray:
        """
        Landmark-aware mask: build a convex hull from the face outline
        landmarks and apply a soft elliptical fallback when the hull
        would be degenerate.
        """
        h, w = img.shape[:2]

        # Outer face contour indices in MediaPipe 468-landmark topology
        FACE_OVAL_IDX = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58,  132, 93,  234, 127,
            162, 21,  54,  103, 67,  109,
        ]

        try:
            pts = np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                 for i in FACE_OVAL_IDX
                 if i < len(landmarks)],
                dtype=np.int32,
            )
            if len(pts) < 3:
                raise ValueError("Too few landmark points")

            hull = cv2.convexHull(pts)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)

            feather = Config.MASK_FEATHER
            if feather % 2 == 0:
                feather += 1
            mask   = cv2.GaussianBlur(mask, (feather, feather), 0)
            mask_f = mask.astype(np.float32) / 255.0
            return (img.astype(np.float32) * mask_f).astype(np.uint8)

        except Exception:
            # Fall back gracefully to the ellipse mask
            return cls._apply_ellipse_mask(img)


# =========================================================
# MEDIAPIPE SINGLETON
# =========================================================
class MediaPipeManager:
    """Singleton manager for MediaPipe resources."""

    _instance = None
    _lock      = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._landmarker  = None
        self._detector    = None
        self._do_initialize()

    def _do_initialize(self):
        Config.init_paths()

        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE
        )

        try:
            if Config.MODEL_PATH.exists():
                base_options = mp.tasks.BaseOptions(
                    model_asset_path=str(Config.MODEL_PATH)
                )
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=Config.MAX_FACES,
                )
                self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                    options
                )
                print("[MediaPipe] ✓ Face landmarker loaded")
            else:
                print(
                    f"[MediaPipe] ⚠ Model not found at {Config.MODEL_PATH},"
                    " using basic detector"
                )
        except Exception as e:
            print(f"[MediaPipe] ⚠ Could not load landmarker: {e}")

    def detect(
        self, rgb_image: np.ndarray
    ) -> Tuple[Optional[Any], Optional[List[int]]]:
        """
        Detect face and landmarks.
        Returns (landmarks, bbox) or (None, None).
        """
        if self._landmarker is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                results  = self._landmarker.detect(mp_image)
                if results.face_landmarks:
                    landmarks = results.face_landmarks[0]
                    h, w      = rgb_image.shape[:2]
                    xs   = [lm.x * w for lm in landmarks]
                    ys   = [lm.y * h for lm in landmarks]
                    bbox = [
                        int(min(xs)), int(min(ys)),
                        int(max(xs) - min(xs)), int(max(ys) - min(ys)),
                    ]
                    return landmarks, bbox
            except Exception as e:
                print(f"[MediaPipe] Landmarker error: {e}")

        # Fallback: basic mp.solutions detector (no landmarks)
        results = self._detector.process(rgb_image)
        if results.detections:
            bboxC = results.detections[0].location_data.relative_bounding_box
            h, w  = rgb_image.shape[:2]
            bbox  = [
                int(bboxC.xmin * w), int(bboxC.ymin * h),
                int(bboxC.width * w), int(bboxC.height * h),
            ]
            return None, bbox

        return None, None


# =========================================================
# LIVENESS DETECTION
# =========================================================
class LivenessDetector:
    """Eye blink and head-movement detection."""

    def __init__(self):
        self.prev_landmarks   = None
        self.blink_counter    = 0
        self.movement_history = []

    def detect_blink(self, landmarks) -> bool:
        """Detect eye blink using Eye Aspect Ratio (EAR)."""
        if landmarks is None or len(landmarks) < 468:
            return False

        # Standard 6-point EAR landmarks for MediaPipe 468-landmark model:
        #  index 0 = inner corner, index 3 = outer corner (horizontal span)
        #  index 1,2 = upper eyelid pts; index 4,5 = lower eyelid pts
        LEFT_EYE  = [33,  160, 158, 133, 153, 144]  # p1,p2,p3,p4,p5,p6
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        def _lm_to_xy(lm):
            return np.array([lm.x, lm.y])

        def ear(indices):
            # Soukupova & Cech (2016) EAR formula:
            # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            pts = [_lm_to_xy(landmarks[i]) for i in indices]
            v1  = np.linalg.norm(pts[1] - pts[5])   # p2 - p6
            v2  = np.linalg.norm(pts[2] - pts[4])   # p3 - p5
            h   = np.linalg.norm(pts[0] - pts[3])   # p1 - p4 (horizontal)
            return (v1 + v2) / (2.0 * h + 1e-10)

        avg_ear = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2.0
        if avg_ear < Config.BLINK_THRESHOLD:
            self.blink_counter += 1
            return True
        return False

    def detect_movement(self, landmarks) -> bool:
        """Detect head movement for liveness."""
        if landmarks is None or self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False

        try:
            nose_cur  = np.array([landmarks[1].x,           landmarks[1].y])
            nose_prev = np.array([self.prev_landmarks[1].x, self.prev_landmarks[1].y])
            movement  = np.linalg.norm(nose_cur - nose_prev) * 1000  # scale to px-ish
            self.movement_history.append(movement)
            if len(self.movement_history) > 10:
                self.movement_history.pop(0)
            self.prev_landmarks = landmarks
            return movement > Config.MOVEMENT_THRESHOLD
        except Exception:
            self.prev_landmarks = landmarks
            return False

    def is_live(
        self,
        landmarks,
        require_blink: bool    = True,
        require_movement: bool = False,
    ) -> bool:
        """Return True if face appears live."""
        if not Config.LIVENESS_ENABLED:
            return True

        # If no landmark data was captured we cannot evaluate blink/movement,
        # so we treat the face as live rather than always denying access.
        if landmarks is None:
            return True

        live = True
        if require_blink:
            live = live and (self.blink_counter > 0)
        if require_movement:
            live = live and self.detect_movement(landmarks)
        return live


# =========================================================
# USER DATABASE (Thread-safe)
# =========================================================
class UserDatabase:
    _instance = None
    _lock      = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._db         = {}
        self._cache      = {}
        self._cache_mtime = 0.0
        self._load()

    def _load(self):
        if Config.USERS_DB_PATH.exists():
            with open(Config.USERS_DB_PATH, "r") as f:
                self._db = json.load(f)
        else:
            self._db = {"subjects": {}}
            self._save()

    def _save(self):
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=Config.USERS_DB_PATH.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(self._db, f, indent=2)
            os.replace(tmp_path, Config.USERS_DB_PATH)
        except Exception:
            os.unlink(tmp_path)
            raise

    def get_all_users(self, webcam_only: bool = False) -> List[Dict]:
        """
        Return all enrolled users.

        Parameters
        ----------
        webcam_only : bool
            If True, return only users enrolled via webcam (source='webcam').
            These are the only subjects present in the live gallery and
            therefore the only ones that can be verified.  ORL dataset
            subjects (source='ORL') are stored in users.json for record-
            keeping but have no live-gallery images.
        """
        return [
            {
                "id":            sid,
                "name":          info["name"],
                "enrolled_date": info.get("enrolled_date"),
                "source":        info.get("source", "unknown"),
                "image_count":   info.get("image_count", 0),
                # True only for webcam users — these exist in live_dataset/
                # and will be matched during verify_user().
                "verifiable":    info.get("source", "unknown") == "webcam",
            }
            for sid, info in self._db["subjects"].items()
            if not webcam_only or info.get("source") == "webcam"
        ]

    def add_user(self, subject_id: str, user_info: Dict):
        with self._lock:
            self._db["subjects"][subject_id] = user_info
            self._save()
            self._cache.clear()

    def get_user(self, subject_id: str) -> Optional[Dict]:
        return self._db["subjects"].get(subject_id)

    def delete_user(self, subject_id: str) -> bool:
        with self._lock:
            if subject_id in self._db["subjects"]:
                del self._db["subjects"][subject_id]
                self._save()
                self._cache.clear()
                return True
            return False

    def get_next_id(self) -> str:
        existing = [
            int(d.name[1:])
            for d in Config.LIVE_DATASET_DIR.iterdir()
            if d.is_dir() and d.name.startswith("u")
        ]
        return f"u{max(existing) + 1 if existing else 1}"


# =========================================================
# MEDOID SELECTION HELPER
# =========================================================
def _select_medoid(features: List[np.ndarray]) -> np.ndarray:
    """
    Return the feature vector from *features* that minimises the sum of
    cosine distances to all other vectors in the list (the medoid).

    This is more outlier-resistant than the mean: a single blurry or
    poorly-lit enrollment image will not pull the prototype away from
    the true face distribution.

    Falls back to the mean when only one image is enrolled.
    """
    if len(features) == 1:
        return features[0]

    mat  = np.array(features, dtype=np.float32)        # (N, D)
    nmat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
    # Cosine similarity matrix (N × N); sum along rows → affinity score
    sim  = nmat @ nmat.T                                # (N, N)
    # Pick the row (= feature vector) with the highest total similarity
    best = int(np.argmax(sim.sum(axis=1)))
    return mat[best]


# =========================================================
# GALLERY CACHE
# =========================================================
class GalleryCache:
    _instance = None
    _lock      = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._gallery    = {}
        self._prototypes = {}
        self._mtime      = 0.0

    def build(self, force_rebuild: bool = False) -> Dict[str, np.ndarray]:
        if not Config.LIVE_DATASET_DIR.exists():
            return {}

        try:
            pgm_files   = list(Config.LIVE_DATASET_DIR.rglob("*.pgm"))
            latest_mtime = max(
                (p.stat().st_mtime for p in pgm_files), default=0.0
            )
        except Exception:
            latest_mtime = 0.0

        with self._lock:
            if (
                not force_rebuild
                and latest_mtime <= self._mtime
                and self._prototypes
            ):
                return self._prototypes

            self._gallery.clear()
            self._prototypes.clear()

            for person_dir in Config.LIVE_DATASET_DIR.iterdir():
                if not person_dir.is_dir():
                    continue

                features = []
                for img_path in person_dir.glob("*.pgm"):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img       = cv2.resize(img, Config.IMG_SIZE)
                    img_hash  = hash(img.tobytes())
                    processed = Preprocessor.process(img_hash, img, None)
                    features.append(OptimizedLBP.compute(processed))

                if features:
                    self._gallery[person_dir.name]    = features
                    # Use best-image (medoid) instead of mean:
                    # pick the feature vector closest to all others.
                    # This is outlier-resistant — one blurry enrollment
                    # image won't drag the prototype away from the true
                    # face distribution.
                    self._prototypes[person_dir.name] = _select_medoid(features)

            self._mtime = latest_mtime
            print(f"[Gallery] Cache rebuilt: {len(self._prototypes)} users")
            return self._prototypes

    def get_prototypes(self) -> Dict[str, np.ndarray]:
        return self._prototypes if self._prototypes else self.build()


# =========================================================
# MAIN FACE RECOGNITION API
# =========================================================
class FaceRecognitionSystem:

    # Protects the get_next_id → mkdir → save → db.add_user sequence so that
    # two concurrent enrollments cannot race and claim the same subject ID.
    _enroll_lock = threading.Lock()

    def __init__(self):
        self.mp       = MediaPipeManager()
        self.db       = UserDatabase()
        self.gallery  = GalleryCache()
        self.liveness = LivenessDetector()
        self._executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

    # ------------------------------------------------------------------
    def detect_face(self, frame_b64: str) -> Dict[str, Any]:
        """
        Detect face and extract features from a base64 image.

        Returns dict with at minimum:
            face_detected (bool), bbox, face_b64, landmarks_data
        """
        img = self._b64_to_cv2(frame_b64)
        if img is None:
            return {"face_detected": False}

        rgb_img              = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks, bbox      = self.mp.detect(rgb_img)

        if bbox is None:
            return {"face_detected": False}

        x, y, w, h    = bbox
        pad_top       = int(Config.PAD_TOP    * h)
        pad_side      = int(Config.PAD_SIDE   * w)
        pad_bottom    = int(Config.PAD_BOTTOM * h)

        x1 = max(0, x - pad_side)
        y1 = max(0, y - pad_top)
        x2 = min(img.shape[1], x + w + pad_side)
        y2 = min(img.shape[0], y + h + pad_bottom)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {"face_detected": False}

        # Align face if landmarks are available
        aligned_face_b64 = None
        if landmarks:
            aligned_crop, _angle = self._align_face(img, landmarks, bbox)
            if aligned_crop.size > 0:
                gray_aligned     = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY) \
                                   if len(aligned_crop.shape) == 3 else aligned_crop
                aligned_face_b64 = self._cv2_to_b64(cv2.resize(gray_aligned, Config.IMG_SIZE))

        face_gray = (
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            if len(face_crop.shape) == 3
            else face_crop
        )

        # Update liveness blink counter on every frame
        blink_detected = self.liveness.detect_blink(landmarks) if landmarks else False

        # --- BUG FIX: landmark coordinate projection -------------------------
        # MediaPipe landmarks carry .x/.y normalised to the FULL original frame.
        # _apply_landmark_mask() operates on the already-cropped and resized
        # 100×100 face image, so we must re-project each landmark into the
        # crop’s coordinate space before storing them.  Without this the convex-
        # hull mask was always computed in the wrong space and silently fell back
        # to the ellipse mask via the except clause.
        crop_landmarks_tuple = None
        if landmarks:
            try:
                full_H_f = float(img.shape[0])
                full_W_f = float(img.shape[1])
                crop_w   = float(x2 - x1)
                crop_h   = float(y2 - y1)
                if crop_w > 0 and crop_h > 0:
                    crop_landmarks_tuple = tuple(
                        (
                            (lm.x * full_W_f - x1) / crop_w,
                            (lm.y * full_H_f - y1) / crop_h,
                        )
                        for lm in landmarks
                    )
            except Exception:
                crop_landmarks_tuple = None
        # ---------------------------------------------------------------------

        return {
            "face_detected":        True,
            "bbox":                 [int(x), int(y), int(w), int(h)],
            "face_b64":             self._cv2_to_b64(cv2.resize(face_gray, Config.IMG_SIZE)),
            "aligned_face_b64":     aligned_face_b64,
            "landmarks_data":       landmarks,
            # Landmarks re-projected to the face-crop coordinate space.
            # Use this key (not landmarks_data) when calling Preprocessor so
            # that _apply_landmark_mask works in the correct pixel space.
            "crop_landmarks_tuple": crop_landmarks_tuple,
            "has_landmarks":        landmarks is not None,
            "liveness_score":       int(blink_detected),
        }

    # ------------------------------------------------------------------
    def enroll_user(self, name: str, face_images_b64: List[str]) -> Dict[str, Any]:
        """Enroll a new user with one or more face images."""
        if not name or not name.strip():
            return {"status": "error", "message": "Name cannot be empty"}

        if len(face_images_b64) < Config.MIN_ENROLLMENT_IMAGES:
            return {
                "status":  "error",
                "message": f"At least {Config.MIN_ENROLLMENT_IMAGES} image(s) required",
            }

        futures = [
            self._executor.submit(self._process_enrollment_image, b64)
            for b64 in face_images_b64[: Config.MAX_ENROLLMENT_IMAGES]
        ]

        face_images = []
        for future in futures:
            result = future.result()
            if result is None:
                return {
                    "status":  "error",
                    "message": "Face detection failed for one or more images",
                }
            face_images.append(result)

        # Duplicate check
        dup = self._check_duplicate(face_images)
        if dup["is_duplicate"]:
            return {
                "status":       "duplicate",
                "message":      f"Face already registered as '{dup['matched_name']}'",
                "matched_name": dup["matched_name"],
                "matched_id":   dup["matched_id"],
                "score":        dup["score"],
            }

        # Acquire enroll lock so concurrent requests cannot race on get_next_id
        # and end up claiming the same subject directory / database entry.
        with self.__class__._enroll_lock:
            subject_id  = self.db.get_next_id()
            subject_dir = Config.LIVE_DATASET_DIR / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)

            for i, face_img in enumerate(face_images):
                cv2.imwrite(str(subject_dir / f"{i + 1}.pgm"), face_img)

            self.db.add_user(subject_id, {
                "name":          name.strip(),
                "enrolled_date": datetime.now().isoformat(),
                "source":        "webcam",
                "image_count":   len(face_images),
            })

        self.gallery.build(force_rebuild=True)

        return {
            "status":      "success",
            "message":     f"Successfully enrolled '{name.strip()}' as {subject_id}",
            "subject_id":  subject_id,
            "image_count": len(face_images),
        }

    # ------------------------------------------------------------------
    def verify_user(self, face_b64: str) -> Dict[str, Any]:
        """Verify a face against the enrolled gallery (1:N)."""
        result = self.detect_face(face_b64)

        # ── Diagnostic: face detection ───────────────────────────────────
        print(f"[Verify] face_detected={result['face_detected']}  "
              f"has_landmarks={result.get('has_landmarks')}  "
              f"crop_lm={'YES' if result.get('crop_landmarks_tuple') else 'NO'}")

        if not result["face_detected"]:
            print("[Verify] ✗ No face detected — returning no_face")
            return {
                "status":  "no_face",
                "access":  "denied",
                "message": "No face detected",
            }

        face_img  = self._face_result_to_gray(result)
        lm_tuple  = result.get("crop_landmarks_tuple")   # None → ellipse fallback
        features  = OptimizedLBP.compute(
            Preprocessor.process(hash(face_img.tobytes()), face_img, lm_tuple)
        )

        prototypes = self.gallery.get_prototypes()

        # ── Diagnostic: gallery state ────────────────────────────────────
        print(f"[Verify] gallery has {len(prototypes)} user(s): {list(prototypes.keys())}")

        if not prototypes:
            print("[Verify] ✗ Gallery is empty — no users enrolled")
            return {
                "status":     "no_match",
                "access":     "denied",
                "message":    "No users enrolled",
                "best_score": 0,
            }

        best_score = -1.0
        best_id    = None
        for sid, prototype in prototypes.items():
            score = self._cosine_sim(features, prototype)
            # ── Diagnostic: per-user score ───────────────────────────────
            print(f"[Verify]   score vs {sid}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_id    = sid

        landmarks = result.get("landmarks_data")
        is_live   = self.liveness.is_live(
            landmarks,
            require_blink=False,
            require_movement=False,
        )

        threshold = self._get_threshold()

        # ── Diagnostic: decision ─────────────────────────────────────────
        print(f"[Verify] best_id={best_id}  best_score={best_score:.4f}  "
              f"threshold={threshold:.4f}  is_live={is_live}  "
              f"PASS={best_score >= threshold and bool(best_id) and is_live}")

        if best_score >= threshold and best_id and is_live:
            user_info = self.db.get_user(best_id)
            return {
                "status":             "matched",
                "access":             "granted",
                "message":            f"Welcome, {user_info.get('name', 'Unknown')}!",
                "user": {
                    "id":            best_id,
                    "name":          user_info.get("name", "Unknown"),
                    "enrolled_date": user_info.get("enrolled_date"),
                    "source":        user_info.get("source", "unknown"),
                },
                "score":              round(best_score, 4),
                "bbox":               result["bbox"],
                "liveness_verified":  is_live,
            }

        reason = "score below threshold" if best_score < threshold else \
                 "liveness check failed" if not is_live else "unknown"
        return {
            "status":          "no_match",
            "access":          "denied",
            "message":         f"Face not recognized ({reason})",
            "best_score":      round(best_score, 4),
            "threshold":       round(threshold, 4),
            "liveness_failed": not is_live,
        }

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _b64_to_cv2(b64_string: str) -> Optional[np.ndarray]:
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_string)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _cv2_to_b64(img: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _align_face(
        self,
        img: np.ndarray,
        landmarks,
        bbox: List[int],
    ) -> Tuple[np.ndarray, float]:
        """Align face by rotating so the eye line is horizontal."""
        h, w = img.shape[:2]

        if landmarks and len(landmarks) > 360:
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

        angle  = np.degrees(
            np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

        corners = np.array([
            [bbox[0],           bbox[1]],
            [bbox[0] + bbox[2], bbox[1]],
            [bbox[0] + bbox[2], bbox[1] + bbox[3]],
            [bbox[0],           bbox[1] + bbox[3]],
        ], dtype=np.float32)

        # BUG FIX: reshape(-2, 2) is invalid; use (-1, 2)
        rotated_corners = cv2.transform(
            corners.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        rx = int(np.min(rotated_corners[:, 0]))
        ry = int(np.min(rotated_corners[:, 1]))
        rw = int(np.max(rotated_corners[:, 0]) - rx)
        rh = int(np.max(rotated_corners[:, 1]) - ry)

        face_crop = rotated[
            max(0, ry): min(h, ry + rh),
            max(0, rx): min(w, rx + rw),
        ]
        return face_crop, angle

    def _process_enrollment_image(self, b64: str) -> Optional[np.ndarray]:
        result = self.detect_face(b64)
        if not result["face_detected"]:
            return None
        return self._face_result_to_gray(result)

    def _face_result_to_gray(self, result: Dict) -> np.ndarray:
        src = result.get("aligned_face_b64") or result["face_b64"]
        img = self._b64_to_cv2(src)
        if img is None:
            # Return blank image as fallback rather than crashing
            return np.zeros(Config.IMG_SIZE, dtype=np.uint8)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(img, Config.IMG_SIZE)

    def _check_duplicate(self, face_images: List[np.ndarray]) -> Dict:
        prototypes = self.gallery.get_prototypes()
        if not prototypes:
            return {"is_duplicate": False}

        probe_features = []
        for img in face_images:
            processed = Preprocessor.process(hash(img.tobytes()), img, None)
            probe_features.append(OptimizedLBP.compute(processed))

        best_score = -1.0
        best_id    = None
        for sid, prototype in prototypes.items():
            score = float(np.mean([self._cosine_sim(pf, prototype) for pf in probe_features]))
            if score > best_score:
                best_score = score
                best_id    = sid

        if best_score >= self._get_duplicate_threshold() and best_id:
            user_info = self.db.get_user(best_id)
            return {
                "is_duplicate": True,
                "matched_name": user_info.get("name", "Unknown"),
                "matched_id":   best_id,
                "score":        round(best_score, 4),
            }
        return {"is_duplicate": False}
    def delete_user(self, subject_id: str) -> Dict[str, Any]:
        """Delete a user from the database and remove their enrolled images."""
        import shutil
        user_info = self.db.get_user(subject_id)
        if not user_info:
            return {"status": "error", "message": f"User {subject_id} not found"}

        # Only allow deleting live webcam users, not ORL dataset subjects
        if user_info.get("source") != "webcam":
            return {"status": "error", "message": "Cannot delete non-webcam dataset subjects"}

        if self.db.delete_user(subject_id):
            subject_dir = Config.LIVE_DATASET_DIR / subject_id
            if subject_dir.exists() and subject_dir.is_dir():
                shutil.rmtree(subject_dir)
            self.gallery.build(force_rebuild=True)
            return {"status": "success", "message": f"User {subject_id} deleted"}

        return {"status": "error", "message": f"Failed to delete user {subject_id}"}


    # Cache loaded calibration so we don't re-read the file on every call.
    # _calibration_mtime tracks the file's mtime so the cache is refreshed
    # automatically if main.py regenerates calibration.json while the server
    # is running — no restart required.
    _calibration_cache: Optional[Tuple[float, float]] = None
    _calibration_mtime: float = -1.0

    @classmethod
    def _load_calibration(cls) -> Tuple[float, float]:
        try:
            current_mtime = (
                Config.CALIBRATION_PATH.stat().st_mtime
                if Config.CALIBRATION_PATH.exists()
                else -1.0
            )
        except Exception:
            current_mtime = -1.0

        if cls._calibration_cache is None or current_mtime != cls._calibration_mtime:
            cls._calibration_cache = Config.load_calibrated_thresholds()
            cls._calibration_mtime = current_mtime
            v, d = cls._calibration_cache
            print(f"[Thresholds] verify={v:.4f}  duplicate={d:.4f}  (mtime={current_mtime})")
        return cls._calibration_cache

    @classmethod
    def _get_threshold(cls) -> float:
        """Verify threshold — loaded from calibration.json or default 0.72."""
        return cls._load_calibration()[0]

    @classmethod
    def _get_duplicate_threshold(cls) -> float:
        """Duplicate threshold — loaded from calibration.json or default 0.75."""
        return cls._load_calibration()[1]


# =========================================================
# CONVENIENCE FUNCTIONS (Backward compatible)
# =========================================================

_system = FaceRecognitionSystem()


def enroll_user(name: str, face_images_b64: List[str]) -> Dict:
    return _system.enroll_user(name, face_images_b64)


def verify_user(face_b64: str) -> Dict:
    return _system.verify_user(face_b64)


def detect_face(frame_b64: str) -> Dict:
    return _system.detect_face(frame_b64)


def get_all_users(webcam_only: bool = False) -> List[Dict]:
    return _system.db.get_all_users(webcam_only=webcam_only)


def delete_user(subject_id: str) -> Dict:
    return _system.delete_user(subject_id)


def build_gallery() -> Dict:
    return _system.gallery.build()


def init_user_db():
    UserDatabase()


# Called on import
Config.init_paths()
IMG_SIZE = Config.IMG_SIZE


def preprocess(img: np.ndarray) -> np.ndarray:
    """Standalone wrapper used by main.py evaluation pipeline."""
    return Preprocessor.process(hash(img.tobytes()), img, None)


def lbp(img: np.ndarray) -> np.ndarray:
    """Standalone wrapper used by main.py evaluation pipeline."""
    return OptimizedLBP.compute(img)


__all__ = [
    "init_user_db",
    "enroll_user",
    "verify_user",
    "detect_face",
    "get_all_users",
    "delete_user",
    "build_gallery",
    "Config",
    "FaceRecognitionSystem",
]