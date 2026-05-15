"""
=============================================================
  Flask Server — Biometric Face Recognition Access Control

  Routes:
    GET  /              → Render SPA
    POST /api/run       → Run PCA/LBP evaluation pipeline
    POST /api/enroll    → Enroll new user (9 poses)
    POST /api/verify    → Verify face (1:N identification)
    POST /api/detect    → Real-time face detection
    GET  /api/users     → List all enrolled users

  FIX (verify route) — now accepts multi-frame input.
  Send {"images": ["<b64>", "<b64>", ...]} for a more robust
  decision (mean score across frames), or the legacy
  {"image": "<b64>"} for a single-frame check.
=============================================================
"""

from flask import Flask, render_template, jsonify, request

from main import run_pipeline
from enrollment import (
    init_user_db, detect_face, enroll_user,
    verify_user, get_all_users,
)

app = Flask(__name__)

# Initialise user database on startup
init_user_db()


# ── Home ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Evaluation pipeline ───────────────────────────────────
@app.route("/api/run", methods=["POST"])
def run_biometrics():
    """Run the full PCA/LBP evaluation pipeline on the ORL dataset."""
    try:
        results = run_pipeline()
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Enroll ────────────────────────────────────────────────
@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """
    Enroll a new user with face images.

    Expected JSON body:
        {
            "name":   "Karim",
            "images": ["<base64>", "<base64>", ...]   // 9 recommended
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        name   = data.get("name", "").strip()
        images = data.get("images", [])

        if not name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        if len(images) < 1:
            return jsonify({"status": "error",
                            "message": "At least 1 face image required"}), 400

        result = enroll_user(name, images)

        if result["status"] == "duplicate":
            return jsonify(result), 409
        elif result["status"] == "error":
            return jsonify(result), 400
        else:
            return jsonify(result), 201

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Verify ────────────────────────────────────────────────
@app.route("/api/verify", methods=["POST"])
def api_verify():
    """
    Verify a face against all enrolled subjects (1:N identification).

    Accepts two JSON formats:

    Single-frame (legacy):
        { "image": "<base64>" }

    Multi-frame (recommended — more robust decision):
        { "images": ["<base64>", "<base64>", "<base64>"] }

    With multi-frame input, verify_user() extracts features from every
    frame in which a face is detected, then scores each enrolled subject
    as the mean similarity across all (probe_frame × gallery_image) pairs.
    This prevents one blurry or accidentally similar frame from granting
    or denying access on its own.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        # Multi-frame path
        if "images" in data and isinstance(data["images"], list):
            payload = data["images"]
            if not payload:
                return jsonify({"status": "error",
                                "message": "images list is empty"}), 400

        # Single-frame legacy path
        else:
            payload = data.get("image", "")
            if not payload:
                return jsonify({"status": "error",
                                "message": "image or images field is required"}), 400

        result = verify_user(payload)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Real-time detection ───────────────────────────────────
@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Detect a face in a single webcam frame.
    Used by the frontend to draw the live bounding-box overlay.

    Expected JSON body:
        { "image": "<base64>" }

    Returns:
        { "face_detected": true/false, "bbox": [x, y, w, h] | null }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"face_detected": False}), 400

        image = data.get("image", "")
        if not image:
            return jsonify({"face_detected": False}), 400

        result = detect_face(image)
        return jsonify({
            "face_detected": result["face_detected"],
            "bbox":          result["bbox"],
        })

    except Exception as e:
        return jsonify({"face_detected": False, "error": str(e)}), 500


# ── Users list ────────────────────────────────────────────
@app.route("/api/users", methods=["GET"])
def api_users():
    """Return all enrolled users from the database."""
    try:
        users = get_all_users()
        return jsonify({"status": "success", "users": users})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)