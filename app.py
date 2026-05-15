"""
=============================================================
  Flask Server — Biometric Face Recognition Access Control

  Routes:
    GET    /                  -> Render SPA
    POST   /api/run           -> Run PCA/LBP evaluation pipeline
    POST   /api/enroll        -> Enroll new user (9 poses)
    POST   /api/verify        -> Verify face (1:N identification)
    POST   /api/detect        -> Real-time face detection
    GET    /api/users         -> List verifiable (webcam) users
    DELETE /api/users/<id>    -> Remove an enrolled user
=============================================================
"""

from flask import Flask, render_template, jsonify, request

from main import run_pipeline
from enrollment import (
    init_user_db, detect_face, enroll_user,
    verify_user, get_all_users, delete_user,
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
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        # Multi-frame path — run verification on every frame, then use
        # majority-vote on the predicted identity and median score for that
        # identity.  Taking the raw maximum score is exploitable by a single
        # lucky outlier frame; median + majority-vote is far more robust.
        if "images" in data and isinstance(data["images"], list):
            frames = data["images"]
            if not frames:
                return jsonify({"status": "error",
                                "message": "images list is empty"}), 400

            from collections import Counter
            import statistics as _stats

            candidates = []          # list of (predicted_id | None, score, result)
            last_result = None
            for frame_b64 in frames:
                r = verify_user(frame_b64)
                last_result = r
                if r.get("status") == "matched" and "user" in r:
                    candidates.append((r["user"]["id"], r.get("score", 0.0), r))
                else:
                    candidates.append((None, r.get("best_score", 0.0), r))

            # Majority vote on predicted identity
            matched_ids = [sid for sid, _, _ in candidates if sid is not None]
            if matched_ids:
                top_id = Counter(matched_ids).most_common(1)[0][0]
                top_scores = [s for sid, s, _ in candidates if sid == top_id]
                median_score = _stats.median(top_scores)
                # Pick the individual result whose score is closest to median
                result = min(
                    [(sid, s, r) for sid, s, r in candidates if sid == top_id],
                    key=lambda t: abs(t[1] - median_score),
                )[2]
                # Overwrite the score field so the frontend sees the median
                result = dict(result)
                result["score"] = round(median_score, 4)
            else:
                # No frame produced a match — return the last denial result
                result = last_result or {"status": "no_face", "access": "denied",
                                         "message": "No face detected in any frame"}

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
            "bbox":          result.get("bbox"),   # None when no face — avoids KeyError
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