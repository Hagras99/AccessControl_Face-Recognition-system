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
=============================================================
"""

from flask import Flask, render_template, jsonify, request

# Import the refactored pipeline from main.py
from main import run_pipeline
from enrollment import (
    init_user_db, detect_face, enroll_user,
    verify_user, get_all_users
)

app = Flask(__name__)

# Initialize user database on startup
init_user_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/run', methods=['POST'])
def run_biometrics():
    """Run the full PCA/LBP evaluation pipeline."""
    try:
        results = run_pipeline()
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """Enroll a new user with face images."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        name = data.get("name", "").strip()
        images = data.get("images", [])

        if not name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        if len(images) < 1:
            return jsonify({"status": "error", "message": "At least 1 face image required"}), 400

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


@app.route('/api/verify', methods=['POST'])
def api_verify():
    """Verify a face against all enrolled subjects."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        image = data.get("image", "")
        if not image:
            return jsonify({"status": "error", "message": "Image is required"}), 400

        result = verify_user(image)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Real-time face detection on a single frame."""
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
            "bbox": result["bbox"],
        })

    except Exception as e:
        return jsonify({"face_detected": False, "error": str(e)}), 500


@app.route('/api/users', methods=['GET'])
def api_users():
    """List all enrolled users."""
    try:
        users = get_all_users()
        return jsonify({"status": "success", "users": users})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
