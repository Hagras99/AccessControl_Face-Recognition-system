from flask import Flask, render_template, jsonify

# Import the refactored pipeline from main.py
from main import run_pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/run', methods=['POST'])
def run_biometrics():
    try:
        results = run_pipeline()
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
