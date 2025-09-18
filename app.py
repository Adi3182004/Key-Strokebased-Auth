"""
Keystroke-based authentication prototype (Flask).

Usage:
  - Install dependencies:
      pip install flask scikit-learn numpy joblib
  - Run: python app.py
  - Open: http://127.0.0.1:5000

Notes:
  - Enroll at least 5 samples (type the passphrase naturally each time).
  - The server auto-trains and auto-tunes a threshold when there are >= 5 samples.
"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.svm import OneClassSVM
from joblib import dump, load
import time

DATA_DIR = "data"
MODEL_DIR = "models"
PASS_PHRASE = "the quick brown fox"  # Must match client phrase exactly

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, static_folder=".")

# -------- Feature extraction (per-key) --------
def extract_features_from_timings(timings, phrase=PASS_PHRASE):
    entries = []
    for ev in timings:
        typ = ev.get("type")
        key = ev.get("key")
        try:
            t = float(ev.get("time"))
        except Exception:
            continue
        if typ == "down":
            entries.append({"key": key, "down": t, "up": None})
        elif typ == "up":
            for e in reversed(entries):
                if e["key"] == key and e["up"] is None:
                    e["up"] = t
                    break
    entries = [e for e in entries if e.get("down") is not None and e.get("up") is not None]
    if len(entries) == 0:
        return None

    phrase_chars = list(phrase)
    N = len(phrase_chars)
    M = min(len(entries), N)
    dwell_list = [entries[i]["up"] - entries[i]["down"] for i in range(M)]
    down_list = [entries[i]["down"] for i in range(M)]
    flight_list = [down_list[i + 1] - down_list[i] for i in range(max(0, M - 1))]

    pad_dwell = float(np.median(dwell_list)) if len(dwell_list) > 0 else 100.0
    pad_flight = float(np.median(flight_list)) if len(flight_list) > 0 else max(30.0, pad_dwell * 0.5)

    dwell_padded = np.array(dwell_list + [pad_dwell] * (N - M), dtype=float)
    flight_padded = np.array(flight_list + [pad_flight] * ((N - 1) - len(flight_list)), dtype=float)

    mean_dwell, std_dwell = float(np.mean(dwell_padded)), float(np.std(dwell_padded))
    mean_flight, std_flight = float(np.mean(flight_padded)), float(np.std(flight_padded))
    median_dwell, median_flight = float(np.median(dwell_padded)), float(np.median(flight_padded))
    total_time = float(entries[M - 1]["up"] - entries[0]["down"]) if len(entries) >= 1 else float(N * (pad_dwell + pad_flight))

    stats = np.array([mean_dwell, std_dwell, mean_flight, std_flight, total_time, float(N), median_dwell, median_flight], dtype=float)
    feature_vector = np.concatenate([dwell_padded, flight_padded, stats])
    return feature_vector

# -------- File helpers --------
def safe_name(username):
    return "".join(c for c in username if c.isalnum() or c in ("-", "_")).strip() or "user"

def file_for_user(username):
    return os.path.join(DATA_DIR, f"{safe_name(username)}.json")

def model_for_user(username):
    return os.path.join(MODEL_DIR, f"{safe_name(username)}_ocsvm.joblib")

def thresh_for_user(username):
    return os.path.join(MODEL_DIR, f"{safe_name(username)}_thresh.json")

def load_samples(username):
    path = file_for_user(username)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_samples(username, samples):
    path = file_for_user(username)
    with open(path, "w") as f:
        json.dump(samples, f)

def save_threshold(username, threshold):
    with open(thresh_for_user(username), "w") as f:
        json.dump({"threshold": threshold}, f)

def load_threshold(username):
    path = thresh_for_user(username)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f).get("threshold")

# -------- Threshold helper --------
def compute_threshold(train_scores):
    p10 = float(np.percentile(train_scores, 10))
    mean = float(np.mean(train_scores))
    tuned = float((p10 + mean) / 2.0 - 0.1)  # loosened margin
    return tuned

# -------- Routes --------
@app.route("/")
def root():
    return send_from_directory(".", "index.html")

@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.get_json() or {}
    username = data.get("username", "").strip() or "user"
    phrase = data.get("phrase", "")
    events = data.get("events", [])

    if phrase != PASS_PHRASE:
        return jsonify({"success": False, "error": "Passphrase mismatch."}), 400

    features = extract_features_from_timings(events)
    if features is None:
        return jsonify({"success": False, "error": "Feature extraction failed."}), 400

    samples = load_samples(username)
    samples.append(features.tolist())
    save_samples(username, samples)

    trained = False
    if len(samples) >= 5:
        try:
            X = np.array(samples)
            oc = OneClassSVM(gamma="auto", nu=0.1).fit(X)
            dump(oc, model_for_user(username))
            train_scores = oc.decision_function(X)
            tuned_thresh = compute_threshold(train_scores)
            save_threshold(username, tuned_thresh)
            trained = True
            print(f"[TRAIN] user={username} samples={len(samples)}")
            print("    train_scores (first 10):", np.round(train_scores[:10], 4))
            print("    tuned_threshold:", tuned_thresh)
        except Exception as e:
            print("Training error:", e)

    return jsonify({"success": True, "num_samples": len(samples), "trained": trained})

@app.route("/train", methods=["POST"])
def train_route():
    data = request.get_json() or {}
    username = data.get("username", "").strip() or "user"
    samples = load_samples(username)
    if len(samples) < 5:
        return jsonify({"success": False, "error": "Need at least 5 samples.", "num_samples": len(samples)}), 400

    X = np.array(samples)
    clf = OneClassSVM(gamma="auto", nu=0.1).fit(X)
    dump(clf, model_for_user(username))
    train_scores = clf.decision_function(X)
    tuned_thresh = compute_threshold(train_scores)
    save_threshold(username, tuned_thresh)
    print(f"[FORCE-TRAIN] user={username} samples={len(samples)}")
    print("    train_scores (first 10):", np.round(train_scores[:10], 4))
    print("    tuned_threshold:", tuned_thresh)

    return jsonify({"success": True, "num_samples": len(samples), "threshold": tuned_thresh})

@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json() or {}
    username = data.get("username", "").strip() or "user"
    phrase = data.get("phrase", "")
    events = data.get("events", [])

    if phrase != PASS_PHRASE:
        return jsonify({"success": False, "error": "Passphrase mismatch."}), 400

    model_path = model_for_user(username)
    if not os.path.exists(model_path):
        return jsonify({"success": False, "error": "Model not trained yet."}), 400

    features = extract_features_from_timings(events)
    if features is None:
        return jsonify({"success": False, "error": "Could not extract features."}), 400

    clf = load(model_path)
    score = float(clf.decision_function(features.reshape(1, -1))[0])
    user_thresh = load_threshold(username)
    if user_thresh is None:
        user_thresh = -0.3

    margin = 0.1  # allow small deviation
    is_genuine = score >= (user_thresh - margin)

    return jsonify(
        {
            "success": True,
            "genuine": bool(is_genuine),
            "score": score,
            "threshold": user_thresh,
        }
    )

@app.route("/status/<username>", methods=["GET"])
def status(username):
    samples = load_samples(username)
    model_exists = os.path.exists(model_for_user(username))
    return jsonify({"num_samples": len(samples), "model_trained": model_exists, "threshold": load_threshold(username)})

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    username = data.get("username", "").strip() or "user"
    for path in [file_for_user(username), model_for_user(username), thresh_for_user(username)]:
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"success": True})

# -------- Test Script --------
def simulate_typing(base_dwell=100, base_flight=80, noise=15):
    """Simulate one sample of typing events for the passphrase."""
    events = []
    t = 0.0
    for c in PASS_PHRASE:
        dwell = base_dwell + np.random.randn() * noise
        events.append({"type": "down", "key": c, "time": t})
        t += dwell
        events.append({"type": "up", "key": c, "time": t})
        flight = base_flight + np.random.randn() * noise
        t += flight
    return events

def run_test():
    username = "alice"
    print("=== Simulating enrollment ===")
    # Reset previous data
    for path in [file_for_user(username), model_for_user(username), thresh_for_user(username)]:
        if os.path.exists(path):
            os.remove(path)

    # Enroll 5 genuine samples
    for i in range(5):
        events = simulate_typing(base_dwell=100, base_flight=80, noise=10)
        features = extract_features_from_timings(events)
        samples = load_samples(username)
        samples.append(features.tolist())
        save_samples(username, samples)
    # Train model
    X = np.array(load_samples(username))
    clf = OneClassSVM(gamma="auto", nu=0.1).fit(X)
    dump(clf, model_for_user(username))
    train_scores = clf.decision_function(X)
    tuned_thresh = compute_threshold(train_scores)
    save_threshold(username, tuned_thresh)
    print(f"Model trained. Threshold: {tuned_thresh:.4f}")

    # Test genuine sample
    genuine_events = simulate_typing(base_dwell=100, base_flight=80, noise=10)
    genuine_features = extract_features_from_timings(genuine_events)
    score = float(clf.decision_function(genuine_features.reshape(1, -1))[0])
    print(f"Genuine test score: {score:.4f} -> {'Genuine' if score>=tuned_thresh else 'Imposter'}")

    # Test imposter sample
    imposter_events = simulate_typing(base_dwell=130, base_flight=120, noise=20)
    imposter_features = extract_features_from_timings(imposter_events)
    score = float(clf.decision_function(imposter_features.reshape(1, -1))[0])
    print(f"Imposter test score: {score:.4f} -> {'Genuine' if score>=tuned_thresh else 'Imposter'}")

if __name__ == "__main__":
    # Start Flask app
    from threading import Thread
    def run_flask():
        app.run(debug=True, use_reloader=False)
    t = Thread(target=run_flask)
    t.start()
    # Give server a moment
    time.sleep(1)
    # Run test
    run_test()
