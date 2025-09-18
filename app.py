"""
Keystroke-based authentication prototype (Flask).

Usage:
  - Run: python app.py
  - Open: http://127.0.0.1:5000 (index.html served)

This app:
 - stores raw timing samples per username
 - after 5 enrollment samples it trains an OneClassSVM (novelty detection)
 - autosaves a per-user scaler + threshold from training scores
 - verifies incoming attempts against the trained model + threshold
 - stricter settings: smaller nu, tighter thresholding, per-user normalization
"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.svm import OneClassSVM
from joblib import dump, load

DATA_DIR = "data"
MODEL_DIR = "models"
PASS_PHRASE = "the quick brown fox"  # Must match client phrase exactly

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, static_folder=".")

# -------- Helpers for file paths --------
def safe_name(username):
    return "".join(c for c in username if c.isalnum() or c in ("-", "_")).strip() or "user"

def file_for_user(username):
    return os.path.join(DATA_DIR, f"{safe_name(username)}.json")

def model_for_user(username):
    return os.path.join(MODEL_DIR, f"{safe_name(username)}_ocsvm.joblib")

def scaler_for_user(username):
    return os.path.join(MODEL_DIR, f"{safe_name(username)}_scaler.json")

def thresh_for_user(username):
    return os.path.join(MODEL_DIR, f"{safe_name(username)}_thresh.json")

# -------- Data persistence --------
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
        json.dump({"threshold": float(threshold)}, f)

def load_threshold(username):
    path = thresh_for_user(username)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f).get("threshold")

def save_scaler(username, mean_vec, std_vec):
    with open(scaler_for_user(username), "w") as f:
        json.dump({"mean": [float(x) for x in mean_vec], "std": [float(x) for x in std_vec]}, f)

def load_scaler(username):
    path = scaler_for_user(username)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        d = json.load(f)
        return np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)

# -------- Feature extraction (fixed-length + stats) --------
def extract_features_from_timings(timings, phrase=PASS_PHRASE):
    """
    Build a compact feature vector from raw down/up event timings.
    - captures per-key dwell times and flight times (padded/truncated to phrase length)
    - includes summary statistics (mean/std/min/max/median) and total_time
    Returns: 1D numpy array of floats or None on failure
    """
    # collect paired down/up by order of events (browser sends performance.now())
    entries = []
    for ev in timings:
        typ = ev.get("type")
        key = ev.get("key")
        try:
            t = float(ev.get("time"))
        except Exception:
            continue
        entries.append({"type": typ, "key": key, "time": t})

    # pair down/up by key in sequence (tolerant: match last unmatched down for same key)
    downs, ups = [], []
    stack = []
    for e in entries:
        if e["type"] == "down":
            stack.append(e)
        elif e["type"] == "up":
            # match to last unmatched down for same key if possible
            match_idx = None
            for i in range(len(stack)-1, -1, -1):
                if stack[i]["key"] == e["key"]:
                    match_idx = i
                    break
            if match_idx is not None:
                d = stack.pop(match_idx)
                downs.append(d["time"])
                ups.append(e["time"])
            else:
                # unmatched up -> skip
                continue

    N = len(list(phrase))
    n = min(len(downs), len(ups), N)
    if n == 0:
        return None

    # compute dwell and flight for the first n characters
    dwell = np.array([ups[i] - downs[i] for i in range(n)], dtype=float)
    # flight: time from down[i] to down[i+1]
    if n >= 2:
        flight = np.array([downs[i+1] - downs[i] for i in range(n-1)], dtype=float)
    else:
        flight = np.array([0.0], dtype=float)

    # pad to fixed length using medians to keep scale similar
    pad_dwell = float(np.median(dwell)) if dwell.size > 0 else 50.0
    pad_flight = float(np.median(flight)) if flight.size > 0 else 30.0

    dwell_full = np.concatenate([dwell, np.full(N - n, pad_dwell)]) if N > n else dwell[:N]
    flight_full = np.concatenate([flight, np.full(max(0, (N - 1) - len(flight)), pad_flight)]) if (N - 1) > len(flight) else flight[: max(0, N-1)]

    # summary statistics (dwell)
    mean_dwell = float(np.mean(dwell_full))
    std_dwell = float(np.std(dwell_full))
    min_dwell = float(np.min(dwell_full))
    max_dwell = float(np.max(dwell_full))
    median_dwell = float(np.median(dwell_full))

    # summary statistics (flight)
    mean_flight = float(np.mean(flight_full))
    std_flight = float(np.std(flight_full))
    min_flight = float(np.min(flight_full))
    max_flight = float(np.max(flight_full))
    median_flight = float(np.median(flight_full))

    total_time = float((ups[n-1] - downs[0]) if n >= 1 else 0.0)
    avg_per_key = total_time / float(n) if n > 0 else 0.0

    # Build vector: per-key dwell (N), per-key flight (N-1), then stats (a few)
    stats = [
        mean_dwell, std_dwell, min_dwell, max_dwell, median_dwell,
        mean_flight, std_flight, min_flight, max_flight, median_flight,
        total_time, avg_per_key, float(n)
    ]

    vec = np.concatenate([dwell_full, flight_full, np.array(stats, dtype=float)])
    return vec

# -------- Threshold computation (stricter) --------
def compute_threshold_from_scores(scores):
    # pick the 10th percentile of genuine scores as cutoff
    return float(np.percentile(scores, 10))


# -------- Routes --------
@app.route("/")
def root():
    # serve index.html from same directory
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
            X = np.array(samples, dtype=float)
            # compute per-feature mean/std for scaling and save (raw)
            mean_vec = np.mean(X, axis=0)
            std_vec = np.std(X, axis=0)
            std_vec[std_vec == 0.0] = 1.0
            save_scaler(username, mean_vec, std_vec)

            # normalize before training
            X_norm = (X - mean_vec) / std_vec

            # stricter OCSVM: smaller nu (fewer allowed outliers), gamma auto
            oc = OneClassSVM(gamma="auto", nu=0.05).fit(X_norm)
            dump(oc, model_for_user(username))

            train_scores = oc.decision_function(X_norm)
            tuned_thresh = compute_threshold_from_scores(train_scores)
            save_threshold(username, tuned_thresh)
            trained = True

            # optionally print diagnostic info
            print(f"[TRAIN] user={username} samples={len(samples)}")
            print("  train_scores (first 10):", np.round(train_scores[:10], 4))
            print("  tuned_threshold:", tuned_thresh)
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

    X = np.array(samples, dtype=float)
    mean_vec = np.mean(X, axis=0)
    std_vec = np.std(X, axis=0)
    std_vec[std_vec == 0.0] = 1.0
    save_scaler(username, mean_vec, std_vec)
    X_norm = (X - mean_vec) / std_vec

    clf = OneClassSVM(gamma="auto", nu=0.05).fit(X_norm)
    dump(clf, model_for_user(username))

    train_scores = clf.decision_function(X_norm)
    tuned_thresh = compute_threshold_from_scores(train_scores)
    save_threshold(username, tuned_thresh)

    print(f"[FORCE-TRAIN] user={username} samples={len(samples)}")
    print("  train_scores (first 10):", np.round(train_scores[:10], 4))
    print("  tuned_threshold:", tuned_thresh)

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
    scaler = load_scaler(username)
    if not os.path.exists(model_path) or scaler is None:
        return jsonify({"success": False, "error": "Model not trained yet."}), 400

    features = extract_features_from_timings(events)
    if features is None:
        return jsonify({"success": False, "error": "Could not extract features."}), 400

    # normalize with user's saved scaler (mean_vec/std_vec are raw)
    mean_vec, std_vec = scaler
    std_vec[std_vec == 0.0] = 1.0
    feat_norm = (features - mean_vec) / std_vec

    clf = load(model_path)
    score = float(clf.decision_function(feat_norm.reshape(1, -1))[0])
    user_thresh = load_threshold(username)
    if user_thresh is None:
        # fallback conservative default
        user_thresh = -0.5

    # stricter decision: must be >= threshold
    is_genuine = score >= user_thresh

    # --- corrected dwell-mean check ---
    # compute mean of per-key dwell values for the attempt (first N entries)
    N = len(PASS_PHRASE)
    # ensure we have at least N dwell entries in the features vector
    if len(features) >= N:
        dwell_mean_feat = float(np.mean(features[:N]))
        # compute training aggregate dwell mean/std from mean_vec/std_vec (first N positions)
        train_dwell_mean = float(np.mean(mean_vec[:N]))
        train_dwell_std = float(np.mean(std_vec[:N]))
        if train_dwell_std <= 0:
            train_dwell_std = 1.0
        # if attempt's mean dwell deviates more than 2 * training dwell std -> imposter
        if abs(dwell_mean_feat - train_dwell_mean) > 2.0 * train_dwell_std:
            is_genuine = False

    # additional defensive check: if total_time differs greatly from training mean, mark imposter
    # total_time is placed in stats near the end: stats contains [ ..., total_time, avg_per_key, n]
    # so total_time index = -3
    total_time_original = float(features[-3]) if len(features) >= 3 else 0.0
    try:
        train_total_mean = float(mean_vec[-3])
        train_total_std = float(std_vec[-3])
        if train_total_std <= 0:
            train_total_std = 1.0
        # if total_time is very different (>1.5 std), mark as imposter
        if abs(total_time_original - train_total_mean) > 1.5 * train_total_std:
            is_genuine = False
    except Exception:
        # if indexing fails or scaler missing elements, ignore this check
        pass

    return jsonify({
        "success": True,
        "genuine": bool(is_genuine),
        "score": score,
        "threshold": user_thresh
    })

@app.route("/status/<username>", methods=["GET"])
def status(username):
    samples = load_samples(username)
    model_exists = os.path.exists(model_for_user(username))
    return jsonify({
        "num_samples": len(samples),
        "model_trained": model_exists,
        "threshold": load_threshold(username)
    })

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    username = data.get("username", "").strip() or "user"
    for path in [file_for_user(username), model_for_user(username), thresh_for_user(username), scaler_for_user(username)]:
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"success": True})

if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True)
