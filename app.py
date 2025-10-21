# app.py
# Keystroke Biometric Banking with Password Strength, Deposits, and History

import os
import re
import json
import hashlib
import secrets
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__, static_folder=".")
CORS(app)

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
USER_DIR = "users"
for directory in [DATA_DIR, MODEL_DIR, USER_DIR]:
    os.makedirs(directory, exist_ok=True)

# Typing threshold
MIN_SPEED_RATIO = 0.50

# Helpers
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def safe_username(username):
    return "".join(c for c in username if c.isalnum() or c in ("-", "_")).strip() or "user"

def user_file(username):
    return os.path.join(USER_DIR, f"{safe_username(username)}.json")

def model_file(username):
    return os.path.join(MODEL_DIR, f"{safe_username(username)}_model.joblib")

def scaler_file(username):
    return os.path.join(MODEL_DIR, f"{safe_username(username)}_scaler.joblib")

def training_data_file(username):
    return os.path.join(DATA_DIR, f"{safe_username(username)}_training.json")

def baseline_file(username):
    return os.path.join(DATA_DIR, f"{safe_username(username)}_baseline.json")

def load_user(username):
    path = user_file(username)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def save_user(username, data):
    with open(user_file(username), "w") as f:
        json.dump(data, f, indent=2)

def user_exists(username):
    return os.path.exists(user_file(username))

def append_transaction(username, tx):
    user = load_user(username)
    if not user:
        return False
    txs = user.get("transactions", [])
    txs.append(tx)
    user["transactions"] = txs
    save_user(username, user)
    return True

def calculate_typing_speed(events):
    if not events or len(events) < 2:
        return None
    events = sorted(events, key=lambda x: x.get('time', 0))
    down_events = [e for e in events if e.get('type') == 'down']
    if len(down_events) < 2:
        return None
    total_time = down_events[-1]['time'] - down_events[0]['time']
    num_chars = len(down_events)
    if total_time <= 0:
        return None
    chars_per_sec = (num_chars / total_time) * 1000
    return {'chars_per_sec': chars_per_sec, 'total_time': total_time, 'num_chars': num_chars}

# Password strength: require 8+ chars with uppercase, lowercase, number, and special
def password_strength(pw):
    if not pw:
        return "weak"
    length = len(pw)
    lower = bool(re.search(r"[a-z]", pw))
    upper = bool(re.search(r"[A-Z]", pw))
    digit = bool(re.search(r"\d", pw))
    special = bool(re.search(r"[^A-Za-z0-9]", pw))
    classes = sum([lower, upper, digit, special])
    if length >= 12 and classes >= 4:
        return "good"
    if length >= 8 and classes >= 4:
        return "medium"
    return "weak"

# Routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# Auth
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    name = data.get("name", "")
    email = data.get("email", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400
    if user_exists(username):
        return jsonify({"success": False, "error": "Username already exists"}), 400

    strength = password_strength(password)
    if strength == "weak":
        return jsonify({"success": False, "error": "Password must be 8+ chars and include uppercase, lowercase, number, and special character."}), 400

    user_data = {
        "username": username,
        "login_password_hash": hash_password(password),
        "transaction_password_hash": None,
        "name": name,
        "email": email,
        "biometric_trained": False,
        "total_samples": 0,
        "transactions": [],
        "created_at": datetime.now().isoformat()
    }
    save_user(username, user_data)
    return jsonify({"success": True})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    login_hash = user.get("login_password_hash") or user.get("password_hash")
    if not login_hash or login_hash != hash_password(password):
        return jsonify({"success": False, "error": "Invalid password"}), 401

    return jsonify({
        "success": True,
        "biometric_trained": user.get("biometric_trained", False),
        "total_samples": user.get("total_samples", 0)
    })

@app.route("/get_training_status", methods=["GET"])
def get_training_status():
    username = request.args.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "error": "Username required"}), 400
    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    return jsonify({
        "success": True,
        "trained": user.get("biometric_trained", False),
        "total_samples": user.get("total_samples", 0)
    })

# Training (transaction password + baseline)
@app.route("/train_biometric", methods=["POST"])
def train_biometric():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    para1_events = data.get("paragraph1_events", [])
    para2_events = data.get("paragraph2_events", [])
    password_samples = data.get("password_samples", [])

    if not username:
        return jsonify({"success": False, "error": "Username required"}), 400

    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    if password_strength(password) == "weak":
        return jsonify({"success": False, "error": "Transaction password must be 8+ chars and include uppercase, lowercase, number, and special character."}), 400

    existing_data = {}
    if os.path.exists(training_data_file(username)):
        with open(training_data_file(username), "r") as f:
            existing_data = json.load(f)

    all_para1 = existing_data.get("paragraph1_events", []) + [para1_events]
    all_para2 = existing_data.get("paragraph2_events", []) + [para2_events]
    all_password = existing_data.get("password_samples", []) + password_samples

    all_speeds = []
    for para_events in all_para1:
        speed = calculate_typing_speed(para_events)
        if speed:
            all_speeds.append(speed['chars_per_sec'])
    for para_events in all_para2:
        speed = calculate_typing_speed(para_events)
        if speed:
            all_speeds.append(speed['chars_per_sec'])
    for pwd_events in all_password:
        speed = calculate_typing_speed(pwd_events)
        if speed:
            all_speeds.append(speed['chars_per_sec'])

    if len(all_speeds) < 7:
        return jsonify({"success": False, "error": "Insufficient training data"}), 400

    baseline_speed = float(np.mean(all_speeds))
    baseline_data = {'chars_per_sec': baseline_speed, 'trained_at': datetime.now().isoformat(), 'total_samples': len(all_speeds)}
    with open(baseline_file(username), "w") as f:
        json.dump(baseline_data, f, indent=2)

    training_data = {"paragraph1_events": all_para1, "paragraph2_events": all_para2, "password_samples": all_password, "trained_at": datetime.now().isoformat()}
    with open(training_data_file(username), "w") as f:
        json.dump(training_data, f)

    user["biometric_trained"] = True
    user["transaction_password_hash"] = hash_password(password)
    user["total_samples"] = len(all_speeds)
    save_user(username, user)

    return jsonify({"success": True, "total_samples": len(all_speeds), "baseline_speed": baseline_speed})

# Transfer Verification (biometrics)
@app.route("/verify_transaction", methods=["POST"])
def verify_transaction():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    keystroke_events = data.get("keystroke_events", [])
    amount = data.get("amount", "")
    recipient = data.get("recipient", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    if not user.get("transaction_password_hash"):
        return jsonify({"success": True, "authorized": False, "impostor": False, "error": "Transaction password not set. Please train biometrics."}), 200

    if user["transaction_password_hash"] != hash_password(password):
        return jsonify({"success": True, "authorized": False, "impostor": False, "error": "Invalid transaction password"}), 200

    if not user.get("biometric_trained", False):
        return jsonify({"success": True, "authorized": False, "impostor": False, "error": "Biometric not set up"}), 200

    if not os.path.exists(baseline_file(username)):
        return jsonify({"success": True, "authorized": False, "impostor": False, "error": "Baseline not found"}), 200

    with open(baseline_file(username), "r") as f:
        baseline = json.load(f)

    current_speed_data = calculate_typing_speed(keystroke_events)
    if not current_speed_data:
        return jsonify({"success": True, "authorized": False, "impostor": False, "error": "Could not analyze typing"}), 200

    baseline_speed = baseline['chars_per_sec']
    current_speed = current_speed_data['chars_per_sec']
    speed_ratio = current_speed / baseline_speed if baseline_speed > 0 else 0
    speed_deviation = ((baseline_speed - current_speed) / baseline_speed * 100) if baseline_speed > 0 else 0

    txn_id = "TXN" + secrets.token_hex(6).upper()

    if speed_ratio >= MIN_SPEED_RATIO:
        tx = {
            "id": txn_id,
            "type": "transfer",
            "recipient": recipient,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "impostor": False,
            "details": {
                "speed_ratio": round(float(speed_ratio), 3),
                "speed_deviation": round(float(speed_deviation), 1),
                "baseline_speed": round(float(baseline_speed), 2),
                "auth_speed": round(float(current_speed), 2)
            }
        }
        append_transaction(username, tx)
        return jsonify({"success": True, "authorized": True, "impostor": False, "transaction_id": txn_id, "speed_ratio": float(speed_ratio)})
    else:
        tx = {
            "id": txn_id,
            "type": "transfer",
            "recipient": recipient,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
            "status": "blocked",
            "impostor": True,
            "details": {
                "reason": "Typing speed too slow",
                "speed_deviation": round(float(speed_deviation), 1),
                "auth_speed": round(float(current_speed), 2),
                "baseline_speed": round(float(baseline_speed), 2)
            }
        }
        append_transaction(username, tx)
        return jsonify({
            "success": True,
            "authorized": False,
            "impostor": True,
            "transaction_id": txn_id,
            "error": "IMPOSTOR DETECTED: Typing speed too slow",
            "speed_deviation": round(float(speed_deviation), 1),
            "auth_speed": round(float(current_speed), 2),
            "training_speed": round(float(baseline_speed), 2)
        }), 200

# Deposit (login password only)
@app.route("/deposit", methods=["POST"])
def deposit():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    amount = data.get("amount", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    login_hash = user.get("login_password_hash") or user.get("password_hash")
    if not login_hash or login_hash != hash_password(password):
        return jsonify({"success": False, "error": "Invalid login password"}), 200

    txn_id = "TXN" + secrets.token_hex(6).upper()
    tx = {
        "id": txn_id,
        "type": "deposit",
        "recipient": "My SecureBank Account",
        "amount": amount,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "impostor": False,
        "details": {"method": "login-password"}
    }
    append_transaction(username, tx)

    return jsonify({"success": True, "transaction_id": txn_id})

# History
@app.route("/transaction_history", methods=["GET"])
def transaction_history():
    username = request.args.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "error": "Username required"}), 400
    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    txs = user.get("transactions", [])
    txs_sorted = sorted(txs, key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"success": True, "transactions": txs_sorted})

# Reset biometric
@app.route("/reset_biometric", methods=["POST"])
def reset_biometric():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "error": "Username required"}), 400

    user = load_user(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    for f in [model_file(username), scaler_file(username), training_data_file(username), baseline_file(username)]:
        if os.path.exists(f):
            os.remove(f)

    user["biometric_trained"] = False
    user["total_samples"] = 0
    user["transaction_password_hash"] = None
    save_user(username, user)

    return jsonify({"success": True})

if __name__ == "__main__":
    print("=" * 80)
    print("🏦 SecureBank - Keystroke Biometric System (₹, Deposits & History)")
    print("=" * 80)
    print("🚀 Server: http://127.0.0.1:5000")
    print()
    app.run(debug=True, host="127.0.0.1", port=5000)