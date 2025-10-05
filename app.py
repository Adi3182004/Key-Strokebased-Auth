"""
Simplified Keystroke Biometric Banking System
- SIMPLE LOGIC: Slow = Impostor, Normal/Fast/Very Fast = Success
- VERY LENIENT: 50% minimum speed (huge buffer)
- Real bank alarm for imposters
- Train more feature (keeps previous samples)
"""

import os
import json
import hashlib
import secrets
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from datetime import datetime

app = Flask(__name__, static_folder=".")
CORS(app)

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
USER_DIR = "users"

for directory in [DATA_DIR, MODEL_DIR, USER_DIR]:
    os.makedirs(directory, exist_ok=True)

# SIMPLE THRESHOLD - Very Lenient
MIN_SPEED_RATIO = 0.50  # Only need 50% of training speed (VERY LENIENT)

# Helper Functions
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

# User Management
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

# Typing Speed Calculation
def calculate_typing_speed(events):
    """Simple speed calculation"""
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
    
    return {
        'chars_per_sec': chars_per_sec,
        'total_time': total_time,
        'num_chars': num_chars
    }

# Routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

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
    
    user_data = {
        "username": username,
        "password_hash": hash_password(password),
        "name": name,
        "email": email,
        "biometric_trained": False,
        "total_samples": 0,
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
    
    if user["password_hash"] != hash_password(password):
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
    
    # Load existing training data if any
    existing_data = {}
    if os.path.exists(training_data_file(username)):
        with open(training_data_file(username), "r") as f:
            existing_data = json.load(f)
    
    # Combine new samples with existing
    all_para1 = existing_data.get("paragraph1_events", []) + [para1_events]
    all_para2 = existing_data.get("paragraph2_events", []) + [para2_events]
    all_password = existing_data.get("password_samples", []) + password_samples
    
    # Calculate baseline from ALL samples
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
    
    baseline_speed = np.mean(all_speeds)
    
    # Save baseline
    baseline_data = {
        'chars_per_sec': float(baseline_speed),
        'trained_at': datetime.now().isoformat(),
        'total_samples': len(all_speeds)
    }
    
    with open(baseline_file(username), "w") as f:
        json.dump(baseline_data, f, indent=2)
    
    # Save all training data
    training_data = {
        "paragraph1_events": all_para1,
        "paragraph2_events": all_para2,
        "password_samples": all_password,
        "trained_at": datetime.now().isoformat()
    }
    
    with open(training_data_file(username), "w") as f:
        json.dump(training_data, f)
    
    # Update user
    user["biometric_trained"] = True
    user["password_hash"] = hash_password(password)
    user["total_samples"] = len(all_speeds)
    save_user(username, user)
    
    print(f"\n{'='*80}")
    print(f"[TRAINING] User: {username}")
    print(f"Total Samples: {len(all_speeds)}")
    print(f"Baseline Speed: {baseline_speed:.2f} chars/sec")
    print(f"Minimum Required: {baseline_speed * MIN_SPEED_RATIO:.2f} chars/sec ({MIN_SPEED_RATIO*100}%)")
    print(f"{'='*80}\n")
    
    return jsonify({
        "success": True,
        "total_samples": len(all_speeds),
        "baseline_speed": float(baseline_speed)
    })

@app.route("/verify_transaction", methods=["POST"])
def verify_transaction():
    """
    SIMPLE LOGIC:
    - If speed >= 50% of baseline → SUCCESS (normal/fast/very fast = OK)
    - If speed < 50% of baseline → IMPOSTOR + ALARM
    """
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
    
    # Check password
    if user["password_hash"] != hash_password(password):
        print(f"[VERIFY] {username}: ❌ Password mismatch")
        return jsonify({
            "success": True,
            "authorized": False,
            "impostor": False,
            "error": "Invalid password"
        }), 200
    
    # Check biometric trained
    if not user.get("biometric_trained", False):
        print(f"[VERIFY] {username}: ⚠️  Not trained")
        return jsonify({
            "success": True,
            "authorized": False,
            "impostor": False,
            "error": "Biometric not set up"
        }), 200
    
    # Load baseline
    if not os.path.exists(baseline_file(username)):
        return jsonify({
            "success": True,
            "authorized": False,
            "impostor": False,
            "error": "Baseline not found"
        }), 200
    
    with open(baseline_file(username), "r") as f:
        baseline = json.load(f)
    
    # Calculate current speed
    current_speed_data = calculate_typing_speed(keystroke_events)
    if not current_speed_data:
        print(f"[VERIFY] {username}: ❌ Could not calculate speed")
        return jsonify({
            "success": True,
            "authorized": False,
            "impostor": False,
            "error": "Could not analyze typing"
        }), 200
    
    baseline_speed = baseline['chars_per_sec']
    current_speed = current_speed_data['chars_per_sec']
    speed_ratio = current_speed / baseline_speed if baseline_speed > 0 else 0
    speed_deviation = ((baseline_speed - current_speed) / baseline_speed * 100) if baseline_speed > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"[VERIFY] User: {username}")
    print(f"Password: ✓ Valid")
    print(f"Baseline Speed: {baseline_speed:.2f} chars/sec")
    print(f"Current Speed: {current_speed:.2f} chars/sec")
    print(f"Speed Ratio: {speed_ratio:.2f} ({speed_deviation:+.1f}% deviation)")
    print(f"Minimum Required: {MIN_SPEED_RATIO*100}%")
    
    # SIMPLE CHECK: Is typing speed >= 50% of baseline?
    if speed_ratio >= MIN_SPEED_RATIO:
        # SUCCESS - Normal/Fast/Very Fast typing
        print(f"✅ AUTHORIZED - Typing speed acceptable")
        print(f"{'='*80}\n")
        
        transaction_id = "TXN" + secrets.token_hex(6).upper()
        
        return jsonify({
            "success": True,
            "authorized": True,
            "impostor": False,
            "transaction_id": transaction_id,
            "speed_ratio": float(speed_ratio)
        })
    else:
        # IMPOSTOR - Too slow
        print(f"🚨 IMPOSTOR DETECTED - Typing TOO SLOW!")
        print(f"   Speed ratio: {speed_ratio:.2f} < {MIN_SPEED_RATIO}")
        print(f"   Deviation: {speed_deviation:.1f}% slower")
        print(f"{'='*80}\n")
        
        return jsonify({
            "success": True,
            "authorized": False,
            "impostor": True,
            "error": "IMPOSTOR DETECTED: Typing speed too slow",
            "speed_deviation": round(speed_deviation, 1),
            "auth_speed": round(current_speed, 2),
            "training_speed": round(baseline_speed, 2)
        }), 200

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
    save_user(username, user)
    
    return jsonify({"success": True})

if __name__ == "__main__":
    print("=" * 80)
    print("🏦 SecureBank - Simplified Keystroke Biometric System")
    print("=" * 80)
    print("🚀 Server: http://127.0.0.1:5000")
    print("=" * 80)
    print("\n⚡ SIMPLE LOGIC:")
    print(f"  • Typing Speed >= {MIN_SPEED_RATIO*100}% of baseline → ✅ SUCCESS")
    print(f"  • Typing Speed < {MIN_SPEED_RATIO*100}% of baseline → 🚨 IMPOSTOR + ALARM")
    print("\n🎯 FEATURES:")
    print("  • Normal/Fast/Very Fast typing = Always SUCCESS")
    print("  • Only SLOW typing triggers impostor alarm")
    print("  • Real bank-style continuous bell alarm")
    print("  • Manual alarm stop button")
    print("  • Train More: keeps previous samples (5→10→15...)")
    print("  • Enter key auto-submit everywhere")
    print("  • Transaction success page with Share & Go Back")
    print("  • Very lenient (50% minimum, huge buffer)")
    print("\n🔔 ALARM FEATURES:")
    print("  • Continuous bell ringing animation")
    print("  • Looping alarm sound")
    print("  • Red flashing screen")
    print("  • Shake effect")
    print("  • Click button to stop")
    print("=" * 80)
    print()
    
    app.run(debug=True, host="127.0.0.1", port=5000)