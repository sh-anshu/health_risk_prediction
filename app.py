import os
import sqlite3
import pickle
import logging
from datetime import datetime
from functools import wraps

import numpy as np
from flask import (
    Flask, request, render_template, redirect, url_for, session, jsonify, g, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------- Configuration --------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change_this_in_production')

DB_PATH = os.environ.get('DP_DB', 'users.db')
MODEL_PKL = os.environ.get('MODEL_PKL', 'model.pkl')
SCALER_PKL = os.environ.get('SCALER_PKL', 'scaler.pkl')
META_PKL = os.environ.get('MODEL_META_PKL', 'model_meta.pkl')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cache for model + scaler
_model_cache = {}

# -------------------- Database helpers --------------------
def get_db():
    """Return sqlite3 connection for request"""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    password TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    mode TEXT,
                    risk REAL,
                    result_text TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )''')
    conn.commit()
    conn.close()
    logger.info("Database initialized (users.db)")

init_db()

# -------------------- Auth helpers --------------------
def login_required_json(f):
    @wraps(f)
    def inner(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'authentication required'}), 401
        return f(*args, **kwargs)
    return inner

def login_required(f):
    @wraps(f)
    def inner(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return inner

# -------------------- Model loading --------------------
def get_model_scaler_meta():
    """
    Lazy-load and cache model, scaler and optional meta.
    Raises RuntimeError if model/scaler files missing.
    """
    if _model_cache:
        return _model_cache['model'], _model_cache['scaler'], _model_cache.get('meta')

    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        raise RuntimeError("model.pkl or scaler.pkl not found. Train model first (run model.py).")

    try:
        with open(MODEL_PKL, 'rb') as mf:
            mdl = pickle.load(mf)
        with open(SCALER_PKL, 'rb') as sf:
            scaler = pickle.load(sf)
    except Exception as e:
        raise RuntimeError(f"Failed loading model/scaler: {e}")

    meta = None
    if os.path.exists(META_PKL):
        try:
            with open(META_PKL, 'rb') as mf:
                meta = pickle.load(mf)
        except Exception:
            meta = None

    _model_cache['model'] = mdl
    _model_cache['scaler'] = scaler
    _model_cache['meta'] = meta
    logger.info("Model and scaler loaded and cached.")
    return mdl, scaler, meta

# -------------------- Utility: save result --------------------
def save_result(user_id, mode, risk, result_text):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO results (user_id, mode, risk, result_text, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, mode, float(risk), result_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
    except Exception as e:
        logger.exception("Failed saving result to DB: %s", e)

# -------------------- Prediction core --------------------
def build_advice_from_prediction(pred, risk):
    # same logic as before, centralized
    if pred == 1:
        if risk < 50:
            result_text = f"Mild Diabetes Risk ({risk}%)."
            advice = {
              "message": "Your risk is mild. Focus on improving your lifestyle.",
              "do": ["Reduce sugar intake.", "Exercise 30 min daily.", "Eat more fiber."],
              "dont": ["Avoid junk food.", "Don’t skip meals."]
            }
        elif 50 <= risk < 80:
            result_text = f"Moderate Diabetes Risk ({risk}%)."
            advice = {
              "message": "Moderate risk. See a doctor soon.",
              "do": ["Follow a diet plan.", "Monitor glucose weekly."],
              "dont": ["Avoid alcohol.", "Don’t consume sweets."]
            }
        else:
            result_text = f"High Diabetes Risk ({risk}%)."
            advice = {
              "message": "High risk. Immediate medical consultation recommended.",
              "do": ["Consult doctor immediately.", "Monitor glucose daily."],
              "dont": ["Avoid sugary food.", "Don’t delay treatment."]
            }
    else:
        result_text = f"You don’t have Diabetes. Risk: {risk}%."
        advice = {
            "message": "Low risk. Keep a healthy lifestyle.",
            "do": ["Exercise regularly.", "Eat well.", "Maintain sleep."],
            "dont": ["Avoid excess sugar."]
        }
    return result_text, advice

def do_lab_predict_from_dict(data):
    """
    Accepts dict with feature names, returns tuple (result_text, risk, advice)
    Raises ValueError on missing/invalid inputs.
    """
    model_obj, scaler_obj, meta = get_model_scaler_meta()

    # Determine expected features
    if meta and isinstance(meta, dict) and meta.get('features'):
        expected_features = meta['features']
    else:
        expected_features = ['glucose', 'bloodpressure', 'insulin', 'age']

    fv = []
    for feat in expected_features:
        raw = data.get(feat)
        # allow some aliases
        if raw is None:
            aliases = {
                'bloodpressure': ['bloodPressure', 'bp', 'blood_pressure'],
                'glucose': ['glucose', 'blood_glucose', 'gl'],
                'insulin': ['insulin', 'serum_insulin'],
                'age': ['age', 'years']
            }.get(feat, [])
            for a in aliases:
                if a in data:
                    raw = data.get(a)
                    break
        if raw in (None, ''):
            raise ValueError(f"Missing required field: '{feat}'")
        try:
            val = float(raw)
        except Exception:
            raise ValueError(f"Invalid numeric value for '{feat}': {raw}")
        fv.append(val)

    X = np.array([fv])

    # Validate with scaler/model expectations if available
    expected_by_scaler = getattr(scaler_obj, "n_features_in_", None)
    expected_by_model = getattr(model_obj, "n_features_in_", None)
    if expected_by_scaler is not None and X.shape[1] != expected_by_scaler:
        raise ValueError(f"Feature length mismatch: built {X.shape[1]} features but scaler expects {expected_by_scaler}.")
    if expected_by_model is not None and X.shape[1] != expected_by_model:
        raise ValueError(f"Feature length mismatch: built {X.shape[1]} features but model expects {expected_by_model}.")

    # transform + predict
    Xs = scaler_obj.transform(X)
    pred = model_obj.predict(Xs)[0]
    prob = model_obj.predict_proba(Xs)[0][1] if hasattr(model_obj, 'predict_proba') else 0.0
    risk = round(float(prob) * 100, 2)

    result_text, advice = build_advice_from_prediction(int(pred), risk)
    return result_text, risk, advice

# -------------------- Web routes --------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        pw = request.form.get('password', '')
        if not name or not email or not pw:
            return render_template('signup.html', error="Please fill all fields.")
        if len(pw) < 8:
            return render_template('signup.html', error="Password must be at least 8 characters.")
        hashed = generate_password_hash(pw)
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Email already exists.")
        except Exception as e:
            logger.exception("Signup failed: %s", e)
            return render_template('signup.html', error="Server error.")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        pw = request.form.get('password', '')
        if not email or not pw:
            return render_template('login.html', error="Please enter both email and password.")
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT id, name, password FROM users WHERE email=?", (email,))
            row = cur.fetchone()
            if row and check_password_hash(row['password'], pw):
                session['user_id'] = int(row['id'])
                session['user_name'] = row['name']
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error="Invalid credentials.")
        except Exception as e:
            logger.exception("Login failed: %s", e)
            return render_template('login.html', error="Server error.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=session.get('user_name'))

@app.route('/admin')
@login_required
def admin():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''SELECT users.name, results.mode, results.risk, results.result_text, results.timestamp
                       FROM results
                       JOIN users ON results.user_id = users.id
                       ORDER BY results.timestamp DESC LIMIT 500''')
        data = cur.fetchall()
        return render_template('admin.html', data=data)
    except Exception as e:
        logger.exception("Admin page failed: %s", e)
        return render_template('admin.html', data=[])

# -------------------- API endpoints (AJAX) --------------------
@app.route('/api/predict', methods=['POST'])
@login_required_json
def api_predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'invalid JSON payload'}), 400

    try:
        result_text, risk, advice = do_lab_predict_from_dict(payload)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except RuntimeError as re:
        return jsonify({'error': str(re)}), 500
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({'error': 'internal prediction error'}), 500

    # best-effort save
    try:
        save_result(session['user_id'], 'Lab', risk, result_text)
    except Exception:
        logger.exception("Failed saving prediction result")

    return jsonify({'prediction_text': result_text, 'risk_level': risk, 'advice': advice})

@app.route('/api/general_predict', methods=['POST'])
@login_required_json
def api_general_predict():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'invalid JSON payload'}), 400

    try:
        responses = [int(data[k]) for k in ['obese','tired','urination','thirst','family']]
    except Exception:
        return jsonify({'error': 'invalid input; expected keys obese,tired,urination,thirst,family with 0/1 values'}), 400

    score = sum(responses)
    if score <= 1:
        risk = 10
        result_text = "Low Risk of Diabetes."
        advice = {
            "message": "Low risk. Maintain healthy habits.",
            "do": ["Exercise regularly.", "Eat whole foods.", "Stay hydrated."],
            "dont": ["Avoid excessive sugar.", "Don’t skip checkups."]
        }
    elif 2 <= score <= 3:
        risk = 50
        result_text = "Moderate Risk of Diabetes."
        advice = {
            "message": "Some symptoms detected. Get a blood sugar test soon.",
            "do": ["Monitor weight.", "Limit refined carbs.", "Increase fiber."],
            "dont": ["Avoid sedentary habits.", "Don’t ignore fatigue."]
        }
    else:
        risk = 85
        result_text = "High Risk of Diabetes."
        advice = {
            "message": "Strong symptoms detected. Consult a doctor immediately.",
            "do": ["Visit a doctor for testing.", "Low-sugar diet.", "Start daily light exercise."],
            "dont": ["Avoid sugar, alcohol, and smoking."]
        }

    try:
        save_result(session['user_id'], 'General', risk, result_text)
    except Exception:
        logger.exception("Failed saving general result")

    return jsonify({'prediction_text': result_text, 'risk_level': risk, 'advice': advice})

@app.route('/api/history')
def api_history():
    # history is public for logged-in user only (front-end expects empty array if not logged in)
    if 'user_id' not in session:
        return jsonify([])
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT results.mode, results.risk, results.result_text, results.timestamp, users.name "
                    "FROM results JOIN users ON results.user_id=users.id WHERE users.id=? ORDER BY results.timestamp DESC LIMIT 10",
                    (session['user_id'],))
        rows = cur.fetchall()
        items = []
        for r in rows:
            items.append({
                'mode': r['mode'],
                'risk': r['risk'],
                'result_text': r['result_text'],
                'timestamp': r['timestamp'],
                'username': r['name']
            })
        return jsonify(items)
    except Exception:
        logger.exception("Failed fetching history")
        return jsonify([])

# -------------------- Optional form POST endpoints (non-AJAX fallback) --------------------
@app.route('/predict', methods=['POST'])
@login_required
def predict_form():
    """
    Accepts classic form POSTs from HTML forms (if JS disabled).
    Uses the same prediction function and renders index with results.
    """
    try:
        # build dict from form
        data = {k: request.form.get(k) for k in request.form.keys()}
        result_text, risk, advice = do_lab_predict_from_dict(data)
    except ValueError as ve:
        return render_template('index.html', prediction_text=f"Error: {ve}", user=session.get('user_name'))
    except Exception as e:
        logger.exception("predict_form error: %s", e)
        return render_template('index.html', prediction_text="Server error during prediction.", user=session.get('user_name'))

    save_result(session['user_id'], 'Lab', risk, result_text)
    return render_template('index.html', prediction_text=result_text, risk_level=risk, user=session.get('user_name'))

@app.route('/general_predict', methods=['POST'])
@login_required
def general_predict_form():
    try:
        keys = ['obese','tired','urination','thirst','family']
        responses = [int(request.form.get(k, 0)) for k in keys]
        score = sum(responses)
        if score <= 1:
            risk = 10
            result_text = "Low Risk of Diabetes."
        elif 2 <= score <= 3:
            risk = 50
            result_text = "Moderate Risk of Diabetes."
        else:
            risk = 85
            result_text = "High Risk of Diabetes."
    except Exception as e:
        logger.exception("general_predict_form error: %s", e)
        return render_template('index.html', prediction_text="Invalid input.", user=session.get('user_name'))

    save_result(session['user_id'], 'General', risk, result_text)
    return render_template('index.html', prediction_text=result_text, risk_level=risk, user=session.get('user_name'))

# -------------------- Run --------------------
if __name__ == '__main__':
    # allow debug toggle via env var
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug)
