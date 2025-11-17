#!/usr/bin/env python
# coding: utf-8
"""
model.py - TRAIN for 4 FEATURES:
['glucose', 'bloodpressure', 'insulin', 'age']

Merges CSVs, cleans target, forces these 4 features,
applies SMOTE if available, compares models, saves best model + scaler + meta.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

RANDOM_STATE = 42

# ---------- CSV files (ensure these exist in same folder) ----------
files = [
    "diabetes.csv",
    "Diabetes_Final_Data_V2.csv",
    "diabetes_prediction_india (1).csv"
]

# ---------- load datasets ----------
datasets = []
for f in files:
    if not os.path.exists(f):
        print(f"⚠️ File not found, skipping: {f}")
        continue
    try:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        datasets.append(df)
        print(f"✅ Loaded {f} | shape: {df.shape}")
    except Exception as e:
        print(f"⚠️ Error loading {f}: {e}")

if not datasets:
    print("❌ No datasets loaded. Put the CSVs in this folder.")
    sys.exit(1)

data = pd.concat(datasets, ignore_index=True)
print(f"\n🔹 Combined shape: {data.shape}")

# ---------- detect target ----------
target_col = None
for col in data.columns:
    if ("outcome" in col) or ("diabetes" in col and col != 'diabetespedigreefunction') or ("label" in col):
        target_col = col
        break

if target_col is None:
    print("❌ Could not detect the target column (expected 'outcome' or 'diabetes' or 'label').")
    print("Columns:", list(data.columns))
    sys.exit(1)

print(f"🎯 Detected target column: {target_col}")
print("Raw target value counts (sample):\n", data[target_col].astype(str).value_counts().head(20))

# ---------- normalize target to 0/1 ----------
def normalize_target(series):
    s = series.astype(str).str.lower().str.strip()
    map_to_1 = {'1','yes','y','true','t','positive','pos','diabetes','diabetic','has_diabetes','1.0'}
    map_to_0 = {'0','no','n','false','f','negative','neg','no_diabetes','not_diabetic','0.0','normal','healthy'}
    mapped = []
    unknown = {}
    for val in s:
        if val in map_to_1:
            mapped.append(1)
        elif val in map_to_0:
            mapped.append(0)
        else:
            try:
                v = float(val)
                if v == 1.0:
                    mapped.append(1)
                elif v == 0.0:
                    mapped.append(0)
                else:
                    mapped.append(np.nan)
                    unknown[val] = unknown.get(val,0) + 1
            except:
                mapped.append(np.nan)
                unknown[val] = unknown.get(val,0) + 1
    return pd.Series(mapped, index=series.index), unknown

y_mapped, unknowns = normalize_target(data[target_col])
if unknowns:
    print("\n⚠️ Found unmapped target labels (examples):")
    for k,v in list(unknowns.items())[:8]:
        print(f"  '{k}': {v}")
# Drop rows with unmapped target
valid_mask = ~y_mapped.isna()
dropped = len(y_mapped) - valid_mask.sum()
if dropped > 0:
    print(f"ℹ️ Dropping {dropped} rows with unknown target labels.")
    data = data.loc[valid_mask].copy()
    data[target_col] = y_mapped.loc[valid_mask].astype(int)
else:
    data[target_col] = y_mapped.astype(int)

print("\n✅ Target counts after mapping:\n", data[target_col].value_counts())

# ---------- FORCE features to the 4 we want ----------
required_order = ['glucose','bloodpressure','insulin','age']

# Verify core features exist
missing_core = [c for c in required_order if c not in data.columns]
if missing_core:
    print(f"❌ Core feature columns missing in data: {missing_core}")
    print("Available columns:", list(data.columns))
    sys.exit(1)

# Coerce to numeric (convert errors to NaN)
for c in required_order:
    data[c] = pd.to_numeric(data[c], errors='coerce')

# Drop rows missing any of the four required features or target
before = data.shape
data = data.dropna(subset=required_order + [target_col])
after = data.shape
print(f"ℹ️ Dropped rows with missing core features: {before[0] - after[0]} (if any). Now: {data.shape}")

# Final feature set (only these 4)
ordered_features = required_order.copy()
print("🧩 Final training features (ordered):", ordered_features)

X = data[ordered_features].astype(float)
y = data[target_col].astype(int)

# ---------- safe stratified split ----------
min_cnt = y.value_counts().min()
if min_cnt >= 2:
    strat = y
    print("\n🔐 Using stratified split.")
else:
    strat = None
    print("\n⚠️ Not enough samples per class for stratify -> using non-stratified split.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
)

print(f"\nSplit sizes -> Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print("Train distribution:", Counter(y_train))
print("Test distribution:", Counter(y_test))

# ---------- scaling ----------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- SMOTE on train (if available & feasible) ----------
apply_smote = False
if IMBLEARN_AVAILABLE:
    try:
        tr_counts = Counter(y_train)
        minor = min(tr_counts.values())
        if minor >= 2:
            print("\n⚖️ Applying SMOTE to training set...")
            sm = SMOTE(random_state=RANDOM_STATE)
            X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
            print("✅ After SMOTE:", Counter(y_train_bal))
            apply_smote = True
        else:
            print("\n⚠️ Not enough minority samples for SMOTE; skipping.")
            X_train_bal, y_train_bal = X_train_scaled, y_train
    except Exception as e:
        print(f"\n⚠️ SMOTE failed: {e}. Proceeding without SMOTE.")
        X_train_bal, y_train_bal = X_train_scaled, y_train
else:
    print("\nℹ️ imbalanced-learn not installed; skipping SMOTE. (pip install imbalanced-learn)")
    X_train_bal, y_train_bal = X_train_scaled, y_train

# ---------- models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
}

results = {}
print("\n🚀 Training models...")
for name, mdl in models.items():
    try:
        mdl.fit(X_train_bal, y_train_bal)
        preds = mdl.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"model": mdl, "accuracy": acc, "f1": f1}
        print(f"\n🔹 {name}\n   Accuracy: {acc*100:.2f}% | F1: {f1:.3f}")
    except Exception as e:
        print(f"\n⚠️ {name} training failed: {e}")

if not results:
    print("❌ No models trained. Exiting.")
    sys.exit(1)

# ---------- select best ----------
best_name = max(results.keys(), key=lambda n: (results[n]["accuracy"], results[n]["f1"]))
best_entry = results[best_name]
best_model = best_entry["model"]
best_acc = best_entry["accuracy"]
best_f1 = best_entry["f1"]

print("\n🏆 Best model:", best_name, f"(Accuracy: {best_acc*100:.2f}%, F1: {best_f1:.3f})")

# report
print("\n📋 Classification report (best model on test set):")
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_best))

# feature importance if applicable
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=ordered_features).sort_values(ascending=False)
    print("\n📊 Feature importances:\n", fi)

# ---------- save model, scaler, meta ----------
pickle.dump(best_model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

meta = {
    "best_model_name": best_name,
    "accuracy": best_acc,
    "f1": best_f1,
    "features": ordered_features,
    "smote_used": bool(apply_smote)
}
pickle.dump(meta, open("model_meta.pkl","wb"))

print("\n💾 Saved model.pkl, scaler.pkl, model_meta.pkl")
print("✅ Training complete. Restart Flask to use the new model.")
