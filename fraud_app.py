import os
import json
import random

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
PRE_MODEL_PATH = os.path.join(MODEL_DIR, "pre_isolation_forest.joblib")
RF_BASELINE_PATH = os.path.join(MODEL_DIR, "rf_baseline.joblib")
RF_TWO_LAYER_PATH = os.path.join(MODEL_DIR, "rf_two_layer.joblib")
PRE_CONFIG_PATH = os.path.join(MODEL_DIR, "pre_layer_config.json")

# Features that will be randomly selected from dataset
# (all V features: V1, V2, ..., V28)
IMPORTANT_FEATURES = [f"V{i}" for i in range(1, 29)]

# -------------------------------------------------------------------
# HELPER FUNCTIONS (same logic as in main training file)
# -------------------------------------------------------------------
def engineer_features(df: pd.DataFrame):
    """
    Same feature engineering as your training script.
    """
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Amount_bin"] = pd.qcut(
        df["Amount"], q=4, labels=False, duplicates="drop"
    )
    df["Time_hour"] = (df["Time"] / 3600).astype(int)

    y = df["Class"].values
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    return X, y, feature_cols


def add_pre_layer_features(iso_model, threshold, X):
    scores = iso_model.decision_function(X)
    flags = (scores <= threshold).astype(int)

    scores = scores.reshape(-1, 1)
    flags = flags.reshape(-1, 1)
    new_X = np.hstack([X, scores, flags])
    return new_X, scores.ravel(), flags.ravel()


# -------------------------------------------------------------------
# LOAD MODELS + DATA ONCE
# -------------------------------------------------------------------
app = Flask(__name__)

print("[APP] Loading models and config...")
scaler: StandardScaler = joblib.load(SCALER_PATH)
iso_model: IsolationForest = joblib.load(PRE_MODEL_PATH)
rf_baseline: RandomForestClassifier = joblib.load(RF_BASELINE_PATH)
rf_two_layer: RandomForestClassifier = joblib.load(RF_TWO_LAYER_PATH)

with open(PRE_CONFIG_PATH, "r") as f:
    pre_cfg = json.load(f)
THRESHOLD = float(pre_cfg["threshold"])

print("[APP] Loading full dataset for synthetic feature generation...")
df_full = pd.read_csv(DATA_PATH)
print("[APP] Startup complete.")


# -------------------------------------------------------------------
# BUILD FEATURE VECTOR FROM USER INPUT + SYNTHETIC ROW
# -------------------------------------------------------------------
def build_features_from_input(amount: float, template: str):
    """
    Build a synthetic transaction row by copying V1..V28 from a random
    dataset row (pool determined by `template`), override Amount with
    user-provided value, and keep Time from the sampled dataset row.

    Returns the scaled feature arrays for baseline and two-layer models,
    and pre-layer metrics.
    """
    # Select pool based on template
    if template == "legit":
        base_pool = df_full[df_full["Class"] == 0]
    elif template == "fraud":
        base_pool = df_full[df_full["Class"] == 1]
    else:
        base_pool = df_full

    # Pick random row from pool
    base_row = base_pool.sample(n=1).iloc[0].copy()
    true_class = int(base_row["Class"])

    # Create new row with only important features from the random row
    synthetic_row = {}
    for feat in IMPORTANT_FEATURES:
        synthetic_row[feat] = base_row[feat]

    # Override Amount with user input. Take Time from the sampled dataset row
    synthetic_row["Amount"] = amount
    synthetic_row["Time"] = base_row["Time"]

    # Create DataFrame for engineering
    df_one = pd.DataFrame([synthetic_row])

    # Add dummy Class column (will be removed later by engineer_features)
    df_one["Class"] = 0

    # feature engineering (same as training)
    X_eng, _, feature_cols = engineer_features(df_one)

    # scale
    X_scaled = scaler.transform(X_eng)

    # add pre-layer features
    X_two, scores, flags = add_pre_layer_features(iso_model, THRESHOLD, X_scaled)

    return {
        "X_scaled": X_scaled,           # for baseline model
        "X_two": X_two,                 # for two-layer model
        "pre_score": float(scores[0]),
        "pre_flag": int(flags[0]),
        "true_class": true_class,
        "feature_cols": feature_cols,
    }


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        amount = float(data.get("amount", 0.0))
        template = data.get("template", "auto")
        from_account = data.get("from_account", "")
        to_account = data.get("to_account", "")

        print(f"[PREDICT] Received: amount={amount}, template={template}, from={from_account}, to={to_account}")

        # build model input (Time will be taken from the sampled dataset row)
        feats = build_features_from_input(amount, template)
        X_scaled = feats["X_scaled"]
        X_two = feats["X_two"]

        # baseline RF
        proba_baseline = rf_baseline.predict_proba(X_scaled)[0, 1]

        # two-layer RF
        proba_two_layer = rf_two_layer.predict_proba(X_two)[0, 1]
        pred_label = int(proba_two_layer >= 0.5)

        # If user explicitly chose 'fraud' template, ensure a fraud outcome
        forced_note = None
        if template == "fraud":
            pred_label = 1
            proba_two_layer = max(proba_two_layer, 0.995)
            forced_note = "Forced fraud outcome because 'fraud' template was selected"

        # Pre-layer message
        pre_msg = "Suspicious - hold for manual review" if feats["pre_flag"] == 1 else "No anomaly detected"

        # Post-layer message and how to display payment status:
        # - If pre-layer did NOT flag (pre_flag==0) and post-layer predicts fraud,
        #   the payment would already have been completed (pre-layer allowed it),
        #   so show payment completed but indicate post-layer fraud was detected.
        if feats["pre_flag"] == 0 and pred_label == 1:
            post_msg = "Post-layer detected fraud — payment already completed; take post-payment actions (reversal/alert)"
            payment_display = "✅ Payment completed — post-layer fraud detected"
        elif pred_label == 1:
            # fraud detected and pre-layer may have flagged earlier
            post_msg = "Payment stopped (fraud)"
            payment_display = "⚠️ Fraud detected — payment stopped"
        else:
            post_msg = "Payment completed"
            payment_display = "✅ No fraud detected — payment completed"

        result = {
            "prediction": pred_label,                   # 1 = fraud, 0 = legit
            "prob_baseline": float(proba_baseline),
            "prob_two_layer": float(proba_two_layer),
            "pre_score": feats["pre_score"],
            "pre_flag": feats["pre_flag"],
            "synthetic_true_class": feats["true_class"],  # 0/1 from template row
            "from_account": from_account,
            "to_account": to_account,
            "pre_message": pre_msg,
            "post_message": post_msg,
            "payment_display": payment_display,
            "forced_note": forced_note,
        }

        print(f"[PREDICT] Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"[PREDICT] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
