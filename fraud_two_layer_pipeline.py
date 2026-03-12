import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import joblib


# CONFIG

DATA_PATH = "C:/Major project/Data/creditcard.csv"   # <<< your path
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)



# DATA LOADING & FEATURE ENGINEERING

def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from {path} ...")
    df = pd.read_csv(path)
    print("[INFO] Data loaded.")
    print(df.head())
    print(df["Class"].value_counts())
    return df


def engineer_features(df: pd.DataFrame):
    """
    Add some simple but useful features:
      - Amount_log: log1p transform of Amount
      - Amount_bin: binned Amount
      - Time_hour: transaction hour (approx)
    """
    df = df.copy()

    # Log-transform of amount
    df["Amount_log"] = np.log1p(df["Amount"])

    # Amount bins (0..3)
    df["Amount_bin"] = pd.qcut(df["Amount"], q=4, labels=False, duplicates="drop")

    # Time in hours (dataset Time is seconds since first transaction)
    df["Time_hour"] = (df["Time"] / 3600).astype(int)

    print("[INFO] Engineered features added.")
    print(df[["Amount", "Amount_log", "Amount_bin", "Time", "Time_hour"]].head())

    # Separate features & target
    y = df["Class"].values
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values

    return X, y, feature_cols


def scale_features(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[INFO] Feature matrix shape after scaling: {X_scaled.shape}")
    return X_scaled, scaler



# PRE-LAYER: ISOLATION FOREST

def train_pre_layer_isolation_forest(X_train, y_train, random_state=42):
    """
    Train IsolationForest on training data only.
    Returns model, chosen threshold and target suspicious fraction.
    """
    # Estimate fraud rate in training data
    fraud_rate = (y_train == 1).mean()
    print(f"[INFO] Estimated fraud rate in training data: {fraud_rate:.6f}")

    # We'll consider ~5x fraud rate as "suspicious" but at least 1%
    target_suspicious_fraction = max(5 * fraud_rate, 0.01)

    print("[INFO] Training pre-layer IsolationForest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",  # we'll still pick our own threshold
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train)
    print("[INFO] Pre-layer model trained.")

    # Higher scores = more normal, lower = more anomalous
    train_scores = iso.decision_function(X_train)

    # Choose threshold so that approximately target_suspicious_fraction are suspicious
    threshold = np.quantile(train_scores, target_suspicious_fraction)
    approx_cnt = int(target_suspicious_fraction * len(X_train))
    print(f"[INFO] Chosen pre-layer anomaly threshold: {threshold:.6f}")
    print(
        f"[INFO] Target suspicious fraction: {target_suspicious_fraction:.6f}, "
        f"~{approx_cnt} of {len(X_train)} training samples"
    )

    return iso, threshold, target_suspicious_fraction, train_scores


def add_pre_layer_features(iso_model, threshold, X, train_scores=None):
    """
    For any feature matrix X, compute:
      - pre_score: IsolationForest decision_function
      - pre_flag: 1 if score <= threshold else 0
    Returns: new_X with 2 extra columns.
    """
    if train_scores is None:
        scores = iso_model.decision_function(X)
    else:
        scores = train_scores

    flags = (scores <= threshold).astype(int)

    scores = scores.reshape(-1, 1)
    flags = flags.reshape(-1, 1)

    new_X = np.hstack([X, scores, flags])
    return new_X, scores.ravel(), flags.ravel()


# ==============================
# POST-LAYER EVALUATION
# ==============================
def evaluate_classifier(y_true, y_pred, y_proba, title=""):
    """
    Print full report + confusion matrix + key metrics.
    Return metrics dict.
    """
    print(f"\n===== {title} EVALUATION =====")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Individual numbers
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # For ROC-AUC, we need probability for class 1
    roc_auc = roc_auc_score(y_true, y_proba)

    # Direct counts for support (avoid NoneType issue)
    support_fraud = int((y_true == 1).sum())
    support_legit = int((y_true == 0).sum())
    total = int(len(y_true))

    metrics = {
        "accuracy": accuracy,
        "precision_fraud": precision,
        "recall_fraud": recall,
        "f1_fraud": f1,
        "roc_auc": roc_auc,
        "support_fraud": support_fraud,
        "support_legit": support_legit,
        "total_samples": total,
    }

    print("\nSummary metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>18}: {v:.4f}")
        else:
            print(f"{k:>18}: {v}")

    return metrics


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    # 1) Load data
    df = load_data(DATA_PATH)

    # 2) Feature engineering
    X, y, feature_cols = engineer_features(df)

    # 3) Scale features
    X_scaled, scaler = scale_features(X)

    # 4) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    print("[INFO] Split data into train and test.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Class distribution for info
    def print_class_dist(name, y_arr):
        unique, counts = np.unique(y_arr, return_counts=True)
        total = len(y_arr)
        print(f"{name} class distribution:")
        for label, cnt in zip(unique, counts):
            print(f"  Class {label}: {cnt} ({cnt / total:.6f})")

    print_class_dist("Train", y_train)
    print_class_dist("Test", y_test)

    # ==========================
    # PRE-LAYER (UNSUPERVISED)
    # ==========================
    iso_model, threshold, target_frac, train_scores = train_pre_layer_isolation_forest(
        X_train, y_train
    )

    # Add pre-layer features to TRAIN and TEST
    X_train_two, train_pre_scores, train_pre_flags = add_pre_layer_features(
        iso_model, threshold, X_train, train_scores
    )
    X_test_two, test_pre_scores, test_pre_flags = add_pre_layer_features(
        iso_model, threshold, X_test
    )

    # ==========================
    # POST-LAYER CLASSIFIERS
    # ==========================

    # (A) Baseline: RandomForest on original scaled features
    print("[INFO] Training post-layer RandomForest classifier (baseline, no pre-layer features)...")
    rf_baseline = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight={0: 1.0, 1: 10.0},  # handle imbalance a bit
    )
    rf_baseline.fit(X_train, y_train)
    print("[INFO] Baseline classifier trained.")

    # (B) Two-layer: RandomForest on (original + pre-layer features)
    print("[INFO] Training post-layer RandomForest classifier (two-layer, with pre-layer features)...")
    rf_two_layer = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight={0: 1.0, 1: 10.0},
    )
    rf_two_layer.fit(X_train_two, y_train)
    print("[INFO] Two-layer classifier trained.")

    # ==========================
    # EVALUATION
    # ==========================

    # Baseline evaluation
    y_post_pred = rf_baseline.predict(X_test)
    y_post_proba = rf_baseline.predict_proba(X_test)[:, 1]

    baseline_metrics = evaluate_classifier(
        y_test, y_post_pred, y_post_proba, title="Post-layer Only (Baseline)"
    )

    # Two-layer evaluation
    y_two_pred = rf_two_layer.predict(X_test_two)
    y_two_proba = rf_two_layer.predict_proba(X_test_two)[:, 1]

    two_layer_metrics = evaluate_classifier(
        y_test, y_two_pred, y_two_proba, title="Two-Layer (Pre + Post)"
    )

    # Side-by-side main numbers
    print("\n===== COMPARISON: BASELINE vs TWO-LAYER =====")
    print(f"Baseline Accuracy      : {baseline_metrics['accuracy'] * 100:.2f}%")
    print(f"Two-layer Accuracy     : {two_layer_metrics['accuracy'] * 100:.2f}%")
    print(f"Baseline F1 (fraud)    : {baseline_metrics['f1_fraud'] * 100:.2f}%")
    print(f"Two-layer F1 (fraud)   : {two_layer_metrics['f1_fraud'] * 100:.2f}%")
    print(f"Baseline ROC-AUC       : {baseline_metrics['roc_auc'] * 100:.2f}%")
    print(f"Two-layer ROC-AUC      : {two_layer_metrics['roc_auc'] * 100:.2f}%")

    # ==========================
    # MODEL SAVING
    # ==========================

    print("\n[INFO] Saving models and scaler to disk...")

    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    pre_model_path = os.path.join(MODEL_DIR, "pre_isolation_forest.joblib")
    rf_baseline_path = os.path.join(MODEL_DIR, "rf_baseline.joblib")
    rf_two_layer_path = os.path.join(MODEL_DIR, "rf_two_layer.joblib")
    pre_config_path = os.path.join(MODEL_DIR, "pre_layer_config.json")

    joblib.dump(scaler, scaler_path)
    joblib.dump(iso_model, pre_model_path)
    joblib.dump(rf_baseline, rf_baseline_path)
    joblib.dump(rf_two_layer, rf_two_layer_path)

    pre_config = {
        "threshold": float(threshold),
        "target_suspicious_fraction": float(target_frac),
        "feature_cols": feature_cols,
    }
    with open(pre_config_path, "w") as f:
        json.dump(pre_config, f, indent=2)

    print("[INFO] Saved:")
    print(f"  Scaler              -> {scaler_path}")
    print(f"  Pre-layer model     -> {pre_model_path}")
    print(f"  Baseline RF model   -> {rf_baseline_path}")
    print(f"  Two-layer RF model  -> {rf_two_layer_path}")
    print(f"  Pre-layer config    -> {pre_config_path}")


if __name__ == "__main__":
    main()


