"""
Training pipeline for the fraud detection model.

- Time-respecting train/test split (no shuffling — avoids leakage across
  the fraud "campaigns" that tend to cluster in time)
- Class imbalance handled via SMOTE oversampling on the train fold only
- Model: XGBoost classifier, tuned for PR-AUC (the right metric for
  severe imbalance — ROC-AUC is misleadingly high on this kind of data)
- Saves the fitted feature engineer + model + metrics as artifacts

Run:
    python src/train.py --data data/transactions.csv --out_dir models/
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from features import FraudFeatureEngineer


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df_sorted = df.sort_values("seconds_elapsed").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def find_best_threshold(y_true, y_scores, min_precision: float = 0.85):
    """Pick the classification threshold that maximizes recall subject to
    a minimum precision constraint — the standard business framing for
    fraud review queues (analysts have limited capacity, so precision floor
    keeps false-positive workload manageable)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    best_thresh, best_recall = 0.5, 0.0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_recall, best_thresh = r, t
    return float(best_thresh), float(best_recall)


def main(args):
    t0 = time.time()
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows, fraud rate {df['is_fraud'].mean()*100:.3f}%")

    train_df, test_df = time_split(df, test_frac=args.test_frac)
    y_train, y_test = train_df["is_fraud"].values, test_df["is_fraud"].values

    fe = FraudFeatureEngineer()
    fe.fit(train_df)
    X_train = fe.transform(train_df)
    X_test = fe.transform(test_df)

    # align columns (one-hot categories may differ slightly between splits)
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
    feature_cols = list(X_train.columns)

    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)")

    print("Applying SMOTE to training fold only...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, max(1, y_train.sum() - 1)))
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Post-SMOTE train size: {len(X_train_res):,} "
          f"({y_train_res.sum():,} fraud, {(y_train_res==0).sum():,} legit)")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    y_scores = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    best_thresh, recall_at_precision = find_best_threshold(y_test, y_scores, min_precision=0.85)

    y_pred = (y_scores >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "decision_threshold": round(best_thresh, 4),
        "recall_at_precision_0.85": round(recall_at_precision, 4),
        "precision_fraud_class": round(report["1"]["precision"], 4),
        "recall_fraud_class": round(report["1"]["recall"], 4),
        "f1_fraud_class": round(report["1"]["f1-score"], 4),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        },
        "train_rows": len(X_train_res),
        "test_rows": len(X_test),
        "train_seconds": round(time.time() - t0, 1),
    }

    print(json.dumps(metrics, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(fe, out_dir / "feature_engineer.joblib")
    joblib.dump(feature_cols, out_dir / "feature_columns.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts saved to {out_dir}/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/transactions.csv")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--test_frac", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
