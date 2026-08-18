"""
Generates evaluation artifacts (precision-recall curve, confusion matrix
heatmap) from a trained model for the model card / README.

Run:
    python src/evaluate.py --data ../data/transactions.csv --model_dir ../models
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from features import FraudFeatureEngineer  # noqa: F401  (needed for unpickling)
from train import time_split


def main(args):
    df = pd.read_csv(args.data)
    _, test_df = time_split(df, test_frac=0.2)
    y_test = test_df["is_fraud"].values

    model = joblib.load(Path(args.model_dir) / "model.joblib")
    fe = joblib.load(Path(args.model_dir) / "feature_engineer.joblib")
    feature_cols = joblib.load(Path(args.model_dir) / "feature_columns.joblib")

    X_test = fe.transform(test_df).reindex(columns=feature_cols, fill_value=0)
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_scores, ax=axes[0])
    axes[0].set_title("Precision-Recall Curve (Fraud Class)")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], cmap="Blues")
    axes[1].set_title("Confusion Matrix @ threshold=0.5")

    plt.tight_layout()
    out_path = Path(args.model_dir) / "evaluation_plots.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved evaluation plots to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/transactions.csv")
    parser.add_argument("--model_dir", type=str, default="../models")
    args = parser.parse_args()
    main(args)
