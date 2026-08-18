"""
Batch scoring CLI: score a CSV of transactions using the trained model
without needing the API running. Useful for scheduled/offline scoring jobs.

Run:
    python src/predict.py --input ../data/transactions.csv --output ../data/scored.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import FraudFeatureEngineer  # noqa: F401  (needed for unpickling)


def main(args):
    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.joblib")
    fe = joblib.load(model_dir / "feature_engineer.joblib")
    feature_cols = joblib.load(model_dir / "feature_columns.joblib")

    import json
    with open(model_dir / "metrics.json") as f:
        threshold = json.load(f).get("decision_threshold", 0.5)

    df = pd.read_csv(args.input)
    X = fe.transform(df).reindex(columns=feature_cols, fill_value=0)
    df["fraud_probability"] = model.predict_proba(X)[:, 1].round(4)
    df["is_fraud_flag"] = df["fraud_probability"] >= threshold

    df.to_csv(args.output, index=False)
    flagged = df["is_fraud_flag"].sum()
    print(f"Scored {len(df):,} transactions -> {flagged:,} flagged as fraud "
          f"({flagged/len(df)*100:.3f}%). Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="scored_transactions.csv")
    parser.add_argument("--model_dir", type=str, default="../models")
    args = parser.parse_args()
    main(args)
