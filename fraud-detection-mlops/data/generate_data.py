"""
Synthetic credit card transaction data generator.

Mimics the statistical properties of real-world fraud datasets
(e.g. severe class imbalance ~0.17% fraud rate, time-of-day patterns,
amount distributions, and merchant-category risk skew) so the pipeline
can be developed and evaluated without relying on a restricted-license
dataset.

Run:
    python data/generate_data.py --n_rows 250000 --out data/transactions.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_transactions(n_rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    customer_ids = rng.integers(10_000, 99_999, size=n_rows)
    merchant_categories = rng.choice(
        ["grocery", "electronics", "travel", "restaurant", "gas_station",
         "online_retail", "jewelry", "entertainment", "utilities", "healthcare"],
        size=n_rows,
        p=[0.20, 0.10, 0.05, 0.15, 0.12, 0.18, 0.03, 0.07, 0.06, 0.04],
    )

    # legitimate transactions: mostly small, daytime, common categories
    legit_amount = np.round(rng.gamma(shape=2.0, scale=35, size=n_legit), 2)
    legit_hour = rng.normal(loc=14, scale=4.5, size=n_legit).clip(0, 23).astype(int)
    legit_seconds = rng.integers(0, 86400 * 30, size=n_legit)  # 30-day window
    legit_distance_from_home = rng.exponential(scale=8, size=n_legit)
    legit_is_online = rng.choice([0, 1], size=n_legit, p=[0.7, 0.3])

    # fraudulent transactions: skew toward high-risk categories, odd hours, larger/round amounts, distant
    fraud_amount = np.round(
        np.concatenate([
            rng.gamma(shape=1.5, scale=250, size=n_fraud // 2),
            rng.choice([1.0, 9.99, 99.99, 500.0, 999.0], size=n_fraud - n_fraud // 2)
            * rng.uniform(0.8, 1.2, size=n_fraud - n_fraud // 2),
        ]), 2
    )
    rng.shuffle(fraud_amount)
    fraud_hour = rng.choice(range(24), size=n_fraud, p=_night_skewed_hour_probs())
    fraud_seconds = rng.integers(0, 86400 * 30, size=n_fraud)
    fraud_distance_from_home = rng.exponential(scale=120, size=n_fraud)
    fraud_is_online = rng.choice([0, 1], size=n_fraud, p=[0.15, 0.85])

    fraud_merchant_categories = rng.choice(
        ["electronics", "jewelry", "online_retail", "travel", "entertainment"],
        size=n_fraud,
        p=[0.30, 0.20, 0.30, 0.12, 0.08],
    )

    df_legit = pd.DataFrame({
        "customer_id": customer_ids[:n_legit],
        "amount": legit_amount,
        "hour_of_day": legit_hour,
        "seconds_elapsed": legit_seconds,
        "merchant_category": merchant_categories[:n_legit],
        "distance_from_home_km": legit_distance_from_home,
        "is_online": legit_is_online,
        "is_fraud": 0,
    })

    df_fraud = pd.DataFrame({
        "customer_id": customer_ids[n_legit:],
        "amount": fraud_amount,
        "hour_of_day": fraud_hour,
        "seconds_elapsed": fraud_seconds,
        "merchant_category": fraud_merchant_categories,
        "distance_from_home_km": fraud_distance_from_home,
        "is_online": fraud_is_online,
        "is_fraud": 1,
    })

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = np.arange(1, len(df) + 1)
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    # a few realistic derived raw fields
    df["day_of_week"] = ((df["seconds_elapsed"] // 86400) % 7).astype(int)

    cols = ["transaction_id", "customer_id", "amount", "hour_of_day", "day_of_week",
            "seconds_elapsed", "merchant_category", "distance_from_home_km",
            "is_online", "is_fraud"]
    return df[cols]


def _night_skewed_hour_probs():
    # Fraud disproportionately occurs late night / early morning
    base = np.array([6, 6, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6], dtype=float)
    return base / base.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=250_000)
    parser.add_argument("--fraud_rate", type=float, default=0.0017)  # ~0.17%, matches real-world card fraud rates
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_transactions(args.n_rows, args.fraud_rate, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['is_fraud'].sum():,} fraud, "
          f"{df['is_fraud'].mean()*100:.3f}% fraud rate) to {args.out}")
