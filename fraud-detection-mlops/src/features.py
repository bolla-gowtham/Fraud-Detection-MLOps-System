"""
Feature engineering for the fraud detection pipeline.

Transforms raw transaction fields into model-ready features:
 - cyclical encoding of time features (hour, day-of-week)
 - customer-level rolling aggregates (spending behavior)
 - risk-encoded categoricals
 - log-scaled monetary features

These transforms are wrapped in a scikit-learn compatible transformer
so the exact same logic runs at training time and inference time
(preventing train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HIGH_RISK_CATEGORIES = {"electronics", "jewelry", "online_retail", "travel", "entertainment"}


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateful feature transformer: learns per-customer stats on fit(),
    applies them (with safe fallbacks for unseen customers) on transform().
    """

    def __init__(self):
        self.customer_avg_amount_: dict | None = None
        self.customer_txn_count_: dict | None = None
        self.global_avg_amount_: float = 0.0

    def fit(self, X: pd.DataFrame, y=None):
        self.customer_avg_amount_ = X.groupby("customer_id")["amount"].mean().to_dict()
        self.customer_txn_count_ = X.groupby("customer_id")["amount"].count().to_dict()
        self.global_avg_amount_ = float(X["amount"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # cyclical time encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)

        # monetary features
        df["log_amount"] = np.log1p(df["amount"])

        # behavioral deviation vs customer's historical average (key fraud signal)
        cust_avg = df["customer_id"].map(self.customer_avg_amount_).fillna(self.global_avg_amount_)
        df["amount_vs_customer_avg"] = df["amount"] / (cust_avg + 1e-6)
        df["customer_txn_count"] = df["customer_id"].map(self.customer_txn_count_).fillna(0)

        # categorical risk encoding
        df["is_high_risk_category"] = df["merchant_category"].apply(
            lambda c: 1 if c in HIGH_RISK_CATEGORIES else 0
        )
        df = pd.get_dummies(df, columns=["merchant_category"], prefix="cat")

        # distance-based signal
        df["log_distance"] = np.log1p(df["distance_from_home_km"])

        drop_cols = [c for c in ["transaction_id", "customer_id", "amount",
                                  "hour_of_day", "day_of_week", "seconds_elapsed",
                                  "distance_from_home_km", "is_fraud"] if c in df.columns]
        return df.drop(columns=drop_cols)

    def feature_names(self, X: pd.DataFrame) -> list[str]:
        return list(self.transform(X.head(5)).columns)
