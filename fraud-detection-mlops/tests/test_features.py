import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import FraudFeatureEngineer  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "customer_id": [100, 100, 200],
        "amount": [50.0, 900.0, 20.0],
        "hour_of_day": [14, 3, 10],
        "day_of_week": [1, 6, 3],
        "seconds_elapsed": [1000, 2000, 3000],
        "merchant_category": ["grocery", "electronics", "travel"],
        "distance_from_home_km": [2.0, 300.0, 15.0],
        "is_online": [0, 1, 0],
        "is_fraud": [0, 1, 0],
    })


def test_fit_transform_shape(sample_df):
    fe = FraudFeatureEngineer()
    fe.fit(sample_df)
    out = fe.transform(sample_df)
    assert len(out) == len(sample_df)
    assert "is_fraud" not in out.columns
    assert "customer_id" not in out.columns


def test_cyclical_encoding_bounds(sample_df):
    fe = FraudFeatureEngineer()
    fe.fit(sample_df)
    out = fe.transform(sample_df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert out[col].between(-1.0, 1.0).all()


def test_unseen_customer_uses_global_fallback(sample_df):
    fe = FraudFeatureEngineer()
    fe.fit(sample_df)
    new_row = pd.DataFrame({
        "transaction_id": [99],
        "customer_id": [999999],  # never seen in fit()
        "amount": [40.0],
        "hour_of_day": [12],
        "day_of_week": [2],
        "seconds_elapsed": [5000],
        "merchant_category": ["grocery"],
        "distance_from_home_km": [1.0],
        "is_online": [0],
        "is_fraud": [0],
    })
    out = fe.transform(new_row)
    assert not out["amount_vs_customer_avg"].isna().any()


def test_night_flag(sample_df):
    fe = FraudFeatureEngineer()
    fe.fit(sample_df)
    out = fe.transform(sample_df)
    # row 1 (hour=3) should be flagged night, row 0 (hour=14) should not
    assert out.loc[1, "is_night"] == 1
    assert out.loc[0, "is_night"] == 0
