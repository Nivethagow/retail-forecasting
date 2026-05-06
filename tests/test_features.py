"""
tests/test_features.py
-----------------------
Unit tests for the Pandas-equivalent feature engineering logic.
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest


# ── Helpers that mirror spark_features.py but in Pandas ───────────────────────

def add_lag_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "date"])
    grp = df.groupby(["store_id", "product_id"])["units_sold"]
    df["lag1_units"] = grp.shift(1)
    df["lag2_units"] = grp.shift(2)
    df["lag3_units"] = grp.shift(3)
    df["rolling_3d_avg"] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    return df


def add_price_gap(df: pd.DataFrame) -> pd.DataFrame:
    df["price_gap"] = df["price"] - df["competitor_pricing"]
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "date"])
    df["next_day_units_sold"] = df.groupby(["store_id", "product_id"])["units_sold"].shift(-1)
    return df


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal 6-row dataframe: 2 stores × 3 dates."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"] * 2),
        "store_id": ["S001"] * 3 + ["S002"] * 3,
        "product_id": ["P0001"] * 6,
        "units_sold": [100, 120, 80, 200, 150, 170],
        "price": [10.0] * 6,
        "competitor_pricing": [9.0] * 6,
    })


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag1_first_row_is_nan(self, sample_df):
        df = add_lag_features_pandas(sample_df)
        # First row per (store, product) has no prior day
        s1 = df[df["store_id"] == "S001"].sort_values("date")
        assert np.isnan(s1.iloc[0]["lag1_units"])

    def test_lag1_second_row_equals_first_units(self, sample_df):
        df = add_lag_features_pandas(sample_df)
        s1 = df[df["store_id"] == "S001"].sort_values("date").reset_index(drop=True)
        assert s1.loc[1, "lag1_units"] == s1.loc[0, "units_sold"]

    def test_lags_dont_bleed_across_stores(self, sample_df):
        df = add_lag_features_pandas(sample_df)
        s2_first = df[df["store_id"] == "S002"].sort_values("date").iloc[0]
        # lag1 of S002's first row must be NaN — not S001's last value
        assert np.isnan(s2_first["lag1_units"])

    def test_rolling_3d_avg_is_correct(self, sample_df):
        df = add_lag_features_pandas(sample_df)
        s1 = df[df["store_id"] == "S001"].sort_values("date").reset_index(drop=True)
        # Third row rolling avg = mean(100, 120, 80)
        expected = np.mean([100, 120, 80])
        assert abs(s1.loc[2, "rolling_3d_avg"] - expected) < 1e-6


class TestPriceGap:
    def test_price_gap_calculation(self, sample_df):
        df = add_price_gap(sample_df)
        assert all(df["price_gap"] == 1.0)

    def test_negative_price_gap(self):
        df = pd.DataFrame({"price": [8.0], "competitor_pricing": [10.0]})
        df = add_price_gap(df)
        assert df["price_gap"].iloc[0] == -2.0


class TestTargetVariable:
    def test_target_is_next_day_units(self, sample_df):
        df = add_target(sample_df)
        s1 = df[df["store_id"] == "S001"].sort_values("date").reset_index(drop=True)
        assert s1.loc[0, "next_day_units_sold"] == s1.loc[1, "units_sold"]

    def test_last_row_target_is_nan(self, sample_df):
        df = add_target(sample_df)
        s1 = df[df["store_id"] == "S001"].sort_values("date").reset_index(drop=True)
        assert np.isnan(s1.iloc[-1]["next_day_units_sold"])


class TestDataIntegrity:
    def test_no_duplicate_store_product_date(self):
        df = pd.read_csv("data/raw/retail_store_inventory.csv")
        dupes = df.duplicated(subset=["Date", "Store ID", "Product ID"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate rows"

    def test_expected_row_count(self):
        df = pd.read_csv("data/raw/retail_store_inventory.csv")
        # 5 stores × 20 products × 731 days = 73,100
        assert len(df) == 73100, f"Expected 73100 rows, got {len(df)}"

    def test_all_stores_present(self):
        df = pd.read_csv("data/raw/retail_store_inventory.csv")
        expected_stores = {"S001", "S002", "S003", "S004", "S005"}
        assert set(df["Store ID"].unique()) == expected_stores

    def test_predictions_cover_full_2024(self):
        df = pd.read_csv("data/predictions/forecasted_2024_predictions.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        assert df["Date"].min() == pd.Timestamp("2024-01-01")
        assert df["Date"].max() == pd.Timestamp("2024-12-31")
