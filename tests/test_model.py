"""
tests/test_model.py
--------------------
Smoke tests and sanity checks for the trained XGBoost model
and the 2024 predictions file.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv("data/predictions/forecasted_2024_predictions.csv")


@pytest.fixture(scope="module")
def actuals():
    df = pd.read_csv("data/raw/retail_store_inventory.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


class TestPredictionsFile:
    def test_predictions_not_empty(self, predictions):
        assert len(predictions) > 0

    def test_expected_columns_present(self, predictions):
        required = {"Date", "Store ID", "Product ID", "predicted_next_day_units_sold"}
        assert required.issubset(set(predictions.columns))

    def test_no_null_predictions(self, predictions):
        nulls = predictions["predicted_next_day_units_sold"].isnull().sum()
        assert nulls == 0, f"Found {nulls} null predictions"

    def test_predictions_are_positive(self, predictions):
        neg = (predictions["predicted_next_day_units_sold"] < 0).sum()
        assert neg == 0, f"Found {neg} negative predictions"

    def test_prediction_range_is_reasonable(self, predictions):
        col = predictions["predicted_next_day_units_sold"]
        # Predictions should stay within a plausible demand range
        assert col.min() > 50, "Suspiciously low minimum prediction"
        assert col.max() < 500, "Suspiciously high maximum prediction"

    def test_mean_prediction_close_to_actuals(self, predictions, actuals):
        """
        The mean prediction for 2024-01-01 should be close to the mean actual
        for the same date (within ±20 units).  From the report: actual=132.44, pred=134.70.
        """
        jan1_pred = predictions[predictions["Date"] == "2024-01-01"]["predicted_next_day_units_sold"].mean()
        jan1_actual = actuals[actuals["Date"] == pd.Timestamp("2024-01-01")]["Units Sold"].mean()
        assert abs(jan1_pred - jan1_actual) < 20, (
            f"Mean prediction ({jan1_pred:.2f}) is too far from mean actual ({jan1_actual:.2f})"
        )


class TestModelConsistency:
    def test_all_stores_covered(self, predictions):
        expected = {"S001", "S002", "S003", "S004", "S005"}
        assert set(predictions["Store ID"].unique()) == expected

    def test_all_products_covered(self, predictions):
        expected = {f"P{str(i).zfill(4)}" for i in range(1, 21)}
        assert set(predictions["Product ID"].unique()) == expected

    def test_predictions_are_not_all_identical(self, predictions):
        """Model should produce some variation — not just a flat mean."""
        std = predictions["predicted_next_day_units_sold"].std()
        assert std > 1.0, f"Predictions have suspiciously low variance: std={std:.4f}"
