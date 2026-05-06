"""
src/modeling/train_xgboost.py
------------------------------
Train and evaluate the XGBoost demand forecasting model.
Can be run locally (reads from data/processed/) or inside a
SageMaker Training Job (reads from /opt/ml/input/data/).

Usage (local):
    python src/modeling/train_xgboost.py \
        --config configs/aws_config.yaml \
        --model-config configs/model_config.yaml \
        --local

Usage (SageMaker job — called automatically by the estimator):
    python src/modeling/train_xgboost.py \
        --config configs/aws_config.yaml \
        --model-config configs/model_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(model_cfg: dict, local: bool) -> pd.DataFrame:
    if local:
        parquet_path = "data/processed/retail_store_inventory_cleaned.parquet"
        print(f"Loading local Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        # SageMaker mounts training data at /opt/ml/input/data/training/
        data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
        parquet_files = list(Path(data_dir).glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {data_dir}")
        print(f"Loading {parquet_files[0]}")
        df = pd.read_parquet(parquet_files[0])

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ── Train/test split ───────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, model_cfg: dict):
    split = model_cfg["train_test_split"]
    date_col = split["split_column"]

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        train = df[df[date_col] <= split["train_end"]]
        test = df[(df[date_col] >= split["test_start"]) & (df[date_col] <= split["test_end"])]
    else:
        # Parquet may not have Date col — use positional 80/20 split
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


# ── Feature/target extraction ──────────────────────────────────────────────────

def get_feature_target(df: pd.DataFrame, model_cfg: dict):
    all_features = (
        model_cfg["features"]["numeric"] + model_cfg["features"]["binary"]
    )
    target = model_cfg["model"]["target"]

    # Use only columns present in df
    available = [c for c in all_features if c in df.columns]
    missing = set(all_features) - set(available)
    if missing:
        print(f"  ⚠ Features not found in data (skipped): {missing}")

    X = df[available]
    y = df[target] if target in df.columns else None
    return X, y, available


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(y_true: pd.Series, y_pred: np.ndarray, threshold: int = 10) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Raw MAPE (will be huge if zeros present)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_mape_arr = np.abs((y_true - y_pred) / y_true)
        raw_mape = float(np.mean(raw_mape_arr.replace([np.inf, -np.inf], np.nan).dropna()) * 100)

    # Filtered MAPE
    mask = y_true > threshold
    filtered_mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ) if mask.sum() > 0 else None

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_all": round(raw_mape, 4),
        f"mape_filtered_gt{threshold}": round(filtered_mape, 4) if filtered_mape else None,
        "test_samples": int(len(y_true)),
        "zero_sale_samples": int((y_true == 0).sum()),
    }
    return metrics


def evaluate_by_group(test_df: pd.DataFrame, y_pred: np.ndarray, target: str, group_col: str) -> pd.DataFrame:
    """Per-group RMSE and MAE breakdown."""
    tmp = test_df[[group_col, target]].copy()
    tmp["y_pred"] = y_pred
    rows = []
    for grp, gdf in tmp.groupby(group_col):
        rows.append({
            group_col: grp,
            "rmse": round(np.sqrt(mean_squared_error(gdf[target], gdf["y_pred"])), 2),
            "mae": round(mean_absolute_error(gdf[target], gdf["y_pred"]), 2),
            "n": len(gdf),
        })
    return pd.DataFrame(rows).sort_values("rmse")


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aws_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--local", action="store_true", help="Run locally instead of in SageMaker")
    parser.add_argument("--output-dir", default="data/predictions", help="Where to save predictions CSV")
    args = parser.parse_args()

    aws_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    hp = model_cfg["hyperparameters"]

    # ── Load & split ──
    df = load_data(model_cfg, local=args.local)
    train_df, test_df = split_data(df, model_cfg)

    X_train, y_train, features_used = get_feature_target(train_df, model_cfg)
    X_test, y_test, _ = get_feature_target(test_df, model_cfg)

    print(f"\nFeatures used ({len(features_used)}): {features_used}")

    # ── Train ──
    print("\nTraining XGBoost model…")
    model = xgb.XGBRegressor(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        random_state=hp["random_state"],
        objective=hp["objective"],
        eval_metric=hp["eval_metric"],
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("  ✓ Training complete")

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    threshold = model_cfg["evaluation"]["filtered_mape_threshold"]
    metrics = evaluate(y_test, y_pred, threshold)

    print("\n── Evaluation Metrics ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    # Per-store and per-product breakdowns (if Date col present)
    if "Store ID" in test_df.columns:
        by_store = evaluate_by_group(test_df, y_pred, model_cfg["model"]["target"], "Store ID")
        print("\n── RMSE by Store ──")
        print(by_store.to_string(index=False))

    if "Product ID" in test_df.columns:
        by_product = evaluate_by_group(test_df, y_pred, model_cfg["model"]["target"], "Product ID")
        print("\n── RMSE by Product ──")
        print(by_product.to_string(index=False))

    # ── Save outputs ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved to {metrics_path}")

    # Model artifact (for SageMaker, also write to /opt/ml/model/)
    model_path = output_dir / "xgboost_model.json"
    model.save_model(str(model_path))
    print(f"  ✓ Model saved to {model_path}")
    if not args.local:
        sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        model.save_model(os.path.join(sm_model_dir, "xgboost_model.json"))

    # Test predictions CSV
    preds_df = test_df.copy()
    preds_df["predicted_next_day_units_sold"] = y_pred
    preds_path = output_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"  ✓ Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
