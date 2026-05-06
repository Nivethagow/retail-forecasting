# Retail Demand Forecasting

> GBA 6430 Big Data Technology in Business  
> California State Polytechnic University, Pomona  
> Instructor: Dr. Mehrdad Koohikamali  
> Author: Nivetha Ramasamy

Next-day SKU-level demand forecasting for a nationwide retail chain using a full AWS + Databricks pipeline.

---

## Architecture Overview

```
Kaggle CSV
    │
    ▼
Amazon S3 (raw/)          ← central data lake
    │
    ▼
AWS Glue Crawler          ← schema inference + Data Catalog
    │
    ▼
Amazon Athena             ← serverless SQL EDA & profiling
    │
    ▼
Databricks / PySpark      ← feature engineering (lag, rolling avg, price gap)
    │  saves Parquet
    ▼
Amazon S3 (processed/)    ← ML-ready feature store
    │
    ▼
Amazon SageMaker          ← XGBoost training + evaluation
    │  logs metrics to CloudWatch
    ▼
Amazon S3 (predictions/)  ← model artifacts + forecast CSVs
    │
    ▼
Tableau / QuickSight      ← interactive dashboards
```

---

## Repository Structure

```
retail-forecasting/
├── data/
│   ├── raw/                         # Original Kaggle CSV (73,101 rows, 2022–2024)
│   ├── processed/                   # Cleaned Parquet + ML-ready CSV with engineered features
│   └── predictions/                 # 2024 forecast output from trained XGBoost model
│
├── notebooks/
│   ├── 01_databricks_feature_engineering.ipynb   # PySpark EDA + feature engineering
│   └── 02_sagemaker_training.ipynb               # XGBoost training, evaluation, visualizations
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── s3_upload.py             # Upload raw CSV to S3, configure IAM
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── spark_features.py        # PySpark feature pipeline (reusable)
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── train_xgboost.py         # SageMaker XGBoost training + evaluation
│   └── visualization/
│       ├── __init__.py
│       └── dashboards.py            # Athena → Tableau/QuickSight helpers
│
├── configs/
│   ├── aws_config.yaml              # S3 buckets, region, IAM role names
│   └── model_config.yaml            # XGBoost hyperparameters + feature list
│
├── tests/
│   ├── test_features.py
│   └── test_model.py
│
├── docs/
│   └── architecture.md              # Detailed AWS architecture notes
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

| File | Description | Rows |
|------|-------------|------|
| `data/raw/retail_store_inventory.csv` | Original Kaggle dataset | 73,101 |
| `data/processed/retail_store_inventory_cleaned.parquet` | Cleaned Parquet with engineered features | 73,100 |
| `data/processed/retail_store_inventory_cleaned_for_ml.csv` | Full ML-ready CSV including all features + target | 73,100 |
| `data/predictions/forecasted_2024_predictions.csv` | XGBoost predictions for 2024 (all stores × products) | 36,600 |

**Coverage:** 5 stores × 20 products × 731 days (2022-01-01 → 2024-01-01)

---

## Engineered Features

| Feature | Description |
|---------|-------------|
| `lag1_units`, `lag2_units`, `lag3_units` | Units sold 1, 2, 3 days prior (per store-product) |
| `rolling_3d_avg` | 3-day rolling average of units sold |
| `price_gap` | Store price minus competitor price |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `month` | Calendar month (1–12) |
| `is_weekend` | Binary flag (Saturday/Sunday) |
| `promo_flag` | Binary: 1 if holiday/promotion active |
| `weather_idx` | Ordinal encoding of weather condition |
| `next_day_units_sold` | **Target variable** — lead(units_sold, 1) |

---

## Model Results (Q4 2023 Test Set)

| Metric | Value | Notes |
|--------|-------|-------|
| MAPE (all samples) | 2.23 × 10¹⁷ % | Inflated by zero-sale days |
| MAPE (actual > 10 units) | 132.84 % | More meaningful business metric |
| RMSE | 108.93 units | Consistent across stores (~108–111) |
| MAE | ~89 units | Avg error vs mean demand of ~132 units |
| Avg actual (2024-01-01) | 132.44 units | Store-level aggregate |
| Avg predicted (2024-01-01) | 134.70 units | <2 unit deviation at aggregate level |

---

## Quick Start

### 1. Configure AWS credentials

```bash
cp configs/aws_config.yaml configs/aws_config.local.yaml
# Fill in your bucket name, region, and IAM role ARN
```

### 2. Upload raw data to S3

```bash
python src/ingestion/s3_upload.py \
  --file data/raw/retail_store_inventory.csv \
  --config configs/aws_config.yaml
```

### 3. Run feature engineering (Databricks)

Open `notebooks/01_databricks_feature_engineering.ipynb` in your Databricks workspace.  
The notebook reads from your S3 bucket and writes `retail_store_inventory_cleaned.parquet` back to `s3://your-bucket/processed/`.

### 4. Train the model (SageMaker)

Open `notebooks/02_sagemaker_training.ipynb` in a SageMaker Studio or Notebook Instance.  
Or run via CLI:

```bash
python src/modeling/train_xgboost.py \
  --config configs/aws_config.yaml \
  --model-config configs/model_config.yaml
```

### 5. Visualize results

Connect Tableau to Athena via ODBC using your IAM Access Key/Secret, or run:

```bash
python src/visualization/dashboards.py
```

---

## Requirements

```
boto3>=1.34
pandas>=2.0
pyarrow>=14.0
xgboost>=2.0
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
awswrangler>=3.6
```

Install: `pip install -r requirements.txt`

---
