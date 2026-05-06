# Architecture — Retail Demand Forecasting Pipeline

**Author:** Nivetha Ramasamy  


---

## Overview

This project builds an end-to-end next-day SKU-level demand forecasting pipeline for a retail chain using a full AWS cloud stack combined with Databricks for distributed feature engineering. The pipeline spans five logical tiers: ingestion, cataloging, processing, modeling, and visualization.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1 — INGESTION                                                 │
│                                                                     │
│   Kaggle CSV (73,100 rows)  ──►  Amazon S3 (data/raw/)             │
│   • 5 stores × 20 products × 731 days (2022–2024)                  │
│   • IAM roles govern all service access                             │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2 — CATALOGING                                                │
│                                                                     │
│   AWS Glue Crawler  ──►  Glue Data Catalog  ──►  Amazon Athena     │
│   • Crawler infers schema from S3 CSV                               │
│   • Athena runs serverless SQL EDA (no data movement)               │
│   • Outputs: descriptive stats, demand distributions, zero-sales    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3 — PROCESSING                                                │
│                                                                     │
│   Databricks / PySpark  ──►  Amazon S3 (data/processed/ Parquet)   │
│   • Spark Window functions for lag features (partitioned by         │
│     store × product to prevent data leakage)                        │
│   • Features: lag1/2/3, rolling_3d_avg, price_gap,                  │
│     day_of_week, month, year, is_weekend, promo_flag, weather_idx   │
│   • Target: next_day_units_sold via lead()                          │
│   • Output saved as Parquet (columnar, compressed)                  │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 4 — MODELING                                                  │
│                                                                     │
│   Amazon SageMaker  ──►  Amazon S3 (data/predictions/)             │
│   • Managed XGBoost container (framework_version: 1.7-1)            │
│   • Train: Jan 2022 – Sep 2023  |  Test: Oct 2023 – Jan 2024       │
│   • Metrics logged to Amazon CloudWatch                             │
│   • Model artifact stored in S3                                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 5 — VISUALIZATION                                             │
│                                                                     │
│   Tableau (active)  ←── ODBC ──  Amazon Athena                     │
│   Amazon QuickSight (planned) ──  Glue Data Catalog                 │
│   • Scatter: predicted vs actual units sold                         │
│   • Line: daily demand trend 2022–2024                              │
│   • Bar: actual vs predicted by product per store                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## AWS Services Reference

| Service | Tier | Purpose | Key Config |
|---------|------|---------|------------|
| Amazon S3 | Ingestion | Central data lake — raw CSV, Parquet, model artifacts, predictions | Bucket: `retail-forecasting-nivetha` |
| IAM | Cross-cutting | Role-based access control for all services | Roles: Glue crawler, SageMaker execution |
| AWS Glue | Cataloging | Schema inference via crawler → Glue Data Catalog | DB: `retail_forecasting_db` |
| Amazon Athena | Cataloging | Serverless SQL EDA directly on S3 data | Workgroup: `primary` |
| Databricks / PySpark | Processing | Distributed feature engineering using Window functions | Community Edition |
| Amazon SageMaker | Modeling | Managed XGBoost training + evaluation + inference | Instance: `ml.m5.xlarge` |
| Amazon CloudWatch | Monitoring | Training metrics (MAPE, RMSE, MAE) logged automatically | Auto-configured by SageMaker |
| Amazon QuickSight | Visualization | Planned BI dashboard (IAM trust policy pending) | Connected to Glue catalog |
| Tableau | Visualization | Active fallback — ODBC connection to Athena | IAM Access Key auth |

---

## Data Flow

```
retail_store_inventory.csv          (raw, 73,100 rows, 15 cols)
        │  S3 upload
        ▼
retail_store_inventory_cleaned.parquet   (processed, 73,100 rows, 12 cols)
        │  SageMaker training
        ▼
retail_store_inventory_cleaned_for_ml.csv  (full ML dataset, 22 cols incl. target)
        │  XGBoost inference
        ▼
forecasted_2024_predictions.csv     (36,600 rows — all stores × products × 365 days)
```

---

## Engineered Features

| Feature | Type | Description |
|---------|------|-------------|
| `lag1_units` | Numeric | Units sold 1 day prior (per store-product partition) |
| `lag2_units` | Numeric | Units sold 2 days prior |
| `lag3_units` | Numeric | Units sold 3 days prior |
| `rolling_3d_avg` | Numeric | 3-day rolling mean of units sold |
| `price_gap` | Numeric | Store price minus competitor price |
| `day_of_week` | Numeric | 0 = Monday … 6 = Sunday |
| `month` | Numeric | Calendar month (1–12) |
| `year` | Numeric | Calendar year |
| `is_weekend` | Binary | 1 if Saturday or Sunday |
| `promo_flag` | Binary | 1 if holiday/promotion active |
| `weather_idx` | Ordinal | Dense-rank encoding of weather condition |
| `next_day_units_sold` | **Target** | `lead(units_sold, 1)` over store-product window |

---

## Model Results

| Metric | Value | Notes |
|--------|-------|-------|
| MAPE (all samples) | 2.23 × 10¹⁷ % | Inflated by zero-sale days in denominator |
| MAPE (actual > 10 units) | 132.84 % | Business-meaningful metric |
| RMSE | ~109 units | Consistent across all 5 stores (108–111) |
| MAE | ~89 units | vs mean actual demand of ~132 units |
| Avg actual (2024-01-01) | 132.44 units | Store-level aggregate |
| Avg predicted (2024-01-01) | 134.70 units | < 2 unit deviation at aggregate level |

### Key Finding
The model performs well at the **aggregate level** (mean prediction within 2 units of mean actual) but struggles with **individual SKU variance** — predictions cluster around 130–150 units while actual sales range from near-zero to 500+. This is a known limitation of static lag features frozen at the training cutoff date.

---

## Known Issues & Planned Improvements

### QuickSight IAM Issue
Amazon QuickSight could not access the Glue Data Catalog due to an IAM trust policy conflict between the QuickSight service role and the Glue crawler role. Tableau was used as a fallback via ODBC. **Fix:** Add `quicksight.amazonaws.com` as a trusted principal in the Glue IAM role.

### Model Limitations
- Lag features are static (frozen at Sept 2023 training cutoff) — rolling updates needed for production
- XGBoost predicts a single mean value per SKU; does not model demand variance
- Dataset is simulated (Kaggle) — all products have identical record counts (3,655 each), which is unrealistic

### Planned Improvements
- Rolling feature updates via a scheduled SageMaker Processing Job
- Segmented models by demand volatility (high/medium/low)
- LSTM or Transformer-based sequence model for better temporal dependency
- Weekly aggregation to reduce noise and improve MAPE
- Real QuickSight dashboard once IAM trust policy is resolved

---

## Repository Structure

```
retail-forecasting/
├── data/
│   ├── raw/                    # Original Kaggle CSV
│   ├── processed/              # Cleaned Parquet + ML-ready CSV
│   └── predictions/            # 2024 XGBoost forecasts
├── notebooks/
│   ├── 01_databricks_feature_engineering.ipynb
│   └── 02_sagemaker_training.ipynb
├── src/
│   ├── ingestion/s3_upload.py
│   ├── feature_engineering/spark_features.py
│   ├── modeling/train_xgboost.py
│   └── visualization/dashboards.py
├── configs/
│   ├── aws_config.yaml         # AWS service config (no secrets)
│   └── model_config.yaml       # XGBoost hyperparameters + feature list
├── tests/
│   ├── test_features.py
│   └── test_model.py
├── docs/
│   └── architecture.md         # This file
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Running the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS
cp configs/aws_config.yaml configs/aws_config.local.yaml
# Edit aws_config.local.yaml with your real values

# 3. Upload raw data to S3
python src/ingestion/s3_upload.py --file data/raw/retail_store_inventory.csv

# 4. Feature engineering → open in Databricks
#    notebooks/01_databricks_feature_engineering.ipynb

# 5. Model training → open in SageMaker Studio
#    notebooks/02_sagemaker_training.ipynb

# 6. Or train locally
python src/modeling/train_xgboost.py --local

# 7. Generate charts
python src/visualization/dashboards.py

# 8. Run tests
pytest tests/
```
