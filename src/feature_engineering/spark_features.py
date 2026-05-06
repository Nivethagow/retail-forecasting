"""
src/feature_engineering/spark_features.py
------------------------------------------
Reusable PySpark feature engineering pipeline for the retail
demand forecasting project. Mirrors the logic in
notebooks/01_databricks_feature_engineering.ipynb but is
structured as importable functions so it can be called from
Databricks Jobs or AWS Glue ETL scripts.

Usage (in a Databricks notebook or Glue job):
    from src.feature_engineering.spark_features import build_feature_df
    df_features = build_feature_df(spark, input_path, output_path)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import re


# ── Column name helpers ────────────────────────────────────────────────────────

def clean_col_name(name: str) -> str:
    """Standardise a column name: lowercase, replace non-alphanum with underscore."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def standardise_columns(df: DataFrame) -> DataFrame:
    """Rename all columns to snake_case."""
    return df.toDF(*[clean_col_name(c) for c in df.columns])


# ── Feature engineering steps ─────────────────────────────────────────────────

def add_temporal_features(df: DataFrame) -> DataFrame:
    """Extract day_of_week, month, year, is_weekend from the date column."""
    return (
        df
        .withColumn("day_of_week", F.dayofweek(F.col("date")) - 2)   # 0=Mon
        .withColumn("month", F.month(F.col("date")))
        .withColumn("year", F.year(F.col("date")))
        .withColumn("is_weekend", (F.col("day_of_week") >= 5).cast("int"))
    )


def add_lag_features(df: DataFrame) -> DataFrame:
    """
    Add lag1/lag2/lag3 units sold and a 3-day rolling average,
    partitioned by store and product so lags don't bleed across entities.
    """
    w = Window.partitionBy("store_id", "product_id").orderBy("date")
    w3 = w.rowsBetween(-2, 0)   # current row + 2 prior rows

    return (
        df
        .withColumn("lag1_units", F.lag("units_sold", 1).over(w))
        .withColumn("lag2_units", F.lag("units_sold", 2).over(w))
        .withColumn("lag3_units", F.lag("units_sold", 3).over(w))
        .withColumn("rolling_3d_avg", F.avg("units_sold").over(w3))
    )


def add_pricing_features(df: DataFrame) -> DataFrame:
    """Compute price_gap = store price - competitor price."""
    return df.withColumn("price_gap", F.col("price") - F.col("competitor_pricing"))


def add_promotion_feature(df: DataFrame) -> DataFrame:
    """
    Cast the holiday_promotion column to a binary promo_flag.
    Handles both string ('Yes'/'No') and numeric (0/1) representations.
    """
    col = F.col("holiday_promotion")
    promo = F.when(
        col.cast("string").isin("1", "yes", "Yes", "YES", "true", "True"), 1
    ).otherwise(0)
    return df.withColumn("promo_flag", promo.cast("int"))


def add_weather_index(df: DataFrame) -> DataFrame:
    """Ordinal-encode the weather_condition column using dense_rank."""
    w = Window.orderBy("weather_condition")
    return df.withColumn("weather_idx", (F.dense_rank().over(w) - 1).cast("int"))


def add_target_variable(df: DataFrame) -> DataFrame:
    """
    Create next_day_units_sold using lead() — the target for supervised learning.
    The last row per (store, product) partition will be null; those rows are
    dropped before writing the ML dataset.
    """
    w = Window.partitionBy("store_id", "product_id").orderBy("date")
    return df.withColumn("next_day_units_sold", F.lead("units_sold", 1).over(w))


# ── Full pipeline ──────────────────────────────────────────────────────────────

FINAL_COLUMNS = [
    "date", "store_id", "product_id", "units_sold",
    "price", "discount", "inventory_level", "competitor_pricing",
    "holiday_promotion", "weather_condition",
    # engineered
    "lag1_units", "lag2_units", "lag3_units", "rolling_3d_avg",
    "price_gap", "day_of_week", "month", "year", "is_weekend",
    "promo_flag", "weather_idx",
    # target
    "next_day_units_sold",
]


def build_feature_df(
    spark: SparkSession,
    input_path: str,
    output_path: str | None = None,
    drop_nulls: bool = True,
) -> DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        spark:        Active SparkSession
        input_path:   S3 or DBFS path to the raw CSV
        output_path:  If provided, write the result as Parquet to this path
        drop_nulls:   Drop rows where next_day_units_sold is null (last row per entity)

    Returns:
        Spark DataFrame with all engineered features
    """
    print(f"Reading raw data from {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df = standardise_columns(df)
    df = df.withColumn("date", F.to_date(F.col("date")))
    df = df.dropDuplicates()

    print("Running feature engineering pipeline…")
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_pricing_features(df)
    df = add_promotion_feature(df)
    df = add_weather_index(df)
    df = add_target_variable(df)

    # Keep only the columns we need, in a consistent order
    df = df.select(*FINAL_COLUMNS)

    if drop_nulls:
        df = df.dropna(subset=["next_day_units_sold"])

    row_count = df.count()
    print(f"Feature engineering complete. Rows: {row_count:,}")

    if output_path:
        print(f"Writing Parquet to {output_path}")
        df.write.mode("overwrite").parquet(output_path)
        print("  ✓ Parquet written")

    return df
