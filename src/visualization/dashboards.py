"""
src/visualization/dashboards.py
---------------------------------
Query Athena for dashboard-ready aggregates and produce
matplotlib charts that replicate the Tableau dashboards.

Run locally to preview charts, or use results to populate
QuickSight / Tableau via the exported CSVs.

Usage:
    python src/visualization/dashboards.py \
        --predictions data/predictions/forecasted_2024_predictions.csv \
        --actuals data/raw/retail_store_inventory.csv \
        --output-dir data/predictions/charts
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Shared style ───────────────────────────────────────────────────────────────

PALETTE = {
    "actual": "#2E75B6",
    "predicted": "#ED7D31",
    "accent": "#70AD47",
    "neutral": "#BFBFBF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Chart 1: Scatter — predicted vs actual ────────────────────────────────────

def plot_scatter_predicted_vs_actual(
    df: pd.DataFrame,
    output_path: Path | None = None,
    date_filter: str | None = "2024-01-01",
) -> None:
    """
    Scatter plot of predicted vs actual units sold.
    Replicates Chart 1 from the report.
    """
    if date_filter:
        plot_df = df[df["Date"] == date_filter].copy()
        title_suffix = f" ({date_filter})"
    else:
        plot_df = df.copy()
        title_suffix = ""

    if plot_df.empty:
        print(f"No data for date_filter={date_filter}")
        return

    x = plot_df["Units Sold"]
    y = plot_df["predicted_next_day_units_sold"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.5, color=PALETTE["actual"], edgecolors="none", s=30)

    # y = x reference line
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=PALETTE["neutral"], linewidth=1, label="Perfect forecast (y = x)")

    ax.set_xlabel("Actual Units Sold", fontsize=11)
    ax.set_ylabel("Predicted Units Sold", fontsize=11)
    ax.set_title(f"Predicted vs. Actual Units Sold{title_suffix}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Annotation: avg error
    mae = np.mean(np.abs(x - y))
    ax.text(0.05, 0.92, f"MAE = {mae:.1f} units", transform=ax.transAxes,
            fontsize=9, color="gray")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  ✓ Saved scatter plot → {output_path}")
    plt.show()
    plt.close()


# ── Chart 2: Line — daily avg units sold over time ────────────────────────────

def plot_daily_demand_trend(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Line chart of daily average units sold (actuals + predictions).
    Replicates Chart 2 from the report.
    """
    actuals["Date"] = pd.to_datetime(actuals["Date"])
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    daily_actual = actuals.groupby("Date")["Units Sold"].mean().reset_index()
    daily_pred = predictions.groupby("Date")["predicted_next_day_units_sold"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(daily_actual["Date"], daily_actual["Units Sold"],
            color=PALETTE["actual"], linewidth=1, label="Actual (2022–2024)", alpha=0.85)
    ax.plot(daily_pred["Date"], daily_pred["predicted_next_day_units_sold"],
            color=PALETTE["predicted"], linewidth=1.5, label="Predicted (2024)", alpha=0.9)

    ax.axvline(pd.Timestamp("2023-10-01"), color="gray", linewidth=0.8,
               linestyle="--", alpha=0.6, label="Train / Test split")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Avg Units Sold", fontsize=11)
    ax.set_title("Daily Average Units Sold — Actuals vs Predictions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  ✓ Saved trend chart → {output_path}")
    plt.show()
    plt.close()


# ── Chart 3: Bar — top 10 products by predicted demand ───────────────────────

def plot_top_products_bar(
    df: pd.DataFrame,
    store_id: str = "S001",
    date_filter: str = "2024-01-01",
    output_path: Path | None = None,
) -> None:
    """
    Grouped bar chart of actual vs predicted per product for one store/date.
    Replicates Chart 3 from the report.
    """
    plot_df = df[(df["Date"] == date_filter) & (df["Store ID"] == store_id)].copy()
    if plot_df.empty:
        print(f"No data for store={store_id} date={date_filter}")
        return

    plot_df = plot_df.sort_values("Units Sold", ascending=False).head(20)

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, plot_df["Units Sold"], width, color=PALETTE["actual"],
           label="Actual", alpha=0.85)
    ax.bar(x + width / 2, plot_df["predicted_next_day_units_sold"], width,
           color=PALETTE["predicted"], label="Predicted", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Product ID"], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Product ID", fontsize=11)
    ax.set_ylabel("Units Sold", fontsize=11)
    ax.set_title(
        f"Actual vs Predicted Units — Store {store_id} ({date_filter})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  ✓ Saved bar chart → {output_path}")
    plt.show()
    plt.close()


# ── Aggregate summary for QuickSight / Tableau export ─────────────────────────

def export_dashboard_aggregates(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write pre-aggregated CSVs that can be imported directly into Tableau or QuickSight."""
    output_dir.mkdir(parents=True, exist_ok=True)

    actuals["Date"] = pd.to_datetime(actuals["Date"])
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    # Daily store-level actuals
    daily_store = (
        actuals.groupby(["Date", "Store ID"])["Units Sold"]
        .agg(["sum", "mean"])
        .rename(columns={"sum": "total_units_sold", "mean": "avg_units_sold"})
        .reset_index()
    )
    path = output_dir / "daily_store_actuals.csv"
    daily_store.to_csv(path, index=False)
    print(f"  ✓ {path}")

    # Daily product-level predictions
    daily_product_pred = (
        predictions.groupby(["Date", "Product ID"])["predicted_next_day_units_sold"]
        .mean()
        .reset_index()
    )
    path = output_dir / "daily_product_predictions.csv"
    daily_product_pred.to_csv(path, index=False)
    print(f"  ✓ {path}")

    # Merged comparison (for scatter / MAPE calculation)
    merged = predictions.merge(
        actuals[["Date", "Store ID", "Product ID", "Units Sold"]],
        on=["Date", "Store ID", "Product ID"],
        how="left",
    )
    merged["abs_error"] = (merged["Units Sold"] - merged["predicted_next_day_units_sold"]).abs()
    path = output_dir / "predictions_vs_actuals.csv"
    merged.to_csv(path, index=False)
    print(f"  ✓ {path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate dashboard charts and exports")
    parser.add_argument("--predictions", default="data/predictions/forecasted_2024_predictions.csv")
    parser.add_argument("--actuals", default="data/raw/retail_store_inventory.csv")
    parser.add_argument("--output-dir", default="data/predictions/charts")
    parser.add_argument("--store", default="S001", help="Store ID for bar chart")
    parser.add_argument("--date", default="2024-01-01", help="Date filter for scatter/bar charts")
    parser.add_argument("--no-show", action="store_true", help="Save charts without displaying")
    args = parser.parse_args()

    if args.no_show:
        plt.switch_backend("Agg")

    print("Loading data…")
    predictions = pd.read_csv(args.predictions)
    actuals = pd.read_csv(args.actuals)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge for scatter and bar
    merged = predictions.merge(
        actuals[["Date", "Store ID", "Product ID", "Units Sold"]],
        on=["Date", "Store ID", "Product ID"],
        how="left",
    )

    print("\nGenerating charts…")
    plot_scatter_predicted_vs_actual(
        merged, output_path=output_dir / "chart1_scatter.png", date_filter=args.date
    )
    plot_daily_demand_trend(
        actuals, predictions, output_path=output_dir / "chart2_trend.png"
    )
    plot_top_products_bar(
        merged, store_id=args.store, date_filter=args.date,
        output_path=output_dir / "chart3_bar.png"
    )

    print("\nExporting Tableau/QuickSight aggregates…")
    export_dashboard_aggregates(actuals, predictions, output_dir / "exports")

    print(f"\n✓ All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
