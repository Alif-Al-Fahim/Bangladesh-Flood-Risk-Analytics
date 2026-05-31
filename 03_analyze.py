"""
Phase 3: Analyse cleaned data — generate insights, charts, and AI summaries
Outputs charts to data/processed/charts/
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

DB_PATH = "data/processed/bangladesh_rainfall.db"
CHART_DIR = "data/processed/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Clean, professional style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
})

DISTRICT_COLORS = {
    "Sylhet":      "#1a6faf",
    "Sunamganj":   "#2196c4",
    "Cox's Bazar": "#e05c2a",
    "Chittagong":  "#e8882a",
    "Dhaka":       "#5b8e3a",
    "Khulna":      "#3a7a5a",
    "Rajshahi":    "#9b59b6",
    "Rangpur":     "#7d3c98",
    "Barisal":     "#c0392b",
    "Mymensingh":  "#e74c3c",
    "Comilla":     "#f39c12",
    "Narayanganj": "#d68910",
}


def load_data():
    conn = sqlite3.connect(DB_PATH)
    daily  = pd.read_sql("SELECT * FROM rainfall_daily",  conn)
    monthly = pd.read_sql("SELECT * FROM monthly_summary", conn)
    annual  = pd.read_sql("SELECT * FROM annual_summary",  conn)
    conn.close()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily, monthly, annual


# ── Chart 1: Annual total rainfall by district ──────────────────────────────
def chart_annual_trends(annual):
    fig, ax = plt.subplots(figsize=(13, 6))
    top_districts = ["Sylhet", "Chittagong", "Sunamganj", "Dhaka",
                     "Khulna", "Rajshahi", "Cox's Bazar"]
    for district in top_districts:
        sub = annual[annual["district"] == district].sort_values("year")
        if sub.empty:
            continue
        color = DISTRICT_COLORS.get(district, "#888")
        ax.plot(sub["year"], sub["total_annual_rainfall_mm"],
                label=district, color=color, linewidth=1.8, marker="o",
                markersize=3, alpha=0.9)

    ax.set_title("Annual total rainfall by district (2000–2023)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total rainfall (mm)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{CHART_DIR}/01_annual_trends.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 2: Average monthly rainfall (seasonal pattern) ────────────────────
def chart_seasonal_pattern(monthly):
    season_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    palette = {"Winter": "#5b8fd4", "Pre-Monsoon": "#f4a923",
               "Monsoon": "#1a6faf",  "Post-Monsoon": "#6aaf2a"}

    avg = (monthly.groupby(["month", "season"])["avg_daily_rainfall_mm"]
                  .mean().reset_index())
    avg["season"] = pd.Categorical(avg["season"], categories=season_order, ordered=True)
    avg.sort_values("month", inplace=True)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(avg["month"], avg["avg_daily_rainfall_mm"],
                  color=[palette[s] for s in avg["season"]],
                  edgecolor="white", linewidth=0.5)

    ax.set_title("Average daily rainfall by month — all Bangladesh (2000–2023)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg daily rainfall (mm)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in palette.items()]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.7)
    plt.tight_layout()
    path = f"{CHART_DIR}/02_seasonal_pattern.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 3: Heavy rain days per year (flood risk indicator) ─────────────────
def chart_heavy_rain_days(annual):
    """Days with >50mm rainfall per year — proxy for flood risk."""
    risk = (annual.groupby("year")["heavy_rain_days"]
                  .mean().reset_index())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(risk["year"], risk["heavy_rain_days"],
                    alpha=0.3, color="#c0392b")
    ax.plot(risk["year"], risk["heavy_rain_days"],
            color="#c0392b", linewidth=2.2, marker="o", markersize=4)

    # 5-year rolling average
    risk["rolling"] = risk["heavy_rain_days"].rolling(5, min_periods=3).mean()
    ax.plot(risk["year"], risk["rolling"],
            color="#7b241c", linewidth=2, linestyle="--", label="5-yr rolling avg")

    ax.set_title("Average heavy-rain days per year (>50 mm/day) — Bangladesh",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Days with >50 mm rainfall")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{CHART_DIR}/03_heavy_rain_days.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 4: District heatmap — avg monthly rainfall ────────────────────────
def chart_district_heatmap(monthly):
    pivot = monthly.groupby(["district", "month"])["avg_daily_rainfall_mm"].mean().unstack()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = month_labels

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(pivot, cmap="Blues", ax=ax, linewidths=0.4,
                linecolor="#ddd", annot=True, fmt=".0f",
                cbar_kws={"label": "Avg daily rainfall (mm)"})
    ax.set_title("Average daily rainfall by district and month (mm)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("District")
    plt.tight_layout()
    path = f"{CHART_DIR}/04_district_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Key statistics for summary ────────────────────────────────────────────
def print_key_stats(daily, monthly, annual):
    print("\n" + "=" * 50)
    print("KEY FINDINGS ")
    print("=" * 50)

    # Wettest district on average
    wettest = (annual.groupby("district")["total_annual_rainfall_mm"]
                     .mean().idxmax())
    wettest_mm = (annual.groupby("district")["total_annual_rainfall_mm"]
                        .mean().max())
    print(f"\n1. Wettest district (avg annual): {wettest} ({wettest_mm:,.0f} mm/year)")

    # Driest
    driest = (annual.groupby("district")["total_annual_rainfall_mm"]
                    .mean().idxmin())
    driest_mm = (annual.groupby("district")["total_annual_rainfall_mm"]
                       .mean().min())
    print(f"2. Driest district (avg annual):  {driest} ({driest_mm:,.0f} mm/year)")

    # Most extreme single day
    peak = daily.loc[daily["rainfall_mm"].idxmax()]
    print(f"3. Highest single-day rainfall:   {peak['rainfall_mm']} mm "
          f"on {peak['date'].date()} near {peak['district']}")

    # Monsoon share
    monsoon = monthly[monthly["season"] == "Monsoon"]["total_monthly_rainfall_mm"].sum()
    total   = monthly["total_monthly_rainfall_mm"].sum()
    pct     = 100 * monsoon / total
    print(f"4. Monsoon share of total rainfall: {pct:.1f}%")

    # Trend in heavy rain days
    early = annual[annual["year"] <= 2010]["heavy_rain_days"].mean()
    late  = annual[annual["year"] >= 2015]["heavy_rain_days"].mean()
    change = late - early
    print(f"5. Heavy-rain days (>50mm): avg {early:.1f}/yr (2000-2010) → "
          f"{late:.1f}/yr (2015-2023)  [change: {change:+.1f} days]")

    print("\n" + "=" * 50)
    print("POWER BI — load these files:")
    print("  data/processed/rainfall_clean.csv")
    print("  data/processed/monthly_summary.csv")
    print("  data/processed/annual_summary.csv")
    print("=" * 50)


def main():
    print("=" * 50)
    print("Bangladesh Rainfall — Analysis")
    print("=" * 50)

    if not os.path.exists(DB_PATH):
        print("Database not found. Run 02_clean_load.py first.")
        return

    print("Loading data from database...")
    daily, monthly, annual = load_data()
    print(f"  Daily rows:   {len(daily):,}")
    print(f"  Monthly rows: {len(monthly):,}")
    print(f"  Annual rows:  {len(annual):,}")

    print("\nGenerating charts...")
    chart_annual_trends(annual)
    chart_seasonal_pattern(monthly)
    chart_heavy_rain_days(annual)
    chart_district_heatmap(monthly)

    print_key_stats(daily, monthly, annual)


if __name__ == "__main__":
    main()