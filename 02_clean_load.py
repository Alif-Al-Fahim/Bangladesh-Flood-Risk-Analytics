"""
Phase 2 (v4): Clean NASA POWER regional CSVs → SQLite + Power BI CSVs
Format: LAT, LON, YEAR, DOY, PRECTOTCORR
DOY = Day of Year (1–365/366). Converted to proper date.
"""

import pandas as pd
import sqlite3
import os
import glob
from io import StringIO

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
DB_PATH       = f"{PROCESSED_DIR}/bangladesh_rainfall.db"
CSV_OUT       = f"{PROCESSED_DIR}/rainfall_clean.csv"

os.makedirs(PROCESSED_DIR, exist_ok=True)

DISTRICTS = {
    "Dhaka":       (23.8103, 90.4125),
    "Chittagong":  (22.3569, 91.7832),
    "Sylhet":      (24.8949, 91.8687),
    "Rajshahi":    (24.3745, 88.6042),
    "Khulna":      (22.8456, 89.5403),
    "Barisal":     (22.7010, 90.3535),
    "Rangpur":     (25.7439, 89.2752),
    "Mymensingh":  (24.7471, 90.4203),
    "Comilla":     (23.4607, 91.1809),
    "Narayanganj": (23.6238, 90.4997),
    "Sunamganj":   (25.0658, 91.3950),
    "Cox's Bazar": (21.4272, 92.0058),
}


def find_nearest_district(lat, lon):
    best, name = float("inf"), "Unknown"
    for d, (dlat, dlon) in DISTRICTS.items():
        dist = ((lat - dlat)**2 + (lon - dlon)**2)**0.5
        if dist < best:
            best, name = dist, d
    return name


def get_season(month):
    if month in [12, 1, 2]:   return "Winter"
    if month in [3, 4, 5]:    return "Pre-Monsoon"
    if month in [6, 7, 8, 9]: return "Monsoon"
    return "Post-Monsoon"


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Find the header line: starts with "LAT"
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("LAT"):
            header_idx = i
            break

    if header_idx is None:
        print(f"\n  Could not find header row.")
        return None

    content = "".join(lines[header_idx:])
    df = pd.read_csv(StringIO(content))
    df.columns = [c.strip().upper() for c in df.columns]

    # Validate required columns
    required = {"LAT", "LON", "YEAR", "DOY", "PRECTOTCORR"}
    if not required.issubset(df.columns):
        print(f"\n  Missing columns: {required - set(df.columns)}. Found: {list(df.columns)}")
        return None

    df = df[["LAT", "LON", "YEAR", "DOY", "PRECTOTCORR"]].copy()
    df.rename(columns={
        "LAT": "lat", "LON": "lon",
        "YEAR": "year", "DOY": "doy",
        "PRECTOTCORR": "rainfall_mm"
    }, inplace=True)

    # Convert YEAR + DOY → proper date, month, day
    # pd.to_datetime with format %Y %j handles leap years correctly
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["doy"].astype(str).str.zfill(3),
        format="%Y%j",
        errors="coerce"
    )
    df["month"] = df["date"].dt.month
    df["day"]   = df["date"].dt.day

    return df


def main():
    print("=" * 52)
    print("Bangladesh Rainfall — Clean & Load  v4")
    print("=" * 52)

    files = sorted(glob.glob(f"{RAW_DIR}/rainfall_*.csv"))
    if not files:
        print("No raw files found. Run 01_collect_data.py first.")
        return

    print(f"Found {len(files)} raw files.\n")

    all_frames = []
    for filepath in files:
        tag = os.path.basename(filepath).replace("rainfall_", "").replace(".csv", "")
        print(f"  {tag}...", end=" ", flush=True)

        df = parse_file(filepath)
        if df is None:
            print("SKIPPED")
            continue

        df.dropna(subset=["date"], inplace=True)

        # Clean rainfall values
        df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        df["rainfall_mm"] = df["rainfall_mm"].replace(-999.0, pd.NA)
        df.loc[df["rainfall_mm"] > 600, "rainfall_mm"] = pd.NA

        # Label nearest district
        df["district"] = df.apply(
            lambda r: find_nearest_district(r["lat"], r["lon"]), axis=1)
        df["season"] = df["month"].apply(get_season)

        all_frames.append(df)
        print(f"{len(df):,} rows ✓")

    if not all_frames:
        print("\nNo data processed.")
        return

    master = pd.concat(all_frames, ignore_index=True)
    master.sort_values(["date", "lat", "lon"], inplace=True)
    master.drop_duplicates(subset=["lat", "lon", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    # Fill missing rainfall with district-month mean
    fill = master.groupby(["district", "month"])["rainfall_mm"].transform("mean")
    master["rainfall_mm"] = master["rainfall_mm"].fillna(fill).round(2)

    print(f"\nTotal rows : {len(master):,}")
    print(f"Years      : {master['year'].min()} – {master['year'].max()}")
    print(f"Districts  : {sorted(master['district'].unique())}")

    # ── SQLite ─────────────────────────────────────────────────────────────
    print(f"\nWriting {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    master.to_sql("rainfall_daily", conn, if_exists="replace", index=False)

    for view in ("monthly_summary", "annual_summary"):
        conn.execute(f"DROP VIEW IF EXISTS {view}")

    conn.execute("""
        CREATE VIEW monthly_summary AS
        SELECT district, year, month, season,
               ROUND(AVG(rainfall_mm), 2) AS avg_daily_rainfall_mm,
               ROUND(SUM(rainfall_mm), 2) AS total_monthly_rainfall_mm,
               COUNT(*)                   AS days_recorded
        FROM rainfall_daily
        GROUP BY district, year, month
    """)
    conn.execute("""
        CREATE VIEW annual_summary AS
        SELECT district, year,
               ROUND(SUM(rainfall_mm), 2) AS total_annual_rainfall_mm,
               ROUND(AVG(rainfall_mm), 2) AS avg_daily_rainfall_mm,
               ROUND(MAX(rainfall_mm), 2) AS peak_daily_rainfall_mm,
               COUNT(CASE WHEN rainfall_mm > 50 THEN 1 END) AS heavy_rain_days
        FROM rainfall_daily
        GROUP BY district, year
    """)
    conn.commit()
    conn.close()

    # ── CSVs for Power BI ──────────────────────────────────────────────────
    master.to_csv(CSV_OUT, index=False)
    conn = sqlite3.connect(DB_PATH)
    pd.read_sql("SELECT * FROM monthly_summary", conn) \
      .to_csv(f"{PROCESSED_DIR}/monthly_summary.csv", index=False)
    pd.read_sql("SELECT * FROM annual_summary", conn) \
      .to_csv(f"{PROCESSED_DIR}/annual_summary.csv", index=False)
    conn.close()

    print("\n✓  All done! Files ready in data/processed/:")
    for f in ["rainfall_clean.csv", "monthly_summary.csv",
              "annual_summary.csv", "bangladesh_rainfall.db"]:
        print(f"   {f}")


if __name__ == "__main__":
    main()