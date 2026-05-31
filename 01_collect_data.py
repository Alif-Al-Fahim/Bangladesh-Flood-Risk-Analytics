"""
Phase 1: Collect Bangladesh rainfall data from NASA POWER API
Pulls daily rainfall (mm/day) for all of Bangladesh, year by year (2000-2023)
"""

import requests
import pandas as pd
import os
import time

# Bangladesh bounding box
LAT_MIN = 20.5
LAT_MAX = 26.7
LON_MIN = 88.0
LON_MAX = 92.7

START_YEAR = 2000
END_YEAR = 2023

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"


def fetch_year(year):
    """Fetch one full year of rainfall data for Bangladesh."""
    filename = f"{RAW_DIR}/rainfall_{year}.csv"

    if os.path.exists(filename):
        print(f"  {year}: already downloaded, skipping.")
        return True

    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "format": "CSV",
        "latitude-min": LAT_MIN,
        "latitude-max": LAT_MAX,
        "longitude-min": LON_MIN,
        "longitude-max": LON_MAX,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "header": "true",
    }

    print(f"  {year}: downloading...", end=" ")
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

        print("done.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("=" * 50)
    print("Bangladesh Rainfall Data Collection")
    print(f"Years: {START_YEAR} - {END_YEAR}")
    print(f"Region: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    print("=" * 50)

    failed = []
    for year in range(START_YEAR, END_YEAR + 1):
        success = fetch_year(year)
        if not success:
            failed.append(year)
        time.sleep(1)  # be polite to the API

    print("\n--- Summary ---")
    print(f"Downloaded: {END_YEAR - START_YEAR + 1 - len(failed)} years")
    if failed:
        print(f"Failed years: {failed}  (re-run the script to retry)")
    else:
        print("All years downloaded successfully!")
    print(f"\nRaw files saved in: {RAW_DIR}/")


if __name__ == "__main__":
    main()