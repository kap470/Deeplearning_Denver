import pandas as pd

# ── 1. Load the raw file ──────────────────────────────────────────────────────
# Change this path to your actual file
INPUT_FILE = "../data/gw_levels.csv"
OUTPUT_FILE = "../data/wells_wide.csv"
SUMMARY_FILE = "../data/wells_summary.csv"

df = pd.read_csv(INPUT_FILE, dtype={"SiteNo": str})

# ── 2. Parse dates and keep only date part (drop timezone offset) ─────────────
df["Time"] = pd.to_datetime(df["Time"], utc=True).dt.date

# ── 3. Keep only the columns we need ─────────────────────────────────────────
df = df[["SiteNo", "Time", "Water_level"]]

# ── 4. Get ordered list of unique wells ──────────────────────────────────────
wells = df["SiteNo"].unique()
print(f"Found {len(wells)} wells: {list(wells)}")

# ── 5. Build summary table and compute missing % per well ────────────────────
PERIOD_START  = pd.Timestamp("1994-03-15").date()
PERIOD_END    = pd.Timestamp("2025-12-31").date()
MISSING_THRESHOLD = 40.0   # discard wells with missing % >= this value

all_months    = pd.date_range(start=PERIOD_START, end=PERIOD_END, freq="MS")
total_months  = len(all_months)

summary_rows = []
for well in wells:
    sub = df[df["SiteNo"] == well].copy()

    sub_period = sub[(sub["Time"] >= PERIOD_START) & (sub["Time"] <= PERIOD_END)].copy()
    sub_period["YearMonth"] = pd.to_datetime(sub_period["Time"]).dt.to_period("M")
    months_with_data = sub_period["YearMonth"].nunique()
    missing_pct = round((1 - months_with_data / total_months) * 100, 2)

    summary_rows.append({
        "SiteNo":               well,
        "Start_date":           sub["Time"].min(),
        "End_date":             sub["Time"].max(),
        "N_records":            len(sub),
        "N_records_in_period":  len(sub_period),
        "Months_with_data":     months_with_data,
        "Total_months":         total_months,
        "Missing_pct":          missing_pct,
        "Included":             missing_pct < MISSING_THRESHOLD,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_FILE, index=False)
print(summary.to_string(index=False))

# ── 6. Filter wells that pass the threshold ───────────────────────────────────
wells_kept     = summary.loc[summary["Included"], "SiteNo"].tolist()
wells_discarded = summary.loc[~summary["Included"], "SiteNo"].tolist()
print(f"\nKept    ({len(wells_kept)}): {wells_kept}")
print(f"Discarded ({len(wells_discarded)}): {wells_discarded}")

# ── 7. Build wide file only for kept wells ────────────────────────────────────
frames = []
for well in wells_kept:
    sub = (df[df["SiteNo"] == well]
           .drop(columns="SiteNo")
           .rename(columns={"Time": well,
                             "Water_level": f"{well}_WL"})
           .reset_index(drop=True))
    frames.append(sub)

wide = pd.concat(frames, axis=1)

# ── 8. Export ─────────────────────────────────────────────────────────────────
wide.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved wide table  → {OUTPUT_FILE}")
print(f"Saved summary     → {SUMMARY_FILE}")