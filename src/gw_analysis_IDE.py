import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Set Up
try: 
    df = pd.read_csv('../data/gw_levels.csv', encoding="utf-8")
except Exception as e:
    print(f'Error reading groundwater data file: {e}')
else: 
    print(f'"df" created successfully from "gw_levels.csv"')

file_path = "../data/gw_levels.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")
print(f'\nDataframe overview, heads and tails: \n{df.head(3)}\n{df.tail(3)}')

# Data Cleaning
df["Time"] = pd.to_datetime(df["Time"], errors="coerce",utc=True)
print(f'\nNumber of null values: \n{df.isnull().sum()}')

results = []
for station, group in df.groupby("SiteNo"):
    values = group["Water_level"]

    stats = {
        "SiteNo":         station,
        "Start_Date":     group["Time"].min(),
        "End_Date":       group["Time"].max(),
        "N_Observations": len(values),
        "Minimum":        values.min(),
        "Maximum":        values.max(),
        "Mean":           values.mean(),
        "Std_Dev":        values.std(),
        "Skewness":       skew(values, bias=False),
        "Kurtosis":       kurtosis(values, bias=False),
    }
    results.append(stats)

# Summary statistics dataframe sorted by mean
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values("Mean").reset_index(drop=True)
ordered_wells = summary_df["SiteNo"].tolist()

# Creates length of timeframe data for longest, shortest, mean of well data
by_well = pd.DataFrame(results)
length_running = (by_well['End_Date']-by_well['Start_Date'])
print(f'\nFirst collection date: {by_well["Start_Date"].min()}')
print(f'\nLast collection date: {by_well["End_Date"].max()}')
print(f'\nLongest running well data: {length_running.max()}')
print(f'Shortest running well data: {length_running.min()}')
print(f'Average time for well monitoring data: {length_running.mean()}\n')

# Statistics for all wells, "Water_level"
print(f"Skewness by well: {by_well['Skewness']}")
print(f"Kurtosis by well: {by_well['Kurtosis']}")
print(f"Average Skewness: {by_well['Skewness'].mean()}")
print(f"Average Kurtosis: {by_well['Kurtosis'].mean()}\n")

# Assign integer IDs starting from 1 across all 23 wells
# The ID just reflects the rank by mean water level (1 = lowest mean)
summary_df["Well_Int_ID"] = range(1, len(summary_df) + 1)

# Build the lookup table: integer ID <-> original SiteNo
lookup_table = summary_df[["Well_Int_ID", "SiteNo"]].copy()
lookup_table = lookup_table.rename(columns={"Well_Int_ID": "ID", "SiteNo": "Well_SiteNo"})

print("=" * 40)
print("Well ID Lookup Table")
print("=" * 40)
print(lookup_table.to_string(index=False))
print("=" * 40)

# Save lookup table to CSV
lookup_table.to_csv("../outputs/well_id_lookup.csv", index=False)
print("Lookup table saved to: ../outputs/well_id_lookup.csv\n")

# Split into two groups by mean
sub_12 = ordered_wells[:12]   # 12 wells with lowest mean
sub_11 = ordered_wells[12:]   # 11 wells with highest mean

# Get the integer IDs for each group (for x-axis labels)
id_map   = dict(zip(summary_df["SiteNo"], summary_df["Well_Int_ID"]))
ids_12   = [id_map[s] for s in sub_12]
ids_11   = [id_map[s] for s in sub_11]

# Boxplot 1 - 12 wells with lowest mean
subset1 = df[df["SiteNo"].isin(sub_12)]
grouped_data1 = [
    subset1[subset1["SiteNo"] == s]["Water_level"]
    for s in sub_12
]

fig1, (ax1_plot, ax1_table) = plt.subplots(
    2, 1, figsize=(14, 10),
    gridspec_kw={"height_ratios": [4, 1]}
)

# Widths closer together: use widths=0.6 (default is 0.5, higher = wider/closer)
ax1_plot.boxplot(grouped_data1, widths=0.6)
ax1_plot.set_xticks(range(1, len(sub_12) + 1))
ax1_plot.set_xticklabels(ids_12, rotation=0, fontsize=11)
ax1_plot.set_ylim(3650, 3900)
ax1_plot.set_xlabel("Well ID", fontsize=14)
ax1_plot.set_ylabel("Water Level (ft)", fontsize=14)
ax1_plot.grid(axis="y", alpha=0.4)

# Table below the plot
table1_data = [ids_12, [str(s) for s in sub_12]]
table1 = ax1_table.table(
    cellText=table1_data,
    rowLabels=["ID", "Well SiteNo"],
    loc="center",
    cellLoc="center"
)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1, 1.6)
ax1_table.axis("off")

plt.tight_layout()
plt.savefig("../Figures/boxplot_group1.png", dpi=150, bbox_inches="tight")
plt.show()

# Boxplot 2 - 11 wells with highest mean
subset2 = df[df["SiteNo"].isin(sub_11)]
grouped_data2 = [
    subset2[subset2["SiteNo"] == s]["Water_level"]
    for s in sub_11
]

fig2, (ax2_plot, ax2_table) = plt.subplots(
    2, 1, figsize=(14, 10),
    gridspec_kw={"height_ratios": [4, 1]}
)

ax2_plot.boxplot(grouped_data2, widths=0.6)
ax2_plot.set_xticks(range(1, len(sub_11) + 1))
ax2_plot.set_xticklabels(ids_11, rotation=0, fontsize=11)
ax2_plot.set_ylim(3800, 4150)
ax2_plot.set_xlabel("Well ID", fontsize=14)
ax2_plot.set_ylabel("Water Level (ft)", fontsize=14)
ax2_plot.grid(axis="y", alpha=0.4)

# Table below the plot
table2_data = [ids_11, [str(s) for s in sub_11]]
table2 = ax2_table.table(
    cellText=table2_data,
    rowLabels=["ID", "Well SiteNo"],
    loc="center",
    cellLoc="center"
)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1, 1.6)
ax2_table.axis("off")

plt.tight_layout()
plt.savefig("../Figures/boxplot_group2.png", dpi=150, bbox_inches="tight")
plt.show()

####################
# Save final cleaned dataset to gw_cleaned.csv
try: 
    datafile.to_csv('../data/gw_cleaned.csv', index = False)
except Exception as e:
    print('Error saving gw_cleaned.csv')
else: 
    print(f'gw_cleaned.csv created successfully')
