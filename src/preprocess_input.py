import pandas as pd
import numpy as np

# file paths
wells_file="wells_wide.csv"
climate_file="Precip_and_Temp.csv"
streamflow_file="streamflow.csv"
output_file="input.csv"

# modeling period
period_start=pd.Timestamp("1994-03-01")
period_end=pd.Timestamp("2025-12-01")

# build a monthly index (first day of each month)
monthly_index=pd.date_range(start=period_start, end=period_end, freq="MS")

# groundwater — wells_wide.csv
#    columns come in pairs: site number (date col) + site number_WL (value col)
gw_raw=pd.read_csv(wells_file, dtype=str)

date_cols=[c for c in gw_raw.columns if not c.endswith("_WL")]
wl_cols=[c for c in gw_raw.columns if c.endswith("_WL")]

gw_monthly=pd.DataFrame(index=monthly_index)
gw_monthly.index.name="YearMonth"

for date_col, wl_col in zip(date_cols, wl_cols):
    well_id=date_col

    pair=gw_raw[[date_col, wl_col]].dropna(subset=[date_col])
    pair=pair.copy()
    pair[date_col]=pd.to_datetime(pair[date_col], errors="coerce")
    pair[wl_col]=pd.to_numeric(pair[wl_col], errors="coerce")
    pair=pair.dropna(subset=[date_col])

    # keep only rows inside the study period
    pair=pair[(pair[date_col] >=period_start) & (pair[date_col] <=period_end)]

    # some months have multiple readings, just average them
    pair["ym"]=pair[date_col].dt.to_period("M").dt.to_timestamp()
    monthly=pair.groupby("ym")[wl_col].mean()

    # fill non data values with linear interpolation
    monthly=monthly.reindex(monthly_index)
    monthly=monthly.interpolate(method="linear", limit_direction="forward")

    gw_monthly[f"GW_{well_id}"]=monthly.values

# monthly averages of monthly precipitation and temperature
clim_raw=pd.read_csv(climate_file, header=0)
clim_raw.columns=[c.strip() for c in clim_raw.columns]

date_col_c=clim_raw.columns[2]
prcp_col=clim_raw.columns[3]
tmax_col=clim_raw.columns[4]
tmin_col=clim_raw.columns[5]

clim_raw[date_col_c]=pd.to_datetime(clim_raw[date_col_c], errors="coerce")
clim_raw[prcp_col]=pd.to_numeric(clim_raw[prcp_col], errors="coerce")
clim_raw[tmax_col]=pd.to_numeric(clim_raw[tmax_col], errors="coerce")
clim_raw[tmin_col]=pd.to_numeric(clim_raw[tmin_col], errors="coerce")

clim_raw=clim_raw.dropna(subset=[date_col_c])
clim_raw=clim_raw[(clim_raw[date_col_c] >=period_start) &
                     (clim_raw[date_col_c] <=period_end)]

clim_raw["ym"]=clim_raw[date_col_c].dt.to_period("M").dt.to_timestamp()

prcp_monthly=clim_raw.groupby("ym")[prcp_col].sum().reindex(monthly_index)

clim_raw["tmean"]=(clim_raw[tmax_col] + clim_raw[tmin_col]) / 2
temp_monthly=clim_raw.groupby("ym")["tmean"].mean().reindex(monthly_index)

# average values from streamflow file
sf_raw=pd.read_csv(streamflow_file, header=0)
sf_raw.columns=[c.strip() for c in sf_raw.columns]

time_col=sf_raw.columns[3]
val_col=sf_raw.columns[4]

sf_raw[time_col]=pd.to_datetime(sf_raw[time_col], errors="coerce")
sf_raw[val_col]=pd.to_numeric(sf_raw[val_col], errors="coerce")

sf_raw=sf_raw.dropna(subset=[time_col])
sf_raw=sf_raw[(sf_raw[time_col] >=period_start) &
                 (sf_raw[time_col] <=period_end)]

sf_raw["ym"]=sf_raw[time_col].dt.to_period("M").dt.to_timestamp()
sf_monthly=sf_raw.groupby("ym")[val_col].mean().reindex(monthly_index)

#Save output file
merged=pd.DataFrame(index=monthly_index)
merged.index.name="Date"

merged["Precip_mm"]=prcp_monthly.values
merged["Temp_C"]=temp_monthly.values
merged["Streamflow"]=sf_monthly.values

for col in gw_monthly.columns:
    merged[col]=gw_monthly[col].values

merged=merged.reset_index()
merged.to_csv(output_file, index=False)