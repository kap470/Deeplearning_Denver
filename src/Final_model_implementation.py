#Objective:  Improve implementation of models with XGBoost, Random Forest, and SVM from scikit learn
#            with hyperparameter tuning (RandomizedSearchCV and TimeSeriesSplit)
#            expanded evaluation metrics (RMSE, R2, MAE, Willmott's d, Percent Bias)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

#Functions
#Willmott's Index of Agreement (ranges from 0 to 1. Closer to 1 = better agreement with observations)
def willmott_d(observed, predicted):
    obs_mean = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((np.abs(predicted - obs_mean) + np.abs(observed - obs_mean)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

#Percent Bias (PBIAS). 
#Positive model underestimates. Negative model overestimates. Close to 0 is good
def percent_bias(observed, predicted):
    return 100 * np.sum(observed - predicted) / np.sum(observed)

#Compute all 5 metrics
def evaluate(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    d = willmott_d(np.array(y_test), np.array(y_pred))
    pbias = percent_bias(np.array(y_test), np.array(y_pred))
    return {"RMSE": rmse, "R2": r2, "MAE": mae, "Willmott_d": d, "PBIAS": pbias}

#load input file with gw level records
df = pd.read_csv("input.csv", parse_dates=["Date"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Rename groundwater wells for easier code
df = df.rename(columns={
    'GW_325002107181201':'GW_1',
    'GW_315712106361803':'GW_2',
    'GW_321740106481002':'GW_3',
    'GW_4904476':'GW_4',
    'GW_321332106443702':'GW_5',
    'GW_321740106481001':'GW_6',
    'GW_321332106443701':'GW_7'
})

wells = ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5", "GW_6", "GW_7"]


# The hyperparameter grids define possible values for the model parameters
xgb_param_grid = {
    "n_estimators":[200, 400, 600, 800],
    "max_depth":[3, 4, 5, 6],
    "learning_rate":[0.01, 0.03, 0.05, 0.1],
    "subsample":[0.7, 0.8, 1.0],
    "colsample_bytree":[0.7, 0.8, 1.0]
}

rf_param_grid = {
    "n_estimators":[200, 300, 500],
    "max_depth":[None, 5, 10, 15],
    "max_features":["sqrt", "log2", 0.5]
}

svr_param_grid = {
    "C":[1, 10, 50, 100, 200],
    "gamma":["scale", "auto"],
    "epsilon":[0.01, 0.05, 0.1, 0.5]
}

tscv = TimeSeriesSplit(n_splits=5)
all_results = []
time_series_store = {}

#main loop (iterates for every well)
for well in wells:

    print(f"Processing {well}")

    #Tuned models
    df_well = df.copy()

    for lag in range(1, 13):
        df_well[f"GW_lag_{lag}"] = df_well[well].shift(lag)

    df_well = df_well.dropna().reset_index(drop=True)
    lag_cols = [f"GW_lag_{lag}" for lag in range(1, 13)]

    X = df_well[["Precip_mm", "Temp_C", "Streamflow"] + lag_cols]
    y = df_well[well]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    test_dates = df_well["Date"].iloc[len(X_train):].values

    #XGBoost tuned
    print(f"  Tuning SXGBoost for {well}...")
    xgb_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=1, verbosity=0),
        param_distributions=xgb_param_grid,
        n_iter=30, cv=tscv, scoring="r2", n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train_std, y_train)
    xgb_best = xgb_search.best_estimator_

    start = time.perf_counter()
    y_pred_xgb = xgb_best.predict(X_test_std)
    xgb_time = time.perf_counter() - start
    #Calculate erros metrics
    xgb_metrics = evaluate(y_test, y_pred_xgb)
    
    
    #Random Forest tuned
    print(f"  Tuning Random Forest for {well}...")
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=1),
        param_distributions=rf_param_grid,
        n_iter=20, cv=tscv, scoring="r2", n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train_std, y_train)
    rf_best = rf_search.best_estimator_

    start = time.perf_counter()
    y_pred_rf = rf_best.predict(X_test_std)
    rf_time = time.perf_counter() - start
    #Calculate erros metrics
    rf_metrics = evaluate(y_test, y_pred_rf)

    #SVR tuned
    print(f"  Tuning SVR for {well}...")
    svr_search = RandomizedSearchCV(
        estimator=SVR(kernel="rbf"),
        param_distributions=svr_param_grid,
        n_iter=20, cv=tscv, scoring="r2", n_jobs=-1, random_state=42
    )
    svr_search.fit(X_train_std, y_train)
    svr_best = svr_search.best_estimator_

    start = time.perf_counter()
    y_pred_svr = svr_best.predict(X_test_std)
    svr_time = time.perf_counter() - start
    #Calculate erros metrics
    svr_metrics = evaluate(y_test, y_pred_svr)

    #Store tuned results
    for model_name, metrics, test_time in [
        ("XGBoost", xgb_metrics, xgb_time),
        ("RandomForest", rf_metrics,  rf_time),
        ("SVR", svr_metrics, svr_time)
    ]:
        row = {"Well": well, "Model": model_name}
        row.update(metrics)
        row["Test_Time_s"] = round(test_time, 6)
        all_results.append(row)

    #Save arrays for time series plots
    time_series_store[well] = {
        "test_dates":test_dates,
        "y_actual":y_test.values,
        "y_pred_xgb":y_pred_xgb,
        "y_pred_rf":y_pred_rf,
        "y_pred_svr":y_pred_svr
    }


#Figure 1 - Time series plots
fig1, axes1 = plt.subplots(7, 1, figsize=(16, 22), sharex=False)

for i, well in enumerate(wells):
    ax = axes1[i]
    ts = time_series_store[well]

    ax.plot(ts["test_dates"], ts["y_actual"], color="black",linewidth=2.5, label="Measured",zorder=5)
    ax.plot(ts["test_dates"], ts["y_pred_xgb"], color="steelblue",linewidth=1.8, label="XGBoost",linestyle="-.")
    ax.plot(ts["test_dates"], ts["y_pred_rf"], color="purple",linewidth=1.8, label="Random Forest",linestyle=":")
    ax.plot(ts["test_dates"], ts["y_pred_svr"], color="darkorange", linewidth=1.8, label="SVR",linestyle=(0, (3, 1, 1, 1)))

    ax.set_title(well, fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("GW Level (ft)", fontsize=16)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc="upper right", fontsize=16, ncol=4)

axes1[-1].set_xlabel("Date", fontsize=16)
plt.tight_layout()
plt.show()

# Figure 2 - Metric bar charts
results_df = pd.DataFrame(all_results)

metrics_list  = ["RMSE", "R2", "MAE", "Willmott_d", "PBIAS"]
metric_labels = {
    "RMSE":"RMSE (ft)",
    "R2":"R² Score",
    "MAE":"MAE (ft)",
    "Willmott_d":"Willmott's d",
    "PBIAS":"Percent Bias (%)"
}

model_colors = {
    "XGBoost":"#4FC3F7",
    "RandomForest":"#66BB6A",
    "SVR":"#EF6C00"
}

x     = np.arange(len(wells))
width = 0.25

fig2, axes2 = plt.subplots(5, 1, figsize=(14, 22))
fig2.patch.set_facecolor("white")

for ax, metric in zip(axes2, metrics_list):
    ax.set_facecolor("white")

    xgb_vals = [results_df[(results_df["Well"] == w) & (results_df["Model"] == "XGBoost")][metric].values[0]      for w in wells]
    rf_vals = [results_df[(results_df["Well"] == w) & (results_df["Model"] == "RandomForest")][metric].values[0] for w in wells]
    svr_vals = [results_df[(results_df["Well"] == w) & (results_df["Model"] == "SVR")][metric].values[0]          for w in wells]

    ax.bar(x - width, xgb_vals, width, label="XGBoost",color=model_colors["XGBoost"],edgecolor="none")
    ax.bar(x, rf_vals,  width, label="Random Forest", color=model_colors["RandomForest"], edgecolor="none")
    ax.bar(x + width,  svr_vals, width, label="SVR",color=model_colors["SVR"],edgecolor="none")

    ax.set_ylabel(metric_labels[metric], color="black", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(wells, color="black", fontsize=16)
    ax.tick_params(colors="black", labelsize=11)
    ax.spines["bottom"].set_color("#aaa")
    ax.spines["left"].set_color("#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(facecolor="white", labelcolor="black", fontsize=16, edgecolor="#aaa")

    if metric in ["PBIAS", "R2"]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

#summary table
results_df = results_df.round({"RMSE": 3, "R2": 3, "MAE": 3, "Willmott_d": 3, "PBIAS": 2, "Test_Time_s": 6})
results_df.to_csv("phase5_results_summary.csv", index=False)