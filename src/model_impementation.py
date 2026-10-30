import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Functions

# Willmott's Index of Agreement (ranges from 0 to 1, closer to 1 = better agreement between predicted values and observed data)
def willmott_d(observed, predicted):
    obs_mean = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((np.abs(predicted - obs_mean) + np.abs(observed - obs_mean)) ** 2)
    if denominator == 0: 
        return np.nan
    return 1 - (numerator / denominator)

# Percent Bias (PBIAS) Positive models indicate underestimation, negative models overestimation. Close to zero is good. 
def percent_bias(observed, predicted):
    return 100 * np.sum(observed - predicted) / np.sum(observed)

#Compute all 5 metrics
def evaluate(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    d = willmott_d(np.asarray(y_test), np.asarray(y_pred))
    pbias = percent_bias(np.asarray(y_test), np.asarray(y_pred))
    return {"RMSE": rmse, "R2": r2, "MAE": mae, "Willmott_d": d, "PBIAS": pbias}

# Load input data file 
df = pd.read_csv('input_model.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Rename groundwater wells for easier code, create list for future iterations
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
# Establishes hyperparameter grid to test possible values for model parameters

xgb_param_grid = {
    "n_estimators":[200, 400, 600, 800],
    "max_depth":[3, 4, 5, 6],
    "learning_rate":[0.01, 0.03, 0.05, 0.1],
    "subsample":[0.7, 0.8, 1.0],
    "colsample_bytree":[0.7, 0.8, 1.0]
}

cv = TimeSeriesSplit(n_splits=5)
results = []
time_series_store={}
# Main XGBoost loop, iterates for each well

for well in wells: 
    print(f'Processing {well}')

    # Model tuning
    df_well = df.copy()

    for lag in range(1, 13):
        df_well[f'GW_lag_{lag}'] = df_well[well].shift(lag)

    df_well = df_well.dropna().reset_index(drop=True)
    lag_cols = [f'GW_lag_{lag}' for lag in range(1, 13)]

    X = df_well[["Precip_mm", "Temp_C", "Streamflow"] + lag_cols]
    y = df_well[well]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    test_dates = df_well['Date'].iloc[len(X_train):].values

    print(f'Tuning model for {well}...')
    xgb_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=1, verbosity=0),
        param_distributions=xgb_param_grid,
        n_iter=30, cv=cv, scoring="r2", n_jobs=-1, random_state=42)
    
    xgb_search.fit(X_train_std, y_train)
    best_fit = xgb_search.best_estimator_

    start = time.time()
    y_pred_xgb = best_fit.predict(X_test_std)
    time_test_xgb = time.time()-start

    # Calculate error metrics
    metrics_xgb = evaluate(y_test, y_pred_xgb)

    # Store tuned results
    for model_name, metrics, test_time in [('XGBoost', metrics_xgb, time_test_xgb)]:
        row = {"Well": well, 'Model': model_name}
        row.update(metrics)                                                                                                 
        row["Test_Time_s"] = round(test_time, 6)
        results.append(row)

    # save arrays for time series plotting
    time_series_store[well] = {
        'test_dates':test_dates,
        'y_actual':y_test.values,
        'y_pred_xgb':y_pred_xgb
    }

# build neural network

# training parameters: 
n_epochs = 50
# Changes window size
lookback = 1

# keeps best model
best_mse = np.inf
best_weights = None
history = []

def create_dataset(X, y, lookback):
    X_seq = []
    y_seq = []

    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

    return X_seq, y_seq

# Defines model: LSTM model and fully connected layer
class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size= 15, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)

        # extracts only the last time step's output
        x = x[:, -1, :]

        x = self.linear(x)
        return x
# Main LSTM loop: 

for well in wells: 
    print(f'Processing {well}')

    # Model tuning
    df_well = df.copy()

    for lag in range(1, 13):
        df_well[f'GW_lag_{lag}'] = df_well[well].shift(lag)

    df_well = df_well.dropna().reset_index(drop=True)
    lag_cols = [f'GW_lag_{lag}' for lag in range(1, 13)]
    feature_cols = ["Precip_mm", "Temp_C", "Streamflow"] + lag_cols
    X = df_well[feature_cols]
    y = df_well[well]
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    X_train_seq, y_train_seq = create_dataset(X_train_std, y_train, lookback)
    X_test_seq, y_test_seq = create_dataset(X_test_std, y_test, lookback)

    print(f'For {well} : ')
    print('X_train shape:', X_train_seq.shape)
    print('y_train shape:', y_train_seq.shape)
    print('X_test shape:', X_test_seq.shape)
    print('y_test shape:', y_test_seq.shape)

    # ChatGPT used here to help add lookback to previous flow for compatibility, while adding benefits of LSTM 
    #   for time sequence modeling, lag kept despite lookback for performance comparisons

    model = lstm_model()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train_seq, y_train_seq), shuffle=True, batch_size = 8)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            y_batch = y_batch.unsqueeze(1)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            start = time.time()
            y_pred_lstm = model(X_test_seq).cpu().numpy()
            time_test_lstm = time.time()-start

        # Calculate error metrics    
        metrics_lstm = evaluate(y_test_seq, y_pred_lstm)

        # Store tuned results
        for model_name, metrics, test_time in [("LSTM", metrics_lstm, time_test_lstm)]:
            row = {"Well": well, 'Model': model_name}
            row.update(metrics)
            row['Test_Time_s'] = round(test_time, 6)
            results.append(row)

        # Save arrays for time series plotting
        time_series_store[well].update({'y_pred_lstm':y_pred_lstm})
        
#Figure 1 - Time series plots
fig1, axes1 = plt.subplots(7, 1, figsize=(16, 22), sharex=False)

for i, well in enumerate(wells):
    ax = axes1[i]
    ts = time_series_store[well]

    ax.plot(ts["test_dates"], ts["y_actual"], color="black", linewidth=2.5, label="Measured",zorder=5)
    ax.plot(ts["test_dates"], ts["y_pred_xgb"], color="steelblue", linewidth=1.8, label="XGBoost",linestyle="dashdot")
    ax.plot(ts['test_dates'][lookback:], ts['y_pred_lstm'], color = 'orangered', linewidth=1.8, label='LSTM', linestyle = 'dotted' )

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
results_df = pd.DataFrame(results)
results_df

# Figure 2 - Metric bar charts
results_df = pd.DataFrame(results)

metrics_list  = ["RMSE", "R2", "MAE", "Willmott_d", "PBIAS"]
metric_labels = {
    "RMSE":"RMSE (ft)",
    "R2":"R² Score",
    "MAE":"MAE (ft)",
    "Willmott_d":"Willmott's d",
    "PBIAS":"Percent Bias (%)"
}

model_colors = {
    "XGBoost":"turquoise",
    "LSTM": "orangered"
}

x     = np.arange(len(wells))
width = 0.25

fig2, axes2 = plt.subplots(5, 1, figsize=(14, 22))
fig2.patch.set_facecolor("white")

for ax, metric in zip(axes2, metrics_list):
    ax.set_facecolor("white")

    vals = [results_df[results_df["Well"] == w][metric].values[0]      for w in wells]

    ax.bar(x - width, vals, width, label="XGBoost",color=model_colors["XGBoost"],edgecolor="none")
    ax.bar(x - width, vals, width, label="LSTM",color=model_colors["LSTM"],edgecolor="none")

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

# ChatGPT used for debugging tensor shape issues