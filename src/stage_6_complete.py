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
time_series_store_tuning = {}
tuned_results = []

# Main XGBoost loop, iterates for each well

for well in wells: 
    print(f'Processing {well}')

    # Model tuning
    cols = ["Date", well, "Precip_mm", "Temp_C", "Streamflow"]
    df_well = df[cols].copy()

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



# Original LSTM loop:
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
    X_seq, y_seq = [], []
    for i in range(lookback - 1, len(X)):
        X_seq.append(X[i - lookback + 1:i + 1])
        y_seq.append(y[i])
    return (torch.tensor(np.array(X_seq), dtype=torch.float32),
            torch.tensor(np.array(y_seq), dtype=torch.float32))

# Defines model: LSTM model and fully connected layer
class lstm_model(nn.Module):
    def __init__(self, n_features, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x[:, -1, :])
    
# Main LSTM loop: 

for well in wells: 
    print(f'Processing {well}')

    # Model tuning
    cols = ["Date", well, "Precip_mm", "Temp_C", "Streamflow"]
    df_well = df[cols].copy()

    for lag in range(1, 13):
        df_well[f'GW_lag_{lag}'] = df_well[well].shift(lag)

    df_well = df_well.dropna().reset_index(drop=True)
    lag_cols = [f'GW_lag_{lag}' for lag in range(1, 13)]
    feature_cols = ["Precip_mm", "Temp_C", "Streamflow"] + lag_cols
    X = df_well[feature_cols]
    y = df_well[well]
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Scale target (fit on train only)
    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_std = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    X_train_seq, y_train_seq = create_dataset(X_train_std, y_train_std, lookback)
    X_test_seq, y_test_seq = create_dataset(X_test_std, y_test_std, lookback)

    print(f'For {well} : ')
    print('X_train shape:', X_train_seq.shape)
    print('y_train shape:', y_train_seq.shape)
    print('X_test shape:', X_test_seq.shape)
    print('y_test shape:', y_test_seq.shape)

    # ChatGPT used here to help add lookback to previous flow for compatibility, while adding benefits of LSTM 
    #   for time sequence modeling, lag kept despite lookback for performance comparisons

    y_test_orig = y_test[lookback - 1:]   # observed values in ft, aligned with sequences

    torch.manual_seed(1)
    model = lstm_model(n_features=len(feature_cols))
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train_seq, y_train_seq),
                             shuffle=True, batch_size=8)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        start = time.time()
        y_pred_std = model(X_test_seq).cpu().numpy()
        time_test_lstm = time.time() - start

    # Back to original units (ft)
    y_pred_lstm = y_scaler.inverse_transform(y_pred_std).ravel()

    metrics_lstm = evaluate(y_test_orig, y_pred_lstm)

    row = {"Well": well, "Model": "LSTM"}
    row.update(metrics_lstm)
    row["Test_Time_s"] = round(time_test_lstm, 6)
    results.append(row)

    time_series_store[well]["y_pred_lstm"] = y_pred_lstm
        
#Figure 1 - Time series plots
fig1, axes1 = plt.subplots(7, 1, figsize=(16, 22), sharex=False)

for i, well in enumerate(wells):
    ax = axes1[i]
    ts = time_series_store[well]

    ax.plot(ts["test_dates"], ts["y_actual"], color="black", linewidth=2.5, label="Measured",zorder=5)
    ax.plot(ts["test_dates"], ts["y_pred_xgb"], color="steelblue", linewidth=1.8, label="XGBoost",linestyle="dashdot")
    ax.plot(ts['test_dates'][lookback-1:], ts['y_pred_lstm'], color = 'orangered', linewidth=1.8, label='LSTM', linestyle = 'dotted' )

    ax.set_title(well, fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("GW Level (ft)", fontsize=16)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc="upper right", fontsize=16, ncol=4)

axes1[-1].set_xlabel("Date", fontsize=16)
plt.tight_layout()
fig1.savefig("fig1_timeseries.png", dpi=300, bbox_inches="tight")
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

    for k, (name, color) in enumerate(model_colors.items()):
        vals = [results_df[(results_df.Well == w) & (results_df.Model == name)][metric].iloc[0] for w in wells]
        ax.bar(x + (k - 0.5) * width, vals, width, label=name, color=color, edgecolor="none")

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
fig2.savefig("fig2_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

#summary table
results_df = results_df.round({"RMSE": 3, "R2": 3, "MAE": 3, "Willmott_d": 3, "PBIAS": 2, "Test_Time_s": 6})
results_df.to_csv("phase6_results_summary.csv", index=False)

# ChatGPT used for debugging tensor shape issues
# Attempted Nested CV: 
# Nested CV Definitions

# Creates nested CV
def nested_cv(X, y):

    # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    outer_split = TimeSeriesSplit(n_splits=5)
    all_splits = []

    for outer_level, (train_idx, val_idx) in enumerate(outer_split.split(X, y)):

        # Outer training and validation
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        # Inner cross-validation
        inner_split = TimeSeriesSplit(n_splits=5)
        inner_splits = []

        for inner_level, (inner_train_idx, test_idx) in enumerate(
            inner_split.split(X_train)
        ):

            # Inner training and test
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]

            X_test = X_train[test_idx]
            y_test = y_train[test_idx]

            inner_splits.append((X_inner_train, y_inner_train, X_test, y_test))

        all_splits.append((X_train, y_train, X_val, y_val, inner_splits))

    return all_splits


# Creates segmented datasets by well to use in the CV:
def dataset_create(well):
    cols = ["Date", well, "Precip_mm", "Temp_C", "Streamflow"]
    df_well = df[cols].copy()

    for lag in range(1, 13):
        df_well[f'GW_lag_{lag}'] = df_well[well].shift(lag)

    df_well = df_well.dropna().reset_index(drop=True)
    lag_cols = [f'GW_lag_{lag}' for lag in range(1, 13)]
    feature_cols = ["Precip_mm", "Temp_C", "Streamflow"] + lag_cols
    no_feat = len(feature_cols)
    X = df_well[feature_cols]
    y = df_well[well]
    y = np.asarray(y)

    splits = nested_cv(X, y)

    X_train, y_train, X_val, y_val, inner_splits = splits[0]
    X_inner_train, y_inner_train, X_test, y_test = inner_splits[0]

    # Scale features (fit on train only)
    scaler1 = StandardScaler()
    X_train_std = scaler1.fit_transform(X_train)
    X_val_std = scaler1.transform(X_val)
    scaler2 = StandardScaler()
    X_inner_train_std = scaler2.fit_transform(X_inner_train)
    X_test_std = scaler2.transform(X_test)

    # Scale target (fit on train only)
    y_scaler1 = StandardScaler()
    y_train_std = y_scaler1.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_std = y_scaler1.transform(y_val.reshape(-1, 1)).ravel()
    y_scaler2 = StandardScaler()
    y_inner_train_std = y_scaler2.fit_transform(y_inner_train.reshape(-1, 1)).ravel()
    y_test_std = y_scaler2.transform(y_test.reshape(-1,1)).ravel()

    # Create outer CV datasets: 
    X_train_seq, y_train_seq = create_dataset(X_train_std, y_train_std, lookback)
    X_val_seq, y_val_seq = create_dataset(X_val_std, y_val_std, lookback)

    # Create inner CV datasets: 
    X_inner_train_seq, y_inner_train_seq = create_dataset(X_inner_train_std, y_inner_train_std, lookback)
    X_test_seq, y_test_seq = create_dataset(X_test_std, y_test_std, lookback)

    print(f'For {well} : ')
    print(f'Outer CV: ')
    print('X_train shape:', X_train_seq.shape)
    print('y_train shape:', y_train_seq.shape)
    print('X_validate shape:', X_val_seq.shape)
    print('y_validate shape:', y_val_seq.shape)
    print(f'Inner CV:')
    print('X_train shape:', X_inner_train_seq.shape)
    print('y_train shape:', y_inner_train_seq.shape)
    print('X_test shape:', X_test_seq.shape)
    print('y_test shape:', y_test_seq.shape)

    # ChatGPT used here to help add lookback to previous flow for compatibility, while adding benefits of LSTM 
    #   for time sequence modeling, lag kept despite lookback for performance comparisons

    y_val_orig = y_val[lookback-1:] # observed values in ft, aligned with sequences
    y_test_orig = y_test[lookback - 1:]

    outer_sets = {'X_train': X_train_seq, 'y_train': y_train_seq, 'X_validate': X_val_seq, 'y_validate': y_val_seq, 'y_scaler' : y_scaler1, 'y_val_orig': y_val_orig}
    inner_sets = {'X_inner_train': X_inner_train_seq, 'y_inner_train': y_inner_train_seq, 'X_test': X_test_seq, 'y_test': y_test_seq, 'y_scaler': y_scaler2, 'y_test_orig' : y_test_orig}
    
    return (outer_sets, inner_sets)

# Defines models by number of hidden units, layers

class lstm_model_1(nn.Module):
    def __init__(self, n_features, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x[:, -1, :])
    
class lstm_model_2(nn.Module):
    def __init__(self, n_features, hidden_size=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x[:, -1, :])
# Defines a train/test run for each of the CV datasets
def train_one_split(dataset, parameter_list, split_name, set,
                    best_score, best_params, well):

    # Extract model configuration
    model_class, n_epochs = parameter_list
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract training and test data
    if set == 'inner_sets':
        X = torch.tensor(dataset['X_inner_train'], dtype=torch.float32)
        y = torch.tensor(dataset['y_inner_train'], dtype=torch.float32)
        X_test = torch.tensor(dataset['X_test'], dtype=torch.float32)
        y_test_orig = dataset['y_test_orig']

    elif set == 'outer_sets':
        X = torch.tensor(dataset['X_train'], dtype=torch.float32)
        y = torch.tensor(dataset['y_train'], dtype=torch.float32)
        X_test = torch.tensor(dataset['X_validate'], dtype=torch.float32)
        y_test_orig = dataset['y_val_orig']

    else:
        raise ValueError("Unknown dataset split")

    # Initialises a new model for every training run
    model = model_class(n_features=no_feat).to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    print('X size:', X.size())
    print('Y size:', y.size())

    loader = data.DataLoader(data.TensorDataset(X, y),shuffle=True,batch_size=8)

    # Training
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)

            # Ensure target and prediction have matching shapes
            loss = loss_fn(y_pred.reshape(-1), y_batch.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append([well, split_name, epoch, loss.item()])

    # Evaluation
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        start = time.time()
        y_pred_std = model(X_test).cpu().numpy()
        time_test_lstm = time.time() - start

    # Transform predictions back to original units
    y_scaler = dataset['y_scaler']
    y_pred_lstm = y_scaler.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()

    # Calculate metrics
    metrics_lstm = evaluate(y_test_orig, y_pred_lstm)

    # Store results
    row = {"Well": well, "Model": model_class.__name__, "Split": split_name, "Epochs": n_epochs}
    row.update(metrics_lstm)
    row["Test_Time_s"] = round(time_test_lstm, 6)
    tuned_results.append(row)

    # Store predictions
    if well not in time_series_store_tuning:
        time_series_store_tuning[well] = {}

    time_series_store_tuning[well][f"y_pred_{split_name}"] = y_pred_lstm

    # Update best parameters
    if metrics_lstm['RMSE'] < best_score:
        best_score = metrics_lstm['RMSE']
        best_params = parameter_list

        print(f"New best model for {well}: \n{model_class.__name__},\n{n_epochs} epochs, \nRMSE = {best_score:.4f}")

    return best_score, best_params

# Hyperparameter configurations
param_1 = [lstm_model_1, 50]
param_2 = [lstm_model_1, 100]
param_3 = [lstm_model_2, 50]
param_4 = [lstm_model_2, 100]

parameter_grid = [param_1,param_2,param_3,param_4]

# General settings
feature_cols = []
no_feat = 15
training_loss = []
lookback = 1

# Tuning
for well in wells:
    print(f'\nProcessing {well}')

    # Reset best score for EACH well
    best_score = float('inf')
    best_params = None

    # Create datasets for this well
    outer_sets, inner_sets = dataset_create(well)

    # Inner-set hyperparameter tuning
    for i, params in enumerate(parameter_grid):
        print(f'\nTesting parameter configuration {i+1}')
        best_score, best_params = train_one_split(
            dataset=inner_sets,
            parameter_list=params,
            split_name=f'Param {i+1}',
            set='inner_sets',
            best_score=best_score,
            best_params=best_params,
            well=well
        )

    print(f'\nBest parameters for {well}:')
    print(best_params)
    print(f'Best inner RMSE: {best_score:.4f}')

    # Train selected configuration on outer training set
    best_score_val, best_params_val = train_one_split(
        dataset=outer_sets,
        parameter_list=best_params,
        split_name='Validation',
        set='outer_sets',
        best_score=float('inf'),
        best_params=None,
        well=well
    )
