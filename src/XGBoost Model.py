#Created by: Fabian Melendez
#Created On:  13 Mar 2026
#Course: CSCI 5420 Applied Machine Learning I 
#Phase 4 
#Objective:  Begin Implementing model trials with XGBoost, Random Forest, and SVM from scikit learn
# 
# 

# Install standard packages and import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# Load dataset
df = pd.read_csv("input.csv", parse_dates=["Date"])
df["Date"] = pd.to_datetime(df["Date"]) #convert date column to datetime object 
df = df.sort_values("Date") #Sort date values 



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

print(df.dtypes) #Confirm Data types with dataset 
print(df.head()) #Confirm output 



# Create lag features for Well_1
#We needed to build in a lag because groundwater levels arent immediatly impacted by precipitation events
#It takes some time for precipitation to reach groundwater levels.  For example the current months rainfall may not immediatly
# the current months GW levels.  This allows us to build a history and train the model on previous months.  This code creates the ability
# for the model to reference data from 1-12 months ago.  This essentially becomes an additional set of features that the model can learn from  

for lag in range(1, 13):         
    df[f"GW_lag_{lag}"] = df["GW_1"].shift(lag) #1 of 2 places Change the Well number in these lines to get specific well metrics 

# Drop rows with missing lag values # We need this because when we shift our GW levels down and it results in the value
#above have a NA or empty value.  We dont want these in our dataset
df = df.dropna()



#Define features (X) and target (y)
#Store column values from 12 lag columns in list, makes it shorter to add these to the X= statement 
lag_cols = [f"GW_lag_{lag}" for lag in range(1, 13)]

#Creates a feature matrix with 15 dimension (3 basic plus the 12 new lag columns)
X = df[["Precip_mm", "Temp_C", "Streamflow"] + lag_cols] #Features
y = df["GW_1"] #Target Variable #2 of 2 places Change the Well number in these lines to get specific well metrics 

#Split test and training data
#X_train- 3 basic features plus lag columns
#X_test- 3 basic features plus lags
#y_train- groundwater levels from earlier timeperiods
#y_test - groundwater levels for testing target values
#Key Point- We need shuffle=False here since we are dealing with time data.  Shuffling would distort data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)



#Standardize features - transform all of our features to bring everything to mean=0 and standard deviation =1
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train) #Standardize training data
X_test_std = scaler.transform(X_test) #Standardize test data not fit on test data


# Call XGBoost model 
model = XGBRegressor(
    n_estimators=800, #number of trees the model builds 
    eta=0.03, #learning rate  
    max_depth=4,# Depth of each tree 
    random_state=1 #Seed 
)

#Training the Model 
model.fit(X_train_std, y_train) #Train model on training data and target training data 
y_pred = model.predict(X_test_std)


#Evaluate model performance
start_test= time.perf_counter()
y_pred_DT_only = model.predict(X_test_std)
end_test=time.perf_counter()
test_time=end_test-start_test



#Root Mean Square Error -computes average squared between actual(y_test)and Predicted(y_pred)
#Answers on average how far off are my predicitions from the real groundwater levels (ft)
#Measures absolute error-how close are my predicitions 
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

#Coefficient of Determination 
#Determinines how much the variance in groundwater levels is explained by the model (Perfect R2=1.0)
#How well the model fits the data
r2 = r2_score(y_test, y_pred)

print("RMSE (ft):", rmse)
print("R2:", r2)
print(f"Testing Time: {test_time:.4e} s")

#Plot actual vs predicted groundwater levels
#Plotting test data-Since we didnt use shuffle test data starts at point where training data ends.
plt.figure(figsize=(14, 6))#(w,h)
plt.plot(df["Date"].iloc[len(X_train):], y_test, label="Actual", linewidth=2) # Actual Ground Waterlevels 
plt.plot(df["Date"].iloc[len(X_train):], y_pred, label="Predicted", linewidth=2)# Predicted Groundwater levls
plt.title("Groundwater Level Prediction (XGBoost)")
plt.xlabel("Date")
plt.ylabel("Groundwater Level (ft)")
plt.legend()
plt.grid(True)
plt.show()

"""
# Add Date back to the test set for grouping
test_results = pd.DataFrame({
    "Date": df["Date"].iloc[len(X_train):].values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

test_results["Year"] = test_results["Date"].dt.year
test_results["Month"] = test_results["Date"].dt.month

# Compute monthly RMSE and R²
monthly_stats = test_results.groupby("Month").apply(
    lambda g: pd.Series({
        "RMSE": np.sqrt(mean_squared_error(g["Actual"], g["Predicted"])),
        "R2": r2_score(g["Actual"], g["Predicted"])
    })
)

print(monthly_stats)
plt.figure(figsize=(12,5))
plt.plot(monthly_stats.index, monthly_stats["RMSE"], marker="o", label="RMSE")
plt.title("Monthly Prediction Error (RMSE)")
plt.xlabel("Month")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(monthly_stats.index, monthly_stats["R2"], marker="o", color="green", label="R²")
plt.title("Monthly Prediction Accuracy (R²)")
plt.xlabel("Month")
plt.ylabel("R² Score")
plt.grid(True)
plt.show()



annual_stats = test_results.groupby("Year").apply(
    lambda g: pd.Series({
        "RMSE": np.sqrt(mean_squared_error(g["Actual"], g["Predicted"])),
        "R2": r2_score(g["Actual"], g["Predicted"])
    })
)

plt.figure(figsize=(12,5))
plt.plot(annual_stats.index, annual_stats["RMSE"], marker="o", label="RMSE")
plt.title("Annual Prediction Error (RMSE)")
plt.xlabel("Year")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(annual_stats.index, annual_stats["R2"], marker="o", color="purple", label="R²")
plt.title("Annual Prediction Accuracy (R2)")
plt.xlabel("Year")
plt.ylabel("R2 Score")
plt.grid(True)
plt.show()




################################################################################################
"""