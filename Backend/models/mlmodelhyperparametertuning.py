# ### Importing Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
warnings.filterwarnings('ignore')

# ### Loading Data

df = pd.read_csv(r"C:\Users\jesme\OneDrive\Desktop\gtbit_minor_project\Car-Price-Prediction\Backend\Data\train.csv")
df.head()

# ### Preprocessing

# #### Checking null values in the dataset

print(df.isnull().sum())

# Checking total number of rows and columns in the dataset

print(df.shape)
print(f"This DataSet Contains {df.shape[0]} rows & {df.shape[1]} columns")

# #### Checking data types of the columns

df.info()

# #### Describing the dataset

df.describe(include="all").T

# ## Data Cleaning

# #### Dropping duplicate values

df1 = df.drop_duplicates(keep='first')
luxury_brands = ['BMW', 'MERCEDES-BENZ', 'LEXUS', 'AUDI', 'PORSCHE', 'JAGUAR', 'LAND ROVER']

print(df1["Manufacturer"])
# #### Checking total number of rows and columns after dropping duplicates

print(df1.shape)
print(f"This DataSet Contains {df1.shape[0]} rows & {df1.shape[1]} columns after dropping duplicate values")

# #### Handling missing or ambiguous data

# Replace '-' with '0' in 'Levy' column
df1["Levy"] = df1["Levy"].replace("-", "0")
df1["Levy"] = df1["Levy"].astype("int64")

# Remove ' km' from 'Mileage' column and convert to integer
df1["Mileage"] = df1["Mileage"].str.replace(" km", "")
df1["Mileage"] = df1["Mileage"].astype("int64")

# Replace ambiguous door values
df1["Doors"] = df1["Doors"].replace({"04-May": "4-5", "02-Mar": "2-3"})
df1["Doors"].unique()

# Remove 'Turbo' from 'Engine volume' and convert to float
df1["Engine volume"] = df1["Engine volume"].str.replace("Turbo", "")
df1["Engine volume"] = df1["Engine volume"].astype("float64")

# Drop 'ID' column as it doesn't provide useful information
df1 = df1.drop(["ID"], axis=1)

# #### Adding 'Condition Score' and 'Safety Score' Features

# Since these features are not present in the dataset, we'll simulate them.
# 'Condition Score' ranges from 1 (Poor) to 10 (Excellent)
# 'Safety Score' ranges from 1 (Low Safety) to 5 (High Safety)

np.random.seed(42)  # For reproducibility

#Adding new features
df1['normalised_Mileage'] = scaler.fit_transform(df1[['Mileage']])
df1['Condition Score'] = df1['normalised_Mileage'] + (2020 - df1['Prod. year'])
# df1['Price_per_Cylinder'] = df1['Price'] / df1['Cylinders']
df1['Safety Score'] = df1['Airbags']/12



df1 = df1.drop(["normalised_Mileage","Airbags","Cylinders"], axis=1)


# Check the DataFrame after adding new features
df1.head()

# ## Handling Outliers

# Function to handle outliers using IQR

def outlier_handle(data):
    df_copy = data.copy()
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    for col_name in numeric_cols:
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        df_copy[col_name] = np.where(data[col_name] > fence_high, fence_high,
                                     np.where(data[col_name] < fence_low, fence_low, data[col_name]))
    return df_copy

df1 = outlier_handle(df1)

# ### Encoding

from sklearn.preprocessing import LabelEncoder

# Copy the DataFrame to avoid modifying the original data
new_df = df1.copy()

# Identify all object-type columns
object_cols = new_df.select_dtypes(include=['object']).columns
print("Categorical columns to encode:", object_cols)

# Initialize a LabelEncoder dictionary
label_encoders = {}

# Encode each categorical column
for col in object_cols:
    le = LabelEncoder()
    new_df[col] = le.fit_transform(new_df[col].astype(str))
    label_encoders[col] = le

# Verify that all columns are now numeric
print(new_df.dtypes)

# ## Data Modeling

# ### Splitting the Target and Data

X = new_df.drop('Price', axis=1)
Y = new_df['Price']

# ### Splitting the Training and Test Data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print("x_train =", x_train.shape)
print("x_test =", x_test.shape)
print("y_train =", y_train.shape)
print("y_test =", y_test.shape)

# ### Scaling the Data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""# ### Training the Random Forest Model

from sklearn.ensemble import RandomForestRegressor
param_grid = {
    'n_estimators': [200,300,400],
    'max_depth': [15, 20,25],
    'max_features': ['auto', 'log2', 'sqrt'],
    'min_samples_split': [2,3,4],
    'min_samples_leaf': [1, 2,3,4,5]
}

# Initialize the Random Forest Regressor
Rf = RandomForestRegressor(random_state=1)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=Rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)

# Fit the GridSearch to the training data
grid_search.fit(x_train, y_train)

# Best parameters from GridSearchCV
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Use the best estimator to make predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)

# Calculate evaluation metrics
mae = round(mean_absolute_error(y_test, y_pred), 3)
mse = round(mean_squared_error(y_test, y_pred), 3)
rmse = round(np.sqrt(mse), 3)
r2_value = round(r2_score(y_test, y_pred), 3)

print('Mean Absolute Error of the tuned model is : {}'.format(mae))
print('Mean Squared Error of the tuned model is : {}'.format(mse))
print('Root Mean Squared Error of the tuned model is : {}'.format(rmse))
print('R-squared value of the tuned model is : {}'.format(r2_value))"""
param_grid = {
    'n_estimators': [300,500],
    'learning_rate': [0.1,0.15,0.2],
    'max_depth': [7,9],
    'min_child_weight': [1,2],
    'subsample': [0.7,0.8,0.9],
    'colsample_bytree': [0.9,1.0,1.1],
    'gamma': [0, 0.01],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
}


# Initialize the XGBRegressor
import xgboost as xgb
xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=1)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)

# Fit the GridSearch to the training data
grid_search.fit(x_train, y_train)

# Best parameters from GridSearchCV
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Use the best estimator to make predictions
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(x_test)

# Calculate evaluation metrics
mae = round(mean_absolute_error(y_test, y_pred), 3)
mse = round(mean_squared_error(y_test, y_pred), 3)
rmse = round(np.sqrt(mse), 3)
r2_value = round(r2_score(y_test, y_pred), 3)

print('Mean Absolute Error of the tuned model is : {}'.format(mae))
print('Mean Squared Error of the tuned model is : {}'.format(mse))
print('Root Mean Squared Error of the tuned model is : {}'.format(rmse))
print('R-squared value of the tuned model is : {}'.format(r2_value))