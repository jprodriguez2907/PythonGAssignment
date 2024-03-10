# Import libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Define the database URI
DB_URI = "sqlite:///C:/Users/User/Desktop/MBD/Term2/PythonII/Group_Assignment/data/processed/database_energy.db"

# Create SQLAlchemy engine
engine = create_engine(DB_URI)

# Read data from the database table
energy_df = pd.read_sql_table("final_data", engine)

# Define target variable and features
target = 'price actual'
y = energy_df[target]
X = energy_df.drop(columns=['price actual','date'])

# Define test index based on the last 6 months of data
test_index = energy_df["date"].iloc[-1] - pd.DateOffset(months=6)

# Split data into training and testing sets
X_train, X_test = X.loc[energy_df['date'] < test_index], X.loc[energy_df['date'] >= test_index]
y_train, y_test = y.loc[energy_df['date'] < test_index], y.loc[energy_df['date'] >= test_index]
test_dates = energy_df.loc[energy_df['date'] >= test_index, 'date']

# Define time series cross-validation strategy
tscv = TimeSeriesSplit(n_splits=5)

# Define the search space for hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Create an XGBoost model
model_XGB = XGBRegressor()

# Create a GridSearchCV instance
grid_search = GridSearchCV(estimator=model_XGB, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv)

# Train the model on the training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameter configuration
best_params = grid_search.best_params_

# Print the best configuration
print("Best parameters:", best_params)

# Get the best validation score
best_score = grid_search.best_score_

# Print the best score
print("Best score:", best_score)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate RMSE, MAPE, MAE and MSE for train and test sets
train_rmse = mean_squared_error(y_train, best_model.predict(X_train), squared=False)
train_mae = mean_absolute_error(y_train, best_model.predict(X_train))
train_mse = mean_squared_error(y_train, best_model.predict(X_train))

test_rmse = mean_squared_error(y_test, y_pred, squared=False)
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

# Print the results
print("-" * 50)
print("Train Results:")
print("RMSE:", train_rmse)
print("MAE:", train_mae)
print("MSE:", train_mse)

print("-" * 50)
print("Test Results:")
print("RMSE:", test_rmse)
print("MAE:", test_mae)
print("MSE:", test_mse)

# Feature importance (if the model supports it)
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_

    # Print the feature importances
    print("-" * 50)
    print("Feature importances:")
    for i, importance in enumerate(feature_importances):
        print(f"{i+1}. {X_train.columns[i]}: {importance}")
else:
    print("The selected model does not support feature importances.")


# Define a function to predict next price energy
def predict_next_price_energy(X, model, n_periods):
    X_pred = X.copy()
    y_pred = np.zeros(n_periods)

    lag_columns = ['price(t-' + str(i) + ')' for i in range(1, 13)]

    for i in range(n_periods):
        X_pred = pd.concat([X_pred, X_pred.iloc[-1:, :]], axis=0, ignore_index=True)
        y_pred[i] = model.predict(X_pred.iloc[-1:])
        for j in range(1, 13):
            lag_column = f'price(t-{j})'
            X_pred[lag_column].iloc[-1] = X_pred[lag_column].iloc[-2]
        X_pred[lag_columns[0]].iloc[-1] = y_pred[i]

    return X_pred, y_pred

# Predict next price energy
X_pred, y_pred = predict_next_price_energy(X_train, grid_search.best_estimator_, n_periods=12)

# Concatenate the predictions to the original dataframe
y_pred = pd.concat([y_train, pd.Series(y_pred)], axis=0)
y_true = pd.concat([y_train, y_test.iloc[0:16]], axis=0)
time = pd.concat([energy_df.loc[energy_df['date'] < test_index, 'date'], test_dates.iloc[0:16]], axis=0)

# Get the last 60 values for plotting
last_60_idx = -60
y_pred_last_60 = y_pred.iloc[last_60_idx:]
y_true_last_60 = y_true.iloc[last_60_idx:]
time_last_60 = time.iloc[last_60_idx:]

# Plot the results
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time_last_60, y_pred_last_60, label='Prediction', color='dodgerblue')
ax.plot(time_last_60, y_true_last_60, label='True', color='darkorange')
ax.set_title('Price of energy: True vs Predicted', fontsize=16)
ax.set_ylabel('Price of energy', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
plt.legend()
plt.show()
