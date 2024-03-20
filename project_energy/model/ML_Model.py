# Import libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import warnings
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer import Typer
import os
warnings.filterwarnings('ignore')

app = Typer()

#Create database and session
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
processed_path = os.path.join(grandparent_dir,'data', 'processed', 'database_energy.db')
db_path = os.path.join(processed_path, 'database_energy.db')
db_path = processed_path.replace('\\', '/')

DB_URI = f'sqlite:///{db_path}'

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

query = text("SELECT * FROM final_data")
with SessionLocal() as session:
    energy_df = pd.DataFrame(session.execute(query).fetchall())
    energy_df = [column[0] for column in session.execute(query).cursor.description]

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

# Create an LGBM model
model = LGBMRegressor(random_state=13)

# Create a GridSearchCV instance
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv)

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
