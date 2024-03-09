# Import libraries
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mape
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import numpy as np


#Load cleaned data from SQLite database into Python object
query = text("SELECT * FROM cleaned_data")

with SessionLocal() as session:
    df_from_database = pd.DataFrame(session.execute(query))

df.head()

df.info()

df.isna().sum()

target = 'price_actual'

y = df[target]
X = df.drop(columns=target)

#scale_X = MinMaxScaler(feature_range=(0, 1))
#scale_y = MinMaxScaler(feature_range=(0, 1))
#scale_X.fit(X[:train_end_idx])
#scale_y.fit(y[:train_end_idx])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7)


# Define the search space
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
grid_search = GridSearchCV(estimator=model_XGB, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

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

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)

# Calculate RMSE, MAPE, MAE and MSE for train and test sets
train_rmse = mean_squared_error(y_train, grid_search.predict(X_train), squared=False)
train_mape = mape(y_train, grid_search.predict(X_train))
train_mae = mean_absolute_error(y_train, grid_search.predict(X_train))
train_mse = mean_squared_error(y_train, grid_search.predict(X_train))

test_rmse = mean_squared_error(y_test, y_pred, squared=False)
test_mape = mape(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

# Print the results
print("-" * 50)
print("Train Results:")
print("RMSE:", train_rmse)
print("MAPE:", train_mape)
print("MAE:", train_mae)
print("MSE:", train_mse)

print("-" * 50)
print("Test Results:")
print("RMSE:", test_rmse)
print("MAPE:", test_mape)
print("MAE:", test_mae)
print("MSE:", test_mse)

# Feature importance
feature_importances = model_XGB.feature_importances_

# Print the feature importances
print("-" * 50)
print("Feature importances:")
for i, importance in enumerate(feature_importances):
    print(f"{i+1}. {X_train.columns[i]}: {importance}")


# Define the time
time = np.arange(len(y_test))

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the real values and predictions
ax.plot(time, y_test, label="Real Values")
ax.plot(time, y_pred, label="Predictions")

# Customize the plot
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()

# Show the plot
plt.show()

'''
idx = 200
aa=[x for x in range(idx)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[:idx], marker='.', label="actual")
plt.plot(aa, test_predict[:idx], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('TOTAL Load', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();


# Import libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import adfuller, normal_ad
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Auto-ARIMA
model_arima = auto_arima(y_train, seasonal=True, trace=True)

# Predict ARIMA
predictions_arima = model_arima.predict(start=test.index[0], end=test.index[-1])



# Plot the time series
plt.plot(y)
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Plot the ACF and PACF
plot_acf(y, lags=40)
plt.show()
plot_pacf(y, lags=40)
plt.show()

# Perform the ADF test for stationarity
adf_result = adfuller(y)
print('ADF test:', adf_result[0])

# Perform the Shapiro test for normality
shapiro_result = normal_ad(y)
print('Shapiro test:', shapiro_result[1])

# Perform the Box test for normality of the residuals
box_result = box_ljung(y, lags=40)
print('Box test:', box_result[1])

'''