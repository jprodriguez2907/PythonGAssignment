import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

# Function to load data from the SQLite database
def load_data():
    DB_URI = "sqlite:///../../data/processed/database_energy.db"
    engine = create_engine(DB_URI, pool_pre_ping=True)
    query = "SELECT * FROM cleaned_data"
    data = pd.read_sql_query(query, engine)
    return data

# Function to train the SARIMA model and make predictions
def train_model(data, start_date, num_days):
    data["date"] = pd.to_datetime(data["date"])
    data.set_index('date', inplace=True)

    target = 'price actual'
    y = data.set_index('date')[target]

    # SARIMA model training
    s = 7
    sar_model = SARIMAX(endog=y, order=(1, 0, 2), seasonal_order=(1, 1, 2, s)).fit()

    # Forecasting
    y_pred = sar_model.forecast(steps=num_days)
    date_range = pd.date_range(start=start_date, periods=num_days, freq='D')

    return y, y_pred, date_range, sar_model

# Function to visualize ACF and PACF plots
def plot_acf_pacf(y):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(y, lags=40, ax=ax[0])
    ax[0].set_title('Autocorrelation Function (ACF)')
    plot_pacf(y, lags=40, ax=ax[1])
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    st.pyplot(fig)

# Function to perform statistical tests
def perform_statistical_tests(y, sar_model):
    # Augmented Dickey-Fuller Test (ADF)
    adf_result = adfuller(y)
    st.write(f'ADF Test:\nStatistic: {adf_result[0]}\nP-value: {adf_result[1]}')

    # Ljung-Box Test
    box_result = acorr_ljungbox(sar_model.resid, lags=[25])
    st.write(f'Ljung-Box Test (residuals):\nP-value: {box_result[1]}')

    # Shapiro-Wilk Test
    shapiro_result = shapiro(sar_model.resid)
    st.write(f'Shapiro-Wilk Test (residuals):\nStatistic: {shapiro_result[0]}\nP-value: {shapiro_result[1]}')

def main():
    st.title("Electricity Price Prediction")

    # Load data
    data = load_data()

    # Sidebar options
    st.sidebar.title("Options")
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2017-01-01"))
    num_days = st.sidebar.slider("Number of days to predict", min_value=1, max_value=30, value=20)

    # Train model and make predictions
    y, y_pred, date_range, sar_model = train_model(data, start_date, num_days)

    # Plot the graph
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['price actual'], label='Actual values')
    ax.plot(date_range, y_pred, label='Predicted values', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Electricity Price')
    ax.legend()
    st.pyplot(fig)

    # Plot ACF and PACF
    st.subheader("Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)")
    plot_acf_pacf(y)

    # Perform statistical tests
    st.subheader("Statistical Tests")
    perform_statistical_tests(y, sar_model)

if __name__ == "__main__":
    main()
