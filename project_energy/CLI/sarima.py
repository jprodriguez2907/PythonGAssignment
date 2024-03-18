import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer import Typer
import os

app = Typer()

#Create database and session
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
processed_path = os.path.join(grandparent_dir,'data', 'processed', 'database_energy.db')
db_path = processed_path.replace('\\', '/')

DB_URI = f'sqlite:///{db_path}'

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Function to train the SARIMA model and make predictions
@app.command()
def train_model(start_date, num_days):

    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]

    data["date"] = pd.to_datetime(data["date"])
    data.set_index('date', inplace=True)

    target = 'price actual'

    y = data[target]

    # SARIMA model training
    s = 7
    sar_model = SARIMAX(endog=y, order=(1, 0, 2), seasonal_order=(1, 1, 2, s)).fit()

    # Forecasting
    y_pred = sar_model.forecast(steps=num_days)
    date_range = pd.date_range(start=start_date, periods=num_days, freq='D')

    return y, y_pred, date_range, sar_model

# Function to visualize ACF and PACF plots
@app.command()
def plot_acf_pacf(y):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))

    for axes in ax:
        axes.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
        axes.tick_params(axis='x', colors='#1c0858')  # X-axis ticks in white color
        axes.tick_params(axis='y', colors='#1c0858')  # Y-axis ticks in white color

    plot_acf(y, lags=40, ax=ax[0], color='#0d9240', alpha=0.2)
    ax[0].set_title('ACF', color='#1c0858')

    plot_pacf(y, lags=40, ax=ax[1], color='#0d9240', alpha=0.2)
    ax[1].set_title('PACF', color='#1c0858')

    st.pyplot(fig)


# Function to perform statistical tests
@app.command()
def perform_statistical_tests(y, sar_model):
    # Augmented Dickey-Fuller Test (ADF)
    adf_result = adfuller(y)

    # Ljung-Box Test
    box_result = acorr_ljungbox(sar_model.resid, lags=[25])

    # Shapiro-Wilk Test
    shapiro_result = shapiro(sar_model.resid)

    # Crear una tabla con los resultados
    results_data = {
        "Test": ["ADF", "Shapiro-Wilk" ,"Ljung-Box"],
        "Statistic": [adf_result[0], shapiro_result[0], box_result],
        "P-value": [adf_result[1], shapiro_result[1]]
    }
    st.table(results_data)
@app.command()
def visualize_forecast(start_date, forecast_steps):
    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    target = 'price actual'
    y = data[target]

    # Filtrar los datos históricos hasta start_date
    y_train = y[y.index <= start_date]

    # Fit SARIMA model to historical data up to start_date
    s = 7
    sar_model = SARIMAX(endog=y_train, order=(1, 0, 2), seasonal_order=(1, 1, 2, s)).fit()

    # Generar el rango de fechas para las predicciones
    forecast_index = pd.date_range(start=start_date, periods=forecast_steps, freq='D')

    # Hacer predicciones para el rango de fechas generado
    forecast = sar_model.forecast(steps=forecast_steps)
    forecast.index = forecast_index  # Asignar el índice de pronóstico al rango de fechas

    # Plot actual values and predicted values
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.plot(y, label='Actual values', color='#1c0858')
    ax.plot(forecast, label='Predicted values', color='#0d9240')
    ax.set_xlabel('Date',fontsize=14, color='#1c0858')
    ax.set_ylabel('Electricity price',fontsize=14, color='#1c0858')
    ax.tick_params(axis='x', rotation=45, colors='#1c0858')
    ax.tick_params(axis='y', rotation=45, colors='#1c0858')# Rotar las fechas en el eje x
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
