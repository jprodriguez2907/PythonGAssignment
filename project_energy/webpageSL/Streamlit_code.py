import streamlit as st
import pandas as pd
import sys

from matplotlib import pyplot as plt

sys.path.append("C:\\Users\\User\\Desktop\\MBD\\Term2\\PythonII\\Group_Assignment\\project_energy\\CLI")
from sarima2 import load_data, train_model, plot_acf_pacf, perform_statistical_tests


def main():
    st.title("Electricity Price Prediction")

    # Load data
    data = load_data()

    # Sidebar options
    st.sidebar.title("Options")
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2017-01-01"))
    num_days = st.sidebar.slider("Number of days to predict", min_value=1, max_value=30, value=20)

    # Train model and make predictions
    y, y_pred, date_range, sar_model = train_model(data, "2017.01.01", num_days)

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


