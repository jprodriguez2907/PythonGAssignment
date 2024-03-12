import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import warnings
warnings.filterwarnings('ignore')

# Create database and session
DB_URI = "sqlite:///../../data/processed/database_energy.db"
engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def Plot_predictions_ML(start_date, end_date, num_days_predicted):
    # Load predicted and actual values from database
    query = text("SELECT * FROM predictions")
    with SessionLocal() as session:
        predictions = pd.DataFrame(session.execute(query).fetchall())
        predictions.columns = [column[0] for column in session.execute(query).cursor.description]

    # Ensure column date is in date format
    predictions['date'] = pd.to_datetime(predictions['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data by date range
    filtered_predictions = predictions[(predictions['date'] >= start_date) & (predictions['date'] <= end_date)]

    # Plot data
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353)) # Set background color to match Streamlit background
    ax.plot(filtered_predictions["date"], filtered_predictions['actual_energy_price'], label='Actual Energy Price', color='white', marker='o')

    # Filter predictions to show only the number of days selected by the user
    filtered_predictions_pred = filtered_predictions.head(num_days_predicted)

    ax.plot(filtered_predictions_pred["date"], filtered_predictions_pred['predicted_energy_price'], label='Predicted Energy Price', color='red', linestyle='--',
             marker='x')
    ax.set_title('Actual vs Predicted Energy Prices', fontsize=16, color='white')  # Title in larger size and white color
    ax.set_xlabel('Date', fontsize=14, color='white')  # X-axis label in larger size and white color
    ax.set_ylabel('Energy Price', fontsize=14, color='white')  # Y-axis label in larger size and white color
    ax.tick_params(axis='x', colors='white')  # X-axis ticks in white color
    ax.tick_params(axis='y', colors='white')  # Y-axis ticks in white color
    ax.legend()
    plt.tight_layout()
    st.pyplot()


if __name__ == "__main__":
    main()
