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

def Plot_predictions_ML(start_date, end_date):
    st.title('Plot Predictions for ML Models')

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

    # Plotear datos
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_predictions["date"], filtered_predictions['actual_energy_price'], label='Actual Energy Price', color='blue', marker='o')
    ax.plot(filtered_predictions["date"], filtered_predictions['predicted_energy_price'], label='Predicted Energy Price', color='red', linestyle='--',
             marker='x')
    ax.set_title('Actual vs Predicted Energy Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Price')
    ax.legend()
    plt.tight_layout()
    st.pyplot()

if __name__ == "__main__":
    main()
