import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
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
@app.command()
def plot_predictions_ML(start_date, end_date, num_days_predicted):
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
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862)) # Set background color to match Streamlit background

    ax.plot(filtered_predictions["date"], filtered_predictions['actual_energy_price'], label='Actual Energy Price', color='#1c0858', marker='o')

    # Filter predictions to show only the number of days selected by the user
    filtered_predictions_pred = filtered_predictions.head(num_days_predicted)

    ax.plot(filtered_predictions_pred["date"], filtered_predictions_pred['predicted_energy_price'], label='Predicted Energy Price', color='#0d9240', linestyle='--',
             marker='x')
    ax.set_title('Actual vs Predicted Energy Prices', fontsize=16, color='#1c0858')
    ax.set_xlabel('Date', fontsize=14, color='#1c0858')
    ax.set_ylabel('Energy Price', fontsize=14, color='#1c0858')
    ax.tick_params(axis='x', colors='#1c0858')  # X-axis ticks in white color
    ax.tick_params(axis='y', colors='#1c0858')  # Y-axis ticks in white color
    ax.legend()
    plt.tight_layout()
    st.pyplot()


if __name__ == "__main__":
    main()
