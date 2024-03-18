from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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
def plot_actual_price_vs_feature(feature, frequency):

    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]

    # Convertir la columna de fecha en DatetimeIndex
    data['date'] = pd.to_datetime(data['date'])

    # Establecer la columna de fecha como índice
    data.set_index('date', inplace=True)

    # Group data based on frequency
    if frequency == 'daily':
        pass  # No se realiza ningún agrupamiento
    elif frequency == 'weekly':
        data = data.groupby(pd.Grouper(freq='W')).mean().reset_index()
    elif frequency == 'monthly':
        data = data.groupby(pd.Grouper(freq='M')).mean().reset_index()
    elif frequency == 'yearly':
        data = data.groupby(pd.Grouper(freq='Y')).mean().reset_index()
    else:
        raise ValueError("Frequency should be 'daily', 'weekly', 'monthly', or 'yearly'.")

    # Variables
    y = data["price actual"]
    x = data.index  # Usar el índice de fecha como variable x
    feature_data = data[feature]

    # Create the figure and first y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))

    # Plot the actual price on the first y-axis
    ax1.plot(x, y, color='#1c0858', label="Actual Price")
    ax1.set_xlabel("Date", color='#1c0858', fontsize=18)
    ax1.set_ylabel("Actual Price", color='#1c0858', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='#1c0858')

    # Create the second y-axis (twin axis)
    ax2 = ax1.twinx()

    # Plot the feature on the second y-axis
    ax2.plot(x, feature_data, color="#0d9240", label=feature)
    ax2.set_ylabel(feature, color="#0d9240",fontsize=18)
    ax2.tick_params(axis='y', labelcolor="#0d9240")

    # Add title and legend
    ax1.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax1.tick_params(axis='x', colors='#1c0858')  # X-axis ticks in white color
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


if __name__ == "__main__":
    main()