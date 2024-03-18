import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
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
def plot_histogram(feature):


    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]


    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_xlabel('Importance', fontsize=15, color='#1c0858')
    ax.set_ylabel('Feature', fontsize=15, color='#1c0858')
    plt.hist(data[feature], bins=20, color='#0d9240', edgecolor="#bcbcc3")
    plt.xlabel(feature, color='#1c0858')
    plt.ylabel('Frequency', color='#1c0858')
    plt.grid(True, color="#bcbcc3")
    ax.tick_params(axis='x', colors='#1c0858')  # X-axis ticks in white color
    ax.tick_params(axis='y', colors='#1c0858')  # Y-axis ticks in white color
    st.pyplot(fig)

if __name__ == "__main__":
    main()
