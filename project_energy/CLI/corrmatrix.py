import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
def plot_correlation_matrix():

    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]

    features_cor = data[list(data.columns[1:33])]
    fig, ax = plt.subplots(figsize=(40, 40))
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_xlabel('Importance', fontsize=20, color='#1c0858')
    ax.set_ylabel('Feature', fontsize=20, color='#1c0858')
    sns.heatmap(round(features_cor.corr(), 1), annot=True, cmap='Greens', linewidth=0.9, ax=ax)
    ax.tick_params(axis='x', colors='#1c0858', labelsize=30, rotation=45)   # X-axis ticks in white color
    ax.tick_params(axis='y', colors='#1c0858',labelsize=30, rotation=45)  # Y-axis ticks in white color
    st.pyplot(fig)

if __name__ == "__main__":
    main()
