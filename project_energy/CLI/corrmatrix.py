import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns

# Function to load data from the SQLite database
def load_data():
    DB_URI = "sqlite:///C:/Users/User/Desktop/MBD/Term2/PythonII/Group_Assignment/data/processed/database_energy.db"
    engine = create_engine(DB_URI)
    data = pd.read_sql_table("final_data", engine)
    return data

def plot_correlation_matrix(data):
    features_cor = data[list(data.columns[1:33])]
    fig, ax = plt.subplots(figsize=(40, 40))
    fig.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_xlabel('Importance', fontsize=20, color='white')
    ax.set_ylabel('Feature', fontsize=20, color='white')
    plt.title('Correlation Matrix ', color='white', fontsize=100)
    sns.heatmap(round(features_cor.corr(), 1), annot=True, cmap='Reds', linewidth=0.9, ax=ax)
    ax.tick_params(axis='x', colors='white', labelsize=30, rotation=45)   # X-axis ticks in white color
    ax.tick_params(axis='y', colors='white',labelsize=30, rotation=45)  # Y-axis ticks in white color
    st.pyplot(fig)

if __name__ == "__main__":
    main()
