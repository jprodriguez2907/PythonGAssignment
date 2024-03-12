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

def plot_histogram(data,feature):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_xlabel('Importance', fontsize=15, color='white')
    ax.set_ylabel('Feature', fontsize=15, color='white')
    plt.hist(data[feature], bins=20, color='red', edgecolor='black')
    plt.title('Histogram of ' + feature, color='white', fontsize=25)
    plt.xlabel(feature, color='white')
    plt.ylabel('Frequency', color='white')
    plt.grid(True, color='white')
    ax.tick_params(axis='x', colors='white')  # X-axis ticks in white color
    ax.tick_params(axis='y', colors='white')  # Y-axis ticks in white color
    st.pyplot(fig)

if __name__ == "__main__":
    main()
