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

def plot_monthprice(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    monthly_mean_price = data.groupby(['year', 'month'])['price actual'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
    ax.set_xlabel('Month', fontsize=14, color='white')
    ax.set_ylabel('Mean Energy Price', fontsize=14, color='white')
    for year in range(2015, 2019):
        monthly_mean_price[year].plot(label=str(year), ax=ax)

    plt.title('Monthly Mean Energy Price for Each Year', fontsize=25, color='white')
    plt.legend(title='Year', fontsize=12)
    plt.grid(True, color='white')
    ax.tick_params(axis='x', colors='white')  # X-axis ticks in white color
    ax.tick_params(axis='y', colors='white')  # Y-axis ticks in white color
    st.pyplot(fig)

if __name__ == "__main__":
    main()
