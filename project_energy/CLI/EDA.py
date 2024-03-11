import seaborn as sns
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_data():
    DB_URI = "sqlite:///C:/Users/User/Desktop/MBD/Term2/PythonII/Group_Assignment/data/processed/database_energy.db"
    engine = create_engine(DB_URI)

    data = pd.read_sql_table("final_data", engine)
    return data

def plot_correlation_matrix():
    print("something")

def newfun():
    print("Hello World")

def plot_actual_price_vs_feature(data, feature, frequency):
    # Group data based on frequency
    if frequency == 'daily':
        pass  # No se realiza ningún agrupamiento
    elif frequency == 'weekly':
        data = data.groupby(pd.Grouper(key='date', freq='W')).mean().reset_index()
    elif frequency == 'monthly':
        data = data.groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
    elif frequency == 'yearly':
        data = data.groupby(pd.Grouper(key='date', freq='Y')).mean().reset_index()
    else:
        raise ValueError("Frequency should be 'daily', 'weekly', 'monthly', or 'yearly'.")

    # Variables
    y = data["price actual"]
    x = data["date"]
    feature_data = data[feature]

    # Create the figure and first y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the actual price on the first y-axis
    ax1.plot(x, y, color='tab:blue', label="Actual Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Actual Price", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create the second y-axis (twin axis)
    ax2 = ax1.twinx()

    # Plot the feature on the second y-axis
    ax2.plot(x, feature_data, color='tab:red', label=feature)
    ax2.set_ylabel(feature, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add title and legend
    plt.title("Actual Price vs. " + feature)
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)





if __name__ == "__main__":
    main()
