import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer import Typer
import os

app = Typer()

# Define paths to access the database
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
processed_path = os.path.join(
    grandparent_dir, "data", "processed", "database_energy.db"
)
db_path = processed_path.replace("\\", "/")

DB_URI = f"sqlite:///{db_path}"

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# Define a command for plotting monthly mean energy price
@app.command()
def plot_monthprice():
    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [
            column[0] for column in session.execute(query).cursor.description
        ]

    # Convert 'date' column to datetime
    data["date"] = pd.to_datetime(data["date"])
    # Extract year and month from 'date' column
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    # Calculate monthly mean energy price
    monthly_mean_price = data.groupby(["year", "month"])["price actual"].mean()

    # Plotting monthly mean energy price for each year
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_facecolor((0.9607843137254902, 0.9568627450980393, 0.9450980392156862))
    ax.set_xlabel("Month", fontsize=14, color="#1c0858")
    ax.set_ylabel("Mean Energy Price", fontsize=14, color="#1c0858")
    for year in range(2015, 2019):
        monthly_mean_price[year].plot(label=str(year), ax=ax)

    plt.legend(title="Year", fontsize=12)
    plt.grid(True, color="#bcbcc3")
    ax.tick_params(axis="x", colors="#1c0858")
    ax.tick_params(axis="y", colors="#1c0858")

    st.pyplot(fig)


if __name__ == "__main__":
    main()
