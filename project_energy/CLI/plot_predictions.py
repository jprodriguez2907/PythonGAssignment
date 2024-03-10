from typer import Typer, Option
#from database_session import SessionLocal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import pandas as pd
import matplotlib.pyplot as plt

app = Typer()

#Create database and session
DB_URI = "sqlite:///../../data/processed/database_energy.db"

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)



@app.command()
def plot_predictions():
    """
    plots predicted and actual energy prices
    """
    # Load predicted and actual values from database
    query = text("SELECT * FROM predictions")
    with SessionLocal() as session:
        predictions = pd.DataFrame(session.execute(query).fetchall())
        predictions.columns = [column[0] for column in session.execute(query).cursor.description]

    #Ensure column date is in date format
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Plot actual vs predicted energy prices
    plt.figure(figsize=(12, 6))
    plt.plot(predictions["date"], predictions['actual_energy_price'], label='Actual Energy Price', color='blue', marker='o')
    plt.plot(predictions["date"], predictions['predicted_energy_price'], label='Predicted Energy Price', color='red', linestyle='--',
             marker='x')
    plt.title('Actual vs Predicted Energy Prices')
    plt.xlabel('Date')
    plt.ylabel('Energy Price')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    app()