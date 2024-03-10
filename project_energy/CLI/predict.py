from sklearn.metrics import mean_squared_error, mean_absolute_error
from typer import Typer, Option
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
#from database_session import SessionLocal
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
import joblib

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
def predict(date: str = Option(..., "--date", "-d",
                               help="The date as of which to make predictions (YYYY.MM.DD)")):
    """
    predicts energy prices as of given date, and saves actual and predicted values to database
    """
    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]

    # Turn input date into datetime
    initial_date = pd.to_datetime(date)
    data["date"] = pd.to_datetime(data["date"])

    # Load the trained model
    model = joblib.load("../model/trained_modelCLI")

    # Filter data from the initial date for predictions
    test = data[data["date"] >= initial_date]

    # Define features and target
    X_test = test.drop(columns=["price actual", "date"])
    y_test = test["price actual"]

    # Make predictions
    predictions = model.predict(X_test)

    # Create a DataFrame with predictions and index from test data
    predictions_df = pd.DataFrame({
        'date': test["date"],
        'predicted_energy_price': predictions,
        'actual_energy_price': test["price actual"]  # Add actual prices to the DataFrame
    }, index=test.index)

    # Save predictions to a CSV file in SQLite database
    with SessionLocal() as session:
        predictions_df.to_sql("predictions", session.get_bind(), if_exists="replace", index=False)
    print("Predictions saved to table predictions")

    return predictions_df



if __name__ == "__main__":
    app()