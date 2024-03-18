from sklearn.metrics import mean_squared_error, mean_absolute_error
from typer import Typer, Option
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
import joblib
import os

app = Typer()

#Create database and session
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
processed_path = os.path.join(grandparent_dir,'data', 'processed', 'database_energy.db')
db_path = os.path.join(processed_path, 'database_energy.db')
db_path = processed_path.replace('\\', '/')

DB_URI = f'sqlite:///{db_path}'

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
@app.command()
def predict_ML(date: str = Option(..., "--date", "-d", help="The date as of which to make predictions (YYYY.MM.DD)")):
    """
    Predict energy prices as of a given date and save actual and predicted values to the database.
    """
    # Load data from SQLite database
    query = "SELECT * FROM final_data"
    with SessionLocal() as session:
        data = pd.read_sql_query(query, session.bind)

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

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)

    # Display evaluation metrics
    print("Evaluation metrics:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    return predictions_df

# Function to calculate MSE
@app.command()
def calculate_mse(predictions_df):
    mse = np.mean((predictions_df["predicted_energy_price"] - predictions_df["actual_energy_price"]) ** 2)
    return mse

# Function to calculate RMSE
@app.command()
def calculate_rmse(predictions_df):
    rmse = np.sqrt(np.mean((predictions_df["predicted_energy_price"] - predictions_df["actual_energy_price"]) ** 2))
    return rmse

# Function to calculate MAE
@app.command()
def calculate_mae(predictions_df):
    mae = np.mean(np.abs(predictions_df["predicted_energy_price"] - predictions_df["actual_energy_price"]))
    return mae


if __name__ == "__main__":
    app()
