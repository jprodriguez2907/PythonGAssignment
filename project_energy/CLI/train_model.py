# Import necessary libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
#from database_session import SessionLocal
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sqlalchemy.sql import text
from typer import Typer, Option
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import joblib

# Initialize Typer app
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
def train_model(model_name: str = Option(..., "--model_name", "-m",help="Choose a model to be used for training from RandomForest, XGBoost, LightGBM or CatBoost"),
                         initial_date: str = Option(..., "--date", "-d",help="Date up to which data is used for training (YYYY.MM.DD)"),
                         random_state: int = Option(42, "--random_state", "-r",help="Choose a random state for your model")):
    """
    Trains a Tree Based Machine Learning Model (RandomForestRegressor, XGBoost,LGBM or CatBoost) with data up to a given initial date and saves the trained model to a file.
    """
    # Load data from SQLite database
    query = text("SELECT * FROM final_data")
    with SessionLocal() as session:
        data = pd.DataFrame(session.execute(query).fetchall())
        data.columns = [column[0] for column in session.execute(query).cursor.description]

    # Turn input date into datetime
    initial_date = pd.to_datetime(initial_date)
    data["date"] = pd.to_datetime(data["date"])


    # Filter data up to the initial date for training
    train = data[data["date"] < initial_date]

    # Define features and target
    X_train = train.drop(columns=["price actual", "date"])
    y_train = train["price actual"]


    # Initialize and train the corresponding model
    if model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {'n_estimators': [50, 100, 200]}
    elif model_name == 'XGBoost':
        model = XGBRegressor(random_state=random_state)
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(random_state=random_state)
        param_grid = {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == 'LightGBM':
        model = LGBMRegressor(random_state=random_state)
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    else:
        raise ValueError("Model name must be one of 'RandomForest', 'XGBoost', 'CatBoost', or 'LightGBM'")

    # Define time series cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv)

    # Train GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get best parameters
    best_params = grid_search.best_params_

    # Get MSE
    y_pred = grid_search.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)

    # Get feature importances
    feature_importances = grid_search.best_estimator_.feature_importances_

    # Get column names
    column_names = X_train.columns

    # Combine column names with feature importances
    feature_importance_dict = {column_names[i]: feature_importances[i] for i in range(len(column_names))}

    # Sort feature importances dictionary by importance values in descending order
    sorted_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    # Save the trained model to a file
    joblib.dump(grid_search.best_estimator_, "../model/trained_modelCLI")
    print("Model saved")

    return {
        'Filename': "trained_modelCLI",
        'Best Parameters': best_params,
        'MSE': mse,
        'Feature Importances': sorted_importance_dict
    }


if __name__ == "__main__":
    app()