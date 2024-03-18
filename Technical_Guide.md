### Welcome to the Technical Guide for Energy Price Prediction Tool

This guide provides step-by-step instructions on how to utilize the energy price prediction tool effectively. Below, you'll find detailed explanations and commands to navigate through the various stages of the tool, from initial setup to data processing and model execution.

## Step 0: Project Setup

Before proceeding with any data processing, several initial setup steps were completed to ensure a smooth workflow:

1. **GitHub Repository Setup:** The project was connected to GitHub using Git, creating a repository to facilitate collaboration and enable others to view and utilize the project.

2. **Virtual Environment Creation:** A specific virtual environment was created for this project to manage dependencies and ensure compatibility across different environments.

3. **Dependency Management:** The project's dependencies were managed using Poetry. This involved creating a `pyproject.toml` file to specify project metadata and dependencies, along with a `poetry.lock` file to lock dependency versions. This setup allows other users to easily replicate the project environment using `poetry install`.

4. **Input Data and Resources:** Input data files, such as CSV datasets and images used in Streamlit, were organized in the `data/raw` directory. This directory structure ensures that all project resources are easily accessible and well-organized.

## Step 1: Data Cleaning
Begin by cleaning the raw data to prepare it for further analysis. Run the `data_cleaning.py` script. This script loads the raw energy and weather datasets, performs data type conversions, handles duplicates, filters data for Madrid, merges datasets, handles missing values, drops irrelevant columns, replaces outliers, scales the variables, and finally loads the cleaned data into an SQLite database.

To execute the data cleaning process, run the following command in your terminal:

```bash
python data_cleaning.py
```

This script ensures that the data is in a suitable format for subsequent analysis and modeling tasks. Once executed, you can proceed to the next step


## Step 2: Database Session Setup
The next step is to set up the database session to utilize the session and access the SQLite database created in `data_cleaning.py`. This session allows for interaction with the database using SQLAlchemy.

```bash
python database_session.py
```

This command initializes the database session, enabling further database operations.

## Step 3: Feature Engineering
Proceed with feature engineering to enhance the dataset for modeling purposes. Run the `feature_engineering.py` script to create new features and downsample the data from hourly to daily frequency.

```bash
python feature_engineering.py
```

This script creates new features such as seasonal indicators and lag features, which provide historical information about energy prices. Additionally, the data is downsampled from hourly to daily frequency to mitigate noise and volatility, making it more suitable for modeling purposes. These preprocessing steps ensure that the dataset has relevant features and formatted appropriately for subsequent analysis and modeling tasks.

**Note:** In the `model` folder, you can find the initial model that was run before incorporating each step into the CLIs. It is a Random Forest model with grid search. However, the final version of the model is included within the CLIs, which we will detail in Step 4.

## Step 4: Command Line Interfaces (CLIs)
Command Line Interfaces (CLIs) are simple text-based tools designed to perform specific tasks in our project, like processing data, training models, and evaluating results. CLIs make it easy to interact with our project by executing tasks with predefined settings. They help us ensure consistency and make it simpler to integrate our models and analyses into other workflows or applications.

These CLIs are all run from our Streamlit code, but we'll provide an explanation of each one so you know what they do.

### 4.1 train_model.py CLI
The `train_model_ML.py` CLI trains tree-based machine learning models (RandomForestRegressor, XGBoost, LightGBM, or CatBoost) using data up to a specified initial date obtained from the SQLite database. 

#### GridSearchCV
It employs GridSearchCV to optimize hyperparameters, exploring a predefined parameter grid. The parameters considered for optimization vary depending on the chosen model:
- RandomForest: `n_estimators` with values [50, 100, 200].
- XGBoost: `n_estimators` with values [50, 100, 200], `learning_rate` with values [0.01, 0.1, 0.2].
- CatBoost: `iterations` with values [50, 100, 200], `learning_rate` with values [0.01, 0.1, 0.2].
- LightGBM: `n_estimators` with values [50, 100, 200], `learning_rate` with values [0.01, 0.1, 0.2].

#### Time Series Split
To account for the temporal nature of the data, a time series split strategy with 5 folds is employed for cross-validation. This ensures that the model is trained on past data and evaluated on future data, preventing data leakage.

#### Model Training and Storage
After training, the best performing model based on MSE is selected and saved to a file named `trained_modelCLI` within the `project_energy/model` directory. 

The CLI provides detailed information on:
- Best parameters found during the grid search.
- Mean squared error (MSE) calculated on the training data.
- Feature importances sorted in descending order.

```bash
python train_model.py --model_name [model_name] --date [initial_date] --random_state [random_state]
```
### 4.2 train_model.py CLI
The `predict.py` CLI enables predicting energy prices as of a specified date and saving the actual and predicted values to the database.

#### Prediction Process
- **Data Retrieval:** Data is fetched from the SQLite database using a SQL query.

- **Data Preparation:** The input date is converted to a datetime object, and the data is filtered to include only records from the input date onwards.

- **Model Loading:** The trained model (saved as 'trained_modelCLI') is loaded from the 'project_energy/model' directory.

- **Feature Selection:** Features are selected for prediction, excluding the target variable ('price actual') and the date column.

- **Prediction:** Predictions are made using the loaded model on the selected features.

- **Evaluation Metrics:** Metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) are calculated to evaluate the performance of the model.

- **Saving Predictions:** Predictions, along with actual values and dates, are saved to a table named 'predictions' in the SQLite database.

#### Evaluation Metrics
- **MSE:** Measures the average squared difference between the actual and predicted values.
- **RMSE:** Represents the square root of MSE, providing a more interpretable measure.
- **MAE:** Indicates the average absolute difference between the actual and predicted values.

The CLI also offers additional commands to calculate MSE, RMSE, and MAE individually, facilitating further analysis of model performance.

### 4.3 plot_predictions.py CLI
The `plot_predictions.py` CLI generates plots of actual and predicted energy prices for a specific date range. It takes the predictions and actual data stored in a SQLite database, filtering predictions within the provided date range and plotting them alongside actual energy prices. Visualization is conducted using Matplotlib and Streamlit, providing a visual representation of the model's performance in predicting energy prices.

Now, moving into the CLI tools for Exploratory Data Analysis (EDA), we'll explore commands tailored to analyze and visualize data trends related to energy pricing.

### 4.4 pricevsfeatures.py CLI
The pricevsfeatures.py CLI generates a plot showing actual energy prices and a selected feature over a specified frequency (daily, weekly, monthly, or yearly). It retrieves data from a SQLite database, groups it based on the chosen frequency, and plots the information using Matplotlib and Streamlit.

### 4.5 hist.py CLI
The `hist.py` CLI generates a histogram plot for a specified feature using data from a SQLite database. It utilizes Matplotlib and Streamlit to visualize the frequency distribution of the selected feature.

### 4.6 corrmatrix.py CLI
The `correlation_matrix.py` CLI generates a correlation matrix plot for the features in the dataset using data from a SQLite database. It utilizes Seaborn and Streamlit to visualize the pairwise correlations between different features.

### 4.7 pricemonth.py CLI
The `plot_monthprice.py` CLI generates a plot showing the mean energy price across different months over the years using data from a SQLite database. It utilizes Matplotlib and Streamlit to visualize the monthly mean energy prices for each year from 2015 to 2018.


### 4.8 sarima.py CLI:

In this section, we delve into the SARIMA (Seasonal Autoregressive Integrated Moving Average) model, offering an alternative to predict energy prices. SARIMA is a time series forecasting model that extends the ARIMA model to account for seasonality in the data. We aim to compare SARIMA predictions with those from our machine learning model.

#### Model Detail:

The SARIMA model is trained using historical energy price data retrieved from a SQLite database. We iteratively define SARIMA parameters, including order and seasonal order, by evaluating various combinations and selecting the model with the lowest Akaike Information Criterion (AIC), a metric for model selection. The SARIMA model is fitted to historical data up to a specified start date, and predictions are made for the desired number of future days.

#### Plots:

- ACF and PACF Plots: Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are generated to analyze autocorrelation and partial autocorrelation of the energy price time series, aiding in determining model parameters.
- Forecast Plot: SARIMA model predictions are visualized alongside actual energy prices, offering insight into the model's performance in forecasting future prices.

## step 5: init and main files
For our CLI structure we have two main documents for running our Python scripts.

- The `__init__.py` file serves as the initialization module for our project. 

- The `__main__.py` file serves as the CLI access point for our project. It enables running the project as a module using the command `python -m PythonGAssignment`. Additionally, the CLI can be accessed directly with the command `PythonGAssignment`.

## step 6: Streamlit
"""
## Streamlit_code.py
The `Streamlit_code.py` file contains the code for building a Streamlit web application to visualize and interact with various aspects of the energy price prediction project. These are the key components of the code:

- **Imports**: The necessary libraries such as Matplotlib, Streamlit, Pandas, Seaborn, and Typer are imported.

- **Initialization**: The application is initialized using Streamlit's `Typer` module.

- **Paths**: File paths for accessing images and modules within the project directory are defined.

- **Model and Visualization Modules**: The necessary modules for model training, prediction, and visualization are imported from the project's CLI directory.

- **Streamlit Layout**: The main function `main()` sets up the layout for the Streamlit application. It includes buttons for different pages, such as Machine Learning Models, Technical Information, SARIMA Model, and Exploratory Data Analysis (EDA).

- **Machine Learning Models Page**: This section allows users to select machine learning models, specify parameters, train the model, and visualize predictions. Evaluation metrics such as MSE, RMSE, and MAE are displayed.

- **Technical Information Page**: Provides details about the best parameters, feature importance, and model evaluation metrics of the trained machine learning model.

- **SARIMA Model Page**: Allows users to interact with the SARIMA model by selecting the start date and the number of days for prediction. It includes plots for actual vs. predicted values, ACF and PACF plots, and statistical tests.

- **Exploratory Data Analysis Page**: Enables exploration of the dataset through visualizations such as actual price vs. selected feature, histograms, correlation matrix, and monthly price per year.

- **Execution**: The `main()` function is executed when the script is run.

This file serves as the backbone of the Streamlit application, providing users with a user-friendly interface to explore and analyze energy price data and machine learning models.

For detailed instructions on using the Streamlit interface and interpreting visualizations, please refer to the `Streamlit_Guide.md` guide.

