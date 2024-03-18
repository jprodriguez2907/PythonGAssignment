### Welcome to the Streamlit Guide for Energy Price Prediction Tool
Here, we will show you how to walk around our Streamlit interface in order to use the energy price prediction tool

## Menu Options
Upon arrival, you'll encounter a menu offering different pathways: ML Models, Technical Info, EDA, or Beyond. Choose the page that you want to see.

## Machine Learning Models
### Overview
We can start with the machine learning models. You can choose from different options to run your model like RandomForest, XGBoost, LightGBM, and CatBoost.
### Getting Started
At first, pick a model from the dropdown menu. Next, specify a training start date. Then, press the "Run Model" button to initiate the training process and witness the model in action.

### Evaluation Metrics
Following model execution, we provide insightful metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). These metrics offer a comprehensive assessment of model performance.

## Technical Information
### Overview
See the technical information about the project you just ran. We'll explore parameters, features, and underlying methodologies in detail.

### Best Parameters
Uncover the optimal parameters derived from our model training process. These parameters are the ones selected by a gridsearch model that had the best performance 

### Feature Importance
Discover the significance of individual features through insightful visualizations. Understand which variables had the greatest influence in our predictive models.

## Beyond Machine Learning Models
### SARIMA Model
For comparison, we decide to run a SARIMA model, a powerful tool for forecasting time series with a seasonality trend.

### Getting Started
Select a star date and specify the forecast horizon. Witness the SARIMA model's predictive capabilities firsthand.

### Graphical Analysis
Visualize actual versus predicted energy prices through compelling graphical representations. Additionally, delve into statistical tests to validate model performance.

## Exploratory Data Analysis (EDA)
### Overview
Our dataset spanning 2015-2019. Uncover trends, patterns, and insights regarding energy generation, consumption, and climate.

### Getting Started
Select a variable of interest and choose a desired frequency for analysis.

### Graphical Analysis
Navigate through histograms, correlation matrices, and other graphical representations. Gain deeper insights into the underlying dynamics of our dataset.

