# Overview
Here is the link for the github repository: 
https://github.com/jprodriguez2907/PythonGAssignment

## Project goal

Accurate energy price predictions are valuable for businesses, policymakers and consumers. The goal of this proejct is to analyze trends in energy generation and to forecast energy prices, using different machine learning and statistical models. Additionally, this project provides a graphical user interface in a web app where users can view the results in an interactive way.

## Acknowledgements

This data is publicly available via ENTSOE and REE and may be found in the following links:
- [ENTSOE](https://transparency.entsoe.eu/)
- [REE Market and Prices](https://www.esios.ree.es/en/market-and-prices) 
- [OpenWeatherMap API](https://openweathermap.org/api)

## Authors
- Juan Pablo Rodriguez
- Thomas Russell
- Arnaud Babey
- Andres Ferrer
- Antonia Murad
- Carlotta Hanten


## Datasets Description

### Dataset - energy

This dataset contains four years of electrical consumption, generation and pricing in Spain.

- **Hourly data, from 01. Jan 2015 to 31 Dec 2018**
- **Target variable: price**
- **Columns explanations:**
  - Time: Datetime index localized to CET 
  - Generation biomass: biomass generation in MW 
  - Generation fossil brown coal/lignite: coal/lignite generation in MW 
  - Generation fossil coal-derived gas: coal gas generation in MW 
  - Generation fossil gas: gas generation in MW 
  - Generation fossil hard coal: coal generation in MW 
  - Generation fossil oil: oil generation in MW 
  - Generation fossil oil shale: shale oil generation in MW 
  - Generation fossil peat: peat generation in MW 
  - Generation geothermal: geothermal generation in MW 
  - Generation hydro pumped storage aggregated: hydro1 generation in MW 
  - Generation hydro pumped storage consumption: hydro2 generation in MW 
  - Generation hydro run-of-river and poundage: hydro3 generation in MW 
  - Generation hydro water reservoir: hydro4 generation in MW 
  - Generation marine: sea generation in MW 
  - Generation nuclear: nuclear generation in MW 
  - Generation other: other generation in MW 
  - Generation other renewable: other renewable generation in MW 
  - Generation solar: solar generation in MW 
  - Generation waste: waste generation in MW 
  - Generation wind offshore: wind offshore generation in MW 
  - Generation wind onshore: wind onshore generation in MW 
  - Forecast solar day ahead: forecasted solar generation 
  - Forecast wind offshore eday ahead: forecasted offshore wind generation 
  - Forecast wind onshore day ahead: forecasted onshore wind generation 
  - Total load forecast: forecasted electrical demand 
  - Total load actual: actual electrical demand 
  - Price day ahead: forecasted price EUR/MWh 
  - Price actual: price in EUR/MWh 

### Dataset - weather 

This dataset contains four years of weather data for five different cities in Spain.

- **Hourly data, from 01. Jan 2015 to 31 Dec 2018**
- **Target variable: price**
- **Columns explanations:**
  - dt_iso: datetime index localized to CET
  - text_formatcity_name: name of city
  - temp: in k
  - temp_min: minimum in k
  - temp_max: maximum in k
  - pressure: pressure in hPa
  - humidity: humidity in %
  - wind_speed: wind speed in m/s
  - wind_deg: wind direction
  - rain_1h: rain in last hour in mm
  - rain_3h: rain last 3 hours in mm
  - snow_3h: show last 3 hours in mm
  - clouds_all: cloud cover in %
  - vpn_keyweather_id: Code used to describe weather
  - text_formatweather_main: Short description of current weather
  - text_formatweather_description: Long description of current weather
  - text_formatweather_icon: Weather icon code for website


# Data preprocessing and models 


## Overall pipeline
The final input for our models is created in two steps: data cleaning and feature selection & engineering. Both the output after data cleaning and the final data input are saved in an SQLite database. After training the chosen model, it is saved to a file within the project, and then used to make predictions and to plot these.

## Data cleaning
Four key steps were taken to clean the data: Fill missing values for columns with <10% missing data with the interpolate method. Clear outliers (>2.58 standard deviations away from the mean) are replaced with the column's mean. Features are standard scaled, and the weather data is reduced from five cities to one city due to high correlation of the data.

## Feature selection & engineering
Columns with either all values equal to zero or equal to the same value are dropped. Four columns (boolean) are created for each season. Lags are created for the past 12 days, and data is downsampled to daily data to avoid noise and overfitting.


## Machine learning models
The user chooses the model to be run, a random state for reproducibility and a date up to which data is used for training the models. 
The following machine learning models are run:
Hyperparameters of the model are tuned through GridSearch.

## Statistical models
In addition to machine learning models, the Seasonal AutoRegressive Integrated Moving Average with eXogenous variables (SARIMAX). The goal is to compare its performance since it is usually suitable for forecasting exercises with seasonal trends.


# Project structure

Key documents for the project are included in the main directory. Besides, there are two key directories included: data and project_energy.


- **Files included in main directory of project:**
  - README: This file, with the goal to give an overview of the project
  - gitignore: Files and directories to be ingored when committing code through Git
  - poetry.lock: Overview of libraries for the project, incl. exact versions
  - pyproject.toml: configuration of the project


- **Files included in directory "data":**
  - raw: raw CSV files as downloaded from the internet
  - processed: SQLite database with preprocessed data


- **Files included in directory "project_energy":**
  - preprocessing: one file for data cleaning steps, one file for feature selection & engineering steps
  - CLI: separate files with the individual CLIs, init file combining CLIs into one app
  - main.py file that is executed when running CLIs
  - model: trained machine learning saved through joblib
  - webpage: code for web app, using streamlit

  
# Guide to install the project dependencies

1. Install poetry by running the following command: pip install poetry
2. Using the terminal, navigate to the directory where the poetry.lock file is located
3. Install dependencies by running the following command from the terminal: poetry install
4. Verify dependencies were successfully installed by running the following command:  poetry show