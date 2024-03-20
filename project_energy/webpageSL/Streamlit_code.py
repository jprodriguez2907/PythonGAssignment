from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import seaborn as sns
import os
from typer import Typer

st.set_page_config(layout="centered")

app = Typer()

# Create paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
CLI_path = os.path.join(grandparent_dir, "project_energy", "CLI")
image1_path = os.path.join(grandparent_dir, "data", "raw", "Energy.png")
image2_path = os.path.join(grandparent_dir, "data", "raw", "Energy2.jpg")
sys.path.append(CLI_path)
from corrmatrix import plot_correlation_matrix
from pricevsfeatures import plot_actual_price_vs_feature
from hist import plot_histogram
from pricemonth import plot_monthprice
from sarima import (
    visualize_forecast,
    train_model,
    plot_acf_pacf,
    perform_statistical_tests,
)
from train_model import train_model_ML
from predict import predict_ML, calculate_mse, calculate_rmse, calculate_mae
from plot_predictions import plot_predictions_ML

st.set_option("deprecation.showPyplotGlobalUse", False)


def main():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("   ")

    with col2:
        st.image(image1_path, width=230, use_column_width=False)

    with col3:
        st.write(" ")

    # Title with larger font size
    st.markdown(
        "<h1 style='text-align: center;'>Energy Price Project</h1>",
        unsafe_allow_html=True,
    )

    # Create buttons for each page
    pages = [
        "Machine Learning Models",
        "Technical Information",
        "Exploratory Data Analysis",
        "Beyond Machine Learning Models",
    ]

    cols = st.columns(len(pages))

    # Render each button in a separate column
    for col, page in zip(cols, pages):
        if col.button(page):
            st.session_state.selected_page = page

    if "selected_page" not in st.session_state:
        st.image(image2_path, use_column_width=True)

    if "selected_page" in st.session_state:
        selected_page = st.session_state.selected_page

        if selected_page == "Machine Learning Models":
            st.title("Machine Learning Models")
            st.write("""
            Welcome to the ML Models page! Here you can explore various machine learning models and their predictions for energy price 
            forecasting. Choose from RandomForest, XGBoost, LightGBM, and CatBoost.

            After running the model, you can visualize the predicted values alongside the actual values over a specified date range. Additionally,
            we provide evaluation metrics to assess the performance of the selected model.""")

            st.sidebar.write(
                "To begin, select a model from the dropdown menu. Then, choose the initial date for the training data and specify"
                " a random state for reproducibility. Once you've made your selections, simply click the Run Model button to train "
                "the model and generate predictions."
            )

            ### Model Evaluation
            # Sidebar options
            model_name = st.sidebar.selectbox(
                "Choose a model to be used for training:",
                ["RandomForest", "XGBoost", "LightGBM", "CatBoost"],
            )
            initial_date = st.sidebar.date_input(
                "Choose the date up to which data is used for training:",
                value=pd.to_datetime("2017-01-01"),
            )
            random_state = st.sidebar.number_input(
                "Choose a random state for your model:", value=13
            )

            num_days_selected = st.sidebar.slider(
                "Number of days to predict", min_value=1, max_value=1000, value=365
            )

            # Layout for date inputs
            st.write(
                "Select the starting and ending date for the graph that you would like to see"
            )
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=pd.to_datetime("2017-01-01"),
                    format="YYYY-MM-DD",
                )
            with col2:
                end_date = st.date_input(
                    "End Date", value=pd.to_datetime("2018-12-31"), format="YYYY-MM-DD"
                )

            # Button to run the model
            if st.sidebar.button("Run Model"):
                # Run the train_model function with user-selected parameters
                train_model_output = train_model_ML(
                    model_name, initial_date, random_state
                )

                # Store the train_model_output in a session state variable
                st.session_state.train_model_output = train_model_output

                # Run the predict command
                predictions_df = predict_ML(initial_date)

                st.title("Plot Predictions for ML Models")

                # Plot predictions
                plot_predictions_ML(start_date, end_date, num_days_selected)

                # Display the metrics
                if predictions_df is not None:
                    mse = calculate_mse(predictions_df)
                    rmse = calculate_rmse(predictions_df)
                    mae = calculate_mae(predictions_df)

                    st.subheader("**Model Evaluation Metrics**")
                    metrics_data = {
                        "Metric": ["MSE", "RMSE", "MAE"],
                        "Value": [mse, rmse, mae],
                        "Description": [
                            "Measures the average squared difference between the predicted values and the actual values.",
                            "The square root of the average of squared differences between predicted and actual values.",
                            "The average of the absolute differences between predicted and actual values.",
                        ],
                        "Range": [
                            "0 to ∞ (lower is better)",
                            "0 to ∞ (lower is better)",
                            "0 to ∞ (lower is better)",
                        ],
                    }
                    st.table(pd.DataFrame(metrics_data))

            st.sidebar.write(
                "Feel free to experiment with different models, start dates, and random states to see how they impact the predictions. "
            )

        elif selected_page == "Technical Information":
            st.title("Technical Information")
            st.write("""
            In this section, you can find relevant technical details about the training and evaluation process of our Machine Learning model for predicting energy prices. Below, we present the best parameters of the model, along with the most important features used by the model for making predictions.
            """)

            if "train_model_output" in st.session_state:
                # Display the train_model_output
                st.write("Filename:", st.session_state.train_model_output["Filename"])

                # Display Best Parameters in a table with centered title
                st.write("## Best Parameters")
                best_params_df = pd.DataFrame.from_dict(
                    st.session_state.train_model_output["Best Parameters"],
                    orient="index",
                    columns=["Value"],
                )
                st.dataframe(
                    best_params_df.style.set_properties(
                        **{"text-align": "center"}
                    ).set_table_styles(
                        [dict(selector="th", props=[("text-align", "center")])]
                    )
                )

                # Display Feature Importances in a vertical bar plot
                st.write("## Feature Importance")
                feature_importances = st.session_state.train_model_output[
                    "Feature Importances"
                ]
                sorted_feature_importances = sorted(
                    feature_importances.items(), key=lambda x: x[1], reverse=True
                )
                feature_names = [item[0] for item in sorted_feature_importances]
                feature_values = [item[1] for item in sorted_feature_importances]

                # Create bar plot
                fig, ax = plt.subplots(figsize=(12, 50))
                fig.set_facecolor(
                    (0.9607843137254902, 0.9568627450980393, 0.9450980392156862)
                )
                ax.set_facecolor(
                    (0.9607843137254902, 0.9568627450980393, 0.9450980392156862)
                )

                sns.barplot(x=feature_values, y=feature_names, ax=ax, palette="crest_r")
                ax.set_xlabel("Importance", fontsize=20, color="#1c0858")
                ax.set_ylabel("Feature", fontsize=20, color="#1c0858")

                ax.tick_params(axis="x", labelsize=25, colors="#1c0858")
                ax.tick_params(axis="y", labelsize=25, colors="#1c0858")

                st.pyplot(fig)

        elif selected_page == "Beyond Machine Learning Models":
            st.title("SARIMA Model")

            st.sidebar.write(
                "To start,select the date from which you want to start making predictions, and then specify the number of days you would like to forecast into the future."
            )

            # Sidebar options
            start_date = st.sidebar.date_input(
                "Start date", value=pd.to_datetime("2019-01-01")
            )
            num_days = st.sidebar.slider(
                "Number of days to predict", min_value=1, max_value=1000, value=50
            )

            y, y_pred, date_range, sar_model = train_model(start_date, num_days)

            # Plot the graph
            st.subheader("Actual vs Predicted Values")
            st.write(
                "Compares actual electricity prices with the predicted values using the SARIMA model depending of the parameters"
            )
            start_date = pd.Timestamp(start_date)
            visualize_forecast(start_date, num_days)

            # Plot ACF and PACF
            st.subheader(
                "Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)"
            )
            st.write(
                "Explore the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to analyze the time dependencies and lags in the data."
            )

            plot_acf_pacf(y)

            # Perform statistical tests
            st.subheader("Statistical Tests")

            st.write(
                "We conduct three key statistical tests to evaluate our SARIMA model:"
            )

            st.write(
                "**1. Augmented Dickey-Fuller Test (ADF):** Checks for stationarity in time series data. A significance level of 0.5 indicates non-stationarity above and stationarity below."
            )

            st.write(
                "**2. Ljung-Box Test:** Assesses autocorrelation in residuals. Significance below 0.5 suggests autocorrelation present."
            )

            st.write(
                "**3. Shapiro-Wilk Test:** Tests normality of residuals. A p-value under 0.5 signifies deviation from normality."
            )

            perform_statistical_tests(y, sar_model)

        elif selected_page == "Exploratory Data Analysis":
            st.title("Exploratory Data Analysis (EDA)")

            st.write(
                "Explore our Energy and Climate EDA! Dive into our dataset spanning 2015-2019 for insights on energy generation, consumption, and climate. Uncover trends and patterns to inform decision-making. Let's explore together!"
            )

            # List of available variables
            variables = [
                "temp",
                "temp_min",
                "temp_max",
                "pressure",
                "humidity",
                "wind_speed",
                "wind_deg",
                "rain_1h",
                "rain_3h",
                "snow_3h",
                "clouds_all",
                "weather_id",
                "generation biomass",
                "generation fossil brown coal/lignite",
                "generation fossil gas",
                "generation fossil hard coal",
                "generation fossil oil",
                "generation hydro pumped storage consumption",
                "generation hydro run-of-river and poundage",
                "generation hydro water reservoir",
                "generation nuclear",
                "generation other",
                "generation other renewable",
                "generation solar",
                "generation waste",
                "generation wind onshore",
                "forecast solar day ahead",
                "forecast wind onshore day ahead",
                "total load forecast",
                "total load actual",
                "price day ahead",
                "price actual",
                "winter",
                "spring",
                "summer",
                "autumn",
            ]

            st.sidebar.write(
                "Please select a variable and the desired frequency to explore insights from our dataset"
            )
            # Show the list of variables in a selectbox in the sidebar
            selected_variable = st.sidebar.selectbox("Select variable:", variables)

            # List of available frequencies
            frequencies = ["daily", "weekly", "monthly", "yearly"]

            # Show the list of frequencies in a selectbox in the sidebar
            frequency_selected = st.sidebar.selectbox("Select frequency:", frequencies)

            st.sidebar.write(
                "Note: This selection will change the actual price vs feature and the histogram graph"
            )
            st.subheader(f"Actual Price vs {selected_variable} by date")
            st.write(
                "Track energy prices alongside any feature over different time spans - daily, weekly, monthly, or yearly"
            )
            plot_actual_price_vs_feature(selected_variable, frequency_selected)

            st.subheader(f"Histogram of {selected_variable}")
            st.write(
                "Take a deeper dive into the distribution and frequency analysis of your chosen feature by exploring it visually through a histogram "
            )
            plot_histogram(selected_variable)

            st.subheader("Correlation Matrix")
            st.write(
                "Delve into the correlation matrix heatmap to uncover connections among features"
            )
            plot_correlation_matrix()

            st.subheader("Monthly Price per year")
            st.write(
                "Visualize the mean energy price by month over the years, gaining insights into seasonal fluctuations"
            )
            plot_monthprice()


if __name__ == "__main__":
    main()
