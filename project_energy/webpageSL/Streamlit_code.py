from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import seaborn as sns

# Add project directory to system path
sys.path.append("C:\\Users\\User\\Desktop\\MBD\\Term2\\PythonII\\Group_Assignment\\project_energy\\CLI")
from corrmatrix import plot_correlation_matrix
from pricevsfeatures import plot_actual_price_vs_feature
from hist import plot_histogram
from pricemonth import plot_monthprice
from sarima import load_data, train_model, plot_acf_pacf, perform_statistical_tests
from train_model import train_model_ML
from predict import Predict_ML, calculate_mse, calculate_rmse, calculate_mae
from plot_predictions import Plot_predictions_ML

# Disable matplotlib's global pyplot warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Apply styles
st.markdown(
    """
    <style>
    body {
        background-color: rgb(245, 244, 241);
    }
    .reportview-container {
        max-width: 90%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Title with larger font size
    st.markdown("<h1 style='text-align: center;'>Energy Price Project</h1>", unsafe_allow_html=True)


    # Create buttons for each page
    pages = ["ML Models", "Technical Info", "EDA", "Beyond ML Models"]

    cols = st.columns(len(pages))

    # Render each button in a separate column
    for col, page in zip(cols, pages):
        if col.button(page):
            st.session_state.selected_page = page

    if 'selected_page' in st.session_state:
        selected_page = st.session_state.selected_page

        if selected_page == "ML Models":
            st.write("""
            ## ML Models

            Welcome to the ML Models page! Here you can explore various machine learning models and their predictions for energy price 
            forecasting. Choose from RandomForest, XGBoost, LightGBM, and CatBoost.

            After running the model, you can visualize the predicted values alongside the actual values over a specified date range. Additionally,
            we provide evaluation metrics to assess the performance of the selected model.""")

            st.sidebar.write(
                "To begin, select a model from the dropdown menu. Then, choose the initial date for the training data and specify"
                " a random state for reproducibility. Once you've made your selections, simply click the Run Model button to train "
                "the model and generate predictions.")

            ### Model Evaluation
            # Sidebar options
            model_name = st.sidebar.selectbox("Choose a model to be used for training:",
                                              ["RandomForest", "XGBoost", "LightGBM", "CatBoost"])
            initial_date = st.sidebar.date_input("Choose the date up to which data is used for training:",
                                                 value=pd.to_datetime("2017-01-01"))
            random_state = st.sidebar.number_input("Choose a random state for your model:", value=13)

            num_days_selected = st.sidebar.slider("Number of days to predict", min_value=1, max_value=1000, value=365)

            # Button to run the model
            if st.sidebar.button("Run Model"):
                # Run the train_model function with user-selected parameters
                train_model_output = train_model_ML(model_name, initial_date, random_state)

                # Store the train_model_output in a session state variable
                st.session_state.train_model_output = train_model_output

                # Run the predict command
                predictions_df = Predict_ML(initial_date)

                st.title('Plot Predictions for ML Models')

                st.write("Select the starting and ending date for the graph that you would like to see")

                # Layout for date inputs
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=pd.to_datetime("2017-01-01"), format="YYYY-MM-DD")
                with col2:
                    end_date = st.date_input("End Date", value=pd.to_datetime("2018-12-31"), format="YYYY-MM-DD")

                # Plot predictions
                Plot_predictions_ML(start_date, end_date,num_days_selected)

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
                            "The average of the absolute differences between predicted and actual values."
                        ],
                        "Range": [
                            "0 to ∞ (lower is better)",
                            "0 to ∞ (lower is better)",
                            "0 to ∞ (lower is better)"
                        ]
                    }
                    st.table(pd.DataFrame(metrics_data))

            st.sidebar.write(
                "Feel free to experiment with different models, start dates, and random states to see how they impact the predictions. ")


        elif selected_page == "Technical Info":
            st.write("""
            ## Technical Information

            In this section, you can find relevant technical details about the training and evaluation process of our Machine Learning model for predicting energy prices. Below, we present the best parameters of the model, along with the most important features used by the model for making predictions.
            """)

            if 'train_model_output' in st.session_state:
                # Display the train_model_output
                st.write("Filename:", st.session_state.train_model_output['Filename'])

                # Display Best Parameters in a table with centered title
                st.write("## Best Parameters")
                best_params_df = pd.DataFrame.from_dict(st.session_state.train_model_output['Best Parameters'],
                                                        orient='index', columns=['Value'])
                st.dataframe(best_params_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
                    [dict(selector='th', props=[('text-align', 'center')])]))

                # Display MSE
                #st.write("MSE:", st.session_state.train_model_output['MSE'])

                # Display Feature Importances in a vertical bar plot
                st.write("## Feature Importance")
                feature_importances = st.session_state.train_model_output['Feature Importances']
                sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
                feature_names = [item[0] for item in sorted_feature_importances]
                feature_values = [item[1] for item in sorted_feature_importances]

                # Create bar plot
                fig, ax = plt.subplots(figsize=(12, 50))  # Ajustar el tamaño de la figura
                fig.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
                ax.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
                sns.barplot(x=feature_values, y=feature_names, ax=ax, palette="RdYlBu")
                ax.set_xlabel('Importance', fontsize=20, color='white')
                ax.set_ylabel('Feature', fontsize=20, color='white')

                # Ajustar el tamaño de la fuente de los ticks
                ax.tick_params(axis='x', labelsize=25, colors='white')  # Tamaño de fuente de los ticks del eje X
                ax.tick_params(axis='y', labelsize=25, colors='white')  # Tamaño de fuente de los ticks del eje Y

                st.pyplot(fig)

        elif selected_page == "Beyond ML Models":
            st.write("Welcome to the Beyond ML Models page.")

            # Load data
            data = load_data()

            # Sidebar options
            start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2019-01-01"))
            num_days = st.sidebar.slider("Number of days to predict", min_value=1, max_value=100, value=20)

            # Train model and make predictions
            y, y_pred, date_range, sar_model = train_model(data, start_date, num_days)

            # Plot the graph
            st.subheader("Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data['price actual'], label='Actual values')
            ax.plot(date_range, y_pred, label='Predicted values', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Electricity Price')
            ax.legend()
            fig.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
            ax.set_facecolor((0.054901960784313725, 0.06666666666666667, 0.09411764705882353))
            ax.tick_params(axis='x', colors='white')  # X-axis ticks in white color
            ax.tick_params(axis='y', colors='white')  # Y-axis ticks in white color
            st.pyplot(fig)

            # Plot ACF and PACF
            st.subheader("Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)")
            plot_acf_pacf(y)

            # Perform statistical tests
            st.subheader("Statistical Tests")
            perform_statistical_tests(y, sar_model)

        elif selected_page == "EDA":
            st.title("Exploratory Data Analysis (EDA)")

            # Load data
            data = load_data()

            # List of available variables
            variables = ['generation fossil brown coal/lignite', 'generation fossil gas',
                         'generation fossil hard coal', 'generation fossil oil',
                         'generation hydro pumped storage consumption',
                         'generation hydro run-of-river and poundage',
                         'generation hydro water reservoir', 'generation nuclear',
                         'generation other', 'generation other renewable', 'generation solar',
                         'generation waste', 'generation wind onshore',
                         'forecast solar day ahead', 'forecast wind onshore day ahead',
                         'total load forecast', 'total load actual', 'price day ahead',
                         'price actual', 'winter', 'spring', 'summer', 'autumn']

            # Show the list of variables in a selectbox in the sidebar
            selected_variable = st.sidebar.selectbox("Select variable:", variables)

            # List of available frequencies
            frequencies = ['daily', 'weekly', 'monthly', 'yearly']

            # Show the list of frequencies in a selectbox in the sidebar
            frequency_selected = st.sidebar.selectbox("Select frequency:", frequencies)

            plot_actual_price_vs_feature(data, selected_variable, frequency_selected)

            plot_correlation_matrix(data)

            st.subheader("Histogram")
            plot_histogram(data, selected_variable)

            st.subheader("Monthly Price per year")
            plot_monthprice(data)


if __name__ == "__main__":
    main()
