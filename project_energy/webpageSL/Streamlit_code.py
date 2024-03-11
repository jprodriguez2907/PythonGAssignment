from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
st.set_option('deprecation.showPyplotGlobalUse', False)

sys.path.append("C:\\Users\\User\\Desktop\\MBD\\Term2\\PythonII\\Group_Assignment\\project_energy\\CLI")
from EDA import  newfun,plot_correlation_matrix,plot_actual_price_vs_feature
from sarima import load_data, train_model, plot_acf_pacf, perform_statistical_tests
from train_model import Train_model_ML
from predict import Predict_ML, calculate_mse, calculate_rmse, calculate_mae
from plot_predictions import Plot_predictions_ML
sys.path.append("C:\\Users\\User\\Desktop\\MBD\\Term2\\PythonII\\Group_Assignment\\project_energy\\CLI")




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
            st.write("Welcome to the ML Models page.")

            # Sidebar options
            model_name = st.sidebar.selectbox("Choose a model to be used for training:",
                                              ["RandomForest", "XGBoost", "LightGBM", "CatBoost"])
            initial_date = st.sidebar.date_input("Choose the initial date for training data:",
                                                 value=pd.to_datetime("2017-01-01"))
            random_state = st.sidebar.number_input("Choose a random state for your model:", value=13)

            # Button to run the model
            if st.sidebar.button("Run Model"):
                # Run the train_model function with user-selected parameters
                train_model_output = Train_model_ML(model_name, initial_date, random_state)

                # Store the train_model_output in a session state variable
                st.session_state.train_model_output = train_model_output

                # Run the predict command
                predictions_df = Predict_ML(initial_date)

                # Layout for date inputs
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=pd.to_datetime("2017-01-01"), format="YYYY-MM-DD")
                with col2:
                    end_date = st.date_input("End Date", value=pd.to_datetime("2018-12-31"), format="YYYY-MM-DD")

                #Plot predictions
                Plot_predictions_ML(start_date, end_date)

                # Display the metrics
                if predictions_df is not None:
                    mse = calculate_mse(predictions_df)
                    rmse = calculate_rmse(predictions_df)
                    mae = calculate_mae(predictions_df)

                    st.subheader("Model Evaluation Metrics")
                    st.write(f"MSE: {mse}")
                    st.write(f"RMSE: {rmse}")
                    st.write(f"MAE: {mae}")

        elif selected_page == "Technical Info":
            st.write("Welcome to the Technical Info page.")
            # Add content for the Technical Info page here

            # Check if train_model_output exists in session state
            if 'train_model_output' in st.session_state:
                # Display the train_model_output
                st.write("Filename:", st.session_state.train_model_output['Filename'])
                st.write("Best Parameters:", st.session_state.train_model_output['Best Parameters'])
                st.write("MSE:", st.session_state.train_model_output['MSE'])
                st.write("Feature Importances:", st.session_state.train_model_output['Feature Importances'])


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
            frequencies = ['daily', 'weekly','monthly', 'yearly']

            # Show the list of frequencies in a selectbox in the sidebar
            frequency_selected = st.sidebar.selectbox("Select frequency:", frequencies)

            newfun()

            plot_actual_price_vs_feature(data, selected_variable, frequency_selected)

            st.subheader("Correlation Matrix")



if __name__ == "__main__":
    main()

