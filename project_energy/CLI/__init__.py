"""
**CLI access point**.

With this module we enable the ``python -m PythonGAssignment`` functionality

The CLI should also be accessible through the command: ``PythonGAssignment``.
"""

from typer import Typer

from project_energy.CLI.train_model import app as train_model_ML
from project_energy.CLI.corrmatrix import app as plot_correlation_matrix
from project_energy.CLI.pricevsfeatures import plot_actual_price_vs_feature
from project_energy.CLI.hist import plot_histogram
from project_energy.CLI.pricemonth import plot_monthprice
from project_energy.CLI.sarima import visualize_forecast
from project_energy.CLI.sarima import train_model
from project_energy.CLI.sarima import plot_acf_pacf
from project_energy.CLI.sarima import perform_statistical_tests
from project_energy.CLI.train_model import app
from project_energy.CLI.predict import predict_ML
from project_energy.CLI.predict import calculate_mse
from project_energy.CLI.predict import calculate_rmse
from project_energy.CLI.predict import calculate_mae
from project_energy.CLI.plot_predictions import plot_predictions_ML

app = Typer()

# Adding the imported Typer apps to the main app
app.add_typer(train_model_ML, name="train_model_ML")
app.add_typer(plot_correlation_matrix, name="plot_correlation_matrix")
app.add_typer(plot_actual_price_vs_feature, name="plot_actual_price_vs_feature")
app.add_typer(plot_histogram, name="plot_histogram")
app.add_typer(plot_monthprice, name="plot_monthprice")
app.add_typer(visualize_forecast, name="visualize_forecast")
app.add_typer(train_model, name="train_model")
app.add_typer(plot_acf_pacf, name="plot_acf_pacf")
app.add_typer(perform_statistical_tests, name="perform_statistical_tests")
app.add_typer(predict_ML, name="predict_ML")
app.add_typer(calculate_mse, name="calculate_mse")
app.add_typer(calculate_rmse, name="calculate_rmse")
app.add_typer(calculate_mae, name="calculate_mae")
app.add_typer(plot_predictions_ML, name="plot_predictions_ML")

if __name__ == "__main__":
    app()
