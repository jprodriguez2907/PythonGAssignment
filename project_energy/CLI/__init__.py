"""
**CLI access point**.

With this module we enable the ``python -m PythonGAssignment`` functionality

The CLI should also be accessible through the command: ``PythonGAssignment``.
"""

from typer import Typer

from project_energy.CLI.train_model import app as train_model_ML
#from project_energy.CLI.predict import app as predict
#from project_energy.CLI.plot_predictions import app as plot_predictions

app = Typer()

# Adding the imported Typer apps to the main app
app.add_typer(train_model_ML, name="train_model_ML")
#app.add_typer(predict, name="predict")
#app.add_typer(plot_predictions, name="plot_predictions")
