"""
**CLI access point**.

With this module we enable the ``python -m PythonGAssignment`` functionality

The CLI should also be accessible through the command: ``PythonGAssignment``.
"""

from typer import Typer

from project_energy import app as train_model
from project_energy import app as predict
from project_energy import app as plot_predictions

app = Typer()

# Adding the imported Typer apps to the main app
app.add_typer(train_model, name="train_model")
app.add_typer(predict, name="predict")
app.add_typer(plot_predictions, name="plot_predictions")
