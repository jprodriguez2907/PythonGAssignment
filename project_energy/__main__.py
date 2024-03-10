"""**CLI access point**.

With this module we enable the ``python -m PythonGAssignment`` functionality.

The CLI should also be accessible through the command: ``PythonGAssignment``.
"""


from project_energy import app

if __name__ == "__main__":
    app()