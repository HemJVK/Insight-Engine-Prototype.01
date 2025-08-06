# src/code_executor.py

import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout

# Import plotting libraries so they are available in the execution scope
import plotly.express as px
import matplotlib.pyplot as plt


def execute_code(code: str, df: pd.DataFrame):
    """
    Executes a string of Python code in a restricted environment and captures the output.

    Args:
        code (str): The Python code to execute.
        df (pd.DataFrame): The pandas DataFrame to be available in the execution scope as `df`.

    Returns:
        tuple: A tuple containing:
            - The output of the code (a string, a DataFrame, or a plot object).
            - The potentially modified DataFrame.
    """
    # Create a copy of the DataFrame to avoid modifying the original in case of an error
    df_copy = df.copy()

    # Define the local environment for exec(). df is the main object the code will interact with.
    local_vars = {"df": df_copy, "pd": pd, "np": np, "px": px, "plt": plt}
    global_vars = {}  # Not providing any global variables for security

    # Use a string buffer to capture any print() statements
    buffer = io.StringIO()

    try:
        # Redirect standard output to the buffer
        with redirect_stdout(buffer):
            # Execute the code. The last expression's result is stored in '_'
            exec(code, global_vars, local_vars)

        # The modified DataFrame is now in local_vars['df']
        new_df = local_vars["df"]

        # Get any printed output from the buffer
        output = buffer.getvalue()

        # Check if a figure object was created (e.g., fig = px.histogram(...))
        # This is a robust way to find the plot object, regardless of its variable name.
        fig_object = None
        for var in local_vars.values():
            if isinstance(
                var, (plt.Figure)
            ) or "plotly.graph_objs._figure.Figure" in str(type(var)):
                fig_object = var
                break

        if fig_object is not None:
            return fig_object, new_df

        if output:
            return output, new_df

        # If there was no print output, check the result of the last expression
        last_expr_result = local_vars.get("_")
        if last_expr_result is not None:
            return last_expr_result, new_df

        return "✅ Code executed successfully (no output).", new_df

    except Exception as e:
        # If an error occurs, return the error message and the original DataFrame
        return f"⚠️ Error executing code:\n{e}", df
