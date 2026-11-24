"""Test that all example notebooks can be executed without error."""

import glob
import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# Parallelization can cause issues with notebook execution
MAX_RETRIES = 5


def execute_notebook(notebook_path: str, retries: int = MAX_RETRIES) -> str | None:
    """Execute a Jupyter notebook and return any errors."""
    for attempt in range(1, retries + 1):
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
            return None  # Success, no error
        except Exception as e:
            error_message = str(e)
            if "Kernel died" in error_message and attempt < retries:
                continue  # Retry
            return error_message
    return "Maximum retries exceeded"


directory = "./examples/"
ipynb_files = glob.glob(os.path.join(directory, "**", "*.ipynb"), recursive=True)


@pytest.mark.parametrize("notebook_path", ipynb_files)
def test_notebook_execution(notebook_path: str) -> None:
    """Test that the notebook can be executed without error."""
    error = execute_notebook(notebook_path)
    assert error is None, f"Error executing {notebook_path}: {error}"
