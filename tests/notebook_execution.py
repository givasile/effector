import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.parametrize("notebook_filename", [
    os.path.join("./../notebooks/synthetic-examples", file)
    for file in os.listdir("./../notebooks/synthetic-examples") if file.endswith(".ipynb") and not file.startswith("06_efficiency")
])



def test_notebook_execution(notebook_filename):
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    executor = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        executor.preprocess(nb, {'metadata': {'path': './../notebooks/synthetic-examples'}})
    except Exception as e:
        pytest.fail(f"Notebook {notebook_filename} failed to execute. Error: {e}")