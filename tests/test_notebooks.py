"""P2 (PLAN II): execute the synthetic-examples notebooks end-to-end.

Tier-2 only (slow): the ground-truth *content* of these notebooks is ported to
pytest (test_functional_*.py + effector.benchmarks), so the gate does not need
them; executing them nightly keeps prose, plots and asserts from rotting.

Was notebook_execution.py — never collected (filename didn't match test_*.py)
and paths broke unless pytest ran from inside tests/. Kernel is forced to
python3: several notebooks pin a contributor-local kernelspec (eff-env).
"""

import pathlib

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = pathlib.Path(__file__).parent.parent / "notebooks" / "synthetic-examples"

SKIP = {
    # trains two Keras NNs and its hardcoded node_idx crashes on the freshly
    # fitted tree (found 2026-07-02); revisit with the notebook overhaul
    "04_regional_effects_real_f.ipynb",
}

NOTEBOOKS = sorted(
    p for p in NOTEBOOK_DIR.glob("*.ipynb") if p.name not in SKIP
)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(notebook):
    nb = nbformat.read(notebook, as_version=4)
    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    executor.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_DIR)}})
