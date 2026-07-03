"""P5 (PLAN II): keep the docstring examples in utils.py executable.

Scoped to utils.py on purpose — it is the only module whose docstrings carry
worked numeric examples; widen module by module as their examples are fixed.
"""

import doctest

import effector.utils


def test_utils_doctests():
    result = doctest.testmod(effector.utils, verbose=False)
    assert result.failed == 0, f"{result.failed} doctest(s) failed in effector.utils"
