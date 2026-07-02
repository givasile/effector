"""F9 (LOGBOOK #8): binning exactness — RHALE vs closed-form ALE at atol 1e-2.

f(x) = 3 + 2 x1 - 4 x2 on U(0, 1)^2. The linear model is load-bearing here:
a linear effect is exactly representable under ANY bin partition, so the
binning-induced error is exactly zero and a tolerance ten times tighter than
the rest of the functional layer is legitimate. What this locks is the
axis-partitioning -> bin-effects -> accumulation -> centering pipeline for all
three binning strategies — the only functional coverage of DynamicProgramming
and Greedy binning.

Replaces test_functional.py::TestExample2 (whose docstring described a
different model, whose seed leaked at import time, and which used N=100k
where 10k gives the same tolerance).
"""

import numpy as np
import pytest

import effector

N = 10_000
XS = np.linspace(0, 1, 1000)
ATOL = 1e-2

# zero-integral-centered closed-form ALE of the linear model
GT = {0: 2 * XS - 1, 1: -4 * XS + 2}


def model(x):
    return 3 + 2 * x[:, 0] - 4 * x[:, 1]


def model_jac(x):
    return np.stack([np.full(x.shape[0], 2.0), np.full(x.shape[0], -4.0)], axis=-1)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 1, size=(N, 2))
    x[0], x[-1] = [0.0, 0.0], [1.0, 1.0]  # pin the axis range
    return x


BINNING = [
    pytest.param(
        effector.axis_partitioning.Fixed(nof_bins=100, min_points_per_bin=0),
        id="fixed-100",
    ),
    pytest.param(
        effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=20, min_points_per_bin=10, cat_limit=1
        ),
        id="dp-20",
    ),
    pytest.param(
        effector.axis_partitioning.Greedy(
            init_nof_bins=100, min_points_per_bin=10, discount=0.2, cat_limit=1
        ),
        id="greedy-100",
    ),
]


@pytest.mark.parametrize("binning", BINNING)
@pytest.mark.parametrize("feature", [0, 1])
def test_rhale_exact_for_linear_model(data, binning, feature):
    rhale = effector.RHALE(data=data, model=model, model_jac=model_jac)
    rhale.fit(features=feature, binning_method=binning, centering=True)
    pred = rhale.eval(feature=feature, xs=XS, centering=True)
    np.testing.assert_allclose(pred, GT[feature], atol=ATOL)
