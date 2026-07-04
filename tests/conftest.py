"""Shared fixtures and the method registry for the contract layer (PLAN II §3.1).

This file is the single place that knows how to construct each of the 11 public
effect classes on the shared tiny datasets (R5 in test form: one table, read by
every contract file).  Test files import the registry via ``tests.conftest``
(tests/ is a package, so pytest and the explicit import resolve to the same
module).

Global methods run on a 3-feature linear model (N=200): the cheapest model
where every method's mean effect is well-defined and nontrivial.  ShapDP gets
*analytic* interventional SHAP values passed to the constructor, so no shap
call is ever made in the contract layer (PLAN II §5 speed rules).

Regional methods run on the gated-linear model from the functional anchor
(``5*x0*1{x1>0, x2==0}``, N=500) with one obvious split, except RegionalShapDP
which runs on N=50 / budget=128 to keep the gate fast.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector

# ---------------------------------------------------------------------------
# autouse: never leak figures between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# global-effect registry
# ---------------------------------------------------------------------------

N_GLOBAL = 200
D_GLOBAL = 3
COEF = np.array([2.0, -3.0, 0.5])

GLOBAL_NAMES = ["pdp", "derpdp", "ale", "rhale", "shapdp"]
REGIONAL_NAMES = [
    "regional_pdp",
    "regional_derpdp",
    "regional_ale",
    "regional_rhale",
    "regional_shapdp",
]


def linear_model(x):
    return x @ COEF


def linear_model_jac(x):
    return np.repeat(COEF[np.newaxis, :], x.shape[0], axis=0)


def make_global_data(n=N_GLOBAL, seed=21):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, D_GLOBAL))


def analytic_shap_values(data):
    """Interventional SHAP values of a linear model with independent features:
    phi_j(x) = a_j * (x_j - mean(x_j)).  Passing them to the ShapDP constructor
    makes the contract layer SHAP-free (0 s of shap computation)."""
    return (data - data.mean(axis=0)) * COEF


def eval_mean(m, feature, xs, centering=False):
    """The mean effect at xs, robust to B11 (ShapDP's tuple-returning eval).

    Contract tests whose substance is *not* the return arity go through this
    helper so they keep testing their own rule today; the arity itself is
    pinned once, by the B11-xfailed C1 test.  Once eval has one return type
    the tuple branch is dead code and this is a passthrough.
    """
    y = m.eval(feature, xs, centering=centering)
    return y[0] if isinstance(y, tuple) else y


def make_global(name, data, **kwargs):
    """Construct a fresh global-effect object of the given kind.

    Everything is passed by keyword; jacobians are analytic so no method takes
    the numerical-jacobian path (B9 territory) inside the contract layer.
    """
    if name == "pdp":
        return effector.PDP(data, linear_model, **kwargs)
    if name == "derpdp":
        return effector.DerPDP(data, linear_model, model_jac=linear_model_jac, **kwargs)
    if name == "ale":
        return effector.ALE(data, linear_model, **kwargs)
    if name == "rhale":
        return effector.RHALE(data, linear_model, model_jac=linear_model_jac, **kwargs)
    if name == "shapdp":
        kwargs.setdefault("shap_values", analytic_shap_values(data))
        return effector.ShapDP(data, linear_model, **kwargs)
    raise ValueError(f"unknown global method: {name}")


@pytest.fixture(scope="session")
def global_data():
    return make_global_data()


# ---------------------------------------------------------------------------
# DataFrame mirrors (R10 contract layer) — pandas imported lazily so the
# numpy-only test path keeps working without it
# ---------------------------------------------------------------------------

DF_COLUMNS = ["a", "b", "c"]


def make_global_df(n=N_GLOBAL, seed=21):
    """Numeric DataFrame mirror of make_global_data (same values, named cols)."""
    import pandas as pd

    return pd.DataFrame(make_global_data(n, seed), columns=DF_COLUMNS)


def make_mixed_df(n=N_GLOBAL, seed=21):
    """A DataFrame with one column per R10 dtype family + its native model."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num": rng.uniform(-1, 1, n),
            "count": rng.integers(0, 3, n),
            "color": pd.Categorical(rng.choice(["r", "g", "b"], n)),
            "size": pd.Categorical(
                rng.choice(["S", "M", "L"], n),
                categories=["S", "M", "L"],
                ordered=True,
            ),
        }
    )


def mixed_df_model(df):
    """Model defined ON the DataFrame — exercises the R10 model-call rule."""
    return (
        2.0 * df["num"].to_numpy()
        + df["count"].to_numpy().astype(float)
        + df["color"].cat.codes.to_numpy().astype(float)
        + 0.5 * df["size"].cat.codes.to_numpy().astype(float)
    )


# ---------------------------------------------------------------------------
# regional-effect registry (gated-linear model from the functional anchor)
# ---------------------------------------------------------------------------

N_REGIONAL = 500


def gated_model(x):
    y = np.zeros_like(x[:, 0])
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind] = 5 * x[ind, 0]
    return y


def gated_model_jac(x):
    y = np.zeros_like(x)
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind, 0] = 5
    return y


def make_regional_data(n=N_REGIONAL, seed=21):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            rng.integers(0, 2, n).astype(float),
        ],
        axis=1,
    )


def make_regional(name, data, **kwargs):
    """Construct a fresh (unfitted) regional-effect object of the given kind."""
    if name == "regional_pdp":
        return effector.RegionalPDP(data, gated_model, **kwargs)
    if name == "regional_derpdp":
        return effector.RegionalDerPDP(
            data, gated_model, model_jac=gated_model_jac, **kwargs
        )
    if name == "regional_ale":
        return effector.RegionalALE(data, gated_model, **kwargs)
    if name == "regional_rhale":
        return effector.RegionalRHALE(
            data, gated_model, model_jac=gated_model_jac, **kwargs
        )
    if name == "regional_shapdp":
        return effector.RegionalShapDP(data, gated_model, **kwargs)
    raise ValueError(f"unknown regional method: {name}")


def fit_regional(name, data):
    """Fit feature 0 the standard way for the contract tests.

    RegionalShapDP: N=50 / budget=128 and seeded explainer, per the runtime
    budget (PLAN II §5) — shap cost stays ~seconds and the tree is stable.
    """
    if name == "regional_shapdp":
        np.random.seed(0)
        reg = make_regional(name, data[:50])
        reg.fit(
            0,
            space_partitioner=effector.space_partitioning.Best(max_depth=2),
            budget=128,
            shap_explainer_kwargs={"seed": 0},
        )
        return reg
    reg = make_regional(name, data)
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    return reg


@pytest.fixture(scope="module")
def regional_data():
    return make_regional_data()


@pytest.fixture(scope="module", params=REGIONAL_NAMES)
def fitted_regional(request, regional_data):
    """One fitted regional object per method, cached for the whole module."""
    return request.param, fit_regional(request.param, regional_data)
