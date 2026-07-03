"""Plot-content layer (PLAN II §3.4): beyond crash-testing, the returned
(fig, ax) is inspected — the drawn mean line must equal ``eval`` (R1), scaling
must be affine (and std-only for derivative plots — B5), y_limits/nof_ice/
legend must be respected.  Runs on the shared tiny models from conftest
(ShapDP via precomputed analytic shap values: 0 s of SHAP)."""

import numpy as np
import pytest

from tests.conftest import GLOBAL_NAMES, eval_mean, make_global

# the label each method gives to its mean-effect line
MEAN_LINE_LABEL = {
    "pdp": "PDP",
    "derpdp": "d-PDP",
    "ale": "average effect",
    "rhale": "average effect",
    "shapdp": "SHAP-DP",
}


def _mean_axis(ret):
    """The axis holding the mean-effect line ((RH)ALE returns (fig, (ax1, ax2)))."""
    fig, ax = ret
    return ax[0] if isinstance(ax, tuple) else ax


def _mean_line(ax, label):
    lines = [ln for ln in ax.get_lines() if ln.get_label() == label]
    assert len(lines) == 1, f"expected one line labeled {label!r}"
    return lines[0]


# ---------------------------------------------------------------------------
# the mean line IS eval (R1: plot is a view over eval)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_mean_line_matches_eval(name, global_data):
    m = make_global(name, global_data)
    ret = m.plot(0, heterogeneity=False, centering=False, show_plot=False)
    line = _mean_line(_mean_axis(ret), MEAN_LINE_LABEL[name])
    xdata, ydata = line.get_xdata(), line.get_ydata()
    np.testing.assert_allclose(ydata, eval_mean(m, 0, xdata), atol=1e-8)


# ---------------------------------------------------------------------------
# scaling is affine on the line data
# ---------------------------------------------------------------------------

SCALE_X = {"mean": 5.0, "std": 2.0}
SCALE_Y = {"mean": 10.0, "std": 3.0}


def test_scale_affine_pdp(global_data):
    m = make_global("pdp", global_data)
    ret = m.plot(
        0,
        heterogeneity=False,
        centering=False,
        scale_x=SCALE_X,
        scale_y=SCALE_Y,
        show_plot=False,
    )
    line = _mean_line(_mean_axis(ret), "PDP")
    xs_orig = (line.get_xdata() - SCALE_X["mean"]) / SCALE_X["std"]
    expected = eval_mean(m, 0, xs_orig) * SCALE_Y["std"] + SCALE_Y["mean"]
    np.testing.assert_allclose(line.get_ydata(), expected, atol=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason="B5: is_derivative never wired -> DerPDP + scale_y shifts the "
    "derivative by the mean instead of scaling by std only",
)
def test_scale_derivative_is_std_only(global_data):
    m = make_global("derpdp", global_data)
    ret = m.plot(
        0,
        heterogeneity=False,
        centering=False,
        scale_x=SCALE_X,
        scale_y=SCALE_Y,
        show_plot=False,
    )
    line = _mean_line(_mean_axis(ret), "d-PDP")
    xs_orig = (line.get_xdata() - SCALE_X["mean"]) / SCALE_X["std"]
    expected = eval_mean(m, 0, xs_orig) * SCALE_Y["std"]  # no mean shift
    np.testing.assert_allclose(line.get_ydata(), expected, atol=1e-8)


# ---------------------------------------------------------------------------
# heterogeneity="std_err" must be the standard error, not the std (B5)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="B5: the 'std_err' band is computed as sqrt(var) — i.e. the std",
)
def test_stderr_band_is_standard_error(global_data):
    m = make_global("pdp", global_data)
    ret = m.plot(0, heterogeneity="std_err", centering=False, show_plot=False)
    ax = _mean_axis(ret)
    assert len(ax.collections) == 1
    verts = ax.collections[0].get_paths()[0].vertices

    line = _mean_line(ax, "PDP")
    x0 = line.get_xdata()[0]
    ys_at_x0 = verts[np.isclose(verts[:, 0], x0), 1]
    band = (ys_at_x0.max() - ys_at_x0.min()) / 2

    # ICE curves of the linear model are parallel lines: their pointwise std
    # is constant = std(-3*x1_i + 0.5*x2_i over instances i)
    n = m.data.shape[0]
    std = np.std(-3 * m.data[:, 1] + 0.5 * m.data[:, 2])
    np.testing.assert_allclose(band, std / np.sqrt(n), rtol=0.05)


# ---------------------------------------------------------------------------
# y_limits / nof_ice / legend
# ---------------------------------------------------------------------------


def test_y_limits_respected(global_data):
    m = make_global("pdp", global_data)
    ret = m.plot(0, y_limits=[-1.0, 2.0], show_plot=False)
    assert _mean_axis(ret).get_ylim() == (-1.0, 2.0)

    m = make_global("ale", global_data)
    ret = m.plot(0, y_limits=[-1.0, 2.0], show_plot=False)
    assert _mean_axis(ret).get_ylim() == (-1.0, 2.0)


def test_nof_ice_count(global_data):
    m = make_global("pdp", global_data)
    ret = m.plot(0, heterogeneity="ice", nof_ice=13, show_plot=False)
    ax = _mean_axis(ret)
    # 1 labeled ICE + 13 ICE curves + 1 PDP line
    assert len(ax.get_lines()) == 13 + 2


def test_legend_labels_stable(global_data):
    m = make_global("pdp", global_data)
    ret = m.plot(0, heterogeneity="ice", show_plot=False)
    labels = {t.get_text() for t in _mean_axis(ret).get_legend().get_texts()}
    assert {"PDP", "ICE"} <= labels

    m = make_global("shapdp", global_data)
    ret = m.plot(0, heterogeneity="shap_values", show_plot=False)
    labels = {t.get_text() for t in _mean_axis(ret).get_legend().get_texts()}
    assert {"SHAP-DP", "SHAP values"} <= labels


# ---------------------------------------------------------------------------
# ShapDP heterogeneity=False draws no band (the visible contract around B4)
# ---------------------------------------------------------------------------


def test_shapdp_band_only_when_asked(global_data):
    m = make_global("shapdp", global_data)
    ret = m.plot(0, heterogeneity=False, show_plot=False)
    assert len(_mean_axis(ret).collections) == 0

    m = make_global("shapdp", global_data)
    ret = m.plot(0, heterogeneity="std", show_plot=False)
    assert len(_mean_axis(ret).collections) == 1
