"""House theme (LOGBOOK #25): the semantic palette colors apply to every plot by
default; the chrome rcParams only mutate global state when the user opts in via
``set_theme``. These tests pin both halves — and guard that heterogeneity is now
gray, not red.
"""

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import effector
from effector import theme
from effector import visualization as vis


@pytest.fixture(autouse=True)
def _restore_theme():
    """No theme test may leak rcParams or the active theme into the next."""
    snapshot = dict(mpl.rcParams)
    yield
    mpl.rcParams.update(snapshot)
    theme._ACTIVE = theme.LIGHT


def _pdp():
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400)])
    model = lambda z: z[:, 0] ** 2 + 0.5 * z[:, 1]  # noqa: E731
    m = effector.PDP(x, model)
    m.fit(0)
    return m


def _pdp_std_band(ax):
    """The fill_between PolyCollection is the only collection on a std PDP."""
    return ax.collections[0].get_facecolor()[0]


def test_default_tokens_applied_without_opt_in():
    # no set_theme() called: palette colors still apply (they are baked per-artist)
    _, ax = _pdp().plot(0, heterogeneity="std", show_plot=False)
    mean = next(ln for ln in ax.get_lines() if ln.get_label() == "PDP")
    assert to_rgba(mean.get_color()) == to_rgba(theme.LIGHT.MEAN)
    band = _pdp_std_band(ax)
    assert np.allclose(
        band, to_rgba(theme.LIGHT.BAND, theme.LIGHT.BAND_ALPHA), atol=1e-6
    )


def test_heterogeneity_is_gray_not_red():
    # the whole point of A9: the loud red band/cloud becomes a quiet gray
    _, ax = _pdp().plot(0, heterogeneity="std", show_plot=False)
    r, g, b, _ = _pdp_std_band(ax)
    assert abs(r - g) < 0.05 and abs(g - b) < 0.05  # gray: channels ~equal
    assert not np.allclose((r, g, b), to_rgba("red")[:3])


def test_no_global_rcparam_leak_without_set_theme():
    before = dict(mpl.rcParams)
    _pdp().plot(0, heterogeneity="std", show_plot=False)
    assert dict(mpl.rcParams) == before


def test_import_is_side_effect_free():
    # importing effector must not have mutated the ambient matplotlib chrome
    for key in ("axes.facecolor", "axes.grid", "font.size"):
        assert mpl.rcParams[key] == mpl.rcParamsDefault[key]


def test_set_theme_switches_chrome_and_tracks_active():
    effector.set_theme("dark")
    assert theme.active().name == "dark"
    assert to_rgba(mpl.rcParams["axes.facecolor"]) == to_rgba(
        theme.DARK.rcparams["axes.facecolor"]
    )
    # a fresh plot's avg line reads the now-active theme's token (dark ink)
    _, ax = _pdp().plot(0, heterogeneity="std", show_plot=False)
    mean = next(ln for ln in ax.get_lines() if ln.get_label() == "PDP")
    assert to_rgba(mean.get_color()) == to_rgba(theme.DARK.MEAN)


def test_set_theme_default_resets_to_stock_matplotlib():
    effector.set_theme("dark")
    effector.set_theme("default")
    assert theme.active().name == "light"
    assert to_rgba(mpl.rcParams["axes.facecolor"]) == to_rgba(
        mpl.rcParamsDefault["axes.facecolor"]
    )


def test_comparison_uses_the_cat_cycle():
    x = np.linspace(-1, 1, 10)
    curves = {"a": x, "b": 2 * x, "c": 3 * x}
    _, ax = vis.plot_effect_comparison(x, 0, curves, show_plot=False)
    colors = [to_rgba(ln.get_color()) for ln in ax.get_lines()]
    assert colors == [to_rgba(theme.LIGHT.CAT[i]) for i in range(3)]


def test_invalid_theme_name_raises():
    with pytest.raises(ValueError, match="unknown theme"):
        effector.set_theme("nope")
