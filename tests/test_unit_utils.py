"""Unit layer for effector.utils (PLAN II §3.2): the numerical kernels the
whole package sits on.  Everything here is exact and µs-fast — the docstring
examples become real asserts, plus hand-computable cases for the ALE pipeline.
"""

import numpy as np
import pytest

from effector import utils

# ---------------------------------------------------------------------------
# compute_bin_effect / compute_bin_variance
# ---------------------------------------------------------------------------


def test_compute_bin_effect_basic_and_empty_bin():
    xs = np.ones([100]) - 0.5
    df_dxs = np.ones_like(xs) * 10
    limits = np.array([0.0, 1.0, 2.0])
    bin_effects, ppb = utils.compute_bin_effect(xs, df_dxs, limits)
    np.testing.assert_array_equal(ppb, [100, 0])
    assert bin_effects[0] == 10.0
    assert np.isnan(bin_effects[1])


def test_compute_bin_effect_mean_per_bin():
    xs = np.array([0.1, 0.4, 0.6, 0.9, 1.1, 1.4])
    df_dxs = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    limits = np.array([0.0, 0.5, 1.0, 1.5])
    bin_effects, ppb = utils.compute_bin_effect(xs, df_dxs, limits)
    np.testing.assert_array_equal(ppb, [2, 2, 2])
    np.testing.assert_allclose(bin_effects, [2.0, 6.0, 10.0])


def test_compute_bin_effect_right_edge_included():
    xs = np.array([0.5, 2.0])
    df_dxs = np.array([1.0, 3.0])
    limits = np.array([0.0, 1.0, 2.0])
    bin_effects, ppb = utils.compute_bin_effect(xs, df_dxs, limits)
    np.testing.assert_array_equal(ppb, [1, 1])
    np.testing.assert_allclose(bin_effects, [1.0, 3.0])


def test_compute_bin_variance_zero_and_nan():
    xs = np.ones([100]) - 0.5
    df_dxs = np.ones_like(xs) * 10
    limits = np.array([0.0, 1.0, 2.0])
    bin_effect_mean, _ = utils.compute_bin_effect(xs, df_dxs, limits)
    bin_variance = utils.compute_bin_variance(xs, df_dxs, limits, bin_effect_mean)
    assert bin_variance[0] == 0.0
    assert np.isnan(bin_variance[1])


def test_compute_bin_variance_value():
    xs = np.ones(4) * 0.5
    df_dxs = np.array([1.0, 3.0, 3.0, 5.0])
    limits = np.array([0, 1, 2.0])
    bin_effect_mean = np.array([np.mean(df_dxs), np.nan])
    bin_variance = utils.compute_bin_variance(xs, df_dxs, limits, bin_effect_mean)
    assert bin_variance[0] == 2.0
    assert np.isnan(bin_variance[1])


def test_compute_bin_variance_single_point_bin_is_nan():
    xs = np.array([0.5, 1.5, 1.6])
    df_dxs = np.array([1.0, 2.0, 4.0])
    limits = np.array([0.0, 1.0, 2.0])
    mean, _ = utils.compute_bin_effect(xs, df_dxs, limits)
    var = utils.compute_bin_variance(xs, df_dxs, limits, mean)
    assert np.isnan(var[0])  # one point -> variance undefined
    np.testing.assert_allclose(var[1], 1.0)


# ---------------------------------------------------------------------------
# fill_nans
# ---------------------------------------------------------------------------


def test_fill_nans_interior():
    np.testing.assert_allclose(
        utils.fill_nans(np.array([1.0, np.nan, 2.0])), [1.0, 1.5, 2.0]
    )
    np.testing.assert_allclose(
        utils.fill_nans(np.array([1.0, np.nan, np.nan, np.nan, 2.0])),
        [1.0, 1.25, 1.5, 1.75, 2.0],
    )


def test_fill_nans_edges():
    np.testing.assert_allclose(
        utils.fill_nans(np.array([0.5, 1.0, np.nan, np.nan, np.nan])),
        [0.5, 1.0, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        utils.fill_nans(np.array([np.nan, np.nan, 1.0, 2.0])), [1.0, 1.0, 1.0, 2.0]
    )


def test_fill_nans_all_nan_raises():
    with pytest.raises(utils.AllBinsHaveAtMostOnePointError):
        utils.fill_nans(np.array([np.nan, np.nan]))


# ---------------------------------------------------------------------------
# apply_bin_value / compute_accumulated_effect
# ---------------------------------------------------------------------------


def test_apply_bin_value_clamps_outside():
    x = np.array([-0.5, 0.5, 1.5, 2.0, 2.5])
    bin_limits = np.array([0.0, 1.0, 2.0])
    bin_value = np.array([5.0, 7.0])
    np.testing.assert_allclose(
        utils.apply_bin_value(x, bin_limits, bin_value), [5.0, 5.0, 7.0, 7.0, 7.0]
    )


def test_compute_accumulated_effect_docstring_examples():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    limits = np.array([0, 1.5, 2.0])
    dx = np.array([1.5, 0.5])
    np.testing.assert_allclose(
        utils.compute_accumulated_effect(x, limits, np.array([1.0, -1.0]), dx),
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        utils.compute_accumulated_effect(x, limits, np.array([1.0, 1.0]), dx),
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0],
    )


def test_compute_accumulated_effect_square():
    x = np.array([0.5, 1.0, 1.5, 2.0])
    limits = np.array([0.0, 1.0, 2.0])
    bin_effect = np.array([1.0, 1.0])
    dx = np.array([1.0, 1.0])
    np.testing.assert_allclose(
        utils.compute_accumulated_effect(x, limits, bin_effect, dx, square=True),
        [0.25, 1.0, 1.25, 2.0],
    )


# ---------------------------------------------------------------------------
# compute_ale_params — the ALE pipeline end-to-end on hand-computable points
# ---------------------------------------------------------------------------


def test_compute_ale_params_six_points():
    xs = np.array([0.1, 0.4, 0.6, 0.9, 1.1, 1.4])
    df_dxs = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    limits = np.array([0.0, 0.5, 1.0, 1.5])
    params = utils.compute_ale_params(xs, df_dxs, limits)
    np.testing.assert_allclose(params["dx"], [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(params["points_per_bin"], [2, 2, 2])
    np.testing.assert_allclose(params["bin_effect"], [2.0, 6.0, 10.0])
    np.testing.assert_allclose(params["bin_variance"], [1.0, 1.0, 1.0])


def test_compute_ale_params_interpolates_empty_bin():
    xs = np.array([0.25, 0.3, 2.5, 2.6])
    df_dxs = np.array([1.0, 3.0, 5.0, 7.0])
    limits = np.array([0.0, 1.0, 2.0, 3.0])
    params = utils.compute_ale_params(xs, df_dxs, limits)
    # middle bin is empty: its effect is the interpolation of the neighbours
    np.testing.assert_allclose(params["bin_effect"], [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(params["points_per_bin"], [2, 0, 2])


# ---------------------------------------------------------------------------
# compute_local_effects
# ---------------------------------------------------------------------------


def test_compute_local_effects_docstring():
    data = np.array([[1, 2], [2, 3.0]])
    model = lambda x: np.sum(x, axis=1)
    limits = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        utils.compute_local_effects(data, model, limits, feature=0), [1.0, 1.0]
    )


def test_compute_local_effects_linear_exact():
    rng = np.random.default_rng(21)
    data = rng.uniform(-1, 1, size=(50, 2))
    model = lambda x: 2 * x[:, 0] - 3 * x[:, 1]
    limits = np.linspace(-1, 1, 6)
    effects = utils.compute_local_effects(data, model, limits, feature=0)
    np.testing.assert_allclose(effects, np.full(50, 2.0), atol=1e-10)


# ---------------------------------------------------------------------------
# compute_jacobian_numerically
# ---------------------------------------------------------------------------


def test_jacobian_numerically_linear():
    rng = np.random.default_rng(21)
    data = rng.uniform(-1, 1, size=(20, 2))
    model = lambda x: 2 * x[:, 0] - 3 * x[:, 1]
    jac = utils.compute_jacobian_numerically(model, data)
    np.testing.assert_allclose(jac[:, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(jac[:, 1], -3.0, atol=1e-5)


def test_jacobian_numerically_quadratic():
    rng = np.random.default_rng(21)
    data = rng.uniform(-1, 1, size=(20, 2))
    model = lambda x: x[:, 0] ** 2 + x[:, 1] ** 3
    jac = utils.compute_jacobian_numerically(model, data)
    np.testing.assert_allclose(jac[:, 0], 2 * data[:, 0], atol=1e-4)
    np.testing.assert_allclose(jac[:, 1], 3 * data[:, 1] ** 2, atol=1e-4)


# ---------------------------------------------------------------------------
# get_feature_types — deprecated delegate to ingestion.infer_feature_types
# ---------------------------------------------------------------------------


def test_get_feature_types_deprecated_three_way():
    n_unique_a, n_unique_b = 9, 10
    col_a = np.tile(np.arange(n_unique_a), 10)[:90]
    col_b = np.tile(np.arange(n_unique_b), 9)[:90]
    data = np.stack([col_a, col_b], axis=1).astype(float)
    with pytest.warns(DeprecationWarning, match="infer_feature_types"):
        types = utils.get_feature_types(data, categorical_limit=10)
    # vocabulary is three-way now; strictly-less-than cat_limit is ordinal
    assert types == ["ordinal", "continuous"]


# ---------------------------------------------------------------------------
# filter_points_in_bin
# ---------------------------------------------------------------------------


def test_filter_points_in_bin():
    xs = np.array([1, 2, 3])
    df_dxs = np.array([32, 34, 36])
    xs_f, df_f = utils.filter_points_in_bin(xs, df_dxs, np.array([1, 2]))
    np.testing.assert_array_equal(xs_f, [1, 2])
    np.testing.assert_array_equal(df_f, [32, 34])
    xs_f, df_f = utils.filter_points_in_bin(xs, None, np.array([1, 2]))
    np.testing.assert_array_equal(xs_f, [1, 2])
    assert df_f is None


def test_mean_1d_linspace():
    assert round(float(utils.mean_1d_linspace(lambda x: 2 * x, 0.0, 1.0)), 10) == 1.0
    # midpoint rule on x^2 over [0, 1]: close to 1/3
    np.testing.assert_allclose(
        utils.mean_1d_linspace(lambda x: x**2, 0.0, 1.0), 1 / 3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# compute_local_effects_level_pairs — the order-free nominal raw material
# ---------------------------------------------------------------------------


def _pairs_toy():
    # 3 levels, 2 instances per level; f = level + 0.1 * x1
    data = np.array(
        [[0.0, 1.0], [0.0, 2.0], [1.0, 3.0], [1.0, 4.0], [2.0, 5.0], [2.0, 6.0]]
    )
    model = lambda x: x[:, 0] + 0.1 * x[:, 1]
    levels = np.array([0.0, 1.0, 2.0])
    return data, model, levels


def test_level_pairs_covers_all_pairs_with_both_side_contributors():
    data, model, levels = _pairs_toy()
    lo, hi, eff, idx = utils.compute_local_effects_level_pairs(data, model, levels, 0)
    # 3 pairs x 4 contributors (2 per side) each
    assert len(eff) == 12
    assert sorted(set(zip(lo.tolist(), hi.tolist()))) == [(0, 1), (0, 2), (1, 2)]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        sel = (lo == a) & (hi == b)
        # contributors are exactly the instances at either level
        expected_rows = np.where(
            np.isclose(data[:, 0], levels[a]) | np.isclose(data[:, 0], levels[b])
        )[0]
        np.testing.assert_array_equal(np.sort(idx[sel]), expected_rows)
        # additive model: the raw difference is exactly b - a for everyone
        np.testing.assert_allclose(eff[sel], levels[b] - levels[a], atol=1e-12)


def test_level_pairs_chain_transitions_are_the_adjacent_subset():
    data, model, levels = _pairs_toy()
    lo, hi, eff, idx = utils.compute_local_effects_level_pairs(data, model, levels, 0)
    pos_c, eff_c, idx_c = utils.compute_local_effects_categorical(
        data, model, levels, 0
    )
    for t in range(1, len(levels)):
        sel = (lo == t - 1) & (hi == t)
        sel_c = np.isclose(pos_c, t - 0.5)
        np.testing.assert_array_equal(idx[sel], idx_c[sel_c])
        np.testing.assert_allclose(eff[sel], eff_c[sel_c], atol=1e-12)


def test_level_pairs_antisymmetry_via_interaction():
    # f = level * x1: the pairwise difference is (v_b - v_a) * x1 per instance,
    # so swapping the pair direction flips the sign instance-by-instance
    data, _, levels = _pairs_toy()
    model = lambda x: x[:, 0] * x[:, 1]
    lo, hi, eff, idx = utils.compute_local_effects_level_pairs(data, model, levels, 0)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        sel = (lo == a) & (hi == b)
        np.testing.assert_allclose(
            eff[sel], (levels[b] - levels[a]) * data[idx[sel], 1], atol=1e-12
        )
