import numpy as np
from .plot_utils import *


def _is_cat(feat_name, categorical_features):
    return categorical_features is not None and feat_name in categorical_features


def _nearest_index(arr, v):
    arr = np.asarray(arr)
    return int(np.argmin(np.abs(arr - v)))


def _interp1d(x_grid, y_grid, x):
    """Linear interpolation with clipping to endpoints."""
    xg = np.asarray(x_grid, dtype=float)
    yg = np.asarray(y_grid, dtype=float)

    if x <= xg[0]:
        return float(yg[0])
    if x >= xg[-1]:
        return float(yg[-1])

    j = int(np.searchsorted(xg, x) - 1)
    x0, x1 = xg[j], xg[j + 1]
    y0, y1 = yg[j], yg[j + 1]
    w = (x - x0) / (x1 - x0)
    return float((1 - w) * y0 + w * y1)


def _bilinear(xg, yg, Z, x, y):
    """
    Bilinear interpolation on a rectilinear grid.
    xg: shape (nx,), yg: shape (ny,), Z: shape (ny, nx)
    """
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # clip
    x = float(np.clip(x, xg[0], xg[-1]))
    y = float(np.clip(y, yg[0], yg[-1]))

    ix = int(np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2))
    iy = int(np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2))

    x0, x1 = xg[ix], xg[ix + 1]
    y0, y1 = yg[iy], yg[iy + 1]

    # avoid divide by zero if degenerate
    wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    return float(
        (1 - wx) * (1 - wy) * z00
        + wx * (1 - wy) * z10
        + (1 - wx) * wy * z01
        + wx * wy * z11
    )


def _get_1d_term(model, feat_idx):
    """
    Returns x_grid, y_scores for the main effect of feat_idx.
    Uses your extract_full_shape_function helper to stay consistent.
    """
    x, y, lower, upper, _, _ = extract_full_shape_function(model, feat_idx)
    return np.array(x), np.array(y)


def _get_interaction_term(model, feat_idx, other_idx):
    """
    Returns a canonical representation where:
      - x-axis corresponds to feat_idx
      - y-axis corresponds to other_idx
      - scores is shape (len(other_vals), len(feat_vals))
    """
    explanation = model.explain_global()
    specific = explanation._internal_obj["specific"]
    term_features = model.term_features_

    # find the term index whose term_features == {feat_idx, other_idx} (order can vary)
    target = None
    for term_idx, term in enumerate(specific):
        if term.get("type") != "interaction":
            continue
        feats = term_features[term_idx]
        if set(feats) == set([feat_idx, other_idx]):
            target = (term_idx, term, feats)
            break

    if target is None:
        raise ValueError(
            f"No interaction found between features {feat_idx} and {other_idx}"
        )

    term_idx, term, feats = target
    scores = np.array(term["scores"])
    left_names = np.array(term["left_names"], dtype=float)
    right_names = np.array(term["right_names"], dtype=float)

    # In EBM internal objects, "left" corresponds to feats[0], "right" corresponds to feats[1].
    left_feat, right_feat = feats[0], feats[1]

    # We want x-axis = feat_idx, y-axis = other_idx.
    if right_feat == feat_idx and left_feat == other_idx:
        # already y=left(other), x=right(feat)
        x_vals = right_names
        y_vals = left_names
        Z = scores  # shape (len(y), len(x))
    elif left_feat == feat_idx and right_feat == other_idx:
        # swap axes
        x_vals = left_names
        y_vals = right_names
        Z = scores.T
    else:
        # Shouldn't happen if set matched, but keep safe.
        raise RuntimeError("Unexpected feature ordering in interaction term.")

    return x_vals, y_vals, Z


def ga2m_shape_value(
    model,
    feat_idx,
    feature_value,
    *,
    feat_labels=None,
    categorical_features=None,
    numerical_features=None,
    center_like_plot=True,
    t=None,
):
    """
    Returns y for the 1D shape function at feature_value.
    - numerical: linear interpolation over grid
    - categorical: exact match (or nearest if numeric codes)
    """
    xg, yg = _get_1d_term(model, feat_idx)

    feat_name = feat_labels[feat_idx] if feat_labels is not None else str(feat_idx)
    is_cat = _is_cat(feat_name, categorical_features)

    # match your plot behavior
    if (not is_cat) and center_like_plot:
        yg = yg - np.mean(yg)

    if t is not None:
        # if feat_labels[feat_idx] in numerical_features:
        #     x = x.astype(float)
        x_lookup = reverse_column_transform(
            values=xg,
            pipeline=t,
            feature_name=feat_name,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        if feat_name == "temp":
            x_lookup = rev_celsius(x_lookup)
    else:
        x_lookup = xg

    if is_cat:
        # categorical: direct match on the tick labels/codes
        # try exact first
        x_list = list(x_lookup)
        if feature_value in x_list:
            j = x_list.index(feature_value)
            return float(yg[j])

        # fallback: if values are numeric-ish, nearest
        try:
            j = _nearest_index(np.asarray(xg, dtype=float), float(feature_value))
            return float(yg[j])
        except Exception as e:
            raise ValueError(
                f"Could not match categorical value={feature_value} for feature '{feat_name}'"
            ) from e

    # numerical: interpolation
    return _interp1d(np.asarray(x_lookup, dtype=float), yg, float(feature_value))


def ga2m_interaction_value(
    model,
    feat_idx,
    other_idx,
    feature_value,
    other_value,
    *,
    feat_labels=None,
    categorical_features=None,
    numerical_features=None,
    t=None,
):
    """
    Returns the interaction heatmap value s(feature_value, other_value)
    for the interaction term between feat_idx and other_idx.

    - numerical/numerical: bilinear interpolation
    - categorical: exact match (fallback nearest for numeric codes)
    """
    xg, yg, Z = _get_interaction_term(model, feat_idx, other_idx)

    feat_name = feat_labels[feat_idx] if feat_labels is not None else str(feat_idx)
    other_name = feat_labels[other_idx] if feat_labels is not None else str(other_idx)

    feat_is_cat = _is_cat(feat_name, categorical_features)
    other_is_cat = _is_cat(other_name, categorical_features)

    if t is not None:
        x_lookup = reverse_column_transform(
            values=xg,
            pipeline=t,
            feature_name=feat_name,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        y_lookup = reverse_column_transform(
            values=yg,
            pipeline=t,
            feature_name=other_name,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        if feat_name == "temp":
            x_lookup = rev_celsius(x_lookup)
    else:
        x_lookup, y_lookup = xg, yg

    # categorical handling
    if feat_is_cat:
        x_list = list(x_lookup)
        if feature_value in x_list:
            ix = x_list.index(feature_value)
        else:
            # fallback nearest in internal numeric codes if possible
            ix = _nearest_index(np.asarray(xg, dtype=float), float(feature_value))
    else:
        # numeric -> use interpolation, so keep raw value
        ix = None

    if other_is_cat:
        y_list = list(y_lookup)
        if other_value in y_list:
            iy = y_list.index(other_value)
        else:
            iy = _nearest_index(np.asarray(yg, dtype=float), float(other_value))
    else:
        iy = None

    # cases:
    if (not feat_is_cat) and (not other_is_cat):
        return _bilinear(
            np.asarray(x_lookup, dtype=float),
            np.asarray(y_lookup, dtype=float),
            Z,
            float(feature_value),
            float(other_value),
        )

    if feat_is_cat and other_is_cat:
        return float(Z[iy, ix])

    if feat_is_cat and (not other_is_cat):
        # interpolate along y for fixed x bin
        col = Z[:, ix]
        return _interp1d(np.asarray(y_lookup, dtype=float), col, float(other_value))

    if (not feat_is_cat) and other_is_cat:
        # interpolate along x for fixed y bin
        row = Z[iy, :]
        return _interp1d(np.asarray(x_lookup, dtype=float), row, float(feature_value))

    raise RuntimeError("Unhandled categorical/numerical combination.")


def ga2m_shift_1d(model, feat_idx, k1, k2, **kwargs):
    """Shift in 1D shape function: y(k2) - y(k1)."""
    y1 = ga2m_shape_value(model, feat_idx, k1, **kwargs)
    y2 = ga2m_shape_value(model, feat_idx, k2, **kwargs)
    return float(y2 - y1)


def ga2m_shift_interaction_wrt_feat(
    model, feat_idx, other_idx, k1, k2, other_value_fixed, **kwargs
):
    """
    Shift along feat axis in the interaction heatmap:
      s(k2, other) - s(k1, other)
    """
    s1 = ga2m_interaction_value(
        model, feat_idx, other_idx, k1, other_value_fixed, **kwargs
    )
    s2 = ga2m_interaction_value(
        model, feat_idx, other_idx, k2, other_value_fixed, **kwargs
    )
    return float(s2 - s1)


def ga2m_shift_interaction_wrt_other(
    model, feat_idx, other_idx, feat_value_fixed, l1, l2, **kwargs
):
    """
    Shift along other axis in the interaction heatmap:
      s(feat, l2) - s(feat, l1)
    """
    s1 = ga2m_interaction_value(
        model, feat_idx, other_idx, feat_value_fixed, l1, **kwargs
    )
    s2 = ga2m_interaction_value(
        model, feat_idx, other_idx, feat_value_fixed, l2, **kwargs
    )
    return float(s2 - s1)


def _resolve_calm_term_idx(conditions, condition_id):
    """
    conditions: list/array of masked-gam term indices for this original feature
    condition_id:
      - if in conditions -> treated as actual term index
      - else -> treated as position into conditions
    """
    conditions = list(conditions)
    if condition_id in conditions:
        return int(condition_id)

    # otherwise interpret as positional
    cid = int(condition_id)
    if cid < 0 or cid >= len(conditions):
        raise ValueError(
            f"condition_id={condition_id} is out of range for {len(conditions)} conditions."
        )
    return int(conditions[cid])


def calm_shape_value(
    calm,
    feat_idx,
    feature_value,
    *,
    condition_id,
    # labels / transforms like your plotting
    feat_labels=None,
    categorical_features=None,
    numerical_features=None,
    center_like_plot=True,
    t=None,
    label_maps=None,
    foi_type=None,
):
    """
    Evaluate CALM conditional 1D shape function for original feature feat_idx at feature_value.

    condition_id chooses which conditional 1D function:
      - either a position (0..n_conditions-1)
      - or a direct masked-gam term index (must be in 'conditions')

    Returns y(feature_value) for the chosen conditional curve.
    """

    if calm is None:
        raise ValueError("calm is None")

    calm_gam, calm_feature_names = calm.masked_gam.model, calm.new_names
    calm_feature_names = simplify_expressions(calm_feature_names)

    # build mapping original feat_idx -> list of conditional term indices
    # (exactly like your plot code)
    if feat_labels is None:
        # same behavior you use: feat_labels over calm_feature_names length
        feat_labels = [rf"$x_{{{i}}}$" for i in range(len(calm_feature_names))]

    if t is not None:
        # optional: only needed if your build_feature_mapping expects scaled-back expressions
        calm_feature_names_scaled = scale_back_expressions(
            calm_feature_names,
            t,
            categorical_features,
            numerical_features,
            labels_map=label_maps,
        )
    else:
        calm_feature_names_scaled = calm_feature_names

    calm_features_conditions_names_map = build_feature_mapping(
        calm_feature_names_scaled, feat_labels
    )

    conditions = calm_features_conditions_names_map[feat_idx]
    term_idx = _resolve_calm_term_idx(conditions, condition_id)

    # now evaluate that 1D term
    xg, yg = _get_1d_term(calm_gam, term_idx)

    feat_name = feat_labels[feat_idx] if feat_labels is not None else str(feat_idx)
    is_cat = _is_cat(feat_name, categorical_features)

    if (not is_cat) and center_like_plot:
        yg = yg - np.mean(yg)

    # optional: apply your plot transform to the x grid (for querying in display scale)
    if t is not None:
        x_lookup = reverse_column_transform(
            values=xg,
            pipeline=t,
            feature_name=feat_name,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        if feat_name == "temp":
            x_lookup = rev_celsius(x_lookup)
    else:
        x_lookup = xg

    # handle ordinal categorical special case like your plot
    if is_cat and foi_type is not None and foi_type == "ordinal":
        try:
            x_lookup = np.asarray(x_lookup).astype(float).astype(int)
        except Exception:
            pass

    # categorical: exact match (optionally via label_maps)
    if is_cat:
        # apply label map to x_lookup for comparison if you display-mapped in plotting
        x_match = list(x_lookup)
        if label_maps and feat_name in label_maps:
            x_match = [label_maps[feat_name].get(v, v) for v in x_match]

        if feature_value in x_match:
            j = x_match.index(feature_value)
            return float(yg[j])

        # fallback: numeric-ish nearest on internal grid
        try:
            j = _nearest_index(np.asarray(xg, dtype=float), float(feature_value))
            return float(yg[j])
        except Exception as e:
            raise ValueError(
                f"Could not match categorical value={feature_value} for feature '{feat_name}'"
            ) from e

    # numeric: interpolate on displayed grid (x_lookup)
    return _interp1d(np.asarray(x_lookup, dtype=float), yg, float(feature_value))


def calm_shift_1d(
    calm,
    feat_idx,
    k1,
    k2,
    *,
    condition_id,
    **kwargs,
):
    """Shift on the SAME conditional CALM curve: y(k2) - y(k1)."""
    y1 = calm_shape_value(calm, feat_idx, k1, condition_id=condition_id, **kwargs)
    y2 = calm_shape_value(calm, feat_idx, k2, condition_id=condition_id, **kwargs)
    return float(y2 - y1)


def calm_condition_switch_delta(
    calm,
    feat_idx,
    k,
    *,
    condition_id_from,
    condition_id_to,
    **kwargs,
):
    """
    Difference between TWO conditional curves at the same x=k:
      y_to(k) - y_from(k)
    Useful if your 'interaction term value' changes and you want the jump.
    """
    y_from = calm_shape_value(
        calm, feat_idx, k, condition_id=condition_id_from, **kwargs
    )
    y_to = calm_shape_value(calm, feat_idx, k, condition_id=condition_id_to, **kwargs)
    return float(y_to - y_from)


def calm_list_conditions(
    calm,
    feat_idx,
    *,
    feat_labels,
    t=None,
    categorical_features=None,
    numerical_features=None,
    label_maps=None,
    beautify=False,
):
    """
    Returns a list of dicts describing the valid conditional branches for original feature feat_idx.

    condition_id can be:
      - 'pos' (0..n-1), stable for UI selection
      - or 'term_idx' (masked-gam feature index used in extract_full_shape_function)

    Output fields:
      - pos
      - term_idx
      - full_label
      - condition_only (part after '|', or '(no condition)')
    """
    calm_gam, calm_feature_names = calm.masked_gam.model, calm.new_names
    calm_feature_names = simplify_expressions(calm_feature_names)

    # scale back expressions so they're readable in original feature space (optional but recommended)
    if t is not None:
        calm_feature_names = scale_back_expressions(
            calm_feature_names,
            t,
            categorical_features,
            numerical_features,
            labels_map=label_maps,
        )

    # if you want the same pretty strings you plot (optional)
    if beautify:
        try:
            calm_feature_names = beautify_condition_latex(calm_feature_names)
            calm_feature_names = fix_latex_and_operator(calm_feature_names)
        except Exception:
            pass

    # build mapping original feature -> conditional term indices
    cond_map = build_feature_mapping(calm_feature_names, feat_labels)
    conditions = list(cond_map[feat_idx])

    rows = []
    for pos, term_idx in enumerate(conditions):
        full_label = str(calm_feature_names[term_idx])
        if "|" in full_label:
            cond_only = full_label.split("|", 1)[1].strip()
        else:
            cond_only = "(no condition)"
        rows.append(
            {
                "pos": pos,
                "term_idx": int(term_idx),
                "full_label": full_label,
                "condition_only": cond_only,
            }
        )

    return rows
