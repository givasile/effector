import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

from interpret import show
from sklearn.base import clone
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import matplotlib.ticker as ticker


def parse_condition(clause):
    match = re.match(
        r"([a-zA-Z_]\w*)\s*(<=|>=|<|>|==|!=)\s*(-?\d+\.?\d*)", clause.strip()
    )
    return match.groups() if match else None


def format_val(val):
    return (
        f"{val:.2f}"
        if isinstance(val, float) and not val.is_integer()
        else str(int(val))
    )


def simplify_and_conditions(and_conditions):
    grouped = {}
    for cond in and_conditions:
        parsed = parse_condition(cond)
        if not parsed:
            continue
        var, op, val = parsed
        val = float(val)
        if var not in grouped:
            grouped[var] = []
        grouped[var].append((op, val))

    simplified = []
    for var, conds in grouped.items():
        le = [v for op, v in conds if op == "<="]
        lt = [v for op, v in conds if op == "<"]
        ge = [v for op, v in conds if op == ">="]
        gt = [v for op, v in conds if op == ">"]
        eq = [v for op, v in conds if op == "=="]
        ne = [v for op, v in conds if op == "!="]

        if eq:
            simplified.append(f"{var} == {format_val(eq[0])}")
            continue

        if ne:
            simplified.append(f"{var} != {format_val(ne[0])}")
            continue

        if le:
            simplified.append(f"{var} <= {format_val(min(le))}")
        elif lt:
            simplified.append(f"{var} < {format_val(min(lt))}")

        if ge:
            simplified.append(f"{var} >= {format_val(max(ge))}")
        elif gt:
            simplified.append(f"{var} > {format_val(max(gt))}")

    return simplified


def simplify_expressions(expr_list):
    simplified = []
    for expr in expr_list:
        if "|" not in expr or "&" not in expr:
            simplified.append(expr)
            continue

        parts = [p.strip() for p in expr.split("|")]
        new_parts = []
        for part in parts:
            if "&" in part:
                ands = [s.strip() for s in part.split("&")]
                simplified_ands = simplify_and_conditions(ands)
                new_parts.append(" & ".join(simplified_ands))
            else:
                new_parts.append(part)
        simplified.append(" | ".join(new_parts))
    return simplified


def plot_ebm_from_calm(calm_model, X_tr, y_tr, feat_labels=None):
    model_clone = clone(calm_model.masked_gam.model)
    X_transformed, _, new_feature_names = calm_model.data_transform(
        X_tr, calm_model.tree, labels=feat_labels
    )
    X_transformed_df = pd.DataFrame(X_transformed, columns=new_feature_names)
    model_clone.fit(X_transformed_df, y_tr)

    show(model_clone.explain_global())


def fit_calm_gam_on_transformed(calm_model, X_tr, y_tr):
    model_clone = clone(calm_model.masked_gam.model)
    X_transformed, mask = calm_model.data_transform(X_tr, calm_model.tree)
    new_feature_names = calm_model.new_names
    X_transformed_df = pd.DataFrame(X_transformed, columns=new_feature_names)
    model_clone.fit(X_transformed_df, y_tr)
    return model_clone, new_feature_names


def extract_full_shape_function(model, feature):
    explanation = model.explain_global()

    if isinstance(feature, str):
        feature_idx = explanation.feature_names.index(feature)
    else:
        feature_idx = feature

    feature_data = explanation._internal_obj["specific"][feature_idx]

    if feature_data["type"] != "univariate":
        raise ValueError(f"Feature {feature} is not univariate.")

    x = np.array(feature_data["names"])
    y = np.array(feature_data["scores"])

    lower_bounds = feature_data.get("lower_bounds", None)
    upper_bounds = feature_data.get("upper_bounds", None)

    if lower_bounds is not None and upper_bounds is not None:
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

    density = feature_data.get("density", None)
    density_x = None
    density_y = None
    if density is not None:
        density_x = np.array(density["names"])
        density_y = np.array(density["scores"])

    if len(x) == len(y) + 1:
        x = (x[:-1] + x[1:]) / 2

    if density_x is not None and len(density_x) == len(density_y) + 1:
        density_x = (density_x[:-1] + density_x[1:]) / 2

    return x, y, lower_bounds, upper_bounds, density_x, density_y


def build_feature_mapping(calm_new_features, features):
    mapping = {}

    for s in calm_new_features:
        if " | " in s:
            main_feature = s.split(" | ")[0].strip()
        else:
            main_feature = s.strip()

        if main_feature not in mapping:
            mapping[main_feature] = []
        mapping[main_feature].append(s)
    idx_mapping = {}
    for k, v in mapping.items():
        idx_mapping[features.index(k)] = [calm_new_features.index(x) for x in v]
    return idx_mapping


def beautify_condition_latex(strings):
    comparison_map = {
        "==": r"=",
        "!=": r"\neq",
        "<=": r"\leq",
        ">=": r"\geq",
        "<": r"<",
        ">": r">",
    }

    beautified = []
    for s in strings:
        s = re.sub(r"\bx(\d+)\b", r"x_{\1}", s)

        for k in sorted(comparison_map.keys(), key=len, reverse=True):
            s = s.replace(k, comparison_map[k])

        s = s.replace("|", r"\mid")

        s = f"${s.strip()}$"
        beautified.append(s)

    return beautified


def get_global_local_ylim(models, model_features):
    ymin = np.inf
    ymax = -np.inf
    for model, feat_idx in zip(models, model_features):
        _, y, _, _, _, _ = extract_full_shape_function(model, feat_idx)
        ymin = min(ymin, np.min(y))
        ymax = max(ymax, np.max(y))

    if ymin == ymax:
        ylim = (ymin - 1, ymax + 1)
    else:
        padding = 0.1 * (ymax - ymin)
        ylim = (ymin - padding, ymax + padding)
    return ylim


def get_global_ylim(models, model_features):
    ymin = np.inf
    ymax = -np.inf
    for model, feat_idx in zip(models, model_features):
        explanation = model.explain_global()

        feature_data = explanation._internal_obj["specific"][feat_idx]

        if feature_data["type"] != "univariate":
            raise ValueError(f"Feature {feat_idx} is not univariate.")

        scores_range = feature_data["scores_range"]
        ymin = min(ymin, scores_range[0])
        ymax = max(ymax, scores_range[1])

    return (ymin, ymax)


def increment_latex_indices(strings):
    updated = []

    for s in strings:
        s_new = re.sub(r"x_\{(\d+)\}", lambda m: f"x_{{{int(m.group(1)) + 1}}}", s)
        updated.append(s_new)

    return updated


def fix_latex_and_operator(strings):
    return [s.replace("&", r"\; & \;") for s in strings]


# def reverse_column_transform(values, pipeline, feature_name, categorical_features, numerical_features):
#     values = np.array(values).reshape(-1, 1)

#     # Determine position in transformed array
#     transformed_feature_order = categorical_features + numerical_features
#     feature_idx = transformed_feature_order.index(feature_name)

#     # Step 1: inverse StandardScaler
#     scaler = pipeline.named_steps["std"]
#     unscaled = values * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]

#     if feature_name in categorical_features:
#         # Build a dummy array with all categorical features
#         column_transformer = pipeline.named_steps["ord"]
#         ordinal_encoder = column_transformer.named_transformers_["cat"]
#         num_samples = unscaled.shape[0]
#         num_cat_features = len(categorical_features)

#         # Create a placeholder array for all categorical features
#         dummy_cat = np.zeros((num_samples, num_cat_features))
#         cat_index = categorical_features.index(feature_name)
#         dummy_cat[:, cat_index] = unscaled.ravel()

#         # Inverse transform full array and return only the column we care about
#         cat_values = ordinal_encoder.inverse_transform(dummy_cat)
#         return cat_values[:, cat_index].astype(int)

#     else:
#         return unscaled.ravel()


def reverse_column_transform(
    values, pipeline, feature_name, categorical_features, numerical_features
):
    values = np.array(values).reshape(-1, 1)

    if feature_name in categorical_features:
        cat_index = categorical_features.index(feature_name)
        ordinal_encoder = pipeline.named_transformers_["cat"]
        dummy_cat = np.zeros((len(values), len(categorical_features)))
        dummy_cat[:, cat_index] = values.ravel()
        cat_values = ordinal_encoder.inverse_transform(dummy_cat)
        return cat_values[:, cat_index]

    elif feature_name in numerical_features:
        num_index = numerical_features.index(feature_name)
        scaler = pipeline.named_transformers_["num"]
        unscaled = values * scaler.scale_[num_index] + scaler.mean_[num_index]
        return unscaled.ravel()

    else:
        raise ValueError(
            f"Feature '{feature_name}' not found in either categorical or numerical lists."
        )


def get_grouped_cond(and_conditions):
    grouped = []
    for cond in and_conditions:
        parsed = parse_condition(cond)
        if not parsed:
            continue
        var, op, val = parsed
        val = float(val)
        grouped.append((var, op, val))
    return grouped


def scale_back_expressions(
    expr_list, t, categorical_features, numerical_features, labels_map=None
):
    scaled_back_expr = []
    for expr in expr_list:
        if "|" not in expr:
            scaled_back_expr.append(expr)
            continue

        parts = [p.strip() for p in expr.split("|")]
        scaled_parts = [parts[0]]
        for part in parts[1:]:

            ands = [s.strip() for s in part.split("&")]
            grouped = get_grouped_cond(ands)

            grouped = [
                (
                    cond[0],
                    cond[1],
                    reverse_column_transform(
                        values=[cond[2]],
                        pipeline=t,
                        feature_name=cond[0],
                        categorical_features=categorical_features,
                        numerical_features=numerical_features,
                    )[0],
                )
                for cond in grouped
            ]

            reconstructed = []
            for var, op, val in grouped:
                if labels_map and var in labels_map:
                    val = labels_map[var].get(val, val)
                if var == "temp":
                    val = rev_celsius(val)
                val_str = str(val)
                if isinstance(val, (int, np.integer)) or (
                    isinstance(val, float) and val.is_integer()
                ):
                    val_str = f"{int(val)}"
                elif isinstance(val, float):
                    val_str = f"{val:.2f}"

                reconstructed.append(f"{var} {op} {val_str}")

            scaled_parts.append(" & ".join(reconstructed))

        scaled_back_expr.append(" | ".join(scaled_parts))
    return scaled_back_expr


def rev_celsius(vals):
    return 47 * vals - 8


def plot_ga2m_for_feature(
    model,
    feat_idx,
    figsize=(6, 5),
    show_confidence=True,
    save_dir=None,
    ylim=None,
    feat_labels=None,
    display_title=True,
    ga2m_fixed_figsize=True,
    t=None,
    categorical_features=None,
    numerical_features=None,
    foi_type=None,
    label_maps=None,
    format_latex=False,
    fontsize=12,
    display_labels_map=None,
    y_display_label=None,
    smooth=True,
    smooth_window=7,
    smooth_polyorder=3,
    save_format='pdf',
):
    def prepare_edges_and_labels(values):
        n = len(values)
        edges = np.arange(n + 1)
        centers = np.arange(n) + 0.5
        return edges, centers

    x, y, lower, upper, _, _ = extract_full_shape_function(model, feat_idx)
    if t is not None:
        # if feat_labels[feat_idx] in numerical_features:
        #     x = x.astype(float)
        x = reverse_column_transform(
            values=x,
            pipeline=t,
            feature_name=feat_labels[feat_idx],
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
        if feat_labels[feat_idx] == "temp":
            x = rev_celsius(x)

    explanation = model.explain_global()
    interactions = explanation._internal_obj["specific"]
    term_features = model.term_features_

    relevant_terms = [
        (idx, term)
        for idx, term in enumerate(interactions)
        if term["type"] == "interaction" and feat_idx in term_features[idx]
    ]

    if not relevant_terms:
        print(f"No interactions found involving feature index {feat_idx}.")

    n_terms = len(relevant_terms)
    if not ga2m_fixed_figsize:
        figsize = (figsize[0], figsize[1] * (n_terms + 1))
    fig, axes = plt.subplots(n_terms + 1, 1, figsize=figsize)
    axes = np.atleast_1d(axes)

    is_categorical = (
        categorical_features is not None
        and feat_labels[feat_idx] in categorical_features
    )

    ax = axes[0]
    nbins = 5 if not ga2m_fixed_figsize else 3
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.4)

    xlabel = (
        feat_labels[feat_idx] if feat_labels is not None else rf"$x_{{{feat_idx + 1}}}$"
    )

    right_rotate = 0
    if is_categorical:
        right_rotate = 45 if len(np.unique(x)) > 10 else right_rotate

        if foi_type is not None and foi_type == "ordinal":
            x = x.astype(float).astype(int)

        if label_maps and xlabel in label_maps:
            x = [label_maps[feat_labels[feat_idx]].get(val, val) for val in x]

        n_cats = len(y)
        bar_centers = np.arange(n_cats)
        bar_width = 0.6

        ax.bar(bar_centers, y, width=bar_width, color="green", alpha=0.7)
        ax.set_xticks(bar_centers)
        ax.set_xticklabels(x, rotation=right_rotate)
    else:
        y = y - np.mean(y)
        x = x.astype(float)
        if smooth and len(x) > 3:
            sort_idx = np.argsort(x)
            xs, ys = x[sort_idx], y[sort_idx]
            w = min(smooth_window, len(ys))
            if w % 2 == 0:
                w -= 1
            p = min(smooth_polyorder, w - 1)
            ys = savgol_filter(ys.astype(float), window_length=w, polyorder=p)
            x_dense = np.linspace(xs[0], xs[-1], 300)
            y_dense = make_interp_spline(xs, ys, k=3)(x_dense)
            ax.plot(x_dense, y_dense, linewidth=2, color="green")
            if show_confidence and lower is not None and upper is not None:
                lower_sm = savgol_filter(lower[sort_idx].astype(float), window_length=w, polyorder=p)
                upper_sm = savgol_filter(upper[sort_idx].astype(float), window_length=w, polyorder=p)
                lower_dense = make_interp_spline(xs, lower_sm, k=3)(x_dense)
                upper_dense = make_interp_spline(xs, upper_sm, k=3)(x_dense)
                ax.fill_between(x_dense, lower_dense, upper_dense, alpha=0.2)
        else:
            ax.plot(x, y, linewidth=2, color="green")
            if show_confidence and lower is not None and upper is not None:
                ax.fill_between(x, lower, upper, alpha=0.2)

    xlabel = rf"${{{xlabel}}}$" if format_latex else xlabel
    xlabel_display = (
        display_labels_map.get(feat_labels[feat_idx], xlabel)
        if display_labels_map
        else xlabel
    )
    ax.set_xlabel(xlabel_display, fontsize=fontsize + 1)
    y_label = y_display_label if y_display_label is not None else "y"
    ax.set_ylabel(y_label, fontsize=fontsize + 1)
    ax.tick_params(
        axis="both",
        labelsize=fontsize,
    )
    if display_title:
        ax.set_title(r"GA$^2$M Shape function")
    if ylim is not None:
        ax.set_ylim(ylim)

    for i, (idx, term) in enumerate(relevant_terms):
        ax = axes[i + 1]
        involved_features = term_features[idx]
        bounds = term["scores_range"]
        scores = np.array(term["scores"])

        left_feat_idx, right_feat_idx = involved_features
        left_feat_values = np.array(term["left_names"], dtype=float)
        right_feat_values = np.array(term["right_names"], dtype=float)

        if left_feat_idx == feat_idx:
            left_feat_idx, right_feat_idx = right_feat_idx, left_feat_idx
            left_feat_values, right_feat_values = right_feat_values, left_feat_values
            scores = scores.T

        xlabel = (
            feat_labels[right_feat_idx]
            if feat_labels is not None
            else rf"$x_{{{right_feat_idx+1}}}$"
        )
        ylabel = (
            feat_labels[left_feat_idx]
            if feat_labels is not None
            else rf"$x_{{{left_feat_idx+1}}}$"
        )

        if t is not None:
            right_feat_values = reverse_column_transform(
                values=right_feat_values,
                pipeline=t,
                feature_name=feat_labels[right_feat_idx],
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
            if feat_labels[right_feat_idx] == "temp":
                right_feat_values = rev_celsius(right_feat_values)
            left_feat_values = reverse_column_transform(
                values=left_feat_values,
                pipeline=t,
                feature_name=feat_labels[left_feat_idx],
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )

            if feat_labels[left_feat_idx] == "temp":
                left_feat_values = rev_celsius(left_feat_values)

        left_is_cat = (
            categorical_features is not None
            and feat_labels[left_feat_idx] in categorical_features
        )
        right_is_cat = (
            categorical_features is not None
            and feat_labels[right_feat_idx] in categorical_features
        )

        if right_is_cat and label_maps and xlabel in label_maps:
            right_feat_values = [
                label_maps[xlabel].get(val, val) for val in right_feat_values
            ]

        if left_is_cat and label_maps and ylabel in label_maps:
            left_feat_values = [
                label_maps[ylabel].get(val, val) for val in left_feat_values
            ]

        if left_is_cat:
            Y_edges, Y_centers = prepare_edges_and_labels(left_feat_values)
            ax.set_yticks(Y_centers)
            ax.set_yticklabels(left_feat_values, fontsize=fontsize)
        else:
            Y_edges = (left_feat_values[:-1] + left_feat_values[1:]) / 2

        if right_is_cat:
            X_edges, X_centers = prepare_edges_and_labels(right_feat_values)
            ax.set_xticks(X_centers)
            ax.set_xticklabels(
                right_feat_values, rotation=right_rotate, fontsize=fontsize
            )
        else:
            X_edges = (right_feat_values[:-1] + right_feat_values[1:]) / 2

        if left_is_cat:
            Y_edges = np.arange(scores.shape[0] + 1)
            Y_tick_pos = np.arange(scores.shape[0]) + 0.5
            ax.set_yticks(Y_tick_pos)
            ax.set_yticklabels(left_feat_values, fontsize=fontsize)
        else:
            Y_edges = (left_feat_values[:-1] + left_feat_values[1:]) / 2
            Y_edges = np.concatenate(
                (
                    [left_feat_values[0] - (Y_edges[1] - Y_edges[0]) / 2],
                    (Y_edges[:-1] + Y_edges[1:]) / 2,
                    [left_feat_values[-1] + (Y_edges[-1] - Y_edges[-2]) / 2],
                )
            )

        if right_is_cat:
            X_edges = np.arange(scores.shape[1] + 1)
            X_tick_pos = np.arange(scores.shape[1]) + 0.5
            ax.set_xticks(X_tick_pos)
            ax.set_xticklabels(
                right_feat_values, rotation=right_rotate, fontsize=fontsize
            )
        else:
            X_edges = (right_feat_values[:-1] + right_feat_values[1:]) / 2
            X_edges = np.concatenate(
                (
                    [right_feat_values[0] - (X_edges[1] - X_edges[0]) / 2],
                    (X_edges[:-1] + X_edges[1:]) / 2,
                    [right_feat_values[-1] + (X_edges[-1] - X_edges[-2]) / 2],
                )
            )

        X, Y = np.meshgrid(X_edges, Y_edges)
        shading = "flat"
        xlabel = rf"${{{xlabel}}}$" if format_latex else xlabel
        xlabel_display = (
            display_labels_map.get(xlabel, xlabel) if display_labels_map else xlabel
        )
        ax.set_xlabel(xlabel_display, fontsize=fontsize + 1)
        ylabel = rf"${{{ylabel}}}$" if format_latex else ylabel
        y_label_display = (
            display_labels_map.get(ylabel, ylabel) if display_labels_map else ylabel
        )
        ax.set_ylabel(y_label_display, fontsize=fontsize + 1)
        if display_title:
            ax.set_title(f"{xlabel_display} & {y_label_display}")

        ax.tick_params(
            axis="both",
            labelsize=fontsize,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pcm = ax.pcolormesh(
            X,
            Y,
            scores,
            shading=shading,
            cmap="plasma",
            vmin=bounds[0],
            vmax=bounds[1],
            rasterized=True,
        )
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.ax.tick_params(labelsize=fontsize)

    fig.tight_layout()
    if save_dir:
        filename = os.path.join(save_dir, f"ga2m_{feat_labels[feat_idx]}.{save_format}")
        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            format=save_format,
        )
        print(f"Saved subplot heatmap to {filename}")
    plt.show()


def plot_shape_functions_multiple(
    models,
    model_features,
    feature_label,
    model_labels=None,
    colors=None,
    title=None,
    save_path=None,
    figsize=(8, 6),
    ylim=None,
    show_confidence=False,
    t=None,
    categorical_features=None,
    numerical_features=None,
    foi_type=None,
    label_maps=None,
    format_latex=False,
    fontsize=12,
    display_labels_map=None,
    y_display_label=None,
    smooth=True,
    smooth_window=7,
    smooth_polyorder=3,
    save_format='pdf',
):

    fig, ax = plt.subplots(figsize=figsize)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.4)

    if colors is None:
        colors = plt.cm.tab10.colors

    is_categorical = (
        categorical_features is not None and feature_label in categorical_features
    )

    if is_categorical:
        all_categories = set()
        y_by_model = []

        for i, model in enumerate(models):
            x, y, _, _, _, _ = extract_full_shape_function(model, model_features[i])

            if t is not None:
                x = reverse_column_transform(
                    values=x,
                    pipeline=t,
                    feature_name=feature_label,
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                )
            if foi_type is not None and foi_type == "ordinal":
                x = x.astype(float).astype(int)

            all_categories.update(x)
            y_by_model.append((x, y))

        all_categories = sorted(all_categories)
        category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
        n_categories = len(all_categories)

        bar_centers = np.arange(n_categories)
        n_models = len(models)
        bar_width = 0.8 / n_models

        for i, (cats, y) in enumerate(y_by_model):
            aligned_y = np.zeros(n_categories)
            aligned_y[:] = np.nan

            for cat, val in zip(cats, y):
                idx = category_to_idx[cat]
                aligned_y[idx] = val

            offset = (i - n_models / 2) * bar_width + bar_width / 2
            positions = bar_centers + offset
            color = colors[i % len(colors)]
            label = model_labels[i] if model_labels is not None else None

            ax.bar(
                positions,
                aligned_y,
                width=bar_width,
                label=label,
                color=color,
                alpha=0.8,
            )

        ax.set_xticks(bar_centers)
        if label_maps and feature_label in label_maps:
            all_categories = [
                label_maps[feature_label].get(val, val) for val in all_categories
            ]

        rotation = 45 if len(all_categories) > 10 else 0
        ax.set_xticklabels(all_categories, rotation=rotation, fontsize=fontsize)

    else:
        for i, model in enumerate(models):
            x, y, lower, upper, _, _ = extract_full_shape_function(
                model, model_features[i]
            )
            y = y - np.mean(y)
            x = x.astype(float)

            if t is not None:
                x = reverse_column_transform(
                    values=x,
                    pipeline=t,
                    feature_name=feature_label,
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                )
                if feature_label == "temp":
                    x = rev_celsius(x)

            color = colors[i % len(colors)]
            label = model_labels[i] if model_labels is not None else None

            if smooth and len(x) > 3:
                sort_idx = np.argsort(x)
                xs, ys = x[sort_idx], y[sort_idx]
                w = min(smooth_window, len(ys))
                if w % 2 == 0:
                    w -= 1
                p = min(smooth_polyorder, w - 1)
                ys = savgol_filter(ys.astype(float), window_length=w, polyorder=p)
                x_dense = np.linspace(xs[0], xs[-1], 300)
                y_dense = make_interp_spline(xs, ys, k=3)(x_dense)
                ax.plot(x_dense, y_dense, label=label, color=color, linewidth=2)
                if show_confidence and lower is not None and upper is not None:
                    lower_sm = savgol_filter(lower[sort_idx].astype(float), window_length=w, polyorder=p)
                    upper_sm = savgol_filter(upper[sort_idx].astype(float), window_length=w, polyorder=p)
                    lower_dense = make_interp_spline(xs, lower_sm, k=3)(x_dense)
                    upper_dense = make_interp_spline(xs, upper_sm, k=3)(x_dense)
                    ax.fill_between(x_dense, lower_dense, upper_dense, color=color, alpha=0.2)
            else:
                ax.plot(x, y, label=label, color=color, linewidth=2)
                if show_confidence and lower is not None and upper is not None:
                    ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    xlabel = rf"${{{feature_label}}}$" if format_latex else feature_label
    xlabel = (
        display_labels_map.get(feature_label, xlabel) if display_labels_map else xlabel
    )
    ax.set_xlabel(xlabel, fontsize=fontsize + 1)
    y_label = y_display_label if y_display_label is not None else "y"
    ax.set_ylabel(y_label, fontsize=fontsize + 1)
    ax.tick_params(
        axis="both",
        labelsize=fontsize,
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    if title:
        title = rf"${{{title}}}$" if format_latex else title
        plt.title(title)

    ax.legend(
        loc="best",
        fontsize=fontsize,
        framealpha=0.4,
    )
    fig.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            format=save_format,
        )
        print(f"Saved combined shape function plot to {save_path}")
    else:
        plt.show()


def plot_calm_gam_shape_function(
    gam,
    calm,
    ga2m,
    feat_idx,
    figsize=(10, 6),
    show_confidence=False,
    save_dir=None,
    feat_labels=None,
    display_title=True,
    ga2m_fixed_figsize=True,
    t=None,
    categorical_features=None,
    numerical_features=None,
    foi_type=None,
    label_maps=None,
    ylim_local=False,
    format_latex=False,
    fontsize=12,
    display_labels_map=None,
    y_display_label=None,
    smooth=True,
    smooth_window=7,
    smooth_polyorder=3,
    save_format='pdf',
    ga2m_figsize=None,
):

    if calm is not None:
        calm_gam, calm_feature_names = calm.masked_gam.model, calm.new_names
        calm_feature_names = simplify_expressions(calm_feature_names)
        if t is not None:
            calm_feature_names = scale_back_expressions(
                calm_feature_names,
                t,
                categorical_features,
                numerical_features,
                labels_map=label_maps,
            )
    # feat_labels_is_none = feat_labels is None
    feat_labels = (
        [rf"$x_{{{i}}}$" for i in range(len(calm_feature_names))]
        if feat_labels is None
        else feat_labels
    )

    if calm is not None:
        calm_features_conditions_names_map = build_feature_mapping(
            calm_feature_names, feat_labels
        )
        calm_feature_names = beautify_condition_latex(calm_feature_names)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    colors = ["blue", "orange", "purple", "cyan"]

    all_models = [gam] if gam is not None else []
    all_feats = [feat_idx] if gam is not None else []

    if calm is not None:
        calm_feature_names = fix_latex_and_operator(calm_feature_names)
        conditions = calm_features_conditions_names_map[feat_idx]
        calm_colors = colors[: len(conditions)]

        all_models += [calm_gam] * len(conditions)
        all_feats += list(conditions)

    ylim = (
        get_global_ylim(all_models, all_feats)
        if not ylim_local
        else get_global_local_ylim(all_models, all_feats)
    )

    feature_label = feat_labels[feat_idx]

    if gam is not None:
        plot_shape_functions_multiple(
            models=[gam],
            colors=["green"],
            model_features=[feat_idx],
            feature_label=feature_label,
            title=f"GAM Shape function" if display_title else None,
            show_confidence=show_confidence,
            figsize=figsize,
            ylim=ylim,
            save_path=(
                os.path.join(save_dir, f"gam_{feature_label}.{save_format}") if save_dir else None
            ),
            t=t,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            foi_type=foi_type,
            label_maps=label_maps,
            format_latex=format_latex,
            fontsize=fontsize,
            display_labels_map=display_labels_map,
            y_display_label=y_display_label,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            save_format=save_format,
        )

    if calm is not None:
        # if len(conditions) > 1:
        plot_shape_functions_multiple(
            models=[calm_gam] * len(conditions),
            model_features=conditions,
            feature_label=feature_label,
            model_labels=np.array(calm_feature_names)[conditions],
            title=f"CALM Shape functions" if display_title else None,
            show_confidence=show_confidence,
            figsize=figsize,
            colors=calm_colors,
            ylim=ylim,
            save_path=(
                os.path.join(save_dir, f"calm_{feature_label}.{save_format}")
                if save_dir
                else None
            ),
            t=t,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            foi_type=foi_type,
            label_maps=label_maps,
            format_latex=format_latex,
            fontsize=fontsize,
            display_labels_map=display_labels_map,
            y_display_label=y_display_label,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            save_format=save_format,
        )

    if ga2m is not None:
        plot_ga2m_for_feature(
            ga2m,
            feat_idx=feat_idx,
            figsize=ga2m_figsize if ga2m_figsize is not None else figsize,
            save_dir=save_dir,
            ylim=ylim,
            feat_labels=feat_labels,
            display_title=display_title,
            show_confidence=show_confidence,
            ga2m_fixed_figsize=ga2m_fixed_figsize,
            t=t,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            foi_type=foi_type,
            label_maps=label_maps,
            format_latex=format_latex,
            fontsize=fontsize,
            display_labels_map=display_labels_map,
            y_display_label=y_display_label,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            save_format=save_format,
        )
