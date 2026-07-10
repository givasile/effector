"""Label-free explained variance — how much of the model an additive surrogate keeps.

The units contract puts `importance`/`heter_score` in output units; this module
adds the one denominator that turns the report into variance accounting:
`Var(f̂(X))`. Two surrogates are read off the fitted effect's cached curves —
zero extra model calls beyond the single `f̂(X)` pass — and scored as

    R² = 1 − E[(f̂ − g)²] / Var(f̂)

- the **GAM surrogate**: `g(x) = c + Σ_j centered global curve_j(x_j)`;
- the **regional (CALM) surrogate**: each partitioned feature's curve becomes
  leaf-conditional under *its own* partition (non-partitioned features keep
  their global curve), with the constant term fitted jointly by least squares
  on region-indicator dummies — the joint solve assigns each between-region
  mean shift once, so overlapping partitions cannot double-count it. The
  reported combined figure greedily selects which partitions to apply (see
  `summarize`), since overlapping partitions can also double-count interaction
  deviations inside the curves, which no offset fit can repair.

This is faithfulness/distillation only: no ground-truth labels ever (the
no-`y` constitution), and evaluation happens at the effect's own training
data, inside `axis_limits` by construction. R² can be negative (correlated
features make additive curves double-count shared variance) — it is reported
as-is, never clamped. The regional R² is not guaranteed ≥ the global one
(only offsets are least-squares-fitted, not the curves), though it is in
practice.

Curves are read through the model-free summary path (`_summary` →
`_eval_payload`, the same read `_importance` uses) — **not** `effect.eval`,
which on (d-)PDP grows the position store / retouches ICE with model calls.

This module is a **leaf**: numpy only; it duck-types the effect
(`._summary`/`._eval_payload`/`.data`/`.model`/`._y_pred`) and `Partition`
(`.leaves`/`.mask(idx)`, `leaf.rule.conditions`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def supports(effect) -> bool:
    """Whether an additive output-scale surrogate makes sense for the method.

    Derivative-scale methods (DerPDP) cache `∂f/∂x_j` curves — summing those
    does not approximate `f̂`, so the whole section is skipped for them.
    """
    return not getattr(effect, "IS_DERIVATIVE", False)


def _centered_contribution(
    effect, feature: int, mask: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """The (masked) rows' centered mean-effect contribution of one feature.

    Reads the summary payload at the raw data values of `feature` over the
    selected rows and centers it there — the data-weighted zero-mean term of
    the additive surrogate. Returns `None` when the axis is degenerate inside
    the region (single level / constant feature → the summarizer raises
    ValueError): the feature is constant there, its centered effect is 0.
    """
    rows = slice(None) if mask is None else mask
    xs = effect.data[rows, feature]
    try:
        params = effect._summary(feature, mask)
        mu = np.asarray(effect._eval_payload(feature, params, xs), dtype=float)
    except ValueError:
        return None
    return mu - mu.mean()


def surrogate_r2(effect, fx, partitions: dict, features: list) -> float:
    """R² of the additive surrogate with the given partitions applied.

    `partitions` maps feature index → bound multi-leaf `Partition`; `{}` gives
    the plain GAM. Each partitioned feature contributes its leaf-conditional
    curves (region-locally centered); every other feature its global centered
    curve; the piecewise-constant offset is fitted to what the curves leave
    over (see `_fit_offsets`).

    Args:
        effect: a fitted global effect (all `features` fitted).
        fx: `(N,)` model predictions on `effect.data`.
        partitions: `{feature: Partition}` — bound, multi-leaf.
        features: feature indices the surrogate sums over.

    Returns:
        `1 − mean((f̂ − g)²) / var(f̂)` — may be negative, never clamped.
    """
    # column-vector model outputs ((N,1), e.g. a raw keras forward) would
    # broadcast `fx - g` to (N,N) and silently corrupt the R²
    fx = np.asarray(fx, dtype=float).reshape(-1)
    n = effect.data.shape[0]
    contrib = np.zeros(n)
    for j in features:
        part = partitions.get(j)
        if part is None:
            c = _centered_contribution(effect, j, None)
            if c is not None:
                contrib += c
        else:
            for leaf in part.leaves:
                m = part.mask(leaf.idx)
                c = _centered_contribution(effect, j, m)
                if c is not None:
                    contrib[m] += c
    g = contrib + _fit_offsets(fx - contrib, partitions, n)
    return 1.0 - float(np.mean((fx - g) ** 2)) / float(np.var(fx))


def _fit_offsets(residual: np.ndarray, partitions: dict, n: int) -> np.ndarray:
    """The surrogate's piecewise-constant term, fitted to the curve residual.

    No partitions → the global mean. Otherwise a joint least-squares fit on
    region-indicator dummies: a global intercept plus, per partition, one
    column per leaf except the last (drop-one — each partition's indicators
    sum to the intercept). With a single partition this reduces exactly to
    per-leaf means; with several it assigns shared mean shifts once — the
    double-counting killer. Identical splits across partitions leave the
    design rank-deficient, but `lstsq`'s minimum-norm solution still yields
    the unique projection, so the fitted values (hence R²) are well-defined.
    """
    if not partitions:
        return np.full(n, residual.mean())
    cols = [np.ones(n)]
    for part in partitions.values():
        for leaf in part.leaves[:-1]:
            cols.append(part.mask(leaf.idx).astype(float))
    Z = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Z, residual, rcond=None)
    return Z @ beta


def summarize(effect, partitions: dict, features: list) -> Optional[dict]:
    """The Report's explained-variance payload — plain floats, JSON-safe.

    Computes `f̂(X)` through `effect._y_pred` (filling it on first use — the
    one model call this module ever costs; plots reuse it). Returns `None`
    when the method is unsupported (derivative-scale) or `Var(f̂) == 0`.

    Args:
        effect: a fitted global effect.
        partitions: `{feature: Partition}` — bound; single-leaf ones ignored.
        features: feature indices the surrogates sum over.

    Returns:
        ``{"gam_r2": float, "regional_r2": float, "gains": [{"feature",
        "name", "on", "n_regions", "delta_r2", "in_combined"}, ...]}`` —
        gains in the given partition order, each measured alone on top of the
        GAM. They need not sum to `regional_r2 − gam_r2`: partitions sharing
        interaction variance each recover part of the same pot.

    `regional_r2` is a **greedy forward selection** over the partitions: add
    the split with the largest R² improvement, keep adding while R² improves
    (`in_combined` marks the selected ones). Applying every partition blindly
    can score *worse* than the best single one — overlapping partitions
    double-count interaction deviations in the leaf-conditional curves
    themselves, which the joint offsets cannot repair — while greedy selection
    guarantees `regional_r2 ≥ gam_r2` and ≥ every single-split figure. It is
    also the honest reading of `find_regions`' own greedy objective.
    """
    if not supports(effect):
        return None
    if effect._y_pred is None:
        effect._y_pred = np.asarray(effect.model(effect.data))
    fx = np.asarray(effect._y_pred, dtype=float).reshape(-1)
    if not np.var(fx) > 0:
        return None

    parts = {j: p for j, p in partitions.items() if len(p.leaves) > 1}
    gam_r2 = surrogate_r2(effect, fx, {}, features)

    single_r2 = {j: surrogate_r2(effect, fx, {j: p}, features) for j, p in parts.items()}

    selected: dict = {}
    regional_r2 = gam_r2
    remaining = dict(parts)
    while remaining:
        scored = {
            j: surrogate_r2(effect, fx, {**selected, j: p}, features)
            for j, p in remaining.items()
        }
        best = max(scored, key=lambda j: scored[j])
        if scored[best] <= regional_r2:
            break
        selected[best] = remaining.pop(best)
        regional_r2 = scored[best]

    gains = []
    for j, part in parts.items():
        conditioning = sorted(
            {
                effect.feature_names[k]
                for leaf in part.leaves
                for k in leaf.rule.conditions
            }
        )
        gains.append(
            {
                "feature": int(j),
                "name": effect.feature_names[j],
                "on": ", ".join(conditioning),
                "n_regions": len(part.leaves),
                "delta_r2": single_r2[j] - gam_r2,
                "in_combined": j in selected,
            }
        )
    return {"gam_r2": gam_r2, "regional_r2": regional_r2, "gains": gains}
