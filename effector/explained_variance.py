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
(`.leaves`/`.mask(idx)`, `leaf.rule.conditions`, and — for the ledger's
heterogeneity column — `part[0].heterogeneity`, `leaf.heterogeneity`,
`leaf.nof_instances`).
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


def _heter_pair(part) -> tuple:
    """A partition's (root, weighted-leaf) heterogeneity, or `(None, None)`.

    The drop is a *static* property of the split — the feature's local effects
    against its own regional curves — so unlike the R² marginals it does not
    depend on which other splits are applied. `None` when the partition
    carries no finite heterogeneity numbers (e.g. hand-built rules).
    """
    try:
        before = float(part[0].heterogeneity)
        w = np.array([leaf.nof_instances for leaf in part.leaves], dtype=float)
        h = np.array([leaf.heterogeneity for leaf in part.leaves], dtype=float)
        after = float((w * h).sum() / w.sum())
    except (TypeError, AttributeError, IndexError):
        return None, None
    if not (np.isfinite(before) and np.isfinite(after)):
        return None, None
    return before, after


def _check_cap(name: str, value) -> Optional[int]:
    """A selection cap: `None` (no cap) or an integer >= 0."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(
            f"select_regions: {name} must be None or an integer >= 0; got {value!r}"
        )
    return int(value)


def _conditioning_features(part) -> set:
    """The features a partition's rules condition on — one conditional
    interaction each (`x1 | x2` and `x2 | x1` are two)."""
    return {k for leaf in part.leaves for k in leaf.rule.conditions}


def _effective_min_gain(min_gain, relative_gain, r2):
    """The hybrid acceptance threshold for one greedy round.

    Absolute rule when `relative_gain` is None; otherwise
    ``min(min_gain, relative_gain * (1 - r2))`` — a split must add `min_gain`
    of total variance *or* `relative_gain` of the still-unexplained variance,
    whichever is less. The `min` also handles r2 <= 0 (1 - r2 > 1: the
    absolute rule wins, so low-R² chains are gated exactly as before; only
    near-saturated chains get the relaxed bar).
    """
    if relative_gain is None:
        return min_gain
    return min(min_gain, relative_gain * (1.0 - r2))


def select(
    effect,
    partitions: dict,
    features: list,
    min_gain: float = 0.01,
    target: Optional[np.ndarray] = None,
    relative_gain: Optional[float] = None,
    max_conditional_interactions: Optional[int] = None,
    max_partitions: Optional[int] = None,
):
    """Greedy forward selection of partitions → a `CalmSequence` of snapshots.

    Computes `f̂(X)` through `effect._y_pred` (filling it on first use — the
    one model call this module ever costs; plots reuse it). Passing `target`
    replaces `f̂(X)` as the selection objective: the surrogate's offsets are
    fitted to and its R² measured against that vector instead (e.g. ground
    truth labels — "which splits *predict* better" rather than "which splits
    explain the teacher"). The curves themselves always come from the effect.

    The chain is a **decision sequence**: each round applies the split with
    the largest R² gain *measured on top of the splits already applied*, and
    stops when no remaining split adds at least `min_gain`. Sequential
    marginals are the intuitive currency — the stage gains sum exactly to
    `regional_r2 − gam_r2`, and a redundant split shows ~0 instead of a
    counterfactual solo figure. Applying every partition blindly can score
    *worse* than the best single one (overlapping partitions double-count
    interaction deviations in the leaf-conditional curves, which the joint
    offsets cannot repair); the greedy pass guarantees `regional_r2 ≥ gam_r2`
    and mirrors `find_regions`' own greedy objective.

    Args:
        effect: a fitted global effect.
        partitions: `{feature_index: Partition}` — bound; single-leaf ones
            ignored.
        features: feature indices the surrogates sum over.
        min_gain: smallest R² marginal worth a stage (default 1%) — below
            it a split is skipped as `below_threshold`.
        relative_gain: optional hybrid gate — when set, a round's threshold
            becomes ``min(min_gain, relative_gain * (1 - cum_r2))``, so a
            near-saturated chain (little variance left) still accepts a split
            that explains at least `relative_gain` of what remains. Never
            stricter than `min_gain` alone; `None` (default) reproduces the
            absolute rule exactly.
        max_conditional_interactions: optional cap on the conditional
            interactions of the selected partitions — one per (feature,
            conditioning feature) pair, so `x1 | x2` and `x2 | x1` are two.
            A partition is accepted whole, so the sequence stops before the
            round that would exceed the cap: the capped chain is a prefix of
            the uncapped one. `None` (default) is no cap; 0 keeps the GAM.
        max_partitions: optional cap on the number of selected partitions
            (split features); the sequence stops once it is reached. `None`
            (default) is no cap.

    Returns:
        a `CalmSequence` — `[GAM, calm1, ...]` with each stage carrying
        `{"feature", "name", "on", "n_regions", "delta_r2", "cum_r2",
        "solo_delta_r2", "heter_before", "heter_after"}` (`delta_r2` is the
        sequential marginal, `cum_r2` the running R²; `solo_delta_r2` — the
        split alone on top of the GAM — is a counterfactual kept for
        programmatic consumers, it does not add up across splits). Rejected
        splits land in `.skipped` with their marginal on top of the final
        selection and a `reason`: `"redundant"` (adds ~nothing — its variance
        is already explained), `"below_threshold"` (real but < `min_gain`),
        or — for a split that clears the gate when a cap stops the sequence —
        `"max_conditional_interactions"` / `"max_partitions"`.

    Raises:
        ValueError: the method is derivative-scale (see `supports`),
            `Var(f̂) == 0` — an additive output-scale surrogate is undefined —
            or a cap is not `None` or an integer >= 0.
    """
    from effector.calm import CALM, CalmSequence  # leaf importing a leaf

    max_conditional_interactions = _check_cap(
        "max_conditional_interactions", max_conditional_interactions
    )
    max_partitions = _check_cap("max_partitions", max_partitions)

    if not supports(effect):
        raise ValueError(
            "select_regions: explained-variance selection is undefined for "
            "derivative-scale methods (the cached curves are ∂f/∂x, summing "
            "them does not approximate f̂)."
        )
    if target is not None:
        fx = np.asarray(target, dtype=float).reshape(-1)
        if fx.shape[0] != effect.data.shape[0]:
            raise ValueError(
                f"select_regions: target has {fx.shape[0]} rows but the "
                f"effect's data has {effect.data.shape[0]}"
            )
    else:
        if effect._y_pred is None:
            effect._y_pred = np.asarray(effect.model(effect.data))
        fx = np.asarray(effect._y_pred, dtype=float).reshape(-1)
    if not np.var(fx) > 0:
        raise ValueError(
            "select_regions: Var(target) == 0 — explained variance is "
            "undefined for a constant objective."
        )

    parts = {j: p for j, p in partitions.items() if len(p.leaves) > 1}
    gam_r2 = surrogate_r2(effect, fx, {}, features)

    single_r2 = {
        j: surrogate_r2(effect, fx, {j: p}, features) for j, p in parts.items()
    }

    def _info(j, part):
        conditioning = sorted(
            effect.feature_names[k] for k in _conditioning_features(part)
        )
        before, after = _heter_pair(part)
        return {
            "feature": int(j),
            "name": effect.feature_names[j],
            "on": ", ".join(conditioning),
            "n_regions": len(part.leaves),
            "solo_delta_r2": single_r2[j] - gam_r2,
            "heter_before": before,
            "heter_after": after,
        }

    calms = [CALM.from_effect(effect, {}, r2=gam_r2, index=0)]
    skipped: list = []
    selected: dict = {}
    regional_r2 = gam_r2
    nof_interactions = 0
    remaining = dict(parts)
    while remaining:
        scored = {
            j: surrogate_r2(effect, fx, {**selected, j: p}, features)
            for j, p in remaining.items()
        }
        best = max(scored, key=lambda j: scored[j])
        gain = scored[best] - regional_r2
        threshold = _effective_min_gain(min_gain, relative_gain, regional_r2)
        # a partition can clear the gate yet carry a leaf with no instances
        # (the leaf contributes nothing to the surrogate, so its R² is
        # finite) — snapshotting such a partition is undefined. Latent under
        # the absolute gate: a candidate like this was never accepted (it
        # would have crashed at CALM.from_effect); the relaxed gate can reach
        # one, so it is skipped explicitly instead.
        if gain > 0 and gain >= threshold:
            part_best = remaining[best]
            if any(not np.any(part_best.mask(leaf.idx)) for leaf in part_best.leaves):
                skipped.append(
                    {
                        **_info(best, part_best),
                        "delta_r2": gain,
                        "reason": "empty_region",
                    }
                )
                remaining.pop(best)
                continue
        if gain <= 0 or gain < threshold:
            for j, p in remaining.items():
                marginal = scored[j] - regional_r2
                skipped.append(
                    {
                        **_info(j, p),
                        "delta_r2": marginal,
                        "reason": (
                            "below_threshold" if marginal > 1e-9 else "redundant"
                        ),
                    }
                )
            break
        # the caps: a partition is accepted whole, so the sequence stops
        # before the round that would exceed one
        nof_best = len(_conditioning_features(remaining[best]))
        capped = None
        if max_partitions is not None and len(selected) >= max_partitions:
            capped = "max_partitions"
        elif (
            max_conditional_interactions is not None
            and nof_interactions + nof_best > max_conditional_interactions
        ):
            capped = "max_conditional_interactions"
        if capped is not None:
            for j, p in remaining.items():
                marginal = scored[j] - regional_r2
                if marginal > 0 and marginal >= threshold:
                    reason = capped
                else:
                    reason = "below_threshold" if marginal > 1e-9 else "redundant"
                skipped.append({**_info(j, p), "delta_r2": marginal, "reason": reason})
            break
        selected[best] = remaining.pop(best)
        nof_interactions += nof_best
        regional_r2 = scored[best]
        stage = {
            **_info(best, selected[best]),
            "delta_r2": gain,
            "cum_r2": regional_r2,
        }
        calms.append(
            CALM.from_effect(
                effect,
                dict(selected),
                r2=regional_r2,
                index=len(calms),
                stage=stage,
            )
        )

    return CalmSequence(
        calms,
        skipped=skipped,
        min_gain=min_gain,
        relative_gain=relative_gain,
        max_conditional_interactions=max_conditional_interactions,
        max_partitions=max_partitions,
    )


def summarize(
    effect, partitions: dict, features: list, min_gain: float = 0.01
) -> Optional[dict]:
    """The flat decision-sequence payload — plain floats, JSON-safe.

    A thin serializer over `select`: same greedy semantics, but returns
    ``{"gam_r2", "regional_r2", "min_gain", "stages", "skipped"}`` and `None`
    (instead of raising) when the method is unsupported (derivative-scale)
    or `Var(f̂) == 0`.
    """
    if not supports(effect):
        return None
    if effect._y_pred is None:
        effect._y_pred = np.asarray(effect.model(effect.data))
    if not np.var(np.asarray(effect._y_pred, dtype=float)) > 0:
        return None
    chain = select(effect, partitions, features, min_gain)
    d = chain.to_dict()
    return {k: d[k] for k in ("gam_r2", "regional_r2", "min_gain", "stages", "skipped")}
