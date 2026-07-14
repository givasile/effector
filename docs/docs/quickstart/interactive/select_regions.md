---
title: select_regions
---

???+ success "Description"

    Part of the [interactive API guide](./../interactive_api.md):
    `select_regions` decides across features which found splits actually
    explain the model, and returns the CALM chain: the value behind the
    report's explained variance ledger.

???+ note "Reading time"

	Approx. 5' to read.

## Which splits earn their keep

[`find_regions`](./find_regions.md) proposes one candidate partition **per
feature**; each resolves its own feature's heterogeneity. `select_regions`
asks the cross feature question: starting from the GAM (every feature
global), each round applies the split with the largest explained variance
gain, measured on top of the splits already applied, and stops when no
remaining split adds at least `min_r2_gain`.

On the [bike sharing rig](./construct_and_fit.md):

```python
parts = pdp.find_regions(features="heterogeneous")
chain = pdp.select_regions(partitions=parts)
chain.show()
```

```text
GAM: R2 = 72.3%
  + hr regions (on temp, workingday, yr) -> R2 = 90.6% (+18.3%)
  + temp regions (on hr, hum) -> R2 = 92.0% (+1.4%)
  x yr regions skipped (redundant, -0.8%)
  x weekday regions skipped (below_threshold, +0.2%)
  x workingday regions skipped (redundant, -4.8%)
  x hum regions skipped (below_threshold, +0.2%)
```

Six candidates went in; two came out. `workingday`'s split genuinely
resolves spread, yet it is **redundant**: `hr`'s split already conditions on
it, so applying it would double count (`-4.8%`). This is
[the rejected splits](./../report/rejected_splits.md) story, computed live.

👉 `partitions=` is optional: `pdp.select_regions()` runs the search itself,
with the same `features` / `finder` / `candidate_conditioning_features`
arguments as `find_regions`, plus `min_r2_gain` (default `0.01`: a split
must buy 1% of `Var(f̂)`).

## The chain is a value

`select_regions` returns a `CalmSequence`: the list `[GAM, calm1, ...]`, one
snapshot per accepted split, R² non decreasing along it.

| access | what it is |
|---|---|
| `chain[0]` / `chain.gam` | the GAM snapshot, no partitions |
| `chain.final` | the last snapshot: what the report renders as §2 |
| `chain.gam_r2` / `chain.regional_r2` | the headline numbers |
| `chain.stages` / `chain.skipped` | accepted and rejected splits, with reasons |
| `chain.show()` | the trace above |
| `chain.to_dict()` / `bind(effect)` | serialize / re attach |

```python
chain.final.r2    # 0.92
```

Each snapshot is a **CALM** (Conditional Additive Local Model): the global
read plus the partitions accepted so far. It scores itself:

```python
calm = chain.final
calm.importances()               # per feature, weighted mean over subregions
calm.importance("hr", per_region=True)   # one value per leaf
calm.heter_scores()              # same, for the spread
calm.plot_triage()               # the plane, at this snapshot
calm.is_gam                      # False once a split is applied
```

???+ note "One prediction pass"

    Beyond `fit`, the only model touch is one `f̂(X)` pass for the variance
    denominator, cached on the engine. The search, the scoring, and every
    snapshot's summaries are model free.

???+ warning "Derivative scale methods cannot play"

    `select_regions` raises `ValueError` on `DerPDP`: its curves live on the
    derivative scale, where summing does not approximate `f̂`, so an
    explained variance surrogate is undefined. Same for a constant model
    (`Var(f̂) == 0`).

## This is the report's ledger

Freeze this chain, print it as a table, draw it as a bar: that is exactly
[the explained variance ledger](./../report/explained_variance.md).
`effector.explain` runs `find_regions` + `select_regions` for you and stores
`chain.to_dict()` in `report.explained_variance`; here you hold the live
value, snapshot by snapshot.

---

## Where to next

- [`compare` and `plot_triage`](./compare_and_triage.md): the last page
- [The interactive API](./../interactive_api.md): back to the guide's map
- [The explained variance ledger](./../report/explained_variance.md): this
  chain, frozen
