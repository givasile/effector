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
  EXPLAINED VARIANCE
  ────────────────────────────────────────────────────────────────────────
    step         split on                 solo     ΔR²      R²       heter
    ──────────────────────────────────────────────────────────────────────
    GAM          (all features global)       —       —   72.3%           —
  + hr           temp, workingday, yr   +18.3%  +18.3%   90.6% 0.47 → 0.26
  + temp         hr, hum                 +2.4%   +1.4%   92.0% 0.22 → 0.19
    ──────────────────────────────────────────────────────────────────────
    FINAL                                                92.0%

  REJECTED SPLITS                                            min gain 1.0%
  ────────────────────────────────────────────────────────────────────────
    feature      split on                 solo     ΔR²    reason
    ──────────────────────────────────────────────────────────────────────
  ✗ yr           hr, workingday          +2.7%   -0.8%    redundant
  ✗ weekday      hr, temp, yr            +0.4%   +0.2%    below threshold
  ✗ workingday   hr, yr                  +6.2%   -4.8%    redundant
  ✗ hum          hr, temp                +2.2%   +0.2%    below threshold

    ✗ redundant: it would explain variance on its own (see solo),
      but the accepted splits already account for it.
```

These are the same two tables the report's `.show()` prints; here they are
computed live. `chain.show(ascii=True)` draws them in plain ASCII for
terminals that mangle box-drawing characters.

Six candidates went in; two came out. `workingday`'s split genuinely
resolves spread (see its `solo`), yet it is **redundant**: `hr`'s split
already conditions on it, so applying it would double count (`-4.8%`). This
is [the rejected splits](./../report/rejected_splits.md) story.

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
| `chain.show()` | the ledger tables above |
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
