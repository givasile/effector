## Summary

`select_regions` decides, across features, which of the splits `find_regions`
found actually explain the model. It returns a `CalmSequence` — the chain
`[GAM, calm1, ...]` produced by greedy forward selection: each step applies
the split with the largest explained-variance gain, and stops when no
remaining split adds at least `min_r2_gain` of `Var(f̂)`.

Each snapshot is a **CALM** (Conditional Additive Local Model): the global
read plus the partitions accepted so far, with its surrogate R² stamped on
it. Like `Partition`, both are **values**, not stored state (design contract
R12): they serialize with `to_dict()` and re-attach with `bind(effect)`.

---

## Usage

```python
pdp = effector.PDP(data=X, model=predict)

parts = pdp.find_regions(features="heterogeneous")   # candidates, per feature
chain = pdp.select_regions(partitions=parts)          # which ones earn their keep

chain.show()          # the explained-variance ledger, as tables
chain.gam_r2          # R² of the pure-GAM read
chain.regional_r2     # R² after the accepted splits
chain.skipped         # rejected splits, with reasons

calm = chain.final    # the selected snapshot — the report's §2
calm.importances()    # per feature, weighted mean over subregions
calm.plot_triage()    # the importance × heterogeneity plane, at this snapshot
```

`partitions=` is optional: `select_regions()` runs the search itself, with
the same `features` / `finder` / `candidate_conditioning_features` arguments
as `find_regions`. The method itself is documented per engine in
[Global effect](./api_global.md); the walkthrough lives in
[the `select_regions` guide](./../quickstart/interactive/select_regions.md).

!!! warning "Derivative-scale methods cannot play"

    `select_regions` raises `ValueError` on `DerPDP`: its curves live on the
    derivative scale, where summing does not approximate `f̂`.

## API

### ::: effector.global_effect.GlobalEffectBase.select_regions
       options:
         show_root_heading: True
         show_symbol_type_toc: True

### ::: effector.calm.CalmSequence
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        members:
          - gam
          - final
          - gam_r2
          - regional_r2
          - stages
          - show
          - bind
          - to_dict
          - from_dict

### ::: effector.calm.CALM
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        members:
          - from_effect
          - is_gam
          - features
          - importance
          - importances
          - heter_score
          - heter_scores
          - plot_triage
          - show
          - bind
          - to_dict
          - from_dict
