## The finder protocol

A **region finder** is any object exposing

```python
find_regions(feature, data, score_fn, *, axis_limits, feature_types, cat_limit,
             candidate_conditioning_features, feature_names, target_name) -> Partition
```

where `score_fn(mask) -> float` scores a boolean subregion (the effect passes
`heter_score(feature, mask)`). The finder owns the min-points / degeneracy guard;
the effect never sees the `BIG_M` vocabulary (design contract R12). `Best` and
`BestLevelWise` below implement this protocol; a new finder (ICE clustering,
subgroup discovery, a user `groupby`) plugs into `find_regions` with no changes
elsewhere.

## API

### ::: effector.space_partitioning.Best
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - __init__
               - find_regions

### ::: effector.space_partitioning.BestLevelWise
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - __init__
               - find_regions
