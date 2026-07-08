# Triage & comparison

## Summary

Two free functions stand *above* the engines. They take fitted effect
objects, query their public verbs, and draw — they compute nothing themselves
and store nothing.

**`plot_triage`** is the survey: importance (x) against heterogeneity (y),
one labeled point per feature. Bottom-left is unimportant; bottom-right is
important and fully described by its mean effect; top-right — important *and*
heterogeneous — is where `find_regions` should look. With
`partitions=effect.find_regions(features="heterogeneous")` it becomes the
before/after story: an arrow runs from each partitioned feature's global
point to each leaf point — leaves of a good partition move right (more
decisive) and down (heterogeneity explained).

```python
pdp = effector.PDP(X, model, schema=schema)

effector.plot_triage(pdp)                          # step (b): the survey
parts = pdp.find_regions(features="heterogeneous") # step (d)
effector.plot_triage(pdp, partitions=parts)        # step (e): the arrows
```

**`compare`** is the cross-examination: overlay the mean effect of several
fitted engines — different methods, or even different models over the same
columns — on one feature, always centered.

```python
effector.compare(pdp, rhale, shapdp, feature="temp")
```

(For the single-model shortcut that builds its own engines, see
`effector.FeatureEffect`.)

---

## API

### ::: effector.visualization.plot_triage
      options:
        show_root_heading: True

### ::: effector.visualization.compare
      options:
        show_root_heading: True
