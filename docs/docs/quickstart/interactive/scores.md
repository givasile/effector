---
title: importance and heter_score
---

???+ success "Description"

    Part of the [interactive API guide](./../interactive_api.md): the twin
    scalars. `importance` for how much the mean effect moves the output,
    `heter_score` for how much the average hides, both in the output's
    units.

???+ note "Reading time"

	Approx. 4' to read.

## Two scalars, one scale

```python
pdp.importance(0)    # 0.0144: how much the MEAN effect moves
pdp.heter_score(0)   # 5.8008: how much the per-instance effects SPREAD
```

Both are **std type quantities in the output's units**, so they are
comparable across features, across feature types, and, in magnitude, across
methods. On the conditional interaction model of
[the plot page](./plot.md), `x_0` scores near zero importance and huge
heterogeneity: its mean effect cancels out while the individual effects are
wildly spread. That is precisely the feature a global average hides.

✅ Read them as **orthogonal axes**, not as one ranking. High importance with
low heterogeneity means the mean curve is the whole story. High
heterogeneity means one curve is not enough, *whatever* the importance. The
two axes drawn together are
[the triage plane](./../report/triage_plane.md).

## What exactly they measure

- **`importance`**: the dispersion of the **mean** effect over the data
  distribution. A flat curve scores ~0; a swinging curve scores high. For a
  linear model it recovers `|coefficient| · std(x)`.
- **`heter_score`**: the RMS of [`eval_heter`](./eval.md) over the feature's
  own data values, frequency weighted over levels for categorical features,
  bridged by the feature's dispersion for the derivative based methods so
  every method lands on the same y unit scale. Read it as *"a typical
  instance's effect deviates from the mean effect by about this much"*.

???+ note "No `y`, ever"

    effector never sees ground truth labels. These are properties of the
    fitted effect, not a loss based or permutation importance.

## On real data

On the [bike sharing rig](./construct_and_fit.md), both axes at once:

```python
pdp.importance("hr")     # 0.6576: the mean effect swings hard
pdp.heter_score("hr")    # 0.4707: and the instances still disagree a lot
```

`hr` is the top right corner of the triage plane: important **and**
heterogeneous, the feature to hunt regions for.

## The whole vector

```python
imp = pdp.importances()                    # (D,)
order = np.argsort(-np.nan_to_num(imp))    # most important first
```

```text
imp  ->  [0.086 0.227 0.051 0.658 0.008 0.029 0.018 0.067 0.212 0.093 0.021]
```

One call, one `(D,)` array: `hr` (0.658), `yr` (0.227) and `temp` (0.212)
carry the model.

???+ warning "NaN means unsupported, not unimportant"

    Feature types a method cannot explain (e.g. `DerPDP` on a nominal
    feature) return `NaN`, with one `UserWarning` naming them.

## The regional bridge

Both scalars take `mask=` / `rule=`, and that is not a convenience; it is
the engine of the regional analysis:

```python
pdp.heter_score("hr")                             # 0.4707  global
pdp.heter_score("hr", rule="workingday == no")    # 0.3274  inside the subregion
```

Conditioning on `workingday` explains away a third of `hr`'s spread. Note
the rule string speaks the schema's **level names** (`no`), not internal
codes.

`heter_score(feature, mask)` is **the quantity
[`find_regions`](./find_regions.md) minimizes**: a good split is one whose
subregions score much lower than the parent. When you read a partition tree,
every chip's `heter` is this call on that node's mask.

---

## Where to next

- [`find_regions`](./find_regions.md): put the scores to work
- [The interactive API](./../interactive_api.md): back to the guide's map
- [`eval`](./eval.md): the same quantities, as curves
