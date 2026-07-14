---
title: eval
---

???+ success "Description"

    Part of the [interactive API guide](./../interactive_api.md): the model
    free queries. `eval` for the mean effect, `eval_heter` for its spread,
    `grid` for the positions, `payload` for the raw fitted object.

???+ note "Reading time"

	Approx. 4' to read.

## Effects as numbers

```python
xs      = np.linspace(-1, 1, 100)
y       = pdp.eval(0, xs)          # (100,) the mean effect
y_heter = pdp.eval_heter(0, xs)    # (100,) the spread around it
```

Identical on every engine; swap `pdp` for `ale`, `rhale`, `shapdp`,
`derpdp`. After the single model touch of [fit](./construct_and_fit.md),
every call on this page is **model free**: answered from cached local
effects, zero model calls, nothing stored.

## `eval`: the mean effect

On the [bike sharing rig](./construct_and_fit.md):

```python
xs = pdp.grid("hr")                                 # the 24 observed levels
y  = pdp.eval("hr", xs)                             # global
y_w = pdp.eval("hr", xs, rule="workingday == no")   # within a subregion
```

```text
y[:4]  ->  [-0.712 -0.817 -0.871 -0.914]
```

👉 `eval` returns numbers in the **model's own output units** (here a
standardized target, hence values near ±1); only `plot` applies the schema's
display scaling.

- `centering`: `None` (the fitted default), `False`, `True`, or
  `"zero_start"`, as in [plot](./plot.md).
- `mask=` / `rule=`: restrict to a subregion; the effect is re summarized
  within it.

???+ note "One array, one type"

    `eval` always returns the mean effect only. The spread has its own
    ladder: `eval_heter` (curve), `heter_score` (scalar), `payload` (the raw
    object). No tuples that change shape with flags.

???+ warning "Discrete features evaluate at observed levels only"

    For an ordinal or nominal feature, `xs` must be observed levels; any
    other value raises `ValueError`. Ask `grid(feature)` for the valid
    positions.

## `eval_heter`: the spread, as a curve

```python
h    = pdp.eval_heter("hr", xs)   # (T,) a variance at each x
band = np.sqrt(h)                 # std-like, plot ready
```

```text
h[:4]  ->  [0.133 0.15  0.156 0.154]
```

It is a **variance in the method's native units**: PDP, variance of the
centered ICE curves; DerPDP, of the d-ICE slopes; ALE/RHALE, the per bin
slope variance as a step function; ShapDP, the interpolated per bin φ
variance. Take the square root for a band. There is deliberately no
`centering` argument: heterogeneity is invariant to it.

## `grid`: where the effect is model free

```python
pdp.grid("hr")     # the 24 observed levels of an ordinal feature
pdp.grid("temp")   # an even grid inside temp's axis limits
```

The positions the cache is built on; the same grid `effector.explain`
evaluates every reported curve on.

## `payload`: the raw fitted object

```python
p = ale.payload("hr")
# e.g. {"limits": ..., "bin_effect": ..., "bin_variance": ...}
```

Everything `eval`/`eval_heter` read from, as plain numpy: per bin effects
and variances for ALE, RHALE and ShapDP; grid summaries for PDP and DerPDP.
It is a copy; mutate freely. This is the escape hatch for custom plots and
custom statistics.

---

## Where to next

- [`importance` and `heter_score`](./scores.md): the curves, as two scalars
- [The interactive API](./../interactive_api.md): back to the guide's map
- [`plot`](./plot.md): the previous verb
