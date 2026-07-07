# One-click report

`effector.explain` runs the whole pipeline — fit, rank features by importance,
draw the effect curves, and search for subregions on the heterogeneous features —
and hands you back a single `Report` value you can print, plot, save as a
self-contained HTML page, or serialize.

```python
import effector

X = ...        # (N, D) numpy array
predict = ...  # numpy -> numpy model

report = effector.explain(X, predict, method="pdp", top_k=5)
```

## Read it in the terminal

```python
report.show()
```

prints an importance-ranked table and, for each heterogeneous feature, the
partition tree that `find_regions` found:

```text
PDP report — target: y
============================================================
feature                   importance     heter  #regions
------------------------------------------------------------
hr                            0.4312    0.4300         7
temp                          0.2811    0.1900         1
...
```

## Plot the importances

```python
report.plot_importance()            # shows a horizontal bar chart
fig, ax = report.plot_importance(show_plot=False)   # or grab the figure
```

## Save a self-contained HTML page

```python
report.to_html("report.html")
```

The page inlines every figure as a base64 PNG — no external assets — so it opens
anywhere and can be emailed or committed as an artifact.

## Serialize the report

A `Report` is a value: it round-trips through a plain dict with no model or effect
attached.

```python
d = report.to_dict()
report2 = effector.Report.from_dict(d)   # text + HTML work without an effect
```

## Choosing what runs

- `method` — `"pdp"`, `"ale"`, `"rhale"`, `"shapdp"`, `"derpdp"` (aliases allowed).
- `top_k` — how many top-importance features to detail.
- `heter_threshold` — the `heter_score` above which `find_regions` runs; `None`
  uses the median across the ranked features.
- `finder` / `candidate_conditioning_features` — passed through to `find_regions`.

Because effector never sees `y`, importance here is a property of the fitted
effect (the dispersion of the mean effect, R13), not a held-out error — so a
report needs only the model and the data.
