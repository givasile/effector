---
title: Configuring the report
---

???+ success "Description"

    The last page of the [report guide](./../report.md): every knob of
    `effector.explain(...)` and what it changes, the output switches
    (`show(ascii=)`, `to_html(share_y=)`), and the report as a value.

???+ note "Reading time"

	Approx. 5' to read.

## The full signature

```python
report = effector.explain(
    data,                                  # (N, D) numpy matrix
    model,                                 # (N, D) -> (N,) callable
    model_jac=None,                        # jacobian; needed by rhale / derpdp
    y=None,                                # ground truth: adds model R² to the header
    schema=None,                           # names, types, level names, units
    method="pdp",                          # "pdp" | "ale" | "rhale" | "shapdp" | "derpdp"
    top_k=5,                               # ceiling on plotted features
    coverage=0.8,                          # importance mass the plots must carry
    heter_threshold=None,                  # None: the median heter_score
    min_r2_gain=0.01,                      # a split must buy 1% of R² to be kept
    finder="best",                         # "best" | "best_level_wise" | an instance
    candidate_conditioning_features="all", # who may define splits
    nof_instances=10_000,                  # subsample the effect is built on
    random_state=21,                       # seed for the subsample
)
```

The same pipeline runs from a fitted engine as `pdp.explain(y=...)`, reusing
its cache instead of fitting again.

## What each knob moves

| knob | what it changes | seen in |
|---|---|---|
| `method` | which engine computes the effects | the title, everywhere |
| `y` | adds the model's own score | [the header](./data_and_model.md) |
| `schema` | names, types, level names, units | rules, ticks, [the header](./data_and_model.md) |
| `top_k`, `coverage` | how many features get **plotted** | [the ranked features](./ranked_features.md) |
| `heter_threshold` | who enters the region search | [the triage plane](./triage_plane.md) hairline |
| `min_r2_gain` | the price of admission for a split | [the ledger](./explained_variance.md), [the rejected splits](./rejected_splits.md) |
| `finder` | how a feature's space is partitioned | [the regional analysis](./regional_analysis.md) trees |
| `candidate_conditioning_features` | who may appear in a rule | the `split on` columns |
| `nof_instances`, `random_state` | the subsample everything runs on | `instances` in [the header](./data_and_model.md) |

???+ note "Search wide, display by coverage"

    `top_k` and `coverage` trim only the **curve plots**. The region search
    and the R² selection run over all heterogeneous features, and the triage
    plane and ranked table always cover every supported feature; a feature
    outside the display cut can still carry the biggest gain.

???+ note "`heter_threshold=None` means the median"

    The default threshold is the median `heter_score` across the supported
    features, the same convention as `find_regions(features="heterogeneous")`.
    The achieved value is printed in the HTML header chips.

???+ warning "DerPDP has no ledger"

    `method="derpdp"` reports effects on the **derivative** scale, where sums
    of curves do not approximate the model's output; the explained variance
    ledger does not apply and is omitted. Every other method has one.

## The output switches

```python
report.show()                    # the tables; unicode box drawing
report.show(ascii=True)          # same tables, plain ASCII
report.to_html("report.html")    # the page; y axes shared across features
report.to_html("report.html", share_y="within")   # shared only within each feature
report.plot_importance()         # a standalone importance bar chart
```

`share_y="across"` (default) puts every effect figure of the page on one y
range, so levels compare at a glance; `"within"` frees each feature to use
its own range, which reads better when importances differ by orders of
magnitude.

## It is a value

```python
d = report.to_dict()                  # plain, serializable
again = effector.Report.from_dict(d)  # no model, no data needed
again.show()                          # identical output
```

A rebuilt report has no live effect behind it: `show()` and the HTML render
from stored values, with per leaf plots replaced by leaf statistics tables
(see [the regional analysis](./regional_analysis.md)). Everything the tables
print also lives on the object: `report.features`, `report.overview`,
`report.explained_variance`, `report.config`, `report.summary`.

---

## Where to next

- [effector's report](./../report.md): back to the guide's map
- [The data & model header](./data_and_model.md): restart the component tour
- [`select_regions`](./../interactive/select_regions.md): the selection,
  driven by hand
