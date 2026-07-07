## Summary

`effector.explain(data, model, ...)` runs the whole explanation pipeline in one
call and returns a `Report` — a serializable value (design contract R12):

1. fit the chosen `method` once
2. rank features by `importance` (R13)
3. compute the mean-effect + heterogeneity curves for the top-`top_k`
4. `find_regions` on the features whose `heter_score` clears the threshold

Every model call happens through the single `fit`; importance, curves, and
`find_regions` are all model-free afterwards.

---

## Usage

```python
import effector

report = effector.explain(X, predict, method="pdp", top_k=5)

report.show()                    # importance-ranked table + region trees
report.plot_importance()         # horizontal bar chart
report.to_html("report.html")    # self-contained page (figures inlined)

d = report.to_dict()             # serialize (no effect needed to round-trip)
report2 = effector.Report.from_dict(d)
```

## API

### ::: effector.report.explain
       options:
         show_root_heading: True
         show_symbol_type_toc: True

### ::: effector.report.Report
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         members:
           - show
           - plot_importance
           - to_html
           - to_dict
           - from_dict

### ::: effector.report.FeatureReport
       options:
         show_root_heading: True
         show_symbol_type_toc: True
