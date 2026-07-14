# Input layer — Schema & from_dataframe

## Summary

effector is numpy-only at the border (design contract R10): `data` is a 2-D
numeric numpy array, the model is a numpy→numpy callable, and **all input
metadata travels in the single `schema=` argument** — an `effector.Schema`
(frozen dataclass, reusable across constructions) or a plain dict with the
same keys. Every field is optional; anything omitted is inferred
conservatively (nominal is never guessed from a numpy matrix).

```python
schema = {
    "feature_names": ["hr", "temp", "workingday"],
    "feature_types": ["ordinal", "continuous", "nominal"],
    "target_name": "count",
}
pdp = effector.PDP(X, model, schema=schema)
```

Coming from pandas, `from_dataframe` reads names, dtypes, and category levels
into `(X, Schema)` — it converts *data* only and never touches the model. The
returned schema is a proposal to inspect: the int-column guess (ordinal vs
continuous vs label-encoded nominal) is the one thing no extractor can know
for sure.

```python
X, schema = effector.from_dataframe(df)
ale = effector.ALE(X, model, schema=schema)
```

(For the model side of the border — sklearn/torch/classifier wrappers — see
[Adapters](./api_adapters.md).)

---

## API

### ::: effector.ingestion.Schema
      options:
        show_root_heading: True
        show_symbol_type_toc: True

### ::: effector.ingestion.from_dataframe
      options:
        show_root_heading: True

### ::: effector.ingestion.FeatureMetadata
      options:
        show_root_heading: True
        show_symbol_type_toc: True
