---
title: The data & model header
---

???+ success "Description"

    The first component of [effector's report](./../report.md): the
    **DATA & MODEL** header. Four rows that anchor everything below: what data
    the explanation ran on, what the model outputs, and how good that model is.

???+ note "Reading time"

	Approx. 3' to read.

## What you see

In `report.show()` it is the first table:

```text
  DATA & MODEL
  ────────────────────────────────────────────────────────────────────────
    instances     2,000
    features      11  ·  5 nominal · 3 ordinal · 3 continuous
    model output  mean 0.0237 · std 0.973 · range [-1.03, 3.72]
    model R²      0.955  (on this subsample)
```

On the HTML page the same facts render as chips under the title, next to a
`target · features · plotted` caption:

```text
target bike-rentals · 11 features · 4 plotted

[ data 10,000 × 11 ]  [ 5 nominal · 3 ordinal · 3 continuous ]
[ model output -0.0755 ± 0.979 in [-1.34, 4.33] ]  [ R² 0.952 on this subsample ]
```

The two examples above show different numbers on purpose: they are two runs of
the same bike-sharing pipeline, one with `nof_instances=2000`, one with the
default `10_000`. Every number in this header is computed **on the subsample
the explanation actually used**, and that is the first thing it tells you.

## Row by row

| row | the question it answers |
|---|---|
| `instances` | how many rows the explanation was computed on |
| `features` | how many columns, and how each one was read |
| `model output` | what the black box predicts, and on what scale |
| `model R²` | how good the model itself is (only if you passed `y=`) |

👉 **`instances`** is the engine's subsample, not your full matrix. `explain`
caps the data at `nof_instances` (default `10_000`) before doing anything else;
the bike-sharing training set has 13,903 rows, so the header says `10,000`.

👉 **`features`** counts the columns and breaks them down by type:
`5 nominal · 3 ordinal · 3 continuous`. The types come straight from your
[schema](./../input_guide.md). With no schema they are inferred from dtypes:
floats are continuous, integer columns with few unique values ordinal, and
**nominal is never guessed**; a categorical read as ordinal stays wrong until
you say otherwise.

???+ danger "Sanity check the type breakdown first"

    If a categorical feature is being read as continuous, every plot and every
    number below answers the wrong question. This row is where you catch it,
    before you have interpreted anything. Fix it in the
    [schema](./../input_guide.md).

👉 **`model output`** is mean, std and range of the model's predictions on the
subsample. The numbers are in the **model's own output units**: this model was
trained on a standardized target, so they hover around 0 ± 1, while the report's
figures rescale to raw bike rentals through the schema's `scale_y`. Use this
row to anchor magnitudes: an importance of 0.74 is large *because* the model's
whole output spread is 0.97.

👉 **`model R²`** appears only when you pass `y=`. It is the model's own
accuracy against the ground truth, scored on this subsample; it is **not** the
explanation's fidelity, which is the next component's job (the
explained-variance ledger). For a binary 0/1 target the row becomes
`model accuracy`, thresholded at 0.5.

???+ warning "The worse-model trap"

    A low `model R²` means the black box itself is weak. effector will still
    explain it faithfully; you will be reading a faithful explanation of a bad
    model. Judge this row before trusting anything downstream; see
    [the input guide](./../input_guide.md) for the full argument.

## The second chip row

The HTML header adds one more chip row that `.show()` does not print: the
**configuration** the report ran with.

```text
[ method pdp ]  [ top_k 5 ]  [ coverage 0.8000 ]  [ heter_threshold 0.1561 ]
[ min_r2_gain 0.0100 ]  [ finder best ]  [ nof_instances 10000 ]  [ random_state 21 ]
```

A report is a value you can mail around, so it carries its own provenance:
anyone opening the file sees exactly which method and which knobs produced it.
Each knob is explained in [configuring the report](./configuration.md).

---

## Where to next

- [The explained variance ledger](./explained_variance.md): the next component
- [effector's report](./../report.md): back to the guide's map
- [The input layer](./../input_guide.md): the schema that names these rows
