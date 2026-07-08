# The mental model

effector has one central object and a handful of values around it. If you
internalize this page, every verb in the API becomes predictable.

```
                ┌── effector.explain(X, model, schema)      ← the one-liner entrance
                │
X, schema ──────┤
model ──────────┤
(via adapters,  │   pdp = effector.PDP(X, model, schema)    ← the workbench entrance
 your pass)     └─► ONE stateful engine per method:
                    data + model + two caches, everything
                    else computed on demand
                        │
                        ├─ importance / heter_score  ──► effector.plot_triage(pdp)
                        ├─ plot(feature, rule=...)
                        ├─ find_regions(features=...) ──► {name: Partition}
                        └─ eval / eval_heter / payload

                    effector.compare(pdp, rhale, feature=...)   ← stands above, stateless
```

## One engine, five methods

An effect object — `PDP`, `DerPDP`, `ALE`, `RHALE`, `ShapDP` — is the one
stateful thing in effector. Think of it the way you think of a pytorch model:
you construct it once, it holds what is expensive, and you live with it for
the whole session, asking it things.

What it holds is deliberately small (design contract
[R14](./design.md#r14-two-block-lifecycle)): the data, the model handle,
and two caches — the model-touching local effects (computed once per feature)
and the cheap summaries derived from them. Everything you ask afterwards —
effects, heterogeneity, importance, subregion searches, plots — is computed
on demand from those caches, without touching the model again.

```python
pdp = effector.PDP(X, model, schema=schema)   # construct once
pdp.plot("hr")                                 # everything else is a query
pdp.importance("temp")
pdp.find_regions("hr")
```

Every verb accepts a feature **index or name** — names come from your schema
(or the synthesized `x_0, x_1, …`).

## Values, not state

Queries return **values** ([R12](./design.md#r12-regions-are-values-not-state)):
`importance` returns a float, `find_regions` returns a `Partition`,
`explain` returns a `Report`. The engine never remembers your analysis behind
your back — the variables in your notebook *are* the session:

```python
parts = pdp.find_regions(features="heterogeneous")   # a dict you hold
pdp.plot("hr", rule=parts["hr"][2].rule)             # plot node 2 of the tree
```

A `Partition` is a tree of `Region`s; each region carries its `rule` (a
predicate like `hr < 12 and workingday == 1`), and `rule=` on any verb
restricts it to that subpopulation — evaluated model-free from the caches.
Don't like a partition? Recompute it with different finder kwargs; nothing
needs resetting, because nothing was stored.

## Two entrances, one engine

There are exactly two ways in, and they share every internal:

- **`effector.explain(X, model, schema=schema)`** — the one-liner. It fits one
  method, ranks features, searches regions for the heterogeneous ones, and
  freezes everything into a `Report` (→ `.to_html()`). Use it when you want
  *the answer*.
- **The workbench** — construct an engine and drive the analysis yourself,
  step by step, intervening wherever the defaults don't convince you. Use it
  when you want *the analysis*.

`explain` is not a different pipeline: it is the workbench walked with
defaults and nobody intervening.

## Adapters: the explicit final pass

Engines take a plain numpy-in / numpy-out callable and a numpy matrix.
`effector.adapters` and `effector.from_dataframe` convert common objects —
but they only *return* the plain objects; **you** make the final pass into
the constructor. Nothing is auto-detected, so nothing can be silently wrong:

```python
X, schema = effector.from_dataframe(df)               # DataFrame -> (matrix, Schema)
model = effector.adapters.from_sklearn(est)           # estimator -> callable
model = effector.adapters.classifier_proba(clf, class_=1)  # classifiers: P(class=k)
effector.adapters.check(model, X)                     # the handshake: probe on 2 rows

pdp = effector.PDP(X, model, schema=schema)           # yours, visibly
```

## The canonical workflow

The whole package folds into five steps — the bike-sharing walkthrough
([notebook](./notebooks/real-examples/01_bike_sharing_dataset.md)) runs them
end to end:

**(a) Scope.** Construct the engine; `fit` the features you care about.

```python
pdp = effector.PDP(X, model, schema=schema)
pdp.fit(features="all")
```

**(b) Triage.** Where is the signal, and where is it hiding something?

```python
effector.plot_triage(pdp)
```

Importance right, heterogeneity up. Bottom-left: ignore. Bottom-right:
important and fully described by its mean effect — done. **Top-right:
important *and* heterogeneous — the mean effect is averaging different
stories. That corner is your to-do list.**

**(c) Look.** Plot the suspects' global effects; the heterogeneity band shows
*that* something varies, not yet *what*.

```python
pdp.plot("hr", heterogeneity="ice")
```

**(d) Explain the spread.** Search for subregions that resolve the
heterogeneity, then look at each leaf:

```python
parts = pdp.find_regions(features="heterogeneous")
parts["hr"].show()
pdp.plot("hr", rule=parts["hr"].leaves[0].rule)
```

**(e) Triage again, with receipts.**

```python
effector.plot_triage(pdp, partitions=parts)
```

Arrows run from each partitioned feature's global point to its leaves. Leaves
of a good partition move **right** (each subpopulation's effect is more
decisive) and **down** (the spread is explained). This figure is the analysis:
what mattered, what was heterogeneous, and what resolved it.

For a second opinion at any point, `effector.compare(pdp, rhale,
feature=...)` overlays fitted engines — different methods, or different
models over the same columns.

## Where to next

- [Quickstart](./quickstart.md) — the same ideas, hands-on
- [The design contract](./design.md) — the formal rules (R1–R14) behind this page
- [Method semantics](./method_semantics.md) — what each method computes per feature type
- [API docs](./api_docs.md) — every verb and value
