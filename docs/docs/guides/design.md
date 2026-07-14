---
title: The design contract
---

???+ success "Description"

    The rules every `effector` class follows, R1 to R14, trimmed to one
    breath each. The full normative text lives in the repository
    ([docs/design.md](https://github.com/givasile/effector/blob/main/docs/design.md));
    the contract test layer (`tests/test_contract_*.py`) enforces most of it
    mechanically.

???+ note "Reading time"

	Approx. 4' to read.

## The rules, one breath each

| rule | name | the essence |
|---|---|---|
| **R1** | Lifecycle | construction ingests; `fit` declares config and warms caches; fit-then-query equals never-fit-just-query, byte for byte |
| **R2** | Heterogeneity | its own surface, an aggregation ladder: `payload` → `eval_heter` (variance curve) → `heter_score` (scalar); never through `eval` |
| **R3** | Centering | one vocabulary: `False`, `"zero_integral"` (=`True`), `"zero_start"`; each class declares its default once |
| **R4** | State | exactly two caches and one declared config; the fitted payload *is* a cache entry, there is no separate fitted state |
| **R5** | Method registry | one `{name: (cls, needs_jac, ...)}` table; per-method if/elif chains are a bug |
| **R6** | String registries | one alias table per concept, one resolver; validation goes through it |
| **R7** | Plot contract | every `.plot` returns `(fig, ax)` when `show_plot=False`, `None` otherwise |
| **R8** | Constructor | canonical order, keyword-only after `model_jac`; `random_state` everywhere; two identical constructions give identical output |
| **R9** | Errors | `ValueError`/`TypeError` for user input, never bare `assert`; warnings, never `print` |
| **R10** | Inputs | numpy-only: data is a 2-D numpy array, model/jacobian are numpy→numpy callables, all metadata in one `schema=`; DataFrames only through `from_dataframe` |
| **R11** | Regional ≡ masked global | a region is the one global object's summary under a boolean mask; frames stay frozen, masked calls are model-free |
| **R12** | Values, not state | `find_regions` returns a `Partition` and stores nothing; a `Region`'s identity is its `Rule` |
| **R13** | Importance | the μ-twin of `heter_score`: std of the mean effect, output units, centering-invariant, no `y` ever |
| **R14** | Two-block lifecycle | cache (a) local effects (the only model touch, frame-gated); cache (b) memoized summaries keyed by epoch, staleness by key structure |

## Why a contract at all

Five methods, three feature types, global and regional surfaces: without a
constitution the combinations drift apart one convenience hack at a time.
The contract pins the load-bearing decisions once, and the test layer makes
a violation a red build instead of a review comment.

Three rules carry most of the weight:

- **R10, numpy-only.** The model is called exactly as given, never wrapped;
  nothing is auto-detected, so nothing can be silently wrong. Everything the
  package knows about your columns travels in one `schema=` argument.
  This is [the input layer](../quickstart/input_guide.md)'s foundation.
- **R11 + R12, regions are masked views and values.** A regional effect is
  never a re-fitted object: same frame, same caches, a mask on top — which is
  why the whole regional search costs
  [zero model calls](../notebooks/guides/efficiency_regional.md). And what
  the search returns is yours to hold, serialize, or discard; the engine
  stores nothing.
- **R14, two blocks.** One block touches the model (once per feature); the
  other is a memo of cheap numpy summaries. Every verb is a query over the
  two, which is what makes
  [the whole session model-free after fit](../notebooks/guides/efficiency_global.md).

???+ note "Where the details live"

    The full text with every sub-rule (frame semantics, the retrigger rule,
    the proposer seam, `mask_key` normalization) is
    [docs/design.md](https://github.com/givasile/effector/blob/main/docs/design.md)
    in the repository, and the per-method exactness formulas are in
    [the methods reference](./methods.md).

---

## Where to next

- [Methods: the math reference](./methods.md): the formulas under the rules
- [The mental model](./mental_model.md): the same design, as intuition
- [API docs](../api_docs.md): every verb and value
