# effector, one click away — the vision

*The destination this project is walking toward. Written 2026-07-05 as a
proposal (`scripts/EFFECTOR_VISION.pdf`), rewritten 2026-07-09 against what
has shipped since. The live backlog is `PLAN.md`; the trail is `LOGBOOK.md`.*

## The premise

A user has a trained model and thirty seconds of patience. The methods
underneath effector are excellent — a coherent global + regional family with
honest heterogeneity semantics that no other package has. The vision is the
last mile: **the layer that turns fitted effect objects into an explanation a
human enjoys reading.** Everything is designed around one target user story:

> *"I trained a model. One call gives me a report that ranks my features,
> shows me every effect, warns me where the average curve lies, splits those
> features into regimes, and lets me drill from any curve down to a single
> row — and I can send it to a colleague as one HTML file."*

## The twofold end goal

**(b) One-click end-to-end auto-explanation — ✅ SHIPPED.**
`effector.explain(X, model, schema=...)` fits once, ranks by importance,
plots the important features, runs `find_regions` on the heterogeneous ones,
and returns a serializable `Report` with a self-contained `to_html()`.

**(a) Interactive exploration — ◻ OPEN (the one big missing piece).**
Change method params, filter by other features manually or automatically, see
new effects on the fly. The architecture was built for it — pure
`(feature, xs, mask, kwargs)` queries over an immutable engine, memoized so a
slider dragged back to a prior state is instant — but no widget layer exists
yet. This is PLAN Part IV §4 (Phase D).

## The architectural bet (made, and it held)

*One engine, values not state.* The effect object is the single stateful
thing — data + model + two caches (R14); everything else is a model-free
query returning a value: a float (`importance`), a `Partition`
(`find_regions`), a `Report` (`explain`). There is **no session/Explainer
class**: the engine IS the session, and `explain` is the workbench walked
with defaults. This is what makes both end goals cheap: the report owns
values and serializes; widgets re-query a pure surface.

## Scorecard — the original A/B items

Part A (visualization) and Part B (API) from the 2026-07-05 proposal:

| item | status |
|---|---|
| A1 one-click HTML report | ✅ `explain` → `Report.to_html()` |
| A2 "model at a glance" overview grid (shared centered y-axis, ranked) | ◻ parking lot |
| A3 visual partition tree (`plot_tree`) | ◻ parking lot — highest leverage-per-effort viz item |
| A4 interactive backend | ◻ = Phase D (PLAN IV §4) |
| A5 method cross-examination | ➖ `compare` shipped; disagreement strip + cause annotation open |
| A6 categorical effects redesign (dot-intervals, evidence margin) | ◻ parking lot |
| A7 triage map | ✅ `plot_triage` (+ before/after arrows with `partitions=`) |
| A8 explain one row (`locate`) | ◻ parking lot |
| A9 house theme | ✅ `set_theme("light"/"dark"/"paper")` |
| B1 facade above the classes | ✅ resolved differently: no new class — engine + `explain` |
| B2 feature importance | ✅ `importance`/`importances` (R13, the μ-twin of `heter_score`) |
| B3 model adapters + classification story | ➖ adapters shipped (`from_sklearn`/`from_torch`/`classifier_proba`/`check`); the classification *story* (units, per-class reports) open |
| B4 one vocabulary | ✅ via the R1–R14 homogenization |
| B5 regional ergonomics (objects, not printouts) | ✅ `Partition`/`Region`/`Rule`, rule-addressable plots |
| B6 effects as data (`to_dataframe`, `save`/`load`) | ➖ `to_dict` on Partition/Report; tidy export + fitted-effect persistence open |
| B7 `method="auto"` + presets | ◻ parking lot |
| B8 trust layer (extrapolation guard, agreement score, evidence flags) | ◻ PLAN IV §3 — the "honest effects package" made visible |

## What "1.0" means

The API is settled when R1–R14 hold, the shell (adapters, names, rules,
triage, compare, report) is documented, and the chain is on `main`. Version
1.0.0 is a statement about *stability*, not completeness: the parking-lot
viz items and the interactive layer can land in 1.x without breaking anyone.

Beyond 1.0, the two directions that make effector more than "effect plots":

1. **Interaction quantification** (PLAN IV §2, §5) — importance says *which*,
   heterogeneity says *that*, the interaction vector/matrix says *why* and
   *with whom*. The D×D matrix falls out of the split search almost for free,
   and it is consistent by construction with what `find_regions` splits on.
2. **Joint regions & interpretability-by-design** — GADGET-style single
   partitions valid for many features at once, feeding CALM-style locally
   additive models (a separate package consuming effector's now-stable seams).

The differentiator to protect throughout: regional effects with honest,
method-agnostic heterogeneity — delivered so well (visual tree, triage map,
one-click report, interactive drill-down) that the picture people screenshot
into slides is an effector figure.
