## Quickstart

Read these in order:

- [What are global and regional effects](./quickstart/global_and_regional_effects.md): the concepts
- [`effector`'s API](./quickstart/simple_api.md): the whole API in 3 minutes
    - [(a) The input layer](./quickstart/input_guide.md): numpy, models, adapters, the schema
    - [(b) `effector`'s report](./quickstart/report.md): the one-liner, `.show()`, and the HTML page
    - [(c) The interactive API](./quickstart/interactive_api.md): the five engines, global and regional effects

## Page by page

Every component of the report and every verb of the interactive API has its own page.

Inside **(b) `effector`'s report**:

- [The data & model header](./quickstart/report/data_and_model.md): what was explained, and how good the model is
- [The explained variance ledger](./quickstart/report/explained_variance.md): how much of the model the explanation captures
- [The rejected splits](./quickstart/report/rejected_splits.md): every split considered and refused, with the reason
- [The ranked features](./quickstart/report/ranked_features.md): the ranking, its units, and the coverage cut
- [The triage plane](./quickstart/report/triage_plane.md): importance versus heterogeneity, where to look first
- [The regional analysis](./quickstart/report/regional_analysis.md): partition trees and per-leaf plots
- [The global baseline](./quickstart/report/global_baseline.md): what you would have believed without regions
- [Configuring the report](./quickstart/report/configuration.md): every knob of `explain(...)`

Inside **(c) the interactive API**:

- [Construct and fit](./quickstart/interactive/construct_and_fit.md): the five engines and `.fit()`
- [Customize `.fit()`](./quickstart/interactive/customize_fit.md): binning, centering, search depth
- [`plot`](./quickstart/interactive/plot.md): effects and their heterogeneity, drawn
- [`eval`](./quickstart/interactive/eval.md): effects as numbers, model free
- [`importance` and `heter_score`](./quickstart/interactive/scores.md): the twin scalars, in output units
- [`find_regions`](./quickstart/interactive/find_regions.md): subregions and the `Partition` value
- [`select_regions`](./quickstart/interactive/select_regions.md): which splits earn their keep
- [`compare` and `plot_triage`](./quickstart/interactive/compare_and_triage.md): the cross-engine views

## Going deeper

- [The mental model](./guides/mental_model.md): the thinking behind the API; one engine, values not state, two entrances
- [Methods: the math reference](./guides/methods.md): how each method defines the effect, its heterogeneity, and the two scalars, per feature type
- [The design contract](./guides/design.md): the rules the API is built on, R1 to R14, one breath each
- [Efficiency of global methods](./notebooks/guides/efficiency_global.md): count the model calls; everything after the fit is free
- [Efficiency of regional methods](./notebooks/guides/efficiency_regional.md): the regional search costs zero model calls
