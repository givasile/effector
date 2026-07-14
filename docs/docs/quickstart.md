**Quickstart tutorials:**

Start here:

- [What are global and regional effects](./global_and_regional_effects/): the concepts
- [`effector`'s API](./simple_api/): the whole API in 3 minutes
    - [(a) The input layer](./input_guide/): numpy, models, adapters, the schema
    - [(b) `effector`'s report](./report/): the one-liner, `.show()`, and the HTML page
        - [The data & model header](./report/data_and_model/): what was explained, and how good the model is
        - [The explained variance ledger](./report/explained_variance/): how much of the model the explanation captures
        - [The rejected splits](./report/rejected_splits/): every split considered and refused, with the reason
        - [The ranked features](./report/ranked_features/): the ranking, its units, and the coverage cut
        - [The triage plane](./report/triage_plane/): importance versus heterogeneity, where to look first
        - [The regional analysis](./report/regional_analysis/): partition trees and per-leaf plots
        - [The global baseline](./report/global_baseline/): what you would have believed without regions
        - [Configuring the report](./report/configuration/): every knob of `explain(...)`
    - [(c) The interactive API](./interactive_api/): the five engines, global and regional effects
        - [Construct and fit](./interactive/construct_and_fit/): the five engines and `.fit()`
        - [Customize `.fit()`](./interactive/customize_fit/): binning, centering, search depth
        - [`plot`](./interactive/plot/): effects and their heterogeneity, drawn
        - [`eval`](./interactive/eval/): effects as numbers, model free
        - [`importance` and `heter_score`](./interactive/scores/): the twin scalars, in output units
        - [`find_regions`](./interactive/find_regions/): subregions and the `Partition` value
        - [`select_regions`](./interactive/select_regions/): which splits earn their keep
        - [`compare` and `plot_triage`](./interactive/compare_and_triage/): the cross-engine views

Going deeper:

- [The mental model](./../guides/mental_model/): the thinking behind the API; one engine, values not state, two entrances
- [The guides](./../guides/): the math reference, the design contract, the efficiency guides

---
