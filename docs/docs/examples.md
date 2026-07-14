???+ success "Description"

    Every example below, synthetic or real, ends the same way: one line,
    `effector.explain(...)`, and the self-contained HTML page it returns.
    The 📄 link in the last column **is** that page. Read the notebook to learn
    how the analysis is built; open the report to see the whole notebook
    summarized in a single file you could email to someone.

## Synthetic examples

The model is known, so every effect has a closed form to check against.

| # | Example | Report |
|:--|:--------|:-------|
| 1(a) | [Regional effects, known black-box function](./notebooks/synthetic-examples/03_regional_effects_synthetic_f.md): same as in the [REPID paper](https://proceedings.mlr.press/v151/herbinger22a/herbinger22a.pdf) | [📄 PDP](./static/reports/03_regional_effects_synthetic_f_pdp.html) |
| 1(b) | [Regional effects, unknown black-box function](./notebooks/synthetic-examples/04_regional_effects_real_f.md): the same data, now explained through a fitted neural network | [📄 PDP](./static/reports/04_regional_effects_real_f_pdp.html) |
| 2(a) | [A conditional interaction: global effects](./notebooks/synthetic-examples/05_conditional_interaction_independent_uniform_global.md) | [📄 PDP](./static/reports/05_conditional_interaction_independent_uniform_global_pdp.html) |
| 2(b) | [A conditional interaction: heterogeneity](./notebooks/synthetic-examples/05_conditional_interaction_independent_uniform_heter.md) | [📄 PDP](./static/reports/05_conditional_interaction_independent_uniform_heter_pdp.html) |
| 2(c) | [A conditional interaction: regional effects](./notebooks/synthetic-examples/05_conditional_interaction_independent_uniform_regional.md) | [📄 PDP](./static/reports/05_conditional_interaction_independent_uniform_regional_pdp.html) |
| 3 | [A general form interaction](./notebooks/synthetic-examples/06_general_interaction_independent_uniform_global.md): $x_1 x_2^2$, where no single split can undo the heterogeneity | [📄 PDP](./static/reports/06_general_interaction_independent_uniform_global_pdp.html) |
| 4 | [A conditional interaction with four regions](./notebooks/synthetic-examples/07_conditional_interaction_4_regions_independent_uniform_global.md) | [📄 PDP](./static/reports/07_conditional_interaction_4_regions_independent_uniform_global_pdp.html) |
| 5 | [Categorical features](./notebooks/synthetic-examples/08_categorical_features.md): ordinal and nominal, as the feature of interest and as split candidates | [📄 PDP](./static/reports/08_categorical_features_pdp.html) |
| 6 | [One-click explanations](./notebooks/synthetic-examples/09_explain_importance_report.md): `importance`, `explain` and the `Report`, on a model whose answer we know | [📄 PDP](./static/reports/09_explain_importance_report_pdp.html) |

## Real examples

The model is a black box, so the report is the whole story.

| # | Example | Report |
|:--|:--------|:-------|
| 1 | [Bike sharing](./notebooks/real-examples/01_bike_sharing_dataset.md): the canonical walkthrough, hour of day against working day | [📄 PDP](./static/reports/01_bike_sharing_dataset_pdp.html) |
| 2 | [California housing](./notebooks/real-examples/02_california_housing.md): eight correlated features, explained with RHALE | [📄 RHALE](./static/reports/02_california_housing_rhale.html) |
| 3 | [California housing with TabPFN](./notebooks/real-examples/03_california_housing_tabpfn.md): a foundation model is just another callable | none, see below |
| 4 | [NO2 concentration](./notebooks/real-examples/04_no2.md): traffic, wind and temperature on a small, noisy dataset | [📄 PDP](./static/reports/04_no2_pdp.html) |
| 5 | [Medical costs](./notebooks/real-examples/05_medical_costs.md): one interaction, two claimants; the decision sequence on the smoker × bmi classic | [📄 PDP](./static/reports/05_medical_costs_pdp.html) |
| 6 | [Airfoil self-noise](./notebooks/real-examples/06_airfoil_self_noise.md): when half the model is one interaction, a single split recovers +28% of explained variance | [📄 PDP](./static/reports/06_airfoil_self_noise_pdp.html) |
| 7 | [Adult census income](./notebooks/real-examples/07_adult_income.md): explaining a classifier's probability, a four-split decision sequence | [📄 PDP](./static/reports/07_adult_income_pdp.html) |

???+ note "Why example 3 has no report"

    TabPFN needs a `TABPFN_TOKEN`, so that notebook cannot be executed in a
    clean checkout and its report cannot be regenerated with the others. Run it
    with your own token and `report.to_html(...)` produces the same page.
