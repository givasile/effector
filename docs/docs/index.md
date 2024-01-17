# Home

`Effector` is a python package for [global](./Feature Effect/01_global_effect_intro/) and [regional](./Feature Effect/02_regional_effect_intro/) effect analysis.

---

![using effector](static/effector_intro.gif)

---
### Installation

`Effector` is compatible with `Python 3.7+`. We recommend to first create a virtual environment with `conda`:

```bash
conda create -n effector python=3.7
conda activate effector
```

and then install `Effector` via `pip`:

```bash
pip install effector
```

---
### Motivation

#### Global Effect 

Global effect is one the simplest ways to interpret a black-box model;
it simply shows how a particular feature relates to the model's output.
Given the dataset `X` (`np.ndarray`) and the black-box predictive function `model` (`callable`), 
you can use `Effector` to get the global effect of a `feature` in a single line of code:

```python
# for Robust and Heterogeneity-aware ALE (RHALE)
RHALE(data=X, model=model).plot(feature)
```

For example, the following code shows the global effect of the feature hour (`hr`) on the 
number of bikes (`cnt`) rent within a day (check [this](Tutorials/real-examples/01_bike_sharing_dataset.ipynb) 
notebook for more details). It is easy to interpret what the black-box model has learned:
There are two peaks in rentals during a day, one in the morning and one in the evening,
where people go to work and return home, respectively:

![Feature effect plot](./Tutorials/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_19_1.png)

--- 

#### Heterogeneity

However, there are cases where the global effect can be misleading. This happens 
when there are many particular instances that deviate from the global effect.
In `Effector`, the user can understand where the global effect is misleading, 
using the argument `heterogeneity`, while plotting:

```python
# for RHALE
RHALE(data=X, model=model).plot(feature, heterogeneity=True)
```

![Feature effect plot](./Tutorials/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_20_0.png)


For more details, check the [global effect tutorial](./Feature Effect/01_global_effect_intro/).

--- 

#### Regional Effect

In this cases, it is useful to search if there are subspaces where the effect.
In `Effector` this can be also done in a single line of code:

```python
RegionalRHALE(data=X, model=model).plot(feature=0, node_idx=1, heterogeneity=True)
RegionalRHALE(data=X, model=model).plot(feature=0, node_idx=2, heterogeneity=True)
```

In the bike sharing dataset, we can see that the effect of the feature hour (`hr`) 
follows to different patterns (regional effects) depending on the day of the week (subspaces).
In weekdays, the effect is similar to the global effect, while in weekends, the effect is
completely different; there is a single peak in the morning when people rent bikes to go for sightseeing.

![Feature effect plot](./../Tutorials/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_26_0.png)
![Feature effect plot](./../Tutorials/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_26_1.png)

For more details, check the [regional effect tutorial](./Feature Effect/02_regional_effect_intro/).

---

## Methods and Publications

### Methods

`Effector` implements the following methods:

| Method   | Global Effect                                                                  | Regional Effect                                                                         |                                                                                                                                
|----------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| PDP      | [`PDP`](./api/#effector.global_effect_pdp.PDP)                                 | [`RegionalPDP`](./../../03_API/#effector.regional_effect_pdp.RegionalPDP)               |
| d-PDP    | [`DerivativePDP`](./../../03_API/#effector.global_effect_pdp.DerivativePDP)    | [`RegionalDerivativePDP`](./../../03_API/#effector.regional_effect_pdp.RegionalDerivativePDP) |
| ALE      | [`ALE`](./../../03_API/#effector.global_effect_ale.ALE)                        | [`RegionalALE`](./../../03_API/#effector.regional_effect_ale.RegionalALE)               |
| RHALE    | [`RHALE`](./../../03_API/#effector.global_effect_ale.RHALE)                    | [`RegionalRHALE`](./../../03_API/#effector.regional_effect_ale.RegionalRHALE)           |
| SHAP-DP  | [`SHAPDependence`](./../../03_API/#effector.global_effect_shap.SHAPDependence) | [`RegionalSHAP`](./../../03_API/#effector.regional_effect_shap.RegionalSHAPDependence)  |

### Publications

The methods above are based on the following publications:


  - PDP and d-PDP: [Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232.](https://projecteuclid.org/euclid.aos/1013203451)
  - ALE: [Apley, Daniel W. "Visualizing the effects of predictor variables in black box supervised learning models." arXiv preprint arXiv:1612.08468 (2016).](https://arxiv.org/abs/1612.08468)
  - RHALE: [Gkolemis, Vasilis, "RHALE](https://ebooks.iospress.nl/doi/10.3233/FAIA230354)
  - SHAP-DP: [Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems. 2017.](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

- Regional Effect:
 
  - [REPID: Regional Effect Plots with implicit Interaction Detection](https://proceedings.mlr.press/v151/herbinger22a.html)
  - [Decomposing Global Feature Effects Based on Feature Interactions](https://arxiv.org/pdf/2306.00541.pdf)
  - [Regionally Additive Models: Explainable-by-design models minimizing feature interactions](https://arxiv.org/abs/2309.12215)