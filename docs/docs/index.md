# Home

`Effector` is a python package for [global](./Feature Effect/01_global_effect_intro/) and [regional](./Feature Effect/02_regional_effect_intro/) effect plots.

---

![using effector](static/effector_intro.gif)

---
### Installation

`Effector` is compatible with `Python 3.7+`. We recommend to first create a virtual environment with `conda`:

```bash
conda create -n effector python=3.11 # any python 3.7+ will work
conda activate effector
```

and then install `Effector` via `pip`:

```bash
pip install effector
```

If you want to also run the Tutorial notebooks, add some more dependencies to the environment:

```bash
pip install -r requirements-dev.txt
```

---
### Motivation

#### Global Effect 

Global effect is one the simplest ways to interpret a black-box model;
it simply shows how a particular feature relates to the model's output.
Given the dataset `X` (`np.ndarray`) and the black-box predictive function `model` (`callable`), 
you can use `effector` to get the global effect of a `feature` in a single line of code:

```python
# for Robust and Heterogeneity-aware ALE (RHALE)
RHALE(data=X, model=model).plot(feature)
```

For example, the following code shows the global effect of the feature hour (`hr`) on the 
number of bikes (`cnt`) rent within a day (check [this](./Tutorials/real-examples/01_bike_sharing_dataset/) 
notebook for more details). It is easy to interpret what the black-box model has learned:
There are two peaks in rentals during a day, one in the morning and one in the evening,
where people go to work and return home, respectively:

![Feature effect plot](./Tutorials/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_19_1.png)

--- 

#### Heterogeneity

However, there are cases where the global effect can be misleading. This happens 
when there are many particular instances that deviate from the global effect.
In `effector`, the user can understand where the global effect is misleading, 
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
In `effector` this can be also done in a single line of code:

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

### Methods

`Effector` implements the following methods:

| Method   | Global Effect                                             | Regional Effect                                                               | Paper                                                                                                                                               |                                                                                                                                
|----------|-----------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| PDP      | [`PDP`](./api/#effector.global_effect_pdp.PDP)            | [`RegionalPDP`](./api/#effector.regional_effect_pdp.RegionalPDP)              | [PDP](https://projecteuclid.org/euclid.aos/1013203451), [ICE](https://arxiv.org/abs/1309.6392), [GAGDET-PD](https://arxiv.org/pdf/2306.00541.pdf)   |
| d-PDP    | [`DerPDP`](./api/#effector.global_effect_pdp.DerPDP)      | [`RegionalDerPDP`](./api/#effector.regional_effect_pdp.RegionalDerPDP)        | [d-PDP, d-ICE](https://arxiv.org/abs/1309.6392)                                                                                                     | 
| ALE      | [`ALE`](./api/#effector.global_effect_ale.ALE)            | [`RegionalALE`](./api/#effector.regional_effect_ale.RegionalALE)              | [ALE](https://academic.oup.com/jrsssb/article/82/4/1059/7056085), [GAGDET-ALE](https://arxiv.org/pdf/2306.00541.pdf)                                |                                                                                    
| RHALE    | [`RHALE`](./api/#effector.global_effect_ale.RHALE)        | [`RegionalRHALE`](./api/#effector.regional_effect_ale.RegionalRHALE)          | [RHALE](https://ebooks.iospress.nl/doi/10.3233/FAIA230354), [DALE](https://proceedings.mlr.press/v189/gkolemis23a/gkolemis23a.pdf)                  |
| SHAP-DP  | [`ShapDP`](./api/#effector.global_effect_shap.ShapDP)     | [`RegionalShapDP`](./api/#effector.regional_effect_shap.RegionalShapDP)       | [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions), [GAGDET-DP](https://arxiv.org/pdf/2306.00541.pdf)   |

