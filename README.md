# Effector

<p align="center">
  <img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/effector_logo.png" width="500"/>
</p>

[![PyPI version](https://badge.fury.io/py/effector.svg?icon=si%3Apython)](https://badge.fury.io/py/effector)
![Execute Tests](https://github.com/givasile/effector/actions/workflows/run_tests.yml/badge.svg)
![Publish Documentation](https://github.com/givasile/effector/actions/workflows/publish_documentation.yml/badge.svg)
[![PyPI Downloads](https://static.pepy.tech/badge/effector)](https://pepy.tech/projects/effector)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

---

`effector` an eXplainable AI package for **tabular data**. It:

- creates [global and regional](https://xai-effector.github.io/quickstart/global_and_regional_effects/) effect plots
- has a [simple API](https://xai-effector.github.io/quickstart/simple_api/) with smart defaults, but can become [flexible](https://xai-effector.github.io/quickstart/flexible_api/) if needed
- is model agnostic; can explain [any underlying ML model](https://xai-effector.github.io/)
- integrates easily with popular ML libraries, like [Scikit-Learn, Tensorflow and Pytorch](https://xai-effector.github.io/quickstart/simple_api/#__tabbed_2_2)
- is fast, for both [global](https://xai-effector.github.io/notebooks/guides/efficiency_global/) and [regional](https://xai-effector.github.io/notebooks/guides/efficiency_global/) methods
- provides a large collection of [global and regional effects methods](https://xai-effector.github.io/#supported-methods)

---

📖 [Documentation](https://xai-effector.github.io/) | 🔍 [Intro to global and regional effects](https://xai-effector.github.io/quickstart/global_and_regional_effects/) | 🔧 [API](https://xai-effector.github.io/api/) | 🏗 [Examples](https://xai-effector.github.io/examples)

---

## Installation

Effector requires Python 3.10+:

```bash
pip install effector
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm`, `shap`.

---

## Quickstart

### Train an ML model

```python
import effector
import keras
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
bike_sharing = effector.datasets.BikeSharing(pcg_train=0.8)
X_train, Y_train = bike_sharing.x_train, bike_sharing.y_train
X_test, Y_test = bike_sharing.x_test, bike_sharing.y_test

# Define and train a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=20, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)
```

### Wrap it in a callable

```python
def predict(x):
    return model(x).numpy().squeeze()
```

### Explain it with global effect plots

```python
# define the global effect method
pdp = effector.PDP(
    X_test,
    predict,
    feature_names=bike_sharing.feature_names,
    target_name=bike_sharing.target_name
)

# plot the effect of the 3rd feature (feature: temperature)
pdp.plot(
    feature=3,
    nof_ice=200,
    scale_x={"mean": bike_sharing.x_test_mu[3], "std": bike_sharing.x_test_std[3]},
    scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},
    centering=True,
    show_avg_output=True,
    y_limits=[-200, 1000]
)
```

![Feature effect plot](https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_3_0.png)

### Explain it with regional effect plots

```python
r_pdp = effector.RegionalPDP(
    X_test,
    predict,
    feature_names=bike_sharing.feature_names,
    target_name=bike_sharing.target_name
)

# summarize the subregions of feature 3
scale_x_list = [{"mean": mu, "std": std} for mu, std in zip(bike_sharing.x_test_mu, bike_sharing.x_test_std)]
r_pdp.summary(
    features=3,
    scale_x_list=scale_x_list
)
```

```
Feature 3 - Full partition tree:
🌳 Full Tree Structure:
───────────────────────
hr 🔹 [id: 0 | heter: 0.43 | inst: 3476 | w: 1.00]
    workingday = 0.00 🔹 [id: 1 | heter: 0.36 | inst: 1129 | w: 0.32]
        temp ≤ 6.50 🔹 [id: 3 | heter: 0.17 | inst: 568 | w: 0.16]
        temp > 6.50 🔹 [id: 4 | heter: 0.21 | inst: 561 | w: 0.16]
    workingday ≠ 0.00 🔹 [id: 2 | heter: 0.28 | inst: 2347 | w: 0.68]
        temp ≤ 6.50 🔹 [id: 5 | heter: 0.19 | inst: 953 | w: 0.27]
        temp > 6.50 🔹 [id: 6 | heter: 0.20 | inst: 1394 | w: 0.40]
--------------------------------------------------
Feature 3 - Statistics per tree level:
🌳 Tree Summary:
─────────────────
Level 0🔹heter: 0.43
    Level 1🔹heter: 0.31 | 🔻0.12 (28.15%)
        Level 2🔹heter: 0.19 | 🔻0.11 (37.10%)
```

The summary of feature `hr` (hour) says that its effect on the output is highly dependent on the value of features:
- `workingday`, wheteher it is a workingday or not
- `temp`, what is the temperature the specific hour

Let's see how the effect changes on these subregions!

---
#### Is it workingday or not?

```python
# plot the regional effects after the first-level splits (workingday or non-workingday)
for node_idx in [1,2]:
    r_pdp.plot(
        feature=3,
        node_idx=node_idx,
        nof_ice=200,
        scale_x_list=[{"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]} for i in range(X_test.shape[1])],
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},
        y_limits=[-200, 1000]
    )
```

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_5_0.png" alt="Feature effect plot"></td>
    <td><img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_5_1.png" alt="Feature effect plot"></td>
  </tr>
</table>


#### Is it hot or cold?

```python
# plot the regional effects after the second-level splits (workingday or non-workingday and hot or cold temperature)
for node_idx in [3,4,5,6]:
    r_pdp.plot(
        feature=3,
        node_idx=node_idx,
        nof_ice=200,
        scale_x_list=[{"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]} for i in range(X_test.shape[1])],
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},
        y_limits=[-200, 1000]
    )

```

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_6_0.png" alt="Feature effect plot"></td>
    <td><img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_6_1.png" alt="Feature effect plot"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_6_2.png" alt="Feature effect plot"></td>
    <td><img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_6_3.png" alt="Feature effect plot"></td>
  </tr>
</table>

---

## Supported Methods

`effector` implements global and regional effect methods:

| Method  | Global Effect  | Regional Effect | Reference | ML model          | Speed                                        |
|---------|----------------|-----------------|-----------|-------------------|----------------------------------------------|
| PDP     | `PDP`          | `RegionalPDP`   | [PDP](https://projecteuclid.org/euclid.aos/1013203451) | any               | Fast for a small dataset                     |
| d-PDP   | `DerPDP`       | `RegionalDerPDP`| [d-PDP](https://arxiv.org/abs/1309.6392) | differentiable    | Fast for a small dataset      |
| ALE     | `ALE`          | `RegionalALE`   | [ALE](https://academic.oup.com/jrsssb/article/82/4/1059/7056085) | any | Fast                                         |
| RHALE   | `RHALE`        | `RegionalRHALE` | [RHALE](https://ebooks.iospress.nl/doi/10.3233/FAIA230354) | differentiable    | Very fast                                    |
| SHAP-DP | `ShapDP`       | `RegionalShapDP`| [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) | any | Fast for a small dataset and a light ML model |

---

## Method Selection Guide

From the runtime persepective there are three criterias:

- is the dataset `small` (N<10K) or `large` (N>10K instances) ? 
- is the ML model `light` (runtime < 0.1s) or `heavy` (runtime > 0.1s) ?
- is the ML model `differentiable` or `non-differentiable` ?

Trust us and follow this guide:

- `light` + `small` + `differentiable` = `any([PDP, RHALE, ShapDP, ALE, DerPDP])` 
- `light` + `small` + `non-differentiable`: `[PDP, ALE, ShapDP]`
- `heavy` + `small` + `differentiable` = `any([PDP, RHALE, ALE, DerPDP])`
- `heavy` + `small` + `non differentiable` = `any([PDP, ALE])`
- `big` +  `not differentiable` = `ALE`
- `big` +  `differentiable` = `RHALE` 

---

## Citation

If you use `effector`, please cite it:

```bibtex
@misc{gkolemis2024effector,
  title={effector: A Python package for regional explanations},
  author={Vasilis Gkolemis et al.},
  year={2024},
  eprint={2404.02629},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## References


- [Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232.](https://projecteuclid.org/euclid.aos/1013203451)
- [Apley, Daniel W. "Visualizing the effects of predictor variables in black box supervised learning models." arXiv preprint arXiv:1612.08468 (2016).](https://arxiv.org/abs/1612.08468)
- [Gkolemis, Vasilis, "RHALE: Robust and Heterogeneity-Aware Accumulated Local Effects"](https://ebooks.iospress.nl/doi/10.3233/FAIA230354)
- [Gkolemis, Vasilis, "DALE: Decomposing Global Feature Effects Based on Feature Interactions"](https://proceedings.mlr.press/v189/gkolemis23a/gkolemis23a.pdf)
- [Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems. 2017.](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [REPID: Regional Effect Plots with implicit Interaction Detection](https://proceedings.mlr.press/v151/herbinger22a.html)
- [Decomposing Global Feature Effects Based on Feature Interactions](https://arxiv.org/pdf/2306.00541.pdf)
- [Regionally Additive Models: Explainable-by-design models minimizing feature interactions](https://arxiv.org/abs/2309.12215)

---

## License

`effector` is released under the [MIT License](https://github.com/givasile/effector/blob/main/LICENSE).
