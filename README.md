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

Effector is a Python package for interpretable Machine Learning that:

- provides [**global** and **regional**](https://xai-effector.github.io/quickstart/global_and_regional_effects/) effects
- works **only** with tabular data
- offers a [simple API](https://xai-effector.github.io/quickstart/simple_api/) with smart defaults, but can be [flexible](https://xai-effector.github.io/quickstart/flexible_api/) if needed
- is model agnostic, so it works [with any ML model](https://xai-effector.github.io/)
- integrates with popular ML libraries, like [Scikit-Learn, Tensorflow and Pytorch](https://xai-effector.github.io/quickstart/simple_api/#__tabbed_2_2)
- is fast, even for both [global](https://xai-effector.github.io/notebooks/guides/efficiency_global/) and [regional](https://xai-effector.github.io/notebooks/guides/efficiency_global/) methods
- supports a variety of [**global** and **regional** effects](https://xai-effector.github.io/#supported-methods)

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

### Global effects

```python
pdp = effector.PDP(X_test, predict, nof_instances=5000, feature_names=bike_sharing.feature_names)
pdp.plot(feature=3, nof_ice=200)
```

![Feature effect plot](https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_1.png)

### Regional effects

```python
r_pdp = effector.RegionalPDP(X_test, predict, nof_instances=5000, feature_names=bike_sharing.feature_names)
r_pdp.summary(features=3)
```

```
Feature 3 - Full partition tree:
Node id: 0, name: hr, heter: 0.44 || nof_instances:  5000 || weight: 1.00
        Node id: 1, name: hr | workingday == 0.00, heter: 0.38 || nof_instances:  1588 || weight: 0.32
                Node id: 3, name: hr | workingday == 0.00 and temp <= 6.81, heter: 0.19 || nof_instances:   785 || weight: 0.16
                Node id: 4, name: hr | workingday == 0.00 and temp > 6.81, heter: 0.22 || nof_instances:   803 || weight: 0.16
        Node id: 2, name: hr | workingday != 0.00, heter: 0.30 || nof_instances:  3412 || weight: 0.68
                Node id: 5, name: hr | workingday != 0.00 and temp <= 6.81, heter: 0.21 || nof_instances:  1467 || weight: 0.29
                Node id: 6, name: hr | workingday != 0.00 and temp > 6.81, heter: 0.21 || nof_instances:  1945 || weight: 0.39
--------------------------------------------------
Feature 3 - Statistics per tree level:
Level 0, heter: 0.44
        Level 1, heter: 0.32 || heter drop : 0.12 (units), 27.28% (pcg)
                Level 2, heter: 0.21 || heter drop : 0.12 (units), 35.73% (pcg)
```

```python
[r_pdp.plot(feature=3, node_idx=i, nof_ice=200) for i in [1, 2]]
```

![Feature effect plot](https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_28_0.png)
![Feature effect plot](https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_28_1.png)

---

## Supported Methods

Effector implements global and regional effect methods:

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

If you use Effector, please cite it:

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

## License

Effector is released under the [MIT License](https://github.com/givasile/effector/blob/main/LICENSE).
