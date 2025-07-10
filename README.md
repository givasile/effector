# Effector

<p align="center">
  <img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/effector_logo.png" width="500"/>
</p>

[![PyPI version](https://badge.fury.io/py/effector.svg?icon=si%3Apython)](https://badge.fury.io/py/effector)
![Execute Tests](https://github.com/givasile/effector/actions/workflows/run_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/givasile/effector/branch/main/graph/badge.svg)](https://codecov.io/gh/givasile/effector)
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
# Initialize the Partial Dependence Plot (PDP) object
pdp = effector.PDP(
    X_test,  # Use the test set as background data
    predict,  # Prediction function
    feature_names=bike_sharing.feature_names,  # (optional) Feature names
    target_name=bike_sharing.target_name  # (optional) Target variable name
)

# Plot the effect of a feature
pdp.plot(
    feature=3,  # Select the 3rd feature (feature: hour)
    nof_ice=200,  # (optional) Number of Individual Conditional Expectation (ICE) curves to plot
    scale_x={"mean": bike_sharing.x_test_mu[3], "std": bike_sharing.x_test_std[3]},  # (optional) Scale x-axis
    scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},  # (optional) Scale y-axis
    centering=True,  # (optional) Center PDP and ICE curves
    show_avg_output=True,  # (optional) Display the average prediction
    y_limits=[-200, 1000]  # (optional) Set y-axis limits
)
```

![Feature effect plot](https://raw.githubusercontent.com/givasile/effector/main/docs/docs/notebooks/quickstart/readme_example_files/readme_example_3_0.png)

### Explain it with regional effect plots

```python
# Initialize the Regional Partial Dependence Plot (RegionalPDP)
r_pdp = effector.RegionalPDP(
    X_test,  # Test set data
    predict,  # Prediction function
    feature_names=bike_sharing.feature_names,  # Feature names
    target_name=bike_sharing.target_name  # Target variable name
)

# Summarize the subregions of the 3rd feature (temperature)
r_pdp.summary(
    features=3,  # Select the 3rd feature for the summary
    scale_x_list=[  # scale each feature with mean and std
        {"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]}
        for i in range(X_test.shape[1])
    ]
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
# Plot regional effects after the first-level split (workingday vs non-workingday)
for node_idx in [1, 2]:  # Iterate over the nodes of the first-level split
    r_pdp.plot(
        feature=3,  # Feature 3 (temperature)
        node_idx=node_idx,  # Node index (1: workingday, 2: non-workingday)
        nof_ice=200,  # Number of ICE curves
        scale_x_list=[  # Scale features by mean and std
            {"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]}
            for i in range(X_test.shape[1])
        ],
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},  # Scale the target
        y_limits=[-200, 1000]  # Set y-axis limits
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
# Plot regional effects after second-level splits (workingday vs non-workingday and hot vs cold temperature)
for node_idx in [3, 4, 5, 6]:  # Iterate over the nodes of the second-level splits
    r_pdp.plot(
        feature=3,  # Feature 3 (temperature)
        node_idx=node_idx,  # Node index (hot/cold temperature and workingday/non-workingday)
        nof_ice=200,  # Number of ICE curves
        scale_x_list=[  # Scale features by mean and std
            {"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]}
            for i in range(X_test.shape[1])
        ],
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},  # Scale target
        y_limits=[-200, 1000]  # Set y-axis limits
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

## Spotlight on `effector`

### 📚 Featured Publications  
- **Gkolemis, Vasilis, et al.**  
  *"Fast and Accurate Regional Effect Plots for Automated Tabular Data Analysis."*  
  [Proceedings of the VLDB Endowment](https://vldb.org/workshops/2024/proceedings/TaDA/TaDA.5.pdf) | ISSN 2150-8097  

### 🎤 Talks & Presentations  
- **LMU-IML Group Talk**  
  [Slides & Materials](https://github.com/givasile/effector-paper/tree/main/presentation-general) | [LMU-IML Research](https://www.slds.stat.uni-muenchen.de/research/explainable-ai.html)  
- **AIDAPT Plenary Meeting**  
  [Deep dive into effector](https://github.com/givasile/presentation-aidapt-xai-effector/tree/main)  
- **XAI World Conference 2024**  
  [Poster](https://github.com/givasile/effector-paper/blob/main/poster-general/main.pdf) | [Paper](https://github.com/givasile/effector-paper/blob/main/xai-conference-submission/effector_xai_conf.pdf)  

### 🌍 Adoption & Collaborations  
- **[AIDAPT Project](https://www.ai-dapt.eu/effector/)**  
  Leveraging `effector` for explainable AI solutions.  

### 🔍 Additional Resources  

- **Medium Post**  
  [Effector: An eXplainability Library for Global and Regional Effects](https://medium.com/@ntipakos/effector-bfe17206672c)

- **Courses & Lists**:  
  [IML Course @ LMU](https://slds-lmu.github.io/iml/)  
  [Awesome ML Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)  
  [Awesome XAI](https://github.com/wangyongjie-ntu/Awesome-explainable-AI)  
  [Best of ML Python](https://github.com/ml-tooling/best-of-ml-python)  

### 📚 Related Publications

Papers that have inspired `effector`:

- **REPID: Regional Effects in Predictive Models**  
  Herbinger et al., 2022 - [Link](https://proceedings.mlr.press/v151/herbinger22a)  

- **Decomposing Global Feature Effects Based on Feature Interactions**  
  Herbinger et al., 2023 - [Link](https://arxiv.org/pdf/2306.00541.pdf)  

- **RHALE: Robust Heterogeneity-Aware Effects**  
  Gkolemis Vasilis et al., 2023 - [Link](https://ebooks.iospress.nl/doi/10.3233/FAIA230354)  

- **DALE: Decomposing Global Feature Effects**  
  Gkolemis Vasilis et al., 2023 - [Link](https://proceedings.mlr.press/v189/gkolemis23a/gkolemis23a.pdf)  

- **Greedy Function Approximation: A Gradient Boosting Machine**  
  Friedman, 2001 - [Link](https://projecteuclid.org/euclid.aos/1013203451)  

- **Visualizing Predictor Effects in Black-Box Models**  
  Apley, 2016 - [Link](https://arxiv.org/abs/1612.08468)  

- **SHAP: A Unified Approach to Model Interpretation**  
  Lundberg & Lee, 2017 - [Link](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)  

- **Regionally Additive Models: Explainable-by-design models minimizing feature interactions**  
  Gkolemis Vasilis et al., 2023 - [Link](https://arxiv.org/abs/2309.12215)

---

## License

`effector` is released under the [MIT License](https://github.com/givasile/effector/blob/main/LICENSE).



---

## Powered by:

- **[AIDAPT](https://www.ai-dapt.eu/)**  
  <img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/aidapt_logo.png" width="130"/>

- **XMANAI**  
<img src="https://raw.githubusercontent.com/givasile/effector/main/docs/docs/static/xmanai_logo.jpg" width="70"/>