# Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects

This repository contains the official code for the paper titled **Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects** implemented as a submodule of Effector.

## Table of Contents
- [Package Installation](#package-installation)
- [Empirical Evaluation](#empirical-evaluation)
- [Configuration File Options](#configuration-file-options)
- [CALM Method](#calm-method)
- [Analysis](#analysis)
- [User Study](#user-study)
- [Concept Visualization](#concept-visualization)
- [GAMI-Net Compatibility Note](#️-note-on-gami-net-compatibility)


## Package Installation

To install the CALM package (as effector module) and its dependencies, follow the steps below.


### Set Up a Python Environment

You can use either **Python's built-in `venv`** or **`conda`**.

#### Option A: Using `venv` (recommended for most users)

```bash
# Create and activate virtual environment
python3.10 -m venv calm-env
source calm-env/bin/activate   # On Linux/Mac
# .\calm-env\Scripts\activate  # On Windows
```

#### Option B: Using Conda
```bash
# Create a new conda environment with Python = 3.10
conda create -n calm-env python=3.10 -y
conda activate calm-env

# Change to project directory
cd code
```

### Upgrade pip and setuptools 

```bash
python -m pip install --upgrade pip setuptools
```

⚠️ **TensorFlow Compatibility Note**

Due to known issues with TensorFlow 2.19.0 on Windows, this project uses optional dependencies to ensure platform compatibility while matching the experimental setup used in the paper (TensorFlow 2.19.0).

### 🪟 Windows users
Use TensorFlow 2.18.0 for compatibility:


#### Normal Installation
To install the package normally (without editing the source code): 

```shell
# From the repo root (the folder that contains pyproject.toml)
pip install 'effector[calm,windows]'
```


#### Editable Installation

If you plan to modify the CALM source code:

```shell
# From the repo root (the folder that contains pyproject.toml)
pip install -e '.[calm,windows]' 
```

Run this from the repo root (the folder that contains pyproject.toml)

### 🐧 Linux / 🍎 macOS users
Use TensorFlow 2.19.0 to match the versions used in the paper:

#### Normal Installation
To install the package normally (without editing the source code): 

```shell
# From the repo root (the folder that contains pyproject.toml)
pip install 'effector[calm,linux-mac]'
```

#### Editable Installation

If you plan to modify the CALM source code:

```shell
# From the repo root (the folder that contains pyproject.toml)
pip install -e '.[calm,linux-mac]'
```

### Register Environment for Jupyter Notebooks

If you plan to run notebooks, register your virtual environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=<env-name> --display-name=<display-name>

   ```
   Then, when you open a notebook, select <virtual-env-display-name> as the kernel.

---

## Empirical Evaluation
The default configuration of the CALM method is:
* **Step 1**: XGBoost as the black-box model.
* **Step 2**: PDP-based heterogeneity with $d_{\text{max}} = 2$, $\epsilon = 0.2$, and $K$ remains unconstrained.
* **Step 3**: Uses standard gradient boosting.
  
All predictive results are obtained using standard 5-fold cross-validation.

### Synthetic example
- Notebook: `calm-experiments/synthetic_example.ipynb`
- Description: Comparison of CALM against EBM, $EB^{2}M$ methods on 3 synthetic regression datasets

### Evaluation on Real Datasets
All scripts should be executed from inside the `calm-experiments/` directory to ensure relative paths and imports work correctly.

#### Main Paper Results
For the main paper results the XGB, EBM, NAM, CALM, $EB^{2}M$, $NODE-GA^{2}M$, GAMI-Net methods were run in both regression and classification datasets. CALM is configured to run with the default configuration.

**Classification**
- script:
  ```bash
  python experiments.py --config configs/experiment_config_classification_main.yaml
  ```
- Description: Runs main methods on all classification datasets and saves results in `calm-experiments/results_classification_main.csv`

**Regression**
- script:
  ```bash
  python experiments.py --config configs/experiment_config_regression_main.yaml
  ```
- Description: Runs main methods on all regression datasets and saves results in `calm-experiments/results_regression_main.csv`

#### Supplementary Detailed Results
For the supplementary paper results the DNN, XGB, RF, EBM, NAM, SPLINE, $EB^{2}M$, $NODE-GA^{2}M$, GAMI-Net methods and 12 CALM variants shown in the below table were run in both regression and classification datasets.

| GAM \ BlackBox |   DNN         |   XGB     |   RF      |
|----------------|:-------------:|:---------:|:---------:|
| **EBM**            | RHALE, PDP    | PDP       | PDP       |
| **NAM**            | RHALE, PDP    | PDP       | PDP       |
| **SPLINE**         | RHALE, PDP    | PDP       | PDP       |


**Classification**
- script:
  ```bash
  python experiments.py --config configs/experiment_config_classification.yaml
  ```
- Description: Runs all methods on all classification datasets and saves results in `calm-experiments/results_classification.csv`

**Regression**
- script:
  ```bash
  python experiments.py --config configs/experiment_config_regression.yaml
  ```
- Description: Runs all methods on all regression datasets and saves results in `calm-experiments/results_regression.csv`

---

## Configuration File Options

Each experiment is controlled via a YAML configuration file (found in `calm-experiments/configs/`). Below are the supported options and their descriptions:

### Global Options

| **Key**            | **Type**  | **Description**                                                                 | **Example**                          |
|--------------------|-----------|----------------------------------------------------------------------------------|--------------------------------------|
| `task`             | string    | Type of learning task.                                                          | `"classification"` or `"regression"` |
| `stop_on_error`    | boolean   | If `true`, stops execution on the first error. If `false`, continues.           | `true`                               |
| `random_seed`      | integer   | Sets a fixed random seed for reproducibility.                                   | `42`                                 |
| `test_size`        | float     | Fraction of data reserved for testing (range: 0–1).                             | `0.2`                                |
| `output_file`      | string    | Path to output CSV where results are saved.                                     | `"results_classification_main.csv"`  |
| `append_to_file`   | boolean   | If `true`, results are appended to the file instead of overwriting.            | `true`                               |
| `kfold_n_splits`   | integer   | Number of folds for k-fold cross-validation (if used).                          | `5`                                  |


---

### Dataset Configuration

After the global configuration, define the datasets to be used in the experiment. Each dataset entry must include a `name` and the `module` path for loading like:

<pre><code>datasets:
  - name: Adult
    module: effector.calm.datasets.adult
  - name: COMPAS
    module: effector.calm.datasets.compas
  - name: HELOC
    module: effector.calm.datasets.heloc
</code></pre>

---

### Method Configuration

Specify which models to run on the selected datasets. Each method has a `name`, a `type`, and optionally `parameters` like:

<pre><code>methods:
  - name: XGBClassifier
    type: blackbox
  - name: NoInteractionsEBMClassifier
    type: maskedgam
    parameters: {}
  - name: CALMClassifier
    type: calm
    parameters:
      region_detector: ["RegionalPDP", "RegionalRHALE"]
      masked_gam_name: ["NoInteractionsEBMClassifier", "PyGAMClassifier", "MaskedNAMClassifier"]
      blackbox_model: ["DNNClassifier", "RFClassifier", "XGBClassifier"]
  - name: EBM2Classifier
    type: competitor
  - name: NodeGAM2Classifier
    type: competitor
    parameters:
      max_time: [300]
</code></pre>

| **Field**     | **Description**                                                                 |
|---------------|----------------------------------------------------------------------------------|
| `name`        | Name of the method (corresponds to a class in the codebase).                    |
| `type`        | Category of the model (`blackbox`, `maskedgam`, `calm`, `competitor`).          |
| `parameters`  | (Optional) Dictionary of method-specific hyperparameters.                       |

---

### 📏 Evaluation Metrics

List the metrics to compute after training as:

<pre><code>metrics: [accuracy, balanced_accuracy, f1]</code></pre>

---

You can create multiple YAML config files to easily run experiments with different dataset-method combinations or evaluation strategies.

---

## CALM Method
The core CALM method implementation is located in `effector/calm/calm.py`

---

## Analysis
To analyze and compare the predictive performance of all methods on real datasets, use the following notebook:

`calm-experiments/results_analysis.ipynb`  
  This notebook loads the results from the classification and regression experiments and generates the summary tables included in the main paper and supplementary material.

---

## User Study

The figures related to the user study are generated using the following notebook:

`calm-experiments/user_study.ipynb`  
  This notebook reproduces the visualizations presented in the user study section of the paper.

---

## Concept Visualization

The notebook `calm-experiments/concept_image.ipynb` reproduces Figures 1–2 from the paper. It visualizes how CALM models conditional feature effects, highlights interaction-based discontinuities, and illustrates region-specific contributions.


## ⚠️ Note on GAMI-Net Compatibility

The `GAMI-Net`, included as a **baseline competitor**, depends on a native binary (`lib_ebmcore_mac_x64.dylib`) compiled for **x86_64** architecture.

As a result, it is **not compatible with Apple Silicon Macs**, and attempting to run it on such systems will raise an architecture mismatch error.

This issue is specific to this external model and does **not affect the core methods or main contributions** of this work.

If you wish to run the experiments on Apple Silicon and skip GAMI-Net, please comment out its entry in the YAML configuration files, like so:
<pre><code>methods:
  # - name: GAMINetClassifier
  #   type: competitor
  # - name: GAMINetRegressor
  #   type: competitor
</code></pre>
