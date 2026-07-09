# Regional Effects (known black-box function)

- Author: [givasile](https://givasile.github.io/)
- Runtime: ~20 s
- Description: A gentle introduction to regional effects: PDP, RHALE and
  SHAP-DP and their regional counterparts applied to a known black-box
  function with an interaction term, under uncorrelated and correlated
  features.

This tutorial provides a gentle overview of Regional Effect methods and introduces the `Effector` package. Regional Effects serve as a bridge between local and global feature effects. Αs shown in [REPID](https://proceedings.mlr.press/v151/herbinger22a/herbinger22a.pdf), regional effect methods split the feature space in subregions where the feature interactions are minimized.

In this tutorial, we show how to use `Effector` to explain a black box function using regional effect plots. The tutorial is organized as follows:

- Introduction of the simulation example, using two datasets, one with uncorrelated and the other with correlated features. 
- Examine how PDP/RHALE/SHAP plots model the feature effect and how their regional counterpart can minimize feature interactions, providing better explanations.
- Show how each of these methods behaves under correlated and uncorrelated features.


```python
import numpy as np
import effector
```

    /home/givasile/github/packages/effector/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## Simulation example

### Data Generating Distribution

We will generate $N=1000$ examples with $D=3$ features, which are uniformly distributed as follows:


| Feature | Description                                | Distribution                 |
|-------|------------------------------------------|------------------------------|
| $x_1$   | Uniformly distributed between $-1$ and $1$ | $x_1 \sim \mathcal{U}(-1,1)$ |
| $x_2$   | Uniformly distributed between $-1$ and $1$ | $x_2 \sim \mathcal{U}(-1,1)$ |
| $x_3$   | Uniformly distributed between $-1$ and $1$ | $x_3 \sim \mathcal{U}(-1,1)$ |


For the correlated setting we keep the distributional assumptions for $x_2$ and $x_3$ but define $x_1$ such that it is identical to $x_3$ by: $x_1 = x_3$.


```python
def generate_dataset_uncorrelated(N):
    x1 = np.random.uniform(-1, 1, size=N)
    x2 = np.random.uniform(-1, 1, size=N)
    x3 = np.random.uniform(-1, 1, size=N)
    return np.stack((x1, x2, x3), axis=-1)

def generate_dataset_correlated(N):
    x3 = np.random.uniform(-1, 1, size=N)
    x2 = np.random.uniform(-1, 1, size=N)
    x1 = x3
    return np.stack((x1, x2, x3), axis=-1)

# generate the dataset for the uncorrelated and correlated setting
N = 1000
X_uncor_train = generate_dataset_uncorrelated(N)
X_uncor_test = generate_dataset_uncorrelated(10000)
X_cor_train = generate_dataset_correlated(N)
X_cor_test = generate_dataset_correlated(10000)
```

### Black-box function

We will use the following linear model with a subgroup-specific interaction term:
 $$ y = 3x_1I_{x_3>0} - 3x_1I_{x_3\leq0} + x_3$$ 
 
On a global level, there is a high heterogeneity for the features $x_1$ and $x_3$ due to their interaction with each other. However, this heterogeneity vanishes to 0 if the feature space is separated into subregions:

<center>

| Feature | Region      | Average Effect | Heterogeneity |
|---------|-------------|----------------|---------------|
| $x_1$   | $x_3>0$     | $3x_1$         | 0             |
| $x_1$   | $x_3\leq 0$ | $-3x_1$        | 0             |
| $x_2$   | all         | 0              | 0             |
| $x_3$   | $x_3>0$     | $x_3$          | 0             |
| $x_3$   | $x_3\leq 0$ | $x_3$          | 0             |

</center>


```python
def model(x):
    f = np.where(x[:,2] > 0, 3*x[:,0] + x[:,2], -3*x[:,0] + x[:,2])
    return f

def model_jac(x):
    dy_dx = np.zeros_like(x)
    
    ind1 = x[:, 2] > 0
    ind2 = x[:, 2] <= 0
    
    dy_dx[ind1, 0] = 3
    dy_dx[ind2, 0] = -3
    dy_dx[:, 2] = 1
    return dy_dx

```


```python
Y_uncor_train = model(X_uncor_train)
Y_uncor_test = model(X_uncor_test)
Y_cor_train = model(X_cor_train)
Y_cor_test = model(X_cor_test)      
```

---
## PDP

The PDP is defined as **_the average of the model's output over the entire dataset, while varying the feature of interest._**:

$$ \text{PDP}(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)] $$ 

and is approximated using the training data: 

$$ \hat{\text{PDP}}(x_s) = \frac{1}{N} \sum_{j=1}^N f(x_s, x^{(i)}_c) =  \frac{1}{N} \sum_{j=1}^N ICE^i(x_s)$$

The PDP is simply the average over the underlying ICE curves (local effects). The ICE curves show how the feature of interest influences the prediction of the ML model *for each single instance*. The ICE curves show the heterogeneity of the local effects.

### Uncorrelated setting

#### Global PDP


```python
pdp = effector.PDP(data=X_uncor_train, model=model, schema={"feature_names": ['x1','x2','x3'], "target_name": "Y"})
[pdp.plot(feature=i, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5]) for i in range(3)]
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_9_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_9_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_9_2.png)
    





    [None, None, None]



#### Feature importance and the one-click report

Before drilling into regional effects manually, we can let `effector` rank the features
for us. `importances()` returns, per feature, the *dispersion of the mean effect* — the
$\mu$-twin of heterogeneity. `effector.explain(...)` runs the whole pipeline in one call:
rank features by importance, plot the top ones, and automatically `find_regions` on the
heterogeneous ones, returning a serializable `Report`.


```python
# per-feature importance (mean-effect dispersion); x1 and x3 carry the interaction
print("PDP importances [x1, x2, x3]:", np.round(pdp.importances(), 3))

# one-click auto-explanation -> Report (values, serializable, self-contained HTML)
report = effector.explain(
    X_uncor_train, model, method="pdp",
    schema={"feature_names": ['x1', 'x2', 'x3'], "target_name": "Y"},
    nof_instances="all",
)
report.show()
```

    PDP importances [x1, x2, x3]: [0.  0.  0.6]
    
    PDP report — target: Y
    ============================================================
    feature                   importance     heter  #regions
    ------------------------------------------------------------
    x3                            0.6005    3.0444         7
    x1                            0.0000    3.2003         3
    x2                            0.0000    0.0000         1
    ============================================================
    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x3 🔹 [id: 0 | heter: 3.04 | inst: 1000 | w: 1.00]
        x1 < -0.00 🔹 [id: 1 | heter: 0.75 | inst: 498 | w: 0.50]
            x1 < -0.50 🔹 [id: 2 | heter: 0.18 | inst: 247 | w: 0.25]
            -0.50 ≤ x1 < -0.00 🔹 [id: 3 | heter: 0.17 | inst: 251 | w: 0.25]
        x1 ≥ -0.00 🔹 [id: 4 | heter: 0.76 | inst: 502 | w: 0.50]
            -0.00 ≤ x1 < 0.50 🔹 [id: 5 | heter: 0.19 | inst: 245 | w: 0.24]
            x1 ≥ 0.50 🔹 [id: 6 | heter: 0.19 | inst: 257 | w: 0.26]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 3.04
        Level 1🔹heter: 0.76 | 🔻2.29 (75.16%)
            Level 2🔹heter: 0.18 | 🔻0.57 (75.95%)
    
    
    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x1 🔹 [id: 0 | heter: 3.20 | inst: 1000 | w: 1.00]
        x3 < -0.00 🔹 [id: 1 | heter: 0.00 | inst: 500 | w: 0.50]
        x3 ≥ -0.00 🔹 [id: 2 | heter: 0.00 | inst: 500 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 3.20
        Level 1🔹heter: 0.00 | 🔻3.20 (100.00%)
    
    


The same survey as one picture: `effector.plot_triage` puts importance on the x-axis and heterogeneity on the y-axis. Here x1 lands **top-left** — its *global* mean effect is flat (near-zero importance) yet its heterogeneity is the highest of all features: the +3/-3 slopes cancel in the average. That corner is exactly where `find_regions` pays off — the effect is hiding, not absent.


```python
effector.plot_triage(pdp)
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_13_0.png)
    


#### Regional PDP

Regional PDP will search for explanations that minimize the interaction-related heterogeneity.


```python
# Regional effects are queried from the *global* effect via `find_regions`,
# which returns `Partition` value objects (nothing is stored on the effect).
# Here we use the *plural* form, `find_regions(features=...)`: one call that
# runs the search per feature and returns a `{feature_name: Partition}` dict
# (`features` also accepts "heterogeneous" to target only the features whose
# heter_score is at or above the median).
pdp = effector.PDP(
    data=X_uncor_train, model=model,
    schema={"feature_names": ['x1', 'x2', 'x3']},
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T,
    nof_instances="all",
)
pdp.fit("all", centering=True)

finder = effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.3, numerical_features_grid_size=10)
parts = pdp.find_regions(features="all", finder=finder)  # {name: Partition} — the plural form
partitions = [parts[name] for name in ["x1", "x2", "x3"]]
```


```python
partitions[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x1 🔹 [id: 0 | heter: 3.21 | inst: 1000 | w: 1.00]
        x3 < 0.00 🔹 [id: 1 | heter: 0.00 | inst: 500 | w: 0.50]
        x3 ≥ 0.00 🔹 [id: 2 | heter: 0.00 | inst: 500 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 3.21
        Level 1🔹heter: 0.00 | 🔻3.21 (100.00%)
    
    



```python
[partitions[0].plot(idx, heterogeneity="ice", centering=True, y_limits=[-5, 5]) for idx in [1, 2]]
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_17_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_17_1.png)
    





    [None, None]




```python
partitions[1].show()
```

    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    



```python
partitions[2].show()
```

    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x3 🔹 [id: 0 | heter: 3.04 | inst: 1000 | w: 1.00]
        x1 < 0.00 🔹 [id: 1 | heter: 0.75 | inst: 498 | w: 0.50]
            x1 < -0.60 🔹 [id: 2 | heter: 0.12 | inst: 205 | w: 0.20]
            -0.60 ≤ x1 < 0.00 🔹 [id: 3 | heter: 0.25 | inst: 293 | w: 0.29]
        x1 ≥ 0.00 🔹 [id: 4 | heter: 0.76 | inst: 502 | w: 0.50]
            0.00 ≤ x1 < 0.40 🔹 [id: 5 | heter: 0.12 | inst: 199 | w: 0.20]
            x1 ≥ 0.40 🔹 [id: 6 | heter: 0.26 | inst: 303 | w: 0.30]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 3.04
        Level 1🔹heter: 0.76 | 🔻2.29 (75.16%)
            Level 2🔹heter: 0.20 | 🔻0.55 (73.32%)
    
    



```python
partitions[2].plot(1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
partitions[2].plot(2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_20_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_20_1.png)
    


Triage, after: with `partitions=` the same plane shows the before→after story — arrows run from x1's global point to its leaves. Both leaves jump **right** (within each subregion the effect is strongly decisive, |slope| = 3) and **down** (the heterogeneity is explained). A hidden effect became two visible ones.


```python
effector.plot_triage(pdp, partitions={"x1": partitions[0]})
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_22_0.png)
    


#### Conclusion

For the Global PDP:

   * the average effect of $x_1$ is $0$ with some heterogeneity implied by the interaction with $x_1$. The heterogeneity is expressed with two opposite lines; $-3x_1$ when $x_1 \leq 0$ and $3x_1$ when $x_1 >0$
   * the average effect of $x_2$ to be $0$ without heterogeneity
   * the average effect of $x_3$ to be $x_3$ with some heterogeneity due to the interaction with $x_1$. The heterogeneity is expressed with a discontinuity around $x_3=0$, with either a positive or a negative offset depending on the value of $x_1^i$

--- 

For the Regional PDP:

* For $x_1$, the algorithm finds two regions, one for $x_3 \leq 0$ and one for $x_3 > 0$
  * when $x_3>0$ the effect is $3x_1$
  * when $x_3 \leq 0$, the effect is $-3x_1$
* For $x_2$ the algorithm does not find any subregion 
* For $x_3$, there is a change in the offset:
  * when $x_1>0$ the line is $x_3 - 3x_1^i$ in the first half and $x_3 + 3x_1^i$ later
  * when $x_1<0$ the line is $x_3 + 3x_1^i$ in the first half and $x_3 - 3x_1^i$ later

### Correlated setting

PDP assumes feature independence, therefore, it is *not* a good explanation method for the correlated case.
Due to this face, we expect the explanations to be identical with the uncorrelated case, which is not correct as we will see later in (RH)ALE plots.

#### Global PDP


```python
pdp = effector.PDP(data=X_cor_train, model=model, schema={"feature_names": ['x1','x2','x3'], "target_name": "Y"})
[pdp.plot(feature=i, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5]) for i in range(3)]
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_26_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_26_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_26_2.png)
    





    [None, None, None]



#### Regional-PDP


```python
pdp = effector.PDP(
    data=X_cor_train, model=model,
    schema={"feature_names": ['x1', 'x2', 'x3']},
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T,
    nof_instances="all",
)
pdp.fit("all", centering=True)

# finder="best" is the default Best() partitioner
partitions = {feat: pdp.find_regions(feat, finder="best") for feat in range(3)}
```


```python
partitions[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x1 🔹 [id: 0 | heter: 3.21 | inst: 1000 | w: 1.00]
        x3 < 0.00 🔹 [id: 1 | heter: 0.00 | inst: 501 | w: 0.50]
        x3 ≥ 0.00 🔹 [id: 2 | heter: 0.00 | inst: 499 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 3.21
        Level 1🔹heter: 0.00 | 🔻3.21 (100.00%)
    
    



```python
partitions[0].plot(1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
partitions[0].plot(2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_30_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_30_1.png)
    



```python
partitions[1].show()
```

    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    



```python
partitions[2].show()
```

    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x3 🔹 [id: 0 | heter: 2.93 | inst: 1000 | w: 1.00]
        x1 < 0.00 🔹 [id: 1 | heter: 0.72 | inst: 501 | w: 0.50]
            x1 < -0.50 🔹 [id: 2 | heter: 0.18 | inst: 245 | w: 0.24]
            -0.50 ≤ x1 < 0.00 🔹 [id: 3 | heter: 0.17 | inst: 256 | w: 0.26]
        x1 ≥ 0.00 🔹 [id: 4 | heter: 0.71 | inst: 499 | w: 0.50]
            0.00 ≤ x1 < 0.50 🔹 [id: 5 | heter: 0.17 | inst: 258 | w: 0.26]
            x1 ≥ 0.50 🔹 [id: 6 | heter: 0.19 | inst: 241 | w: 0.24]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 2.93
        Level 1🔹heter: 0.72 | 🔻2.21 (75.48%)
            Level 2🔹heter: 0.18 | 🔻0.54 (75.18%)
    
    



```python
partitions[2].plot(1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
partitions[2].plot(2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_33_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_33_1.png)
    


#### Conclusion

As expected, the global and the regional PDP explanations are identical with the uncorrelated case.

## (RH)ALE

(RH)ALE defines the feature effect as *the integral of the partial derivative of the model's output with respect to the feature of interest*:

$$\text{ALE}(x_s) = \int_{z=0}^{x_s} \mathbb{E}_{x_c|x_s=z}\left [ \frac{\partial f}{\partial x_s} (z, x_c) \right ] \partial z$$

The approximation is defined as:

$$\hat{\text{ALE}}(x_s) = \sum_{k=1}^{k_{x_s}} \frac{1}{| \mathcal{S}_k |} \sum_{i: x^{(i)} \in \mathcal{S}_k} \left [ f(z_k, x_c) - f(z_{k-1}, x_c) \right ]$$

$\hat{\text{ALE}}(x_s)$ uses a Riemannian sum to approximate the integral of $\text{ALE}(x_s)$. The axis of the $s$-th feature is split in $K$ bins (intervals) of equal size. In each bin, the average effect of the feature of interest is estimated using the instances that fall in the bin. The average effect in each bin is called bin-effect. 

Robust and Heterogeneity-aware ALE (RHALE) is a variant of ALE, proposed by [Gkolemis et. al](https://arxiv.org/abs/2309.11193), where the local effects are computed using automatic differentiation:

$$\hat{\text{RHALE}}(x_s) = \sum_{k=1}^{k_{x_s}} \frac{1}{ \left | \mathcal{S}_k \right |} \sum_{i: x^{(i)} \in \mathcal{S}_k} \frac{\partial f}{\partial x_s} (x_s^{(i)}, x_c^{(i)})$$

 In their paper, [Gkolemis et. al](https://arxiv.org/abs/2309.11193) showed that RHALE has specific advantages over ALE: (a) it ensures on-distribution sampling (b) an unbiased estimation of the heterogeneity and (c) an optimal trade-off between bias and variance. In our example, we will use the RHALE approximation.

### Uncorrelated setting

#### Global RHALE


```python
rhale = effector.RHALE(data=X_uncor_train, model=model, model_jac=model_jac, schema={"feature_names": ['x1','x2','x3'], "target_name": "Y"})

binning_method = effector.axis_partitioning.Fixed(10, min_points_per_bin=0)
rhale.fit(features="all", binning_method=binning_method, centering=True)

rhale.plot(feature=0, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
rhale.plot(feature=1, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
rhale.plot(feature=2, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_37_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_37_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_37_2.png)
    


#### Regional RHALE

The disadvantage of RHALE plot is that it does not reveal the type of heterogeneity. Therefore, Regional (RH)ALE plots are very helpful to identify the type of heterogeneity. Let's see that in practice:


```python
rhale = effector.RHALE(
    data=X_uncor_train, model=model, model_jac=model_jac,
    schema={"feature_names": ['x1', 'x2', 'x3']},
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T,
    nof_instances="all",
)
binning_method = effector.axis_partitioning.Fixed(11, min_points_per_bin=0)
rhale.fit("all", binning_method=binning_method, centering=True)

finder = effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.3, numerical_features_grid_size=10)
partitions = {feat: rhale.find_regions(feat, finder=finder) for feat in range(3)}
```


```python
partitions[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x1 🔹 [id: 0 | heter: 8.85 | inst: 1000 | w: 1.00]
        x3 < 0.00 🔹 [id: 1 | heter: 0.00 | inst: 500 | w: 0.50]
        x3 ≥ 0.00 🔹 [id: 2 | heter: 0.00 | inst: 500 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 8.85
        Level 1🔹heter: 0.00 | 🔻8.85 (100.00%)
    
    



```python
partitions[0].plot(1, heterogeneity="std", centering=True, y_limits=[-5, 5])
partitions[0].plot(2, heterogeneity="std", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_41_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_41_1.png)
    



```python
partitions[1].show()
```

    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    



```python
partitions[2].show()
```

    
    
    Feature 2 - Full partition tree:
    No splits found for feature 2
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    No splits found for feature 2
    
    


#### Conclusion

The explanations are similar to the ones obtained with the PDP plots. The average effect of $x_1$ is $0$ with some heterogeneity due to the interaction with $x_1$. The heterogeneity is shown with the red vertical bars. The average effect of $x_2$ is $0$ without heterogeneity. The average effect of $x_3$ is $x_3$, but in contrast with the PDP plots, there is no heterogeneity. The regional RHALE plots explain the type of the heterogeneity for $x_1$.

### Correlated setting

In the correlated setting $x_3=x_1$, therefore the model's formula becomes:

 $$ y = 3x_1I_{x_1>0} - 3x_1I_{x_1\leq0} + x_3$$ 

#### Global RHALE

RHALE plots respect feature correlations, therefore we expect the explanations to follow the formula above.


```python
rhale = effector.RHALE(data=X_cor_train, model=model, model_jac=model_jac, 
                       schema={"feature_names": ['x1','x2','x3'], "target_name": "Y"}, 
                       axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T)
binning_method = effector.axis_partitioning.Fixed(10, min_points_per_bin=0)
rhale.fit(features="all", binning_method=binning_method, centering=True)
```


```python
[rhale.plot(feature=i, show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5]) for i in range(3)]
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_47_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_47_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_47_2.png)
    





    [None, None, None]



#### Regional RHALE


```python
rhale = effector.RHALE(
    data=X_cor_train, model=model, model_jac=model_jac,
    schema={"feature_names": ['x1', 'x2', 'x3']},
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T,
    nof_instances="all",
)
binning_method = effector.axis_partitioning.Fixed(10, min_points_per_bin=0)
rhale.fit("all", binning_method=binning_method, centering=True)

finder = effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.3, numerical_features_grid_size=10)
partitions = {feat: rhale.find_regions(feat, finder=finder) for feat in range(3)}
```


```python
partitions[0].show()
```

    
    
    Feature 0 - Full partition tree:
    No splits found for feature 0
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    No splits found for feature 0
    
    



```python
partitions[1].show()
```

    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    



```python
partitions[2].show()
```

    
    
    Feature 2 - Full partition tree:
    No splits found for feature 2
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    No splits found for feature 2
    
    


#### Conclusion

The global RHALE plots follow the formula obtained after setting $x_1=x_3$ while the Regional (RH)ALE plot do not find any subregions in the correlated case.

## SHAP DP

### Uncorrelated setting

#### Global SHAP DP


```python
shap = effector.ShapDP(data=X_uncor_train, model=model, schema={"feature_names": ['x1','x2','x3'], "target_name": "Y"})
binning_method = effector.axis_partitioning.Fixed(nof_bins=5, min_points_per_bin=0)
shap.fit("all", binning_method=binning_method)
[shap.plot(feature=i, show_avg_output=False, y_limits=[-3, 3]) for i in range(3)]

```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_56_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_56_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_56_2.png)
    





    [None, None, None]



#### Regional SHAP-DP


```python
shap = effector.ShapDP(
    data=X_uncor_train, model=model,
    schema={"feature_names": ['x1', 'x2', 'x3']},
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T,
    nof_instances="all",
)
shap.fit("all", binning_method=effector.axis_partitioning.Fixed(nof_bins=5, min_points_per_bin=0))

finder = effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.6, numerical_features_grid_size=10)
partitions = {feat: shap.find_regions(feat, finder=finder) for feat in range(3)}
```


```python
partitions[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x1 🔹 [id: 0 | heter: 0.85 | inst: 1000 | w: 1.00]
        x3 < 0.00 🔹 [id: 1 | heter: 0.03 | inst: 500 | w: 0.50]
        x3 ≥ 0.00 🔹 [id: 2 | heter: 0.03 | inst: 500 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.85
        Level 1🔹heter: 0.03 | 🔻0.82 (96.47%)
    
    



```python
partitions[0].plot(1, heterogeneity="std", centering=True, y_limits=[-5, 5])
partitions[0].plot(2, heterogeneity="std", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_60_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_60_1.png)
    



```python
partitions[1].show()
```

    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    



```python
partitions[2].show()
```

    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x3 🔹 [id: 0 | heter: 0.77 | inst: 1000 | w: 1.00]
        x1 < 0.00 🔹 [id: 1 | heter: 0.28 | inst: 498 | w: 0.50]
        x1 ≥ 0.00 🔹 [id: 2 | heter: 0.30 | inst: 502 | w: 0.50]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.77
        Level 1🔹heter: 0.29 | 🔻0.48 (61.98%)
    
    


#### Conclusion

Global SHAP-DP:

   * the average effect of $x_1$ is $0$ with some heterogeneity implied by the interaction with $x_1$. The heterogeneity is expressed with two opposite lines; $-3x_1$ when $x_1 \leq 0$ and $3x_1$ when $x_1 >0$
   * the average effect of $x_2$ to be $0$ without heterogeneity
   * the average effect of $x_3$ to be $x_3$ with some heterogeneity due to the interaction with $x_1$. In contrast with other methods, SHAP spread the heterogeneity along the x-axis.
  
Regional SHAP-DP:


### Correlated setting

#### Global SHAP-DP


```python
shap = effector.ShapDP(data=X_cor_train, model=model, schema={"feature_names": ['x1','x2','x3'], "target_name": "Y"})

[shap.plot(feature=i, y_limits=[-3, 3]) for i in range(3)]
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_65_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_65_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_65_2.png)
    





    [None, None, None]



#### Regional SHAP


```python
shap = effector.ShapDP(
    data=X_cor_train, model=model,
    schema={"feature_names": ['x1', 'x2', 'x3']},
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T,
    nof_instances="all",
)
shap.fit("all")

finder = effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.6, numerical_features_grid_size=10)
partitions = {feat: shap.find_regions(feat, finder=finder) for feat in range(3)}
```


```python
partitions[0].show()
```

    
    
    Feature 0 - Full partition tree:
    No splits found for feature 0
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    No splits found for feature 0
    
    



```python
partitions[1].show()
```

    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    



```python
partitions[2].show()
```

    
    
    Feature 2 - Full partition tree:
    No splits found for feature 2
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    No splits found for feature 2
    
    

