# Regional Effects (known black-box function)

This tutorial provides a gentle overview of Regional Effect methods and introduces the `Effector` package. Regional Effects serve as a bridge between local and global feature effects. Αs shown in [REPID](https://proceedings.mlr.press/v151/herbinger22a/herbinger22a.pdf), regional effect methods split the feature space in subregions where the feature interactions are minimized.

In this tutorial, we show how to use `Effector` to explain a black box function using regional effect plots. The tutorial is organized as follows:

- Introduction of the simulation example, using two datasets, one with uncorrelated and the other with correlated features. 
- Examine how PDP/RHALE/SHAP plots model the feature effect and how their regional counterpart can minimize feature interactions, providing better explanations.
- Show how each of these methods behaves under correlated and uncorrelated features.


```python
import numpy as np
import effector
```

## Simulation example

### Data Generating Distribution

We will generate $N=1000$ examples with $D=3$ features, which are uniformly distributed as follows:

<center>

| Feature | Description                                | Distribution                 |
|-------|------------------------------------------|------------------------------|
| $x_1$   | Uniformly distributed between $-1$ and $1$ | $x_1 \sim \mathcal{U}(-1,1)$ |
| $x_2$   | Uniformly distributed between $-1$ and $1$ | $x_2 \sim \mathcal{U}(-1,1)$ |
| $x_3$   | Uniformly distributed between $-1$ and $1$ | $x_3 \sim \mathcal{U}(-1,1)$ |

</center>

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
regional_rhale = effector.RegionalPDP(data=X_uncor_train, model=model, feature_names=['x1','x2','x3'], axis_limits=np.array([[-1,1],[-1,1],[-1,1]]).T)
regional_rhale.fit(heter_pcg_drop_thres=0.3, nof_candidate_splits_for_numerical=11, centering=True)
effector.binning_methods.DynamicProgramming()
```

    100%|██████████| 3/3 [00:00<00:00, 26.42it/s]





    <effector.binning_methods.DynamicProgramming at 0x7219c27ae500>




```python
pdp = effector.PDP(data=X_uncor_train, model=model, feature_names=['x1','x2','x3'], target_name="Y")
pdp.plot(feature=0, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
pdp.plot(feature=1, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
pdp.plot(feature=2, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_10_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_10_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_10_2.png)
    


#### Regional PDP

Regional PDP will search for explanations that minimize the interaction-related heterogeneity.


```python
regional_pdp = effector.RegionalPDP(data=X_uncor_train, model=model, feature_names=['x1','x2','x3'], axis_limits=np.array([[-1,1],[-1,1],[-1,1]]).T)
regional_pdp.fit(features="all", heter_pcg_drop_thres=0.3, nof_candidate_splits_for_numerical=11, centering=True)
```

    100%|██████████| 3/3 [00:00<00:00, 30.26it/s]



```python
regional_pdp.summary(features=0)
```

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x1, heter: 3.44 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x1 | x3 <= 0.09, heter: 0.94 || nof_instances:   562 || weight: 0.56
            Node id: 2, name: x1 | x3  > 0.09, heter: 0.07 || nof_instances:   438 || weight: 0.44
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 3.44
            Level 1, heter: 0.56 || heter drop : 2.88 (units), 83.72% (pcg)
    
    



```python
regional_pdp.plot(feature=0, node_idx=1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
regional_pdp.plot(feature=0, node_idx=2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_14_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_14_1.png)
    



```python
regional_pdp.summary(features=1)
```

    
    
    Feature 1 - Full partition tree:
    Node id: 0, name: x2, heter: 3.43 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 3.43
    
    



```python
regional_pdp.summary(features=2)
```

    
    
    Feature 2 - Full partition tree:
    Node id: 0, name: x3, heter: 3.10 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x3 | x1 <= -0.09, heter: 0.62 || nof_instances:   455 || weight: 0.46
            Node id: 2, name: x3 | x1  > -0.09, heter: 0.95 || nof_instances:   545 || weight: 0.55
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 3.10
            Level 1, heter: 0.80 || heter drop : 2.30 (units), 74.21% (pcg)
    
    



```python
regional_pdp.plot(feature=2, node_idx=1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
regional_pdp.plot(feature=2, node_idx=2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_17_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_17_1.png)
    


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
pdp = effector.PDP(data=X_cor_train, model=model, feature_names=['x1','x2','x3'], target_name="Y")
pdp.plot(feature=0, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
pdp.plot(feature=1, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
pdp.plot(feature=2, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_21_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_21_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_21_2.png)
    


#### Regional-PDP


```python
regional_pdp = effector.RegionalPDP(data=X_cor_train, model=model, feature_names=['x1','x2','x3'], axis_limits=np.array([[-1,1],[-1,1],[-1,1]]).T)
regional_pdp.fit(features="all", heter_pcg_drop_thres=0.4, nof_candidate_splits_for_numerical=11)
```

    100%|██████████| 3/3 [00:00<00:00, 32.24it/s]



```python
regional_pdp.summary(features=0)
```

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x1, heter: 3.44 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x1 | x3 <= -0.09, heter: 0.07 || nof_instances:   475 || weight: 0.47
            Node id: 2, name: x1 | x3  > -0.09, heter: 0.95 || nof_instances:   525 || weight: 0.53
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 3.44
            Level 1, heter: 0.53 || heter drop : 2.91 (units), 84.61% (pcg)
    
    



```python
regional_pdp.plot(feature=0, node_idx=1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
regional_pdp.plot(feature=0, node_idx=2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_25_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_25_1.png)
    



```python
regional_pdp.summary(features=1)
```

    
    
    Feature 1 - Full partition tree:
    Node id: 0, name: x2, heter: 0.94 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x2 | x1 <= 0.45, heter: 0.31 || nof_instances:   748 || weight: 0.75
            Node id: 2, name: x2 | x1  > 0.45, heter: 0.41 || nof_instances:   252 || weight: 0.25
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 0.94
            Level 1, heter: 0.33 || heter drop : 0.61 (units), 64.71% (pcg)
    
    



```python
regional_pdp.summary(features=2)
```

    
    
    Feature 2 - Full partition tree:
    Node id: 0, name: x3, heter: 2.90 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x3 | x1 <= -0.09, heter: 0.61 || nof_instances:   475 || weight: 0.47
            Node id: 2, name: x3 | x1  > -0.09, heter: 0.80 || nof_instances:   525 || weight: 0.53
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 2.90
            Level 1, heter: 0.71 || heter drop : 2.19 (units), 75.48% (pcg)
    
    



```python
regional_pdp.plot(feature=2, node_idx=1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
regional_pdp.plot(feature=2, node_idx=2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_28_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_28_1.png)
    


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
rhale = effector.RHALE(data=X_uncor_train, model=model, model_jac=model_jac, feature_names=['x1','x2','x3'], target_name="Y")

binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=0)
rhale.fit(features="all", binning_method=binning_method, centering=True)

rhale.plot(feature=0, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
rhale.plot(feature=1, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
rhale.plot(feature=2, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_32_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_32_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_32_2.png)
    


#### Regional RHALE

The disadvantage of RHALE plot is that it does not reveal the type of heterogeneity. Therefore, Regional (RH)ALE plots are very helpful to identify the type of heterogeneity. Let's see that in practice:


```python
regional_rhale = effector.RegionalRHALE(
    data=X_uncor_train, 
    model=model, 
    model_jac= model_jac, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T) 

binning_method = effector.binning_methods.Fixed(11, min_points_per_bin=0)
regional_rhale.fit(
    features="all",
    heter_pcg_drop_thres=0.6,
    binning_method=binning_method,
    nof_candidate_splits_for_numerical=11
)

```

    100%|██████████| 3/3 [00:00<00:00, 10.67it/s]

    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.


    



```python
regional_rhale.summary(features=0)
```

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x1, heter: 8.91 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x1 | x3 <= 0.09, heter: 2.29 || nof_instances:   562 || weight: 0.56
            Node id: 2, name: x1 | x3  > 0.09, heter: 0.00 || nof_instances:   438 || weight: 0.44
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 8.91
            Level 1, heter: 1.29 || heter drop : 7.62 (units), 85.52% (pcg)
    
    



```python
regional_rhale.plot(feature=0, node_idx=1, heterogeneity="std", centering=True, y_limits=[-5, 5])
regional_rhale.plot(feature=0, node_idx=2, heterogeneity="std", centering=True, y_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_36_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_36_1.png)
    



```python
regional_rhale.summary(features=1)
```

    
    
    Feature 1 - Full partition tree:
    Node id: 0, name: x2, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 0.00
    
    



```python
regional_rhale.summary(features=2)
```

    
    
    Feature 2 - Full partition tree:
    Node id: 0, name: x3, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 0.00
    
    


#### Conclusion

The explanations are similar to the ones obtained with the PDP plots. The average effect of $x_1$ is $0$ with some heterogeneity due to the interaction with $x_1$. The heterogeneity is shown with the red vertical bars. The average effect of $x_2$ is $0$ without heterogeneity. The average effect of $x_3$ is $x_3$, but in contrast with the PDP plots, there is no heterogeneity. The regional RHALE plots explain the type of the heterogeneity for $x_1$.

### Correlated setting

In the correlated setting $x_3=x_1$, therefore the model's formula becomes:

 $$ y = 3x_1I_{x_1>0} - 3x_1I_{x_1\leq0} + x_3$$ 

#### Global RHALE

RHALE plots respect feature correlations, therefore we expect the explanations to follow the formula above.


```python
rhale = effector.RHALE(data=X_cor_train, model=model, model_jac=model_jac, 
                       feature_names=['x1','x2','x3'], 
                       target_name="Y", 
                       axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T)
binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=0)
rhale.fit(features="all", binning_method=binning_method, centering=True)
```


```python
rhale.plot(feature=0, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
rhale.plot(feature=1, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
rhale.plot(feature=2, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_42_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_42_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_42_2.png)
    


#### Regional RHALE


```python
regional_rhale = effector.RegionalRHALE(
    data=X_cor_train, 
    model=model, 
    model_jac= model_jac, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T) 

binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=0)
regional_rhale.fit(
    features="all",
    heter_pcg_drop_thres=0.6,
    binning_method=binning_method,
    nof_candidate_splits_for_numerical=10
)

```

     67%|██████▋   | 2/3 [00:00<00:00, 13.53it/s]

    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.


    100%|██████████| 3/3 [00:00<00:00, 13.21it/s]



```python
regional_rhale.summary(features=0)
```

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x1, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 0.00
    
    



```python
regional_rhale.summary(features=1)
```

    
    
    Feature 1 - Full partition tree:
    Node id: 0, name: x2, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 0.00
    
    



```python
regional_rhale.summary(features=2)
```

    
    
    Feature 2 - Full partition tree:
    Node id: 0, name: x3, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 0.00
    
    


#### Conclusion

The global RHALE plots follow the formula obtained after setting $x_1=x_3$ while the Regional (RH)ALE plot do not find any subregions in the correlated case.

## SHAP DP

### Uncorrelated setting

#### Global SHAP DP


```python
shap = effector.ShapDP(data=X_uncor_train, model=model, feature_names=['x1','x2','x3'], target_name="Y")

shap.plot(feature=0, centering=True, heterogeneity="shap_values", show_avg_output=False, y_limits=[-3, 3])
shap.plot(feature=1, centering=True, heterogeneity="shap_values", show_avg_output=False, y_limits=[-3, 3])
shap.plot(feature=2, centering=True, heterogeneity="shap_values", show_avg_output=False, y_limits=[-3, 3])
```

    /home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/numpy/_core/fromnumeric.py:4008: RuntimeWarning: Degrees of freedom <= 0 for slice
      return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    /home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/numpy/_core/_methods.py:175: RuntimeWarning: invalid value encountered in divide
      arrmean = um.true_divide(arrmean, div, out=arrmean,
    /home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/numpy/_core/_methods.py:210: RuntimeWarning: invalid value encountered in scalar divide
      ret = ret.dtype.type(ret / rcount)



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_51_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_51_2.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_51_3.png)
    


#### Regional SHAP-DP


```python
regional_shap = effector.RegionalShapDP(
    data=X_uncor_train, 
    model=model, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T) 

regional_shap.fit(
    features="all",
    heter_pcg_drop_thres=0.6,
    nof_candidate_splits_for_numerical=11
)
```

      0%|          | 0/3 [00:00<?, ?it/s]

    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.


     33%|███▎      | 1/3 [00:00<00:00,  3.13it/s]

    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.


     67%|██████▋   | 2/3 [00:00<00:00,  3.54it/s]

    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.


    100%|██████████| 3/3 [00:00<00:00,  3.59it/s]

    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.
    Error: Input array contains only NaN values. This is probably because in all bins there is at most one point, which is not enough to compute the bin variance. Please consider decreasing the number of bins or changing the bin splitting strategy.


    



```python
regional_shap.summary(0)
```

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x1, heter: 0.65 || nof_instances:   100 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 0.65
    
    



```python
regional_shap.plot(feature=0, node_idx=1, heterogeneity="std", centering=True, y_limits=[-5, 5])
regional_shap.plot(feature=0, node_idx=2, heterogeneity="std", centering=True, y_limits=[-5, 5])
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[35], line 1
    ----> 1 regional_shap.plot(feature=0, node_idx=1, heterogeneity="std", centering=True, y_limits=[-5, 5])
          2 regional_shap.plot(feature=0, node_idx=2, heterogeneity="std", centering=True, y_limits=[-5, 5])


    File ~/github/packages/effector/effector/regional_effect_shap.py:201, in RegionalShapDP.plot(self, feature, node_idx, heterogeneity, centering, nof_points, scale_x_list, scale_y, nof_shap_values, show_avg_output, y_limits, only_shap_values)
        199 kwargs = locals()
        200 kwargs.pop("self")
    --> 201 return self._plot(kwargs)


    File ~/github/packages/effector/effector/regional_effect.py:251, in RegionalEffectBase._plot(self, kwargs)
        248 self.refit(kwargs["feature"])
        250 # select only the three out of all kwargs
    --> 251 fe_method = self._create_fe_object(kwargs["feature"], kwargs["node_idx"], kwargs["scale_x_list"])
        253 kwargs_fitting = copy.deepcopy(self.kwargs_fitting)
        254 kwargs_fitting['centering'] = kwargs["centering"]


    File ~/github/packages/effector/effector/regional_effect.py:152, in RegionalEffectBase._create_fe_object(self, feature, node_idx, scale_x_list)
        150 assert feature_tree is not None, "Feature {} has no splits".format(feature)
        151 node = feature_tree.get_node_by_idx(node_idx)
    --> 152 name = feature_tree.scale_node_name(node.name, scale_x_list)
        153 data = node.info["data"]
        154 data_effect = node.info["data_effect"]


    AttributeError: 'NoneType' object has no attribute 'name'



```python
regional_shap.summary(features=1)
```

    Feature 1 - Full partition tree:
    Node id: 0, name: x2, heter: 0.00 || nof_instances:   100 || weight: 1.00
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 0.00



```python
regional_shap.summary(features=2)
```

    Feature 2 - Full partition tree:
    Node id: 0, name: x3, heter: 0.73 || nof_instances:   100 || weight: 1.00
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 0.73


#### Conclusion

Global SHAP-DP:

   * the average effect of $x_1$ is $0$ with some heterogeneity implied by the interaction with $x_1$. The heterogeneity is expressed with two opposite lines; $-3x_1$ when $x_1 \leq 0$ and $3x_1$ when $x_1 >0$
   * the average effect of $x_2$ to be $0$ without heterogeneity
   * the average effect of $x_3$ to be $x_3$ with some heterogeneity due to the interaction with $x_1$. In contrast with other methods, SHAP spread the heterogeneity along the x-axis.
  
Regional SHAP-DP:


### Correlated setting

#### Global SHAP-DP


```python
shap = effector.ShapDP(data=X_cor_train, model=model, feature_names=['x1','x2','x3'], target_name="Y")

shap.plot(feature=0, centering=True, heterogeneity="shap_values", show_avg_output=False, y_limits=[-3, 3])
shap.plot(feature=1, centering=True, heterogeneity="shap_values", show_avg_output=False, y_limits=[-3, 3])
shap.plot(feature=2, centering=True, heterogeneity="shap_values", show_avg_output=False, y_limits=[-3, 3])
```


    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_60_0.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_60_1.png)
    



    
![png](03_regional_effects_synthetic_f_files/03_regional_effects_synthetic_f_60_2.png)
    


#### Regional SHAP


```python
regional_shap = effector.RegionalShapDP(
    data=X_cor_train, 
    model=model, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T) 

regional_shap.fit(
    features="all",
    heter_pcg_drop_thres=0.6,
    nof_candidate_splits_for_numerical=11
)
```

    100%|██████████| 3/3 [00:03<00:00,  1.11s/it]



```python
regional_shap.summary(0)
```

    Feature 0 - Full partition tree:
    Node id: 0, name: x1, heter: 0.09 || nof_instances:   100 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 0.09



```python
regional_shap.summary(1)
```

    Feature 1 - Full partition tree:
    Node id: 0, name: x2, heter: 0.00 || nof_instances:   100 || weight: 1.00
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 0.00



```python
regional_shap.summary(2)
```

    Feature 2 - Full partition tree:
    Node id: 0, name: x3, heter: 0.09 || nof_instances:   100 || weight: 1.00
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 0.09


#### Conclusion


