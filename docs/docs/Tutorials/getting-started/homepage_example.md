```python
import numpy as np
import effector
```


```python
np.random.seed(21)
```


```python
X_test = np.random.uniform(-1, 1, (1000, 2))
axis_limits = np.array([[-1, -1], [1, 1]])
```


```python
def predict(x):
    y = np.zeros(x.shape[0])
    ind = x[:, 1] > 0
    y[ind] = 10*x[ind, 0]
    y[~ind] = -10*x[~ind, 0]
    return y + np.random.normal(0, 1, x.shape[0])
```


```python
def jacobian(x):
    J = np.zeros((x.shape[0], 2))
    ind = x[:, 1] > 0
    J[ind, 0] = 10
    J[~ind, 0] = -10
    return J
```


```python
y_limits = [-15, 15]
dy_limits = [-25, 25]
```

## Global Effect

### PDP


```python
effector.PDP(X_test, predict, axis_limits=axis_limits).plot(feature=0, heterogeneity="ice", y_limits=y_limits)
```

### d-PDP


```python
der_pdp = effector.DerPDP(X_test, predict, model_jac=jacobian, axis_limits=axis_limits)
der_pdp.plot(feature=0, heterogeneity="ice", dy_limits=dy_limits)
```

### RHALE


```python
binning_method = effector.binning_methods.Greedy(init_nof_bins=20, min_points_per_bin=10)
rhale = effector.RHALE(X_test, predict, jacobian, axis_limits=axis_limits)
rhale.fit(features=0, binning_method=binning_method)
rhale.plot(feature=0, heterogeneity=True, y_limits=y_limits, dy_limits=dy_limits)
```

### ALE


```python
binning_method = effector.binning_methods.Fixed(nof_bins=20)
ale = effector.ALE(X_test, predict, axis_limits=axis_limits)
ale.fit(features=0, binning_method=binning_method)
ale.plot(feature=0, heterogeneity=True, y_limits=y_limits, dy_limits=dy_limits)
```

### SHAP-DP


```python
shap_dp = effector.ShapDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
binning_method = effector.binning_methods.Greedy(init_nof_bins=20)
shap_dp.fit(features=0, binning_method=binning_method)
shap_dp.plot(feature=0, heterogeneity="shap_values", y_limits=y_limits)
```

## Regional Effect

### RegionalPDP


```python
effector.RegionalPDP(X_test, predict, axis_limits=axis_limits).summary(features=0)
```


```python
[effector.RegionalPDP(X_test, predict, axis_limits=axis_limits).plot(feature=0, node_idx=i, heterogeneity="ice", y_limits=y_limits) for i in range(3)]
```

### RegionalDerPDP


```python
effector.RegionalDerPDP(X_test, predict, jacobian, axis_limits=axis_limits).summary(features=0)
```


```python
[effector.RegionalDerPDP(X_test, predict, jacobian, axis_limits=axis_limits).plot(feature=0, node_idx=i, heterogeneity="ice", dy_limits=dy_limits) for i in range(3)]
```

### Regional RHALE


```python
effector.RegionalRHALE(X_test, predict, jacobian, axis_limits=axis_limits).summary(features=0)
```


```python
[effector.RegionalRHALE(X_test, predict, jacobian, axis_limits=axis_limits).plot(feature=0, centering=True, node_idx=i, y_limits=y_limits, dy_limits=dy_limits) for i in range(3)]
```

### RegionalALE


```python
effector.RegionalALE(X_test, predict, axis_limits=axis_limits).summary(features=0)
```


```python
[effector.RegionalALE(X_test, predict, axis_limits=axis_limits).plot(feature=0, centering=True, node_idx=i, y_limits=y_limits, dy_limits=dy_limits) for i in range(3)]
```

### RegionalShapDP


```python
reg_shapdp = effector.RegionalShapDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
binning_method = effector.binning_methods.Greedy(init_nof_bins=20)
reg_shapdp.fit(features=0, binning_method=binning_method)


```


```python
reg_shapdp.summary(features=0)
```


```python
[reg_shapdp.plot(feature=0, node_idx=i, heterogeneity="shap_values") for i in range(3)]
```
