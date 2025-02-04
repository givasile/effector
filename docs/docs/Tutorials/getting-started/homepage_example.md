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
    return y + np.random.normal(0, 1, x.shape[0])*.3
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


    
![png](homepage_example_files/homepage_example_8_0.png)
    


### d-PDP


```python
der_pdp = effector.DerPDP(X_test, predict, model_jac=jacobian, axis_limits=axis_limits)
der_pdp.plot(feature=0, heterogeneity="ice", dy_limits=dy_limits)
```


    
![png](homepage_example_files/homepage_example_10_0.png)
    


### RHALE


```python
binning_method = effector.binning_methods.Greedy(init_nof_bins=20, min_points_per_bin=10)
rhale = effector.RHALE(X_test, predict, jacobian, axis_limits=axis_limits)
rhale.fit(features=0, binning_method=binning_method)
rhale.plot(feature=0, heterogeneity=True, y_limits=y_limits, dy_limits=dy_limits)
```


    
![png](homepage_example_files/homepage_example_12_0.png)
    


### ALE


```python
binning_method = effector.binning_methods.Fixed(nof_bins=20)
ale = effector.ALE(X_test, predict, axis_limits=axis_limits)
ale.fit(features=0, binning_method=binning_method)
ale.plot(feature=0, heterogeneity=True, y_limits=y_limits, dy_limits=dy_limits)
```


    
![png](homepage_example_files/homepage_example_14_0.png)
    


### SHAP-DP


```python
shap_dp = effector.ShapDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
binning_method = effector.binning_methods.Greedy(init_nof_bins=20)
shap_dp.fit(features=0, binning_method=binning_method)
shap_dp.plot(feature=0, heterogeneity="shap_values", y_limits=y_limits)
```


    
![png](homepage_example_files/homepage_example_16_0.png)
    


## Regional Effect

### RegionalPDP


```python
effector.RegionalPDP(X_test, predict, axis_limits=axis_limits).summary(features=0)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 80.99it/s]

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 34.79 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.09 || nof_instances:  1000 || weight: 1.00
            Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.09 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 34.79
            Level 1, heter: 0.18 || heter drop : 34.61 (units), 99.48% (pcg)
    
    


    



```python
[effector.RegionalPDP(X_test, predict, axis_limits=axis_limits).plot(feature=0, node_idx=i, heterogeneity="ice", y_limits=y_limits) for i in range(3)]
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 68.90it/s]



    
![png](homepage_example_files/homepage_example_20_1.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 59.20it/s]



    
![png](homepage_example_files/homepage_example_20_3.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 63.94it/s]



    
![png](homepage_example_files/homepage_example_20_5.png)
    





    [None, None, None]



### RegionalDerPDP


```python
effector.RegionalDerPDP(X_test, predict, jacobian, axis_limits=axis_limits).summary(features=0)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 95.52it/s]

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 100.00 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
            Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 100.00
            Level 1, heter: 0.00 || heter drop : 100.00 (units), 100.00% (pcg)
    
    


    



```python
[effector.RegionalDerPDP(X_test, predict, jacobian, axis_limits=axis_limits).plot(feature=0, node_idx=i, heterogeneity="ice", dy_limits=dy_limits) for i in range(3)]
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 89.27it/s]



    
![png](homepage_example_files/homepage_example_23_1.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 94.77it/s]



    
![png](homepage_example_files/homepage_example_23_3.png)
    


    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 102.22it/s]



    
![png](homepage_example_files/homepage_example_23_5.png)
    





    [None, None, None]



### Regional RHALE


```python
effector.RegionalRHALE(X_test, predict, jacobian, axis_limits=axis_limits).summary(features=0)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.82it/s]

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 93.45 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
            Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 93.45
            Level 1, heter: 0.00 || heter drop : 93.45 (units), 100.00% (pcg)
    
    


    



```python
[effector.RegionalRHALE(X_test, predict, jacobian, axis_limits=axis_limits).plot(feature=0, centering=True, heterogeneity=True, node_idx=i, y_limits=y_limits, dy_limits=dy_limits) for i in range(3)]
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.92it/s]



    
![png](homepage_example_files/homepage_example_26_1.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.05it/s]



    
![png](homepage_example_files/homepage_example_26_3.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.03it/s]



    
![png](homepage_example_files/homepage_example_26_5.png)
    





    [None, None, None]



### RegionalALE


```python
effector.RegionalALE(X_test, predict, axis_limits=axis_limits).summary(features=0)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 72.76it/s]

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 114.57 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x_0 | x_1 <= 0.0, heter: 16.48 || nof_instances:  1000 || weight: 1.00
            Node id: 2, name: x_0 | x_1  > 0.0, heter: 17.41 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 114.57
            Level 1, heter: 33.89 || heter drop : 80.68 (units), 70.42% (pcg)
    
    


    



```python
[effector.RegionalALE(X_test, predict, axis_limits=axis_limits).plot(feature=0, centering=True, heterogeneity=True, node_idx=i, y_limits=y_limits, dy_limits=dy_limits) for i in range(3)]
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 75.34it/s]



    
![png](homepage_example_files/homepage_example_29_1.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.24it/s]



    
![png](homepage_example_files/homepage_example_29_3.png)
    


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.84it/s]



    
![png](homepage_example_files/homepage_example_29_5.png)
    





    [None, None, None]



### RegionalShapDP


```python
reg_shapdp = effector.RegionalShapDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
binning_method = effector.binning_methods.Greedy(init_nof_bins=10)
reg_shapdp.fit(features=0, binning_method=binning_method)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.77it/s]



```python
reg_shapdp.summary(features=0)
```

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 8.33 || nof_instances:  1000 || weight: 1.00
            Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
            Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 8.33
            Level 1, heter: 0.00 || heter drop : 8.33 (units), 99.94% (pcg)
    
    



```python
[reg_shapdp.plot(feature=0, node_idx=i, heterogeneity="shap_values") for i in range(3)]
```


    
![png](homepage_example_files/homepage_example_33_0.png)
    



    
![png](homepage_example_files/homepage_example_33_1.png)
    



    
![png](homepage_example_files/homepage_example_33_2.png)
    





    [None, None, None]




```python

```
