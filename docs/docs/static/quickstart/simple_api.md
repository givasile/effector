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
pdp = effector.PDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
pdp.plot(feature=0, y_limits=y_limits)
```


    
![png](simple_api_files/simple_api_8_0.png)
    


### RHALE


```python
rhale = effector.RHALE(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
rhale.plot(feature=0, y_limits=y_limits, dy_limits=dy_limits)
```


    
![png](simple_api_files/simple_api_10_0.png)
    


### SHAP-DP


```python
shapdp = effector.ShapDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
shapdp.plot(feature=0, y_limits=y_limits)
```


    
![png](simple_api_files/simple_api_12_0.png)
    


### ALE


```python
ale = effector.ALE(X_test, predict, axis_limits=axis_limits, nof_instances="all")
ale.plot(feature=0, y_limits=y_limits, dy_limits=dy_limits)
```


    
![png](simple_api_files/simple_api_14_0.png)
    


### d-PDP


```python
dpdp = effector.DerPDP(X_test, predict, model_jac=jacobian, axis_limits=axis_limits, nof_instances="all")
dpdp.plot(feature=0, dy_limits=dy_limits)
```


    
![png](simple_api_files/simple_api_16_0.png)
    


## Regional Effect

### RegionalPDP


```python
r_pdp = effector.RegionalPDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
r_pdp.summary(features=0)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 36.79it/s]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 35.71 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.09 | inst: 501 | w: 0.50]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.09 | inst: 499 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 35.71
        Level 1🔹heter: 0.09 | 🔻35.62 (99.74%)
    
    


    



```python
[r_pdp.plot(feature=0, node_idx=i, y_limits=y_limits) for i in range(1, 3)]
```


    
![png](simple_api_files/simple_api_20_0.png)
    



    
![png](simple_api_files/simple_api_20_1.png)
    





    [None, None]



### Regional RHALE


```python
r_rhale = effector.RegionalRHALE(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
r_rhale.summary(features=0)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.60it/s]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 97.38 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.00 | inst: 501 | w: 0.50]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.00 | inst: 499 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 97.38
        Level 1🔹heter: 0.00 | 🔻97.38 (100.00%)
    
    


    



```python
[r_rhale.plot(feature=0, node_idx=i, y_limits=y_limits, dy_limits=dy_limits) for i in range(1, 3)]
```


    
![png](simple_api_files/simple_api_23_0.png)
    



    
![png](simple_api_files/simple_api_23_1.png)
    





    [None, None]



### RegionalShapDP


```python
r_shapdp = effector.RegionalShapDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
r_shapdp.summary(features=0)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.36it/s]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 8.83 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.02 | inst: 501 | w: 0.50]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.02 | inst: 499 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 8.83
        Level 1🔹heter: 0.02 | 🔻8.81 (99.76%)
    
    


    



```python
[r_shapdp.plot(feature=0, node_idx=i) for i in range(1, 3)]
```


    
![png](simple_api_files/simple_api_26_0.png)
    



    
![png](simple_api_files/simple_api_26_1.png)
    





    [None, None]



### RegionalALE


```python
r_ale = effector.RegionalALE(X_test, predict, axis_limits=axis_limits, nof_instances="all")
r_ale.summary(features=0)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 33.36it/s]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 113.77 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 15.71 | inst: 501 | w: 0.50]
            x_1 ≤ -0.50 🔹 [id: 2 | heter: 12.91 | inst: 259 | w: 0.26]
            x_1 > -0.50 🔹 [id: 3 | heter: 15.36 | inst: 242 | w: 0.24]
        x_1 > 0.00 🔹 [id: 4 | heter: 17.01 | inst: 499 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 113.77
        Level 1🔹heter: 16.36 | 🔻97.42 (85.62%)
            Level 2🔹heter: 7.06 | 🔻9.29 (56.82%)
    
    


    



```python
[r_ale.plot(feature=0, node_idx=i, y_limits=y_limits, dy_limits=dy_limits) for i in range(1, 3)]
```


    
![png](simple_api_files/simple_api_29_0.png)
    



    
![png](simple_api_files/simple_api_29_1.png)
    





    [None, None]



### RegionalDerPDP


```python
r_dpdp = effector.RegionalDerPDP(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
r_dpdp.summary(features=0)
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 103.09it/s]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 100.00 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.00 | inst: 501 | w: 0.50]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.00 | inst: 499 | w: 0.50]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 100.00
        Level 1🔹heter: 0.00 | 🔻100.00 (100.00%)
    
    


    



```python
[r_dpdp.plot(feature=0, node_idx=i, dy_limits=dy_limits) for i in range(1, 3)]
```


    
![png](simple_api_files/simple_api_32_0.png)
    



    
![png](simple_api_files/simple_api_32_1.png)
    





    [None, None]




```python

```
