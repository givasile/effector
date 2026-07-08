# California Housing with TabPFN

- Author: [givasile](https://givasile.github.io/)
- Runtime: ~8 min (TabPFN on CPU)
- Description: Global and regional ALE effects on the California-housing
  dataset, using a TabPFN foundation model as the black box — a showcase of
  `effector` explaining a model that is expensive to query.


```python
import numpy as np
import keras
import tensorflow as tf
import effector
from sklearn.datasets import fetch_california_housing
import tabpfn
import time

california_housing = fetch_california_housing(as_frame=True)
```


```python
np.random.seed(21)
```


```python
print(california_housing.DESCR)
```


```python
feature_names = california_housing.feature_names
target_name= california_housing.target_names[0]
df = type(california_housing.frame)
```


```python
X = california_housing.data
y = california_housing.target
```


```python
print("Design matrix shape: {}".format(X.shape))
print("---------------------------------")
for col_name in X.columns:
    print("Feature: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(X[col_name].unique()), X[col_name].mean(), X[col_name].std(), X[col_name].min(), X[col_name].max()))
    
print("\nTarget shape: {}".format(y.shape))
print("---------------------------------")
print("Target: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(y.name, len(y.unique()), y.mean(), y.std(), y.min(), y.max()))
```


```python
def preprocess(X, y):
    # Compute mean and std for outlier detection
    X_mean = X.mean()
    X_std = X.std()
    
    # Exclude instances with any feature 2 std away from the mean
    mask = (X - X_mean).abs() <= 2 * X_std
    mask = mask.all(axis=1)
    
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Standardize X
    X_mean = X_filtered.mean()
    X_std = X_filtered.std()
    X_standardized = (X_filtered - X_mean) / X_std

    # Standardize y
    y_mean = y_filtered.mean()
    y_std = y_filtered.std()
    y_standardized = (y_filtered - y_mean) / y_std

    return X_standardized, y_standardized, X_mean, X_std, y_mean, y_std



# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = preprocess(X, y)
```


```python
def split(X_df, Y_df):
    # shuffle indices
    indices = np.arange(len(X_df))
    np.random.shuffle(indices)
    
    # data split
    train_size = int(0.8 * len(X_df))
    
    X_train = X_df.iloc[indices[:train_size]]
    Y_train = Y_df.iloc[indices[:train_size]]
    X_test = X_df.iloc[indices[train_size:]]
    Y_test = Y_df.iloc[indices[train_size:]]
    
    return X_train, Y_train, X_test, Y_test

# train/test split
X_train, Y_train, X_test, Y_test = split(X_df, Y_df)
```


```python
X_train = X_train[:500].to_numpy()
Y_train = Y_train[:500].to_numpy()
X_test = X_test[:500].to_numpy()
Y_test = Y_test[:500].to_numpy()
```


```python
model = tabpfn.TabPFNRegressor(n_jobs=7, device="cpu")
model.fit(X_train, Y_train)
```


```python
def model_forward(x):
    return model.predict(x)
```


```python
scale_y = {"mean": y_mean, "std": y_std}
scale_x_list =[{"mean": x_mean.iloc[i], "std": x_std.iloc[i]} for i in range(len(x_mean))]
```


```python
y_limits = [0, 4]
dy_limits = [-3, 3]
```

## Global effects


```python
ale = effector.ALE(data=X_test, model=model_forward, schema={"feature_names": feature_names, "target_name": target_name}, nof_instances="all")
```


```python
tic = time.time()
ale.fit("all", centering=True)
toc = time.time()
print(toc - tic)
```


```python
for i in range(8):
    ale.plot(feature=i, centering=True, scale_x=scale_x_list[i], scale_y=scale_y, y_limits=y_limits, dy_limits=dy_limits)
```

## Feature importance & one-click explanation

`importances()` ranks features by the dispersion of their mean ALE effect (the
μ-twin of heterogeneity), and `effector.explain(...)` runs the whole pipeline
(fit → rank → regional split of the heterogeneous features) into a single
serializable `Report`. TabPFN is expensive to query, so we pass a modest
`nof_instances` for the auto-explanation.


```python
# per-feature importance = dispersion of the mean ALE effect (already fitted above)
print("importances:", np.round(ale.importances(), 3))

# one-click auto-explanation -> Report (serializable; self-contained HTML)
report = effector.explain(
    X_test,
    model_forward,
    method="ale",
    schema={"feature_names": feature_names, "target_name": target_name},
    nof_instances=200,
)
report.show()
```

## Regional Effects


```python
ale_reg = effector.ALE(
    data=X_train,
    model=model_forward,
    schema={"feature_names": feature_names, "target_name": target_name},
    nof_instances="all",
)
ale_reg.fit("all", centering=True)
finder = "best"
partitions = {feat: ale_reg.find_regions(feat, finder=finder) for feat in [6, 7]}
```

## Latitude (south to north)


```python
partitions[6].show(scale_x_list=scale_x_list)
```


```python
partitions[6].plot(0, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

**Global Trend:** House prices decrease as we move north.  


```python
for node_idx in [1, 4]:
    partitions[6].plot(node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


```python
for node_idx in [2, 3, 5, 6]:
    partitions[6].plot(node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

**Global Trend:** House prices decrease as we move north.  

**Regional Trends:** Moreorless the same, with minor different curves.

## Longitude (west to east)


```python
partitions[7].show(scale_x_list=scale_x_list)
```


```python
partitions[7].plot(0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```

**Global Trend:** House prices decrease as we move east.  


```python
for node_idx in [1, 4]:
    partitions[7].plot(node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


```python
for node_idx in [2, 3, 5, 6]:
    partitions[7].plot(node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

**Global Trend:** House prices decrease as we move eastward.  

**Regional Trends:**  
- **South (latitude <= 35.85):** Prices drop more sharply in the second half from west to east.
  - **Latitude <= 33.65:** A different pattern emerges: prices drop sharply in the first half, then increase in the second half, suggesting a U-shaped trend.
  - **Latitude > 33.65:** The decrease in the price is almost linear as we move to east.
- **North (latitude > 35.85):** Prices drop more sharply in the first half from west to east.  
  - **AveRooms <= 6.19:** The pattern follows the overall northern trend, with a steep early drop.
  - **AveRooms > 6.19:** The decline becomes smoother and more linear
