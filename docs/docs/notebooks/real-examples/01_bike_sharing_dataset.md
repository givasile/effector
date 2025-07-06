# Bike-Sharing Dataset

This notebook analyzes the Capital Bikeshare system's rental data from 2011-2012. We'll explore how various factors influence bike rental patterns using advanced machine learning techniques.
The [Bike-Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) contains:
- 17,379 hourly records
- 14 features including temporal and weather information
- Target variable: hourly bike rental count


```python
import effector
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

    2025-06-24 12:12:26.707704: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


## Preprocess the data


```python
from ucimlrepo import fetch_ucirepo
bike_sharing_dataset = fetch_ucirepo(id=275)
X = bike_sharing_dataset.data.features
y = bike_sharing_dataset.data.targets
```


```python
X = X.drop(["dteday", "atemp"], axis=1)
```


```python
print("Design matrix shape: {}".format(X.shape))
print("---------------------------------")
for i, col_name in enumerate(X.columns):
    print("x_{} {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(i, col_name, len(X[col_name].unique()), X[col_name].mean(), X[col_name].std(), X[col_name].min(), X[col_name].max()))
    
print("\nTarget shape: {}".format(y.shape))
print("---------------------------------")
for col_name in y.columns:
    print("Target: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(y[col_name].unique()), y[col_name].mean(), y[col_name].std(), y[col_name].min(), y[col_name].max()))
```

    Design matrix shape: (17379, 11)
    ---------------------------------
    x_0 season         , unique:    4, Mean:   2.50, Std:   1.11, Min:   1.00, Max:   4.00
    x_1 yr             , unique:    2, Mean:   0.50, Std:   0.50, Min:   0.00, Max:   1.00
    x_2 mnth           , unique:   12, Mean:   6.54, Std:   3.44, Min:   1.00, Max:  12.00
    x_3 hr             , unique:   24, Mean:  11.55, Std:   6.91, Min:   0.00, Max:  23.00
    x_4 holiday        , unique:    2, Mean:   0.03, Std:   0.17, Min:   0.00, Max:   1.00
    x_5 weekday        , unique:    7, Mean:   3.00, Std:   2.01, Min:   0.00, Max:   6.00
    x_6 workingday     , unique:    2, Mean:   0.68, Std:   0.47, Min:   0.00, Max:   1.00
    x_7 weathersit     , unique:    4, Mean:   1.43, Std:   0.64, Min:   1.00, Max:   4.00
    x_8 temp           , unique:   50, Mean:   0.50, Std:   0.19, Min:   0.02, Max:   1.00
    x_9 hum            , unique:   89, Mean:   0.63, Std:   0.19, Min:   0.00, Max:   1.00
    x_10 windspeed      , unique:   30, Mean:   0.19, Std:   0.12, Min:   0.00, Max:   0.85
    
    Target shape: (17379, 1)
    ---------------------------------
    Target: cnt            , unique:  869, Mean: 189.46, Std: 181.39, Min:   1.00, Max: 977.00



```python
def preprocess(X, y):
    # Standarize X
    X_df = X
    x_mean = X_df.mean()
    x_std = X_df.std()
    X_df = (X_df - X_df.mean()) / X_df.std()

    # Standarize Y
    Y_df = y
    y_mean = Y_df.mean()
    y_std = Y_df.std()
    Y_df = (Y_df - Y_df.mean()) / Y_df.std()
    return X_df, Y_df, x_mean, x_std, y_mean, y_std

# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = preprocess(X, y)
```


```python
def split(X_df, Y_df):
    # shuffle indices
    indices = X_df.index.tolist()
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

## Fit a Neural Network


```python
# Train - Evaluate - Explain a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=20, verbose=1)
model.evaluate(X_train, Y_train, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)

```

    Epoch 1/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - loss: 0.6231 - mae: 0.5745 - root_mean_squared_error: 0.7853
    Epoch 2/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.3870 - mae: 0.4506 - root_mean_squared_error: 0.6219
    Epoch 3/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - loss: 0.2976 - mae: 0.3851 - root_mean_squared_error: 0.5454
    Epoch 4/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - loss: 0.2237 - mae: 0.3326 - root_mean_squared_error: 0.4728
    Epoch 5/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step - loss: 0.1619 - mae: 0.2836 - root_mean_squared_error: 0.4023
    Epoch 6/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.1193 - mae: 0.2386 - root_mean_squared_error: 0.3451
    Epoch 7/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step - loss: 0.0906 - mae: 0.2075 - root_mean_squared_error: 0.3009
    Epoch 8/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0753 - mae: 0.1895 - root_mean_squared_error: 0.2745
    Epoch 9/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 0.0669 - mae: 0.1784 - root_mean_squared_error: 0.2586
    Epoch 10/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.0610 - mae: 0.1703 - root_mean_squared_error: 0.2469
    Epoch 11/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - loss: 0.0554 - mae: 0.1614 - root_mean_squared_error: 0.2353
    Epoch 12/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.0500 - mae: 0.1524 - root_mean_squared_error: 0.2235
    Epoch 13/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 0.0462 - mae: 0.1459 - root_mean_squared_error: 0.2149
    Epoch 14/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.0431 - mae: 0.1407 - root_mean_squared_error: 0.2075
    Epoch 15/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 0.0410 - mae: 0.1372 - root_mean_squared_error: 0.2026
    Epoch 16/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 14ms/step - loss: 0.0399 - mae: 0.1360 - root_mean_squared_error: 0.1996
    Epoch 17/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step - loss: 0.0381 - mae: 0.1325 - root_mean_squared_error: 0.1952
    Epoch 18/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.0378 - mae: 0.1323 - root_mean_squared_error: 0.1945
    Epoch 19/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 14ms/step - loss: 0.0376 - mae: 0.1325 - root_mean_squared_error: 0.1940
    Epoch 20/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - loss: 0.0378 - mae: 0.1331 - root_mean_squared_error: 0.1944
    [1m435/435[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0420 - mae: 0.1422 - root_mean_squared_error: 0.2048
    [1m109/109[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0701 - mae: 0.1725 - root_mean_squared_error: 0.2633





    [0.06317276507616043, 0.16644978523254395, 0.25134193897247314]



We train a deep fully-connected Neural Network with 3 hidden layers for \(20\) epochs. 
The model achieves a root mean squared error on the test of about $0.24$ units, that corresponds to approximately \(0.26 * 181 = 47\) counts.

## Explain


```python
def model_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()

def model_forward(x):
    return model(x).numpy().squeeze()
```


```python
scale_y = {"mean": y_mean.iloc[0], "std": y_std.iloc[0]}
scale_x_list =[{"mean": x_mean.iloc[i], "std": x_std.iloc[i]} for i in range(len(x_mean))]
scale_x = scale_x_list[3]
feature_names = X_df.columns.to_list()
target_name = "bike-rentals"
y_limits=[-200, 800]
dy_limits = [-300, 300]
```


```python
scale_x_list[8]["mean"] += 8
scale_x_list[8]["std"] *= 47

scale_x_list[9]["std"] *= 100
scale_x_list[10]["std"] *= 67
```

## PDP - analyze all features

We start by examining all relevant features. Feature effect methods are generally more meaningful for numerical features, so we focus on them.
Relevant features:

- `month`  
- `hr`  
- `weekday`  
- `workingday`  
- `temp`  
- `humidity`  
- `windspeed`


```python
pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=2000)
for i in [2, 3, 8, 9, 10]:
    pdp.plot(feature=i, centering=True, scale_x=scale_x_list[i], scale_y=scale_y, show_avg_output=True, nof_ice=200, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_16_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_16_1.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_16_2.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_16_3.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_16_4.png)
    


We observe that features: `hour`, `temperature` and `humidity` have an intersting structure. Out of them `hour` has by far the most influence on the output, so it makes sensce to focus on it further.

## Feature `hour`

### PDP - global


```python
pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=5000)
pdp.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True, nof_ice=200)
```

    2025-06-24 12:12:58.719189: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-06-24 12:12:58.843474: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-06-24 12:12:58.937027: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-06-24 12:12:59.827863: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-06-24 12:12:59.939911: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_20_1.png)
    


### PDP - regional


```python
regional_pdp = effector.RegionalPDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, nof_instances=5_000)
regional_pdp.summary(features=3, scale_x_list=scale_x_list)
```

    100%|██████████| 1/1 [00:02<00:00,  2.26s/it]

    
    
    Feature 3 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    hr 🔹 [id: 0 | heter: 0.24 | inst: 5000 | w: 1.00]
        workingday = 0.00 🔹 [id: 1 | heter: 0.13 | inst: 1548 | w: 0.31]
            temp ≤ 4.50 🔹 [id: 2 | heter: 0.06 | inst: 618 | w: 0.12]
            temp > 4.50 🔹 [id: 3 | heter: 0.10 | inst: 930 | w: 0.19]
        workingday ≠ 0.00 🔹 [id: 4 | heter: 0.12 | inst: 3452 | w: 0.69]
            yr = 0.00 🔹 [id: 5 | heter: 0.06 | inst: 1719 | w: 0.34]
            yr ≠ 0.00 🔹 [id: 6 | heter: 0.11 | inst: 1733 | w: 0.35]
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.24
        Level 1🔹heter: 0.13 | 🔻0.12 (48.45%)
            Level 2🔹heter: 0.08 | 🔻0.04 (32.86%)
    
    


    



```python
for node_idx in [1, 4]:
    regional_pdp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_23_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_23_1.png)
    



```python
for node_idx in [2, 3, 5, 6]:
    regional_pdp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_24_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_24_1.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_24_2.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_24_3.png)
    


### RHALE - global


```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
rhale.plot(feature=3, heterogeneity="std", centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_26_0.png)
    


### RHALE - regional


```python
regional_rhale = effector.RegionalRHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
regional_rhale.summary(features=3, scale_x_list=scale_x_list)
```

    100%|██████████| 1/1 [00:03<00:00,  3.60s/it]

    
    
    Feature 3 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    hr 🔹 [id: 0 | heter: 5.68 | inst: 13903 | w: 1.00]
        workingday = 0.00 🔹 [id: 1 | heter: 0.75 | inst: 4385 | w: 0.32]
            temp ≤ 6.81 🔹 [id: 2 | heter: 0.44 | inst: 2187 | w: 0.16]
            temp > 6.81 🔹 [id: 3 | heter: 0.60 | inst: 2198 | w: 0.16]
        workingday ≠ 0.00 🔹 [id: 4 | heter: 5.44 | inst: 9518 | w: 0.68]
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 5.68
        Level 1🔹heter: 3.96 | 🔻1.71 (30.22%)
            Level 2🔹heter: 0.16 | 🔻3.80 (95.84%)
    
    


    



```python
for node_idx in [1, 4]:
    regional_rhale.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_1.png)
    



```python
for node_idx in [2, 3]:
    regional_rhale.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_30_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_30_1.png)
    


### SHAPDP - global


```python
shap_dp = effector.ShapDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=500)
shap_dp.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)
```

    PermutationExplainer explainer: 501it [02:43,  2.91it/s]                         



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_32_1.png)
    


### SHAPDP - regional


```python
regional_shap_dp = effector.RegionalShapDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, nof_instances=500)
regional_shap_dp.summary(features=3, scale_x_list=scale_x_list)
```

      0%|          | 0/1 [00:00<?, ?it/s]
    PermutationExplainer explainer:   7%|▋         | 37/500 [00:00<?, ?it/s][A
    PermutationExplainer explainer:   8%|▊         | 39/500 [00:10<01:09,  6.62it/s][A
    PermutationExplainer explainer:   8%|▊         | 40/500 [00:10<01:53,  4.06it/s][A
    PermutationExplainer explainer:   8%|▊         | 41/500 [00:11<02:15,  3.38it/s][A
    PermutationExplainer explainer:   8%|▊         | 42/500 [00:11<02:20,  3.25it/s][A
    PermutationExplainer explainer:   9%|▊         | 43/500 [00:11<02:21,  3.22it/s][A
    PermutationExplainer explainer:   9%|▉         | 44/500 [00:12<02:22,  3.20it/s][A
    PermutationExplainer explainer:   9%|▉         | 45/500 [00:12<02:22,  3.19it/s][A
    PermutationExplainer explainer:   9%|▉         | 46/500 [00:12<02:22,  3.18it/s][A
    PermutationExplainer explainer:   9%|▉         | 47/500 [00:13<02:20,  3.21it/s][A
    PermutationExplainer explainer:  10%|▉         | 48/500 [00:13<02:21,  3.18it/s][A
    PermutationExplainer explainer:  10%|▉         | 49/500 [00:13<02:16,  3.31it/s][A
    PermutationExplainer explainer:  10%|█         | 50/500 [00:13<02:11,  3.42it/s][A
    PermutationExplainer explainer:  10%|█         | 51/500 [00:14<02:07,  3.52it/s][A
    PermutationExplainer explainer:  10%|█         | 52/500 [00:14<02:08,  3.49it/s][A
    PermutationExplainer explainer:  11%|█         | 53/500 [00:14<02:05,  3.56it/s][A
    PermutationExplainer explainer:  11%|█         | 54/500 [00:14<02:05,  3.55it/s][A
    PermutationExplainer explainer:  11%|█         | 55/500 [00:15<02:07,  3.50it/s][A
    PermutationExplainer explainer:  11%|█         | 56/500 [00:15<02:08,  3.46it/s][A
    PermutationExplainer explainer:  11%|█▏        | 57/500 [00:15<02:09,  3.43it/s][A
    PermutationExplainer explainer:  12%|█▏        | 58/500 [00:16<02:09,  3.42it/s][A
    PermutationExplainer explainer:  12%|█▏        | 59/500 [00:16<02:08,  3.42it/s][A
    PermutationExplainer explainer:  12%|█▏        | 60/500 [00:16<02:06,  3.47it/s][A
    PermutationExplainer explainer:  12%|█▏        | 61/500 [00:17<02:07,  3.45it/s][A
    PermutationExplainer explainer:  12%|█▏        | 62/500 [00:17<02:06,  3.47it/s][A
    PermutationExplainer explainer:  13%|█▎        | 63/500 [00:17<02:06,  3.45it/s][A
    PermutationExplainer explainer:  13%|█▎        | 64/500 [00:17<02:10,  3.35it/s][A
    PermutationExplainer explainer:  13%|█▎        | 65/500 [00:18<02:12,  3.27it/s][A
    PermutationExplainer explainer:  13%|█▎        | 66/500 [00:18<02:14,  3.24it/s][A
    PermutationExplainer explainer:  13%|█▎        | 67/500 [00:19<02:33,  2.81it/s][A
    PermutationExplainer explainer:  14%|█▎        | 68/500 [00:19<02:46,  2.60it/s][A
    PermutationExplainer explainer:  14%|█▍        | 69/500 [00:19<02:52,  2.49it/s][A
    PermutationExplainer explainer:  14%|█▍        | 70/500 [00:20<02:41,  2.67it/s][A
    PermutationExplainer explainer:  14%|█▍        | 71/500 [00:20<02:37,  2.73it/s][A
    PermutationExplainer explainer:  14%|█▍        | 72/500 [00:20<02:25,  2.94it/s][A
    PermutationExplainer explainer:  15%|█▍        | 73/500 [00:21<02:15,  3.15it/s][A
    PermutationExplainer explainer:  15%|█▍        | 74/500 [00:21<02:17,  3.10it/s][A
    PermutationExplainer explainer:  15%|█▌        | 75/500 [00:21<02:09,  3.28it/s][A
    PermutationExplainer explainer:  15%|█▌        | 76/500 [00:22<02:05,  3.37it/s][A
    PermutationExplainer explainer:  15%|█▌        | 77/500 [00:22<02:06,  3.35it/s][A
    PermutationExplainer explainer:  16%|█▌        | 78/500 [00:22<02:13,  3.15it/s][A
    PermutationExplainer explainer:  16%|█▌        | 79/500 [00:23<02:14,  3.12it/s][A
    PermutationExplainer explainer:  16%|█▌        | 80/500 [00:23<02:13,  3.16it/s][A
    PermutationExplainer explainer:  16%|█▌        | 81/500 [00:23<02:13,  3.13it/s][A
    PermutationExplainer explainer:  16%|█▋        | 82/500 [00:23<02:16,  3.06it/s][A
    PermutationExplainer explainer:  17%|█▋        | 83/500 [00:24<02:14,  3.10it/s][A
    PermutationExplainer explainer:  17%|█▋        | 84/500 [00:24<02:19,  2.98it/s][A
    PermutationExplainer explainer:  17%|█▋        | 85/500 [00:24<02:16,  3.05it/s][A
    PermutationExplainer explainer:  17%|█▋        | 86/500 [00:25<02:12,  3.13it/s][A
    PermutationExplainer explainer:  17%|█▋        | 87/500 [00:25<02:10,  3.16it/s][A
    PermutationExplainer explainer:  18%|█▊        | 88/500 [00:25<02:05,  3.29it/s][A
    PermutationExplainer explainer:  18%|█▊        | 89/500 [00:26<02:02,  3.37it/s][A
    PermutationExplainer explainer:  18%|█▊        | 90/500 [00:26<01:58,  3.45it/s][A
    PermutationExplainer explainer:  18%|█▊        | 91/500 [00:26<01:56,  3.51it/s][A
    PermutationExplainer explainer:  18%|█▊        | 92/500 [00:27<02:08,  3.17it/s][A
    PermutationExplainer explainer:  19%|█▊        | 93/500 [00:27<02:18,  2.93it/s][A
    PermutationExplainer explainer:  19%|█▉        | 94/500 [00:27<02:31,  2.67it/s][A
    PermutationExplainer explainer:  19%|█▉        | 95/500 [00:28<02:30,  2.69it/s][A
    PermutationExplainer explainer:  19%|█▉        | 96/500 [00:28<02:21,  2.86it/s][A
    PermutationExplainer explainer:  19%|█▉        | 97/500 [00:28<02:10,  3.10it/s][A
    PermutationExplainer explainer:  20%|█▉        | 98/500 [00:29<02:20,  2.87it/s][A
    PermutationExplainer explainer:  20%|█▉        | 99/500 [00:29<02:25,  2.76it/s][A
    PermutationExplainer explainer:  20%|██        | 100/500 [00:30<02:32,  2.62it/s][A
    PermutationExplainer explainer:  20%|██        | 101/500 [00:30<02:59,  2.22it/s][A
    PermutationExplainer explainer:  20%|██        | 102/500 [00:31<03:39,  1.81it/s][A
    PermutationExplainer explainer:  21%|██        | 103/500 [00:31<03:28,  1.90it/s][A
    PermutationExplainer explainer:  21%|██        | 104/500 [00:32<03:16,  2.01it/s][A
    PermutationExplainer explainer:  21%|██        | 105/500 [00:32<03:04,  2.14it/s][A
    PermutationExplainer explainer:  21%|██        | 106/500 [00:33<02:56,  2.23it/s][A
    PermutationExplainer explainer:  21%|██▏       | 107/500 [00:33<02:45,  2.38it/s][A
    PermutationExplainer explainer:  22%|██▏       | 108/500 [00:33<02:35,  2.53it/s][A
    PermutationExplainer explainer:  22%|██▏       | 109/500 [00:34<02:22,  2.75it/s][A
    PermutationExplainer explainer:  22%|██▏       | 110/500 [00:34<02:10,  2.98it/s][A
    PermutationExplainer explainer:  22%|██▏       | 111/500 [00:34<02:02,  3.18it/s][A
    PermutationExplainer explainer:  22%|██▏       | 112/500 [00:34<01:59,  3.23it/s][A
    PermutationExplainer explainer:  23%|██▎       | 113/500 [00:35<02:20,  2.76it/s][A
    PermutationExplainer explainer:  23%|██▎       | 114/500 [00:35<02:32,  2.54it/s][A
    PermutationExplainer explainer:  23%|██▎       | 115/500 [00:36<02:31,  2.53it/s][A
    PermutationExplainer explainer:  23%|██▎       | 116/500 [00:36<02:33,  2.51it/s][A
    PermutationExplainer explainer:  23%|██▎       | 117/500 [00:37<02:17,  2.79it/s][A
    PermutationExplainer explainer:  24%|██▎       | 118/500 [00:37<02:05,  3.03it/s][A
    PermutationExplainer explainer:  24%|██▍       | 119/500 [00:37<01:58,  3.22it/s][A
    PermutationExplainer explainer:  24%|██▍       | 120/500 [00:37<01:55,  3.29it/s][A
    PermutationExplainer explainer:  24%|██▍       | 121/500 [00:38<01:50,  3.44it/s][A
    PermutationExplainer explainer:  24%|██▍       | 122/500 [00:38<01:45,  3.57it/s][A
    PermutationExplainer explainer:  25%|██▍       | 123/500 [00:38<01:43,  3.64it/s][A
    PermutationExplainer explainer:  25%|██▍       | 124/500 [00:38<01:42,  3.68it/s][A
    PermutationExplainer explainer:  25%|██▌       | 125/500 [00:39<01:40,  3.71it/s][A
    PermutationExplainer explainer:  25%|██▌       | 126/500 [00:39<01:40,  3.74it/s][A
    PermutationExplainer explainer:  25%|██▌       | 127/500 [00:39<01:41,  3.69it/s][A
    PermutationExplainer explainer:  26%|██▌       | 128/500 [00:39<01:40,  3.72it/s][A
    PermutationExplainer explainer:  26%|██▌       | 129/500 [00:40<01:40,  3.68it/s][A
    PermutationExplainer explainer:  26%|██▌       | 130/500 [00:40<01:40,  3.68it/s][A
    PermutationExplainer explainer:  26%|██▌       | 131/500 [00:40<01:39,  3.70it/s][A
    PermutationExplainer explainer:  26%|██▋       | 132/500 [00:41<01:38,  3.75it/s][A
    PermutationExplainer explainer:  27%|██▋       | 133/500 [00:41<01:38,  3.73it/s][A
    PermutationExplainer explainer:  27%|██▋       | 134/500 [00:41<01:38,  3.73it/s][A
    PermutationExplainer explainer:  27%|██▋       | 135/500 [00:41<01:38,  3.70it/s][A
    PermutationExplainer explainer:  27%|██▋       | 136/500 [00:42<01:38,  3.69it/s][A
    PermutationExplainer explainer:  27%|██▋       | 137/500 [00:42<01:40,  3.63it/s][A
    PermutationExplainer explainer:  28%|██▊       | 138/500 [00:42<01:38,  3.68it/s][A
    PermutationExplainer explainer:  28%|██▊       | 139/500 [00:42<01:36,  3.74it/s][A
    PermutationExplainer explainer:  28%|██▊       | 140/500 [00:43<01:36,  3.71it/s][A
    PermutationExplainer explainer:  28%|██▊       | 141/500 [00:43<01:41,  3.54it/s][A
    PermutationExplainer explainer:  28%|██▊       | 142/500 [00:43<01:43,  3.46it/s][A
    PermutationExplainer explainer:  29%|██▊       | 143/500 [00:44<01:52,  3.18it/s][A
    PermutationExplainer explainer:  29%|██▉       | 144/500 [00:44<01:58,  3.01it/s][A
    PermutationExplainer explainer:  29%|██▉       | 145/500 [00:44<01:55,  3.07it/s][A
    PermutationExplainer explainer:  29%|██▉       | 146/500 [00:45<01:54,  3.10it/s][A
    PermutationExplainer explainer:  29%|██▉       | 147/500 [00:45<01:52,  3.13it/s][A
    PermutationExplainer explainer:  30%|██▉       | 148/500 [00:45<01:50,  3.17it/s][A
    PermutationExplainer explainer:  30%|██▉       | 149/500 [00:46<01:50,  3.17it/s][A
    PermutationExplainer explainer:  30%|███       | 150/500 [00:46<01:49,  3.20it/s][A
    PermutationExplainer explainer:  30%|███       | 151/500 [00:46<01:48,  3.21it/s][A
    PermutationExplainer explainer:  30%|███       | 152/500 [00:46<01:43,  3.35it/s][A
    PermutationExplainer explainer:  31%|███       | 153/500 [00:47<01:39,  3.49it/s][A
    PermutationExplainer explainer:  31%|███       | 154/500 [00:47<01:37,  3.56it/s][A
    PermutationExplainer explainer:  31%|███       | 155/500 [00:47<01:35,  3.62it/s][A
    PermutationExplainer explainer:  31%|███       | 156/500 [00:48<01:33,  3.69it/s][A
    PermutationExplainer explainer:  31%|███▏      | 157/500 [00:48<01:31,  3.75it/s][A
    PermutationExplainer explainer:  32%|███▏      | 158/500 [00:48<01:31,  3.75it/s][A
    PermutationExplainer explainer:  32%|███▏      | 159/500 [00:48<01:31,  3.73it/s][A
    PermutationExplainer explainer:  32%|███▏      | 160/500 [00:49<01:36,  3.53it/s][A
    PermutationExplainer explainer:  32%|███▏      | 161/500 [00:49<01:35,  3.54it/s][A
    PermutationExplainer explainer:  32%|███▏      | 162/500 [00:49<01:36,  3.52it/s][A
    PermutationExplainer explainer:  33%|███▎      | 163/500 [00:50<01:36,  3.51it/s][A
    PermutationExplainer explainer:  33%|███▎      | 164/500 [00:50<01:35,  3.53it/s][A
    PermutationExplainer explainer:  33%|███▎      | 165/500 [00:50<01:34,  3.55it/s][A
    PermutationExplainer explainer:  33%|███▎      | 166/500 [00:50<01:35,  3.51it/s][A
    PermutationExplainer explainer:  33%|███▎      | 167/500 [00:51<01:34,  3.53it/s][A
    PermutationExplainer explainer:  34%|███▎      | 168/500 [00:51<01:35,  3.47it/s][A
    PermutationExplainer explainer:  34%|███▍      | 169/500 [00:51<01:34,  3.50it/s][A
    PermutationExplainer explainer:  34%|███▍      | 170/500 [00:51<01:34,  3.49it/s][A
    PermutationExplainer explainer:  34%|███▍      | 171/500 [00:52<01:37,  3.39it/s][A
    PermutationExplainer explainer:  34%|███▍      | 172/500 [00:52<01:36,  3.42it/s][A
    PermutationExplainer explainer:  35%|███▍      | 173/500 [00:52<01:33,  3.52it/s][A
    PermutationExplainer explainer:  35%|███▍      | 174/500 [00:53<01:30,  3.59it/s][A
    PermutationExplainer explainer:  35%|███▌      | 175/500 [00:53<01:29,  3.65it/s][A
    PermutationExplainer explainer:  35%|███▌      | 176/500 [00:53<01:28,  3.68it/s][A
    PermutationExplainer explainer:  35%|███▌      | 177/500 [00:53<01:26,  3.74it/s][A
    PermutationExplainer explainer:  36%|███▌      | 178/500 [00:54<01:26,  3.70it/s][A
    PermutationExplainer explainer:  36%|███▌      | 179/500 [00:54<01:26,  3.70it/s][A
    PermutationExplainer explainer:  36%|███▌      | 180/500 [00:54<01:29,  3.58it/s][A
    PermutationExplainer explainer:  36%|███▌      | 181/500 [00:55<01:27,  3.64it/s][A
    PermutationExplainer explainer:  36%|███▋      | 182/500 [00:55<01:27,  3.64it/s][A
    PermutationExplainer explainer:  37%|███▋      | 183/500 [00:55<01:28,  3.60it/s][A
    PermutationExplainer explainer:  37%|███▋      | 184/500 [00:55<01:29,  3.53it/s][A
    PermutationExplainer explainer:  37%|███▋      | 185/500 [00:56<01:28,  3.55it/s][A
    PermutationExplainer explainer:  37%|███▋      | 186/500 [00:56<01:30,  3.49it/s][A
    PermutationExplainer explainer:  37%|███▋      | 187/500 [00:56<01:32,  3.38it/s][A
    PermutationExplainer explainer:  38%|███▊      | 188/500 [00:57<01:32,  3.39it/s][A
    PermutationExplainer explainer:  38%|███▊      | 189/500 [00:57<01:31,  3.40it/s][A
    PermutationExplainer explainer:  38%|███▊      | 190/500 [00:57<01:32,  3.36it/s][A
    PermutationExplainer explainer:  38%|███▊      | 191/500 [00:57<01:30,  3.41it/s][A
    PermutationExplainer explainer:  38%|███▊      | 192/500 [00:58<01:29,  3.44it/s][A
    PermutationExplainer explainer:  39%|███▊      | 193/500 [00:58<01:27,  3.50it/s][A
    PermutationExplainer explainer:  39%|███▉      | 194/500 [00:58<01:27,  3.48it/s][A
    PermutationExplainer explainer:  39%|███▉      | 195/500 [00:59<01:26,  3.52it/s][A
    PermutationExplainer explainer:  39%|███▉      | 196/500 [00:59<01:24,  3.60it/s][A
    PermutationExplainer explainer:  39%|███▉      | 197/500 [00:59<01:23,  3.61it/s][A
    PermutationExplainer explainer:  40%|███▉      | 198/500 [00:59<01:21,  3.69it/s][A
    PermutationExplainer explainer:  40%|███▉      | 199/500 [01:00<01:23,  3.62it/s][A
    PermutationExplainer explainer:  40%|████      | 200/500 [01:00<01:21,  3.67it/s][A
    PermutationExplainer explainer:  40%|████      | 201/500 [01:00<01:20,  3.71it/s][A
    PermutationExplainer explainer:  40%|████      | 202/500 [01:00<01:19,  3.74it/s][A
    PermutationExplainer explainer:  41%|████      | 203/500 [01:01<01:21,  3.66it/s][A
    PermutationExplainer explainer:  41%|████      | 204/500 [01:01<01:21,  3.64it/s][A
    PermutationExplainer explainer:  41%|████      | 205/500 [01:01<01:20,  3.65it/s][A
    PermutationExplainer explainer:  41%|████      | 206/500 [01:02<01:21,  3.59it/s][A
    PermutationExplainer explainer:  41%|████▏     | 207/500 [01:02<01:21,  3.61it/s][A
    PermutationExplainer explainer:  42%|████▏     | 208/500 [01:02<01:24,  3.47it/s][A
    PermutationExplainer explainer:  42%|████▏     | 209/500 [01:02<01:21,  3.57it/s][A
    PermutationExplainer explainer:  42%|████▏     | 210/500 [01:03<01:19,  3.63it/s][A
    PermutationExplainer explainer:  42%|████▏     | 211/500 [01:03<01:18,  3.70it/s][A
    PermutationExplainer explainer:  42%|████▏     | 212/500 [01:03<01:18,  3.69it/s][A
    PermutationExplainer explainer:  43%|████▎     | 213/500 [01:03<01:17,  3.72it/s][A
    PermutationExplainer explainer:  43%|████▎     | 214/500 [01:04<01:16,  3.72it/s][A
    PermutationExplainer explainer:  43%|████▎     | 215/500 [01:04<01:17,  3.69it/s][A
    PermutationExplainer explainer:  43%|████▎     | 216/500 [01:04<01:22,  3.45it/s][A
    PermutationExplainer explainer:  43%|████▎     | 217/500 [01:05<01:20,  3.50it/s][A
    PermutationExplainer explainer:  44%|████▎     | 218/500 [01:05<01:20,  3.50it/s][A
    PermutationExplainer explainer:  44%|████▍     | 219/500 [01:05<01:19,  3.53it/s][A
    PermutationExplainer explainer:  44%|████▍     | 220/500 [01:05<01:17,  3.61it/s][A
    PermutationExplainer explainer:  44%|████▍     | 221/500 [01:06<01:16,  3.63it/s][A
    PermutationExplainer explainer:  44%|████▍     | 222/500 [01:06<01:16,  3.62it/s][A
    PermutationExplainer explainer:  45%|████▍     | 223/500 [01:06<01:16,  3.63it/s][A
    PermutationExplainer explainer:  45%|████▍     | 224/500 [01:07<01:14,  3.70it/s][A
    PermutationExplainer explainer:  45%|████▌     | 225/500 [01:07<01:13,  3.76it/s][A
    PermutationExplainer explainer:  45%|████▌     | 226/500 [01:07<01:12,  3.79it/s][A
    PermutationExplainer explainer:  45%|████▌     | 227/500 [01:07<01:13,  3.71it/s][A
    PermutationExplainer explainer:  46%|████▌     | 228/500 [01:08<01:14,  3.64it/s][A
    PermutationExplainer explainer:  46%|████▌     | 229/500 [01:08<01:13,  3.71it/s][A
    PermutationExplainer explainer:  46%|████▌     | 230/500 [01:08<01:13,  3.69it/s][A
    PermutationExplainer explainer:  46%|████▌     | 231/500 [01:08<01:13,  3.66it/s][A
    PermutationExplainer explainer:  46%|████▋     | 232/500 [01:09<01:13,  3.66it/s][A
    PermutationExplainer explainer:  47%|████▋     | 233/500 [01:09<01:12,  3.68it/s][A
    PermutationExplainer explainer:  47%|████▋     | 234/500 [01:09<01:12,  3.68it/s][A
    PermutationExplainer explainer:  47%|████▋     | 235/500 [01:10<01:10,  3.74it/s][A
    PermutationExplainer explainer:  47%|████▋     | 236/500 [01:10<01:09,  3.78it/s][A
    PermutationExplainer explainer:  47%|████▋     | 237/500 [01:10<01:11,  3.70it/s][A
    PermutationExplainer explainer:  48%|████▊     | 238/500 [01:10<01:10,  3.73it/s][A
    PermutationExplainer explainer:  48%|████▊     | 239/500 [01:11<01:10,  3.69it/s][A
    PermutationExplainer explainer:  48%|████▊     | 240/500 [01:11<01:09,  3.73it/s][A
    PermutationExplainer explainer:  48%|████▊     | 241/500 [01:11<01:09,  3.72it/s][A
    PermutationExplainer explainer:  48%|████▊     | 242/500 [01:11<01:08,  3.76it/s][A
    PermutationExplainer explainer:  49%|████▊     | 243/500 [01:12<01:08,  3.75it/s][A
    PermutationExplainer explainer:  49%|████▉     | 244/500 [01:12<01:08,  3.72it/s][A
    PermutationExplainer explainer:  49%|████▉     | 245/500 [01:12<01:09,  3.68it/s][A
    PermutationExplainer explainer:  49%|████▉     | 246/500 [01:12<01:09,  3.65it/s][A
    PermutationExplainer explainer:  49%|████▉     | 247/500 [01:13<01:08,  3.69it/s][A
    PermutationExplainer explainer:  50%|████▉     | 248/500 [01:13<01:08,  3.66it/s][A
    PermutationExplainer explainer:  50%|████▉     | 249/500 [01:13<01:08,  3.64it/s][A
    PermutationExplainer explainer:  50%|█████     | 250/500 [01:14<01:07,  3.70it/s][A
    PermutationExplainer explainer:  50%|█████     | 251/500 [01:14<01:07,  3.66it/s][A
    PermutationExplainer explainer:  50%|█████     | 252/500 [01:14<01:06,  3.70it/s][A
    PermutationExplainer explainer:  51%|█████     | 253/500 [01:14<01:07,  3.64it/s][A
    PermutationExplainer explainer:  51%|█████     | 254/500 [01:15<01:07,  3.65it/s][A
    PermutationExplainer explainer:  51%|█████     | 255/500 [01:15<01:07,  3.62it/s][A
    PermutationExplainer explainer:  51%|█████     | 256/500 [01:15<01:06,  3.69it/s][A
    PermutationExplainer explainer:  51%|█████▏    | 257/500 [01:15<01:04,  3.74it/s][A
    PermutationExplainer explainer:  52%|█████▏    | 258/500 [01:16<01:04,  3.74it/s][A
    PermutationExplainer explainer:  52%|█████▏    | 259/500 [01:16<01:05,  3.67it/s][A
    PermutationExplainer explainer:  52%|█████▏    | 260/500 [01:16<01:07,  3.54it/s][A
    PermutationExplainer explainer:  52%|█████▏    | 261/500 [01:17<01:10,  3.38it/s][A
    PermutationExplainer explainer:  52%|█████▏    | 262/500 [01:17<01:15,  3.15it/s][A
    PermutationExplainer explainer:  53%|█████▎    | 263/500 [01:17<01:15,  3.15it/s][A
    PermutationExplainer explainer:  53%|█████▎    | 264/500 [01:18<01:13,  3.20it/s][A
    PermutationExplainer explainer:  53%|█████▎    | 265/500 [01:18<01:12,  3.25it/s][A
    PermutationExplainer explainer:  53%|█████▎    | 266/500 [01:18<01:12,  3.24it/s][A
    PermutationExplainer explainer:  53%|█████▎    | 267/500 [01:19<01:13,  3.18it/s][A
    PermutationExplainer explainer:  54%|█████▎    | 268/500 [01:19<01:13,  3.15it/s][A
    PermutationExplainer explainer:  54%|█████▍    | 269/500 [01:19<01:11,  3.21it/s][A
    PermutationExplainer explainer:  54%|█████▍    | 270/500 [01:19<01:09,  3.33it/s][A
    PermutationExplainer explainer:  54%|█████▍    | 271/500 [01:20<01:05,  3.47it/s][A
    PermutationExplainer explainer:  54%|█████▍    | 272/500 [01:20<01:04,  3.52it/s][A
    PermutationExplainer explainer:  55%|█████▍    | 273/500 [01:20<01:05,  3.45it/s][A
    PermutationExplainer explainer:  55%|█████▍    | 274/500 [01:21<01:06,  3.40it/s][A
    PermutationExplainer explainer:  55%|█████▌    | 275/500 [01:21<01:04,  3.51it/s][A
    PermutationExplainer explainer:  55%|█████▌    | 276/500 [01:21<01:02,  3.61it/s][A
    PermutationExplainer explainer:  55%|█████▌    | 277/500 [01:21<01:01,  3.62it/s][A
    PermutationExplainer explainer:  56%|█████▌    | 278/500 [01:22<01:00,  3.68it/s][A
    PermutationExplainer explainer:  56%|█████▌    | 279/500 [01:22<00:59,  3.74it/s][A
    PermutationExplainer explainer:  56%|█████▌    | 280/500 [01:22<00:59,  3.69it/s][A
    PermutationExplainer explainer:  56%|█████▌    | 281/500 [01:23<01:00,  3.60it/s][A
    PermutationExplainer explainer:  56%|█████▋    | 282/500 [01:23<01:00,  3.60it/s][A
    PermutationExplainer explainer:  57%|█████▋    | 283/500 [01:23<01:01,  3.55it/s][A
    PermutationExplainer explainer:  57%|█████▋    | 284/500 [01:23<01:01,  3.53it/s][A
    PermutationExplainer explainer:  57%|█████▋    | 285/500 [01:24<01:00,  3.53it/s][A
    PermutationExplainer explainer:  57%|█████▋    | 286/500 [01:24<01:00,  3.54it/s][A
    PermutationExplainer explainer:  57%|█████▋    | 287/500 [01:24<01:01,  3.48it/s][A
    PermutationExplainer explainer:  58%|█████▊    | 288/500 [01:25<01:00,  3.51it/s][A
    PermutationExplainer explainer:  58%|█████▊    | 289/500 [01:25<00:59,  3.54it/s][A
    PermutationExplainer explainer:  58%|█████▊    | 290/500 [01:25<00:59,  3.53it/s][A
    PermutationExplainer explainer:  58%|█████▊    | 291/500 [01:25<00:58,  3.55it/s][A
    PermutationExplainer explainer:  58%|█████▊    | 292/500 [01:26<00:58,  3.56it/s][A
    PermutationExplainer explainer:  59%|█████▊    | 293/500 [01:26<00:58,  3.55it/s][A
    PermutationExplainer explainer:  59%|█████▉    | 294/500 [01:26<00:58,  3.55it/s][A
    PermutationExplainer explainer:  59%|█████▉    | 295/500 [01:26<00:57,  3.59it/s][A
    PermutationExplainer explainer:  59%|█████▉    | 296/500 [01:27<00:55,  3.65it/s][A
    PermutationExplainer explainer:  59%|█████▉    | 297/500 [01:27<00:54,  3.70it/s][A
    PermutationExplainer explainer:  60%|█████▉    | 298/500 [01:27<00:54,  3.69it/s][A
    PermutationExplainer explainer:  60%|█████▉    | 299/500 [01:28<00:54,  3.68it/s][A
    PermutationExplainer explainer:  60%|██████    | 300/500 [01:28<00:53,  3.73it/s][A
    PermutationExplainer explainer:  60%|██████    | 301/500 [01:28<00:52,  3.78it/s][A
    PermutationExplainer explainer:  60%|██████    | 302/500 [01:28<00:52,  3.80it/s][A
    PermutationExplainer explainer:  61%|██████    | 303/500 [01:29<00:52,  3.73it/s][A
    PermutationExplainer explainer:  61%|██████    | 304/500 [01:29<00:54,  3.63it/s][A
    PermutationExplainer explainer:  61%|██████    | 305/500 [01:29<00:55,  3.53it/s][A
    PermutationExplainer explainer:  61%|██████    | 306/500 [01:29<00:55,  3.47it/s][A
    PermutationExplainer explainer:  61%|██████▏   | 307/500 [01:30<00:56,  3.44it/s][A
    PermutationExplainer explainer:  62%|██████▏   | 308/500 [01:30<00:55,  3.48it/s][A
    PermutationExplainer explainer:  62%|██████▏   | 309/500 [01:30<00:54,  3.51it/s][A
    PermutationExplainer explainer:  62%|██████▏   | 310/500 [01:31<00:55,  3.41it/s][A
    PermutationExplainer explainer:  62%|██████▏   | 311/500 [01:31<00:55,  3.39it/s][A
    PermutationExplainer explainer:  62%|██████▏   | 312/500 [01:31<00:56,  3.33it/s][A
    PermutationExplainer explainer:  63%|██████▎   | 313/500 [01:32<00:55,  3.40it/s][A
    PermutationExplainer explainer:  63%|██████▎   | 314/500 [01:32<00:54,  3.40it/s][A
    PermutationExplainer explainer:  63%|██████▎   | 315/500 [01:32<00:52,  3.52it/s][A
    PermutationExplainer explainer:  63%|██████▎   | 316/500 [01:32<00:51,  3.57it/s][A
    PermutationExplainer explainer:  63%|██████▎   | 317/500 [01:33<00:50,  3.64it/s][A
    PermutationExplainer explainer:  64%|██████▎   | 318/500 [01:33<00:49,  3.71it/s][A
    PermutationExplainer explainer:  64%|██████▍   | 319/500 [01:33<00:48,  3.73it/s][A
    PermutationExplainer explainer:  64%|██████▍   | 320/500 [01:33<00:48,  3.70it/s][A
    PermutationExplainer explainer:  64%|██████▍   | 321/500 [01:34<00:50,  3.55it/s][A
    PermutationExplainer explainer:  64%|██████▍   | 322/500 [01:34<00:50,  3.52it/s][A
    PermutationExplainer explainer:  65%|██████▍   | 323/500 [01:34<00:50,  3.53it/s][A
    PermutationExplainer explainer:  65%|██████▍   | 324/500 [01:35<00:49,  3.58it/s][A
    PermutationExplainer explainer:  65%|██████▌   | 325/500 [01:35<00:47,  3.66it/s][A
    PermutationExplainer explainer:  65%|██████▌   | 326/500 [01:35<00:47,  3.64it/s][A
    PermutationExplainer explainer:  65%|██████▌   | 327/500 [01:35<00:47,  3.67it/s][A
    PermutationExplainer explainer:  66%|██████▌   | 328/500 [01:36<00:46,  3.73it/s][A
    PermutationExplainer explainer:  66%|██████▌   | 329/500 [01:36<00:45,  3.76it/s][A
    PermutationExplainer explainer:  66%|██████▌   | 330/500 [01:36<00:45,  3.77it/s][A
    PermutationExplainer explainer:  66%|██████▌   | 331/500 [01:36<00:44,  3.79it/s][A
    PermutationExplainer explainer:  66%|██████▋   | 332/500 [01:37<00:44,  3.75it/s][A
    PermutationExplainer explainer:  67%|██████▋   | 333/500 [01:37<00:44,  3.72it/s][A
    PermutationExplainer explainer:  67%|██████▋   | 334/500 [01:37<00:45,  3.68it/s][A
    PermutationExplainer explainer:  67%|██████▋   | 335/500 [01:38<00:45,  3.65it/s][A
    PermutationExplainer explainer:  67%|██████▋   | 336/500 [01:38<00:44,  3.68it/s][A
    PermutationExplainer explainer:  67%|██████▋   | 337/500 [01:38<00:44,  3.69it/s][A
    PermutationExplainer explainer:  68%|██████▊   | 338/500 [01:38<00:43,  3.69it/s][A
    PermutationExplainer explainer:  68%|██████▊   | 339/500 [01:39<00:43,  3.67it/s][A
    PermutationExplainer explainer:  68%|██████▊   | 340/500 [01:39<00:43,  3.71it/s][A
    PermutationExplainer explainer:  68%|██████▊   | 341/500 [01:39<00:42,  3.75it/s][A
    PermutationExplainer explainer:  68%|██████▊   | 342/500 [01:39<00:41,  3.78it/s][A
    PermutationExplainer explainer:  69%|██████▊   | 343/500 [01:40<00:41,  3.74it/s][A
    PermutationExplainer explainer:  69%|██████▉   | 344/500 [01:40<00:42,  3.70it/s][A
    PermutationExplainer explainer:  69%|██████▉   | 345/500 [01:40<00:42,  3.63it/s][A
    PermutationExplainer explainer:  69%|██████▉   | 346/500 [01:40<00:42,  3.64it/s][A
    PermutationExplainer explainer:  69%|██████▉   | 347/500 [01:41<00:41,  3.69it/s][A
    PermutationExplainer explainer:  70%|██████▉   | 348/500 [01:41<00:41,  3.64it/s][A
    PermutationExplainer explainer:  70%|██████▉   | 349/500 [01:41<00:40,  3.71it/s][A
    PermutationExplainer explainer:  70%|███████   | 350/500 [01:42<00:40,  3.70it/s][A
    PermutationExplainer explainer:  70%|███████   | 351/500 [01:42<00:40,  3.70it/s][A
    PermutationExplainer explainer:  70%|███████   | 352/500 [01:42<00:39,  3.74it/s][A
    PermutationExplainer explainer:  71%|███████   | 353/500 [01:42<00:38,  3.78it/s][A
    PermutationExplainer explainer:  71%|███████   | 354/500 [01:43<00:39,  3.73it/s][A
    PermutationExplainer explainer:  71%|███████   | 355/500 [01:43<00:38,  3.74it/s][A
    PermutationExplainer explainer:  71%|███████   | 356/500 [01:43<00:38,  3.72it/s][A
    PermutationExplainer explainer:  71%|███████▏  | 357/500 [01:43<00:39,  3.64it/s][A
    PermutationExplainer explainer:  72%|███████▏  | 358/500 [01:44<00:38,  3.68it/s][A
    PermutationExplainer explainer:  72%|███████▏  | 359/500 [01:44<00:38,  3.62it/s][A
    PermutationExplainer explainer:  72%|███████▏  | 360/500 [01:44<00:41,  3.41it/s][A
    PermutationExplainer explainer:  72%|███████▏  | 361/500 [01:45<00:40,  3.42it/s][A
    PermutationExplainer explainer:  72%|███████▏  | 362/500 [01:45<00:40,  3.44it/s][A
    PermutationExplainer explainer:  73%|███████▎  | 363/500 [01:45<00:38,  3.54it/s][A
    PermutationExplainer explainer:  73%|███████▎  | 364/500 [01:45<00:37,  3.62it/s][A
    PermutationExplainer explainer:  73%|███████▎  | 365/500 [01:46<00:37,  3.61it/s][A
    PermutationExplainer explainer:  73%|███████▎  | 366/500 [01:46<00:36,  3.67it/s][A
    PermutationExplainer explainer:  73%|███████▎  | 367/500 [01:46<00:36,  3.68it/s][A
    PermutationExplainer explainer:  74%|███████▎  | 368/500 [01:47<00:35,  3.72it/s][A
    PermutationExplainer explainer:  74%|███████▍  | 369/500 [01:47<00:35,  3.70it/s][A
    PermutationExplainer explainer:  74%|███████▍  | 370/500 [01:47<00:34,  3.73it/s][A
    PermutationExplainer explainer:  74%|███████▍  | 371/500 [01:47<00:34,  3.72it/s][A
    PermutationExplainer explainer:  74%|███████▍  | 372/500 [01:48<00:34,  3.73it/s][A
    PermutationExplainer explainer:  75%|███████▍  | 373/500 [01:48<00:34,  3.73it/s][A
    PermutationExplainer explainer:  75%|███████▍  | 374/500 [01:48<00:33,  3.77it/s][A
    PermutationExplainer explainer:  75%|███████▌  | 375/500 [01:48<00:32,  3.81it/s][A
    PermutationExplainer explainer:  75%|███████▌  | 376/500 [01:49<00:32,  3.76it/s][A
    PermutationExplainer explainer:  75%|███████▌  | 377/500 [01:49<00:32,  3.80it/s][A
    PermutationExplainer explainer:  76%|███████▌  | 378/500 [01:49<00:32,  3.70it/s][A
    PermutationExplainer explainer:  76%|███████▌  | 379/500 [01:50<00:33,  3.57it/s][A
    PermutationExplainer explainer:  76%|███████▌  | 380/500 [01:50<00:34,  3.46it/s][A
    PermutationExplainer explainer:  76%|███████▌  | 381/500 [01:50<00:35,  3.36it/s][A
    PermutationExplainer explainer:  76%|███████▋  | 382/500 [01:50<00:37,  3.17it/s][A
    PermutationExplainer explainer:  77%|███████▋  | 383/500 [01:51<00:36,  3.20it/s][A
    PermutationExplainer explainer:  77%|███████▋  | 384/500 [01:51<00:36,  3.20it/s][A
    PermutationExplainer explainer:  77%|███████▋  | 385/500 [01:51<00:35,  3.20it/s][A
    PermutationExplainer explainer:  77%|███████▋  | 386/500 [01:52<00:35,  3.19it/s][A
    PermutationExplainer explainer:  77%|███████▋  | 387/500 [01:52<00:36,  3.13it/s][A
    PermutationExplainer explainer:  78%|███████▊  | 388/500 [01:52<00:35,  3.13it/s][A
    PermutationExplainer explainer:  78%|███████▊  | 389/500 [01:53<00:34,  3.24it/s][A
    PermutationExplainer explainer:  78%|███████▊  | 390/500 [01:53<00:33,  3.31it/s][A
    PermutationExplainer explainer:  78%|███████▊  | 391/500 [01:53<00:31,  3.44it/s][A
    PermutationExplainer explainer:  78%|███████▊  | 392/500 [01:53<00:30,  3.54it/s][A
    PermutationExplainer explainer:  79%|███████▊  | 393/500 [01:54<00:30,  3.47it/s][A
    PermutationExplainer explainer:  79%|███████▉  | 394/500 [01:54<00:29,  3.58it/s][A
    PermutationExplainer explainer:  79%|███████▉  | 395/500 [01:54<00:29,  3.52it/s][A
    PermutationExplainer explainer:  79%|███████▉  | 396/500 [01:55<00:29,  3.51it/s][A
    PermutationExplainer explainer:  79%|███████▉  | 397/500 [01:55<00:28,  3.59it/s][A
    PermutationExplainer explainer:  80%|███████▉  | 398/500 [01:55<00:28,  3.60it/s][A
    PermutationExplainer explainer:  80%|███████▉  | 399/500 [01:55<00:27,  3.65it/s][A
    PermutationExplainer explainer:  80%|████████  | 400/500 [01:56<00:27,  3.58it/s][A
    PermutationExplainer explainer:  80%|████████  | 401/500 [01:56<00:27,  3.61it/s][A
    PermutationExplainer explainer:  80%|████████  | 402/500 [01:56<00:27,  3.58it/s][A
    PermutationExplainer explainer:  81%|████████  | 403/500 [01:57<00:27,  3.57it/s][A
    PermutationExplainer explainer:  81%|████████  | 404/500 [01:57<00:26,  3.56it/s][A
    PermutationExplainer explainer:  81%|████████  | 405/500 [01:57<00:26,  3.55it/s][A
    PermutationExplainer explainer:  81%|████████  | 406/500 [01:57<00:26,  3.54it/s][A
    PermutationExplainer explainer:  81%|████████▏ | 407/500 [01:58<00:26,  3.55it/s][A
    PermutationExplainer explainer:  82%|████████▏ | 408/500 [01:58<00:26,  3.49it/s][A
    PermutationExplainer explainer:  82%|████████▏ | 409/500 [01:58<00:25,  3.52it/s][A
    PermutationExplainer explainer:  82%|████████▏ | 410/500 [01:59<00:25,  3.53it/s][A
    PermutationExplainer explainer:  82%|████████▏ | 411/500 [01:59<00:25,  3.51it/s][A
    PermutationExplainer explainer:  82%|████████▏ | 412/500 [01:59<00:25,  3.48it/s][A
    PermutationExplainer explainer:  83%|████████▎ | 413/500 [01:59<00:25,  3.43it/s][A
    PermutationExplainer explainer:  83%|████████▎ | 414/500 [02:00<00:25,  3.39it/s][A
    PermutationExplainer explainer:  83%|████████▎ | 415/500 [02:00<00:24,  3.48it/s][A
    PermutationExplainer explainer:  83%|████████▎ | 416/500 [02:00<00:23,  3.58it/s][A
    PermutationExplainer explainer:  83%|████████▎ | 417/500 [02:01<00:22,  3.67it/s][A
    PermutationExplainer explainer:  84%|████████▎ | 418/500 [02:01<00:22,  3.67it/s][A
    PermutationExplainer explainer:  84%|████████▍ | 419/500 [02:01<00:21,  3.72it/s][A
    PermutationExplainer explainer:  84%|████████▍ | 420/500 [02:01<00:21,  3.77it/s][A
    PermutationExplainer explainer:  84%|████████▍ | 421/500 [02:02<00:21,  3.65it/s][A
    PermutationExplainer explainer:  84%|████████▍ | 422/500 [02:02<00:21,  3.59it/s][A
    PermutationExplainer explainer:  85%|████████▍ | 423/500 [02:02<00:21,  3.57it/s][A
    PermutationExplainer explainer:  85%|████████▍ | 424/500 [02:02<00:21,  3.51it/s][A
    PermutationExplainer explainer:  85%|████████▌ | 425/500 [02:03<00:21,  3.54it/s][A
    PermutationExplainer explainer:  85%|████████▌ | 426/500 [02:03<00:20,  3.56it/s][A
    PermutationExplainer explainer:  85%|████████▌ | 427/500 [02:03<00:20,  3.51it/s][A
    PermutationExplainer explainer:  86%|████████▌ | 428/500 [02:04<00:20,  3.47it/s][A
    PermutationExplainer explainer:  86%|████████▌ | 429/500 [02:04<00:22,  3.21it/s][A
    PermutationExplainer explainer:  86%|████████▌ | 430/500 [02:04<00:22,  3.18it/s][A
    PermutationExplainer explainer:  86%|████████▌ | 431/500 [02:05<00:21,  3.23it/s][A
    PermutationExplainer explainer:  86%|████████▋ | 432/500 [02:05<00:20,  3.32it/s][A
    PermutationExplainer explainer:  87%|████████▋ | 433/500 [02:05<00:20,  3.32it/s][A
    PermutationExplainer explainer:  87%|████████▋ | 434/500 [02:05<00:19,  3.36it/s][A
    PermutationExplainer explainer:  87%|████████▋ | 435/500 [02:06<00:18,  3.49it/s][A
    PermutationExplainer explainer:  87%|████████▋ | 436/500 [02:06<00:18,  3.53it/s][A
    PermutationExplainer explainer:  87%|████████▋ | 437/500 [02:06<00:17,  3.54it/s][A
    PermutationExplainer explainer:  88%|████████▊ | 438/500 [02:07<00:17,  3.50it/s][A
    PermutationExplainer explainer:  88%|████████▊ | 439/500 [02:07<00:17,  3.58it/s][A
    PermutationExplainer explainer:  88%|████████▊ | 440/500 [02:07<00:16,  3.65it/s][A
    PermutationExplainer explainer:  88%|████████▊ | 441/500 [02:07<00:16,  3.64it/s][A
    PermutationExplainer explainer:  88%|████████▊ | 442/500 [02:08<00:16,  3.58it/s][A
    PermutationExplainer explainer:  89%|████████▊ | 443/500 [02:08<00:16,  3.45it/s][A
    PermutationExplainer explainer:  89%|████████▉ | 444/500 [02:08<00:15,  3.56it/s][A
    PermutationExplainer explainer:  89%|████████▉ | 445/500 [02:09<00:15,  3.66it/s][A
    PermutationExplainer explainer:  89%|████████▉ | 446/500 [02:09<00:14,  3.70it/s][A
    PermutationExplainer explainer:  89%|████████▉ | 447/500 [02:09<00:14,  3.75it/s][A
    PermutationExplainer explainer:  90%|████████▉ | 448/500 [02:09<00:13,  3.78it/s][A
    PermutationExplainer explainer:  90%|████████▉ | 449/500 [02:10<00:13,  3.75it/s][A
    PermutationExplainer explainer:  90%|█████████ | 450/500 [02:10<00:13,  3.71it/s][A
    PermutationExplainer explainer:  90%|█████████ | 451/500 [02:10<00:13,  3.71it/s][A
    PermutationExplainer explainer:  90%|█████████ | 452/500 [02:10<00:12,  3.74it/s][A
    PermutationExplainer explainer:  91%|█████████ | 453/500 [02:11<00:12,  3.66it/s][A
    PermutationExplainer explainer:  91%|█████████ | 454/500 [02:11<00:12,  3.70it/s][A
    PermutationExplainer explainer:  91%|█████████ | 455/500 [02:11<00:12,  3.62it/s][A
    PermutationExplainer explainer:  91%|█████████ | 456/500 [02:11<00:11,  3.70it/s][A
    PermutationExplainer explainer:  91%|█████████▏| 457/500 [02:12<00:11,  3.68it/s][A
    PermutationExplainer explainer:  92%|█████████▏| 458/500 [02:12<00:11,  3.72it/s][A
    PermutationExplainer explainer:  92%|█████████▏| 459/500 [02:12<00:11,  3.67it/s][A
    PermutationExplainer explainer:  92%|█████████▏| 460/500 [02:13<00:10,  3.71it/s][A
    PermutationExplainer explainer:  92%|█████████▏| 461/500 [02:13<00:10,  3.75it/s][A
    PermutationExplainer explainer:  92%|█████████▏| 462/500 [02:13<00:10,  3.77it/s][A
    PermutationExplainer explainer:  93%|█████████▎| 463/500 [02:13<00:09,  3.79it/s][A
    PermutationExplainer explainer:  93%|█████████▎| 464/500 [02:14<00:09,  3.69it/s][A
    PermutationExplainer explainer:  93%|█████████▎| 465/500 [02:14<00:09,  3.72it/s][A
    PermutationExplainer explainer:  93%|█████████▎| 466/500 [02:14<00:09,  3.74it/s][A
    PermutationExplainer explainer:  93%|█████████▎| 467/500 [02:14<00:08,  3.78it/s][A
    PermutationExplainer explainer:  94%|█████████▎| 468/500 [02:15<00:08,  3.76it/s][A
    PermutationExplainer explainer:  94%|█████████▍| 469/500 [02:15<00:08,  3.76it/s][A
    PermutationExplainer explainer:  94%|█████████▍| 470/500 [02:15<00:07,  3.79it/s][A
    PermutationExplainer explainer:  94%|█████████▍| 471/500 [02:15<00:07,  3.79it/s][A
    PermutationExplainer explainer:  94%|█████████▍| 472/500 [02:16<00:07,  3.79it/s][A
    PermutationExplainer explainer:  95%|█████████▍| 473/500 [02:16<00:07,  3.78it/s][A
    PermutationExplainer explainer:  95%|█████████▍| 474/500 [02:16<00:06,  3.74it/s][A
    PermutationExplainer explainer:  95%|█████████▌| 475/500 [02:17<00:06,  3.74it/s][A
    PermutationExplainer explainer:  95%|█████████▌| 476/500 [02:17<00:06,  3.76it/s][A
    PermutationExplainer explainer:  95%|█████████▌| 477/500 [02:17<00:06,  3.78it/s][A
    PermutationExplainer explainer:  96%|█████████▌| 478/500 [02:17<00:05,  3.75it/s][A
    PermutationExplainer explainer:  96%|█████████▌| 479/500 [02:18<00:05,  3.74it/s][A
    PermutationExplainer explainer:  96%|█████████▌| 480/500 [02:18<00:05,  3.70it/s][A
    PermutationExplainer explainer:  96%|█████████▌| 481/500 [02:18<00:05,  3.72it/s][A
    PermutationExplainer explainer:  96%|█████████▋| 482/500 [02:18<00:04,  3.65it/s][A
    PermutationExplainer explainer:  97%|█████████▋| 483/500 [02:19<00:04,  3.69it/s][A
    PermutationExplainer explainer:  97%|█████████▋| 484/500 [02:19<00:04,  3.71it/s][A
    PermutationExplainer explainer:  97%|█████████▋| 485/500 [02:19<00:04,  3.68it/s][A
    PermutationExplainer explainer:  97%|█████████▋| 486/500 [02:19<00:03,  3.72it/s][A
    PermutationExplainer explainer:  97%|█████████▋| 487/500 [02:20<00:03,  3.71it/s][A
    PermutationExplainer explainer:  98%|█████████▊| 488/500 [02:20<00:03,  3.68it/s][A
    PermutationExplainer explainer:  98%|█████████▊| 489/500 [02:20<00:02,  3.70it/s][A
    PermutationExplainer explainer:  98%|█████████▊| 490/500 [02:21<00:02,  3.74it/s][A
    PermutationExplainer explainer:  98%|█████████▊| 491/500 [02:21<00:02,  3.75it/s][A
    PermutationExplainer explainer:  98%|█████████▊| 492/500 [02:21<00:02,  3.75it/s][A
    PermutationExplainer explainer:  99%|█████████▊| 493/500 [02:21<00:01,  3.69it/s][A
    PermutationExplainer explainer:  99%|█████████▉| 494/500 [02:22<00:01,  3.69it/s][A
    PermutationExplainer explainer:  99%|█████████▉| 495/500 [02:22<00:01,  3.71it/s][A
    PermutationExplainer explainer:  99%|█████████▉| 496/500 [02:22<00:01,  3.69it/s][A
    PermutationExplainer explainer:  99%|█████████▉| 497/500 [02:22<00:00,  3.71it/s][A
    PermutationExplainer explainer: 100%|█████████▉| 498/500 [02:23<00:00,  3.62it/s][A
    PermutationExplainer explainer: 100%|█████████▉| 499/500 [02:23<00:00,  3.54it/s][A
    PermutationExplainer explainer: 100%|██████████| 500/500 [02:23<00:00,  3.45it/s][A
    PermutationExplainer explainer: 501it [02:24,  3.22it/s]                         [A
    100%|██████████| 1/1 [02:25<00:00, 145.14s/it]

    
    
    Feature 3 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    hr 🔹 [id: 0 | heter: 0.05 | inst: 500 | w: 1.00]
        workingday = 0.00 🔹 [id: 1 | heter: 0.02 | inst: 158 | w: 0.32]
            temp ≤ 9.11 🔹 [id: 2 | heter: 0.02 | inst: 92 | w: 0.18]
            temp > 9.11 🔹 [id: 3 | heter: 0.01 | inst: 66 | w: 0.13]
        workingday ≠ 0.00 🔹 [id: 4 | heter: 0.03 | inst: 342 | w: 0.68]
            yr = 0.00 🔹 [id: 5 | heter: 0.02 | inst: 169 | w: 0.34]
            yr ≠ 0.00 🔹 [id: 6 | heter: 0.02 | inst: 173 | w: 0.35]
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.05
        Level 1🔹heter: 0.03 | 🔻0.03 (51.02%)
            Level 2🔹heter: 0.02 | 🔻0.01 (30.09%)
    
    


    



```python
for node_idx in [1, 4]:
    regional_shap_dp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_35_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_35_1.png)
    



```python
for node_idx in [2, 3, 5, 6]:
    regional_shap_dp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

    /home/givasile/github/packages/effector/effector/global_effect_shap.py:469: RuntimeWarning: invalid value encountered in sqrt
      np.sqrt(self.feature_effect["feature_" + str(feature)]["spline_std"](x))



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_36_1.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_36_2.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_36_3.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_36_4.png)
    


## Conclusion

**Global effect of hour**
All methods agree: *hour* has a strong influence on bike rentals, showing two clear peaks—around **8:00** and **17:00**. This likely reflects commute times. But the exact shape of the effect varies between methods, hinting that local (regional) patterns could help explain these differences.

**Regional effect of hour**
When we zoom in using regional methods, two patterns emerge:

* **On working days**, the effect follows the global trend, with peaks at **8:00** and **17:00**—again, probably due to commuting.
* **On non-working days**, we see a single peak around **13:00**, which makes sense if people are out enjoying leisure activities or sightseeing.

All methods agree up to this point.

**Interactions**
Looking deeper, we see some interesting (but weaker) interactions. Most methods highlight either **temperature** or **year** (whether it’s the first or second year of data) as relevant.

For example, RHALE shows that on **non-working days**, the midday peak (12:00–14:00) becomes even stronger when the **temperature is higher**. That fits our intuition—people are more likely to rent bikes when it’s warm and sunny.
