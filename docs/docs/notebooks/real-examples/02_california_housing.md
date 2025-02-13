```python
import numpy as np
import keras
import tensorflow as tf
import effector
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
```

    2025-02-09 00:14:28.581561: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2025-02-09 00:14:28.585557: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2025-02-09 00:14:28.598939: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1739056468.621639  256803 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1739056468.627939  256803 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-02-09 00:14:28.649223: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
np.random.seed(21)
```


```python
print(california_housing.DESCR)
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
    :Number of Instances: 20640
    
    :Number of Attributes: 8 numeric, predictive attributes and the target
    
    :Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude
    
    :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    
    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of dollars ($100,000).
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    A household is a group of people residing within a home. Since the average
    number of rooms and bedrooms in this dataset are provided per household, these
    columns may take surprisingly large values for block groups with few households
    and many empty houses, such as vacation resorts.
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. rubric:: References
    
    - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
      Statistics and Probability Letters, 33 (1997) 291-297
    



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

    Design matrix shape: (20640, 8)
    ---------------------------------
    Feature: MedInc         , unique: 12928, Mean:   3.87, Std:   1.90, Min:   0.50, Max:  15.00
    Feature: HouseAge       , unique:   52, Mean:  28.64, Std:  12.59, Min:   1.00, Max:  52.00
    Feature: AveRooms       , unique: 19392, Mean:   5.43, Std:   2.47, Min:   0.85, Max: 141.91
    Feature: AveBedrms      , unique: 14233, Mean:   1.10, Std:   0.47, Min:   0.33, Max:  34.07
    Feature: Population     , unique: 3888, Mean: 1425.48, Std: 1132.46, Min:   3.00, Max: 35682.00
    Feature: AveOccup       , unique: 18841, Mean:   3.07, Std:  10.39, Min:   0.69, Max: 1243.33
    Feature: Latitude       , unique:  862, Mean:  35.63, Std:   2.14, Min:  32.54, Max:  41.95
    Feature: Longitude      , unique:  844, Mean: -119.57, Std:   2.00, Min: -124.35, Max: -114.31
    
    Target shape: (20640,)
    ---------------------------------
    Target: MedHouseVal    , unique: 3842, Mean:   2.07, Std:   1.15, Min:   0.15, Max:   5.00



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
# Train - Evaluate - Explain a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=1024, epochs=20, verbose=1)
model.evaluate(X_train, Y_train, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)
```

    Epoch 1/20


    2025-02-09 00:14:32.300792: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2025-02-09 00:14:32.300826: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:137] retrieving CUDA diagnostic information for host: givasile-ubuntu-XPS-15-9500
    2025-02-09 00:14:32.300836: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:144] hostname: givasile-ubuntu-XPS-15-9500
    2025-02-09 00:14:32.301020: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:168] libcuda reported version is: 560.35.5
    2025-02-09 00:14:32.301050: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:172] kernel reported version is: 550.120.0
    2025-02-09 00:14:32.301058: E external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:262] kernel version 550.120.0 does not match DSO version 560.35.5 -- cannot find working devices in this configuration


    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 26ms/step - loss: 0.6131 - mae: 0.5806 - root_mean_squared_error: 0.7777
    Epoch 2/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.3267 - mae: 0.4119 - root_mean_squared_error: 0.5716
    Epoch 3/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.3024 - mae: 0.3919 - root_mean_squared_error: 0.5499
    Epoch 4/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.2882 - mae: 0.3810 - root_mean_squared_error: 0.5369
    Epoch 5/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.2794 - mae: 0.3725 - root_mean_squared_error: 0.5285
    Epoch 6/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.2760 - mae: 0.3723 - root_mean_squared_error: 0.5253
    Epoch 7/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 0.2748 - mae: 0.3716 - root_mean_squared_error: 0.5241
    Epoch 8/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.2585 - mae: 0.3587 - root_mean_squared_error: 0.5084
    Epoch 9/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.2591 - mae: 0.3572 - root_mean_squared_error: 0.5090
    Epoch 10/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.2549 - mae: 0.3503 - root_mean_squared_error: 0.5048
    Epoch 11/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.2451 - mae: 0.3445 - root_mean_squared_error: 0.4951
    Epoch 12/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.2506 - mae: 0.3519 - root_mean_squared_error: 0.5006
    Epoch 13/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.2288 - mae: 0.3318 - root_mean_squared_error: 0.4783
    Epoch 14/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.2267 - mae: 0.3319 - root_mean_squared_error: 0.4761
    Epoch 15/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.2281 - mae: 0.3307 - root_mean_squared_error: 0.4776
    Epoch 16/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.2234 - mae: 0.3284 - root_mean_squared_error: 0.4727
    Epoch 17/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.2328 - mae: 0.3441 - root_mean_squared_error: 0.4825
    Epoch 18/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.2136 - mae: 0.3226 - root_mean_squared_error: 0.4621
    Epoch 19/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.2185 - mae: 0.3208 - root_mean_squared_error: 0.4674
    Epoch 20/20
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.2107 - mae: 0.3188 - root_mean_squared_error: 0.4590
    [1m456/456[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2044 - mae: 0.3113 - root_mean_squared_error: 0.4521
    [1m114/114[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2989 - mae: 0.3614 - root_mean_squared_error: 0.5461





    [0.27413058280944824, 0.3489225506782532, 0.5235748291015625]




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
scale_y = {"mean": y_mean, "std": y_std}
scale_x_list =[{"mean": x_mean.iloc[i], "std": x_std.iloc[i]} for i in range(len(x_mean))]
```


```python
y_limits = [0, 4]
dy_limits = [-3, 3]
```

## Global effects


```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name, nof_instances="all")
for i in range(len(feature_names)):
    rhale.plot(feature=i, centering=True, scale_x=scale_x_list[i], scale_y=scale_y, y_limits=y_limits, dy_limits=dy_limits)
```


    
![png](02_california_housing_files/02_california_housing_13_0.png)
    



    
![png](02_california_housing_files/02_california_housing_13_1.png)
    



    
![png](02_california_housing_files/02_california_housing_13_2.png)
    



    
![png](02_california_housing_files/02_california_housing_13_3.png)
    



    
![png](02_california_housing_files/02_california_housing_13_4.png)
    



    
![png](02_california_housing_files/02_california_housing_13_5.png)
    



    
![png](02_california_housing_files/02_california_housing_13_6.png)
    



    
![png](02_california_housing_files/02_california_housing_13_7.png)
    


## Regional Effects

```python
reg_rhale = effector.RegionalRHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac,
                                   feature_names=feature_names, target_name=target_name, nof_instances="all")
reg_rhale.fit("all", space_partitioner=effector.space_partitioning.Best(heter_pcg_drop_thres=0.25))
reg_rhale.summary(features="all", scale_x_list=scale_x_list)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:40<00:00,  5.08s/it]

    
    
    Feature 0 - Full partition tree:
    Node id: 0, name: MedInc, heter: 0.06 || nof_instances: 14576 || weight: 1.00
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 0.06
    
    
    
    
    Feature 1 - Full partition tree:
    Node id: 0, name: HouseAge, heter: 0.05 || nof_instances: 14576 || weight: 1.00
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    Level 0, heter: 0.05
    
    
    
    
    Feature 2 - Full partition tree:
    Node id: 0, name: AveRooms, heter: 0.04 || nof_instances: 14576 || weight: 1.00
            Node id: 1, name: AveRooms | MedInc <= 3.73, heter: 0.03 || nof_instances:  8289 || weight: 0.57
            Node id: 2, name: AveRooms | MedInc > 3.73, heter: 0.03 || nof_instances:  6287 || weight: 0.43
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    Level 0, heter: 0.04
            Level 1, heter: 0.03 || heter drop : 0.01 (units), 25.46% (pcg)
    
    
    
    
    Feature 3 - Full partition tree:
    Node id: 0, name: AveBedrms, heter: 0.01 || nof_instances: 14576 || weight: 1.00
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    Level 0, heter: 0.01
    
    
    
    
    Feature 4 - Full partition tree:
    Node id: 0, name: Population, heter: 0.02 || nof_instances: 14576 || weight: 1.00
    --------------------------------------------------
    Feature 4 - Statistics per tree level:
    Level 0, heter: 0.02
    
    
    
    
    Feature 5 - Full partition tree:
    Node id: 0, name: AveOccup, heter: 0.06 || nof_instances: 14576 || weight: 1.00
            Node id: 1, name: AveOccup | MedInc <= 3.73, heter: 0.03 || nof_instances:  8289 || weight: 0.57
            Node id: 2, name: AveOccup | MedInc > 3.73, heter: 0.04 || nof_instances:  6287 || weight: 0.43
    --------------------------------------------------
    Feature 5 - Statistics per tree level:
    Level 0, heter: 0.06
            Level 1, heter: 0.04 || heter drop : 0.02 (units), 39.17% (pcg)
    
    
    
    
    Feature 6 - Full partition tree:
    Node id: 0, name: Latitude, heter: 0.68 || nof_instances: 14576 || weight: 1.00
            Node id: 1, name: Latitude | Longitude <= -121.55, heter: 0.52 || nof_instances:  3810 || weight: 0.26
            Node id: 2, name: Latitude | Longitude > -121.55, heter: 0.31 || nof_instances: 10766 || weight: 0.74
    --------------------------------------------------
    Feature 6 - Statistics per tree level:
    Level 0, heter: 0.68
            Level 1, heter: 0.36 || heter drop : 0.32 (units), 46.55% (pcg)
    
    
    
    
    Feature 7 - Full partition tree:
    Node id: 0, name: Longitude, heter: 0.53 || nof_instances: 14576 || weight: 1.00
            Node id: 1, name: Longitude | Latitude <= 34.75, heter: 0.20 || nof_instances:  7954 || weight: 0.55
            Node id: 2, name: Longitude | Latitude > 34.75, heter: 0.38 || nof_instances:  6622 || weight: 0.45
    --------------------------------------------------
    Feature 7 - Statistics per tree level:
    Level 0, heter: 0.53
            Level 1, heter: 0.28 || heter drop : 0.25 (units), 47.01% (pcg)
    
    


    


**AveOccup: average number of people residing in a house**


```python
reg_rhale.plot(feature=5, node_idx=0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_17_0.png)
    



```python
for node_idx in [1, 2]:
    reg_rhale.plot(feature=5, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_18_0.png)
    



    
![png](02_california_housing_files/02_california_housing_18_1.png)
    


## Latitude (south to north)


```python
reg_rhale.plot(feature=6, node_idx=0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_20_0.png)
    


**Global Trend:** House prices decrease as we move north.  


```python
for node_idx in [1, 2]:
    reg_rhale.plot(feature=6, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_22_0.png)
    



    
![png](02_california_housing_files/02_california_housing_22_1.png)
    


**Global Trend:** House prices decrease as we move north.  

**Regional Trends:** Moreorless the same, with minor different curves.

## Longitude (west to east)


```python
reg_rhale.plot(feature=7, node_idx=0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](02_california_housing_files/02_california_housing_25_0.png)
    


**Global Trend:** House prices decrease as we move east.  


```python
for node_idx in [1, 2]:
    reg_rhale.plot(feature=7, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_27_0.png)
    



    
![png](02_california_housing_files/02_california_housing_27_1.png)
    


**Global Trend:** House prices decrease as we move east.  

**Regional Trends:**  
- **North (latitude > 35.85):** Prices drop more sharply in the first half from east to west.  
- **South (latitude < 35.85):** Prices drop more sharply in the second half from east to west.


```python

```
