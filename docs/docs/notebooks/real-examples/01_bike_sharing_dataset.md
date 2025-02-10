# Bike-Sharing Dataset

The [Bike-Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information. The dataset contains 14 features with information about the day-type, e.g., month, hour, which day of the week, whether it is working-day, and the weather conditions, e.g., temperature, humidity, wind speed, etc. The target variable is the number of bike rentals per hour. The dataset contains 17,379 instances. 


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

    The history saving thread hit an unexpected error (OperationalError('attempt to write a readonly database')).History will not be written to the database.


    2025-02-10 14:25:29.177546: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2025-02-10 14:25:29.180211: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2025-02-10 14:25:29.189115: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1739193929.204583  564943 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1739193929.209098  564943 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-02-10 14:25:29.224189: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
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
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>hum</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.81</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.80</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.80</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.75</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.75</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Design matrix shape: {}".format(X.shape))
print("---------------------------------")
for col_name in X.columns:
    print("Feature: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(X[col_name].unique()), X[col_name].mean(), X[col_name].std(), X[col_name].min(), X[col_name].max()))
    
print("\nTarget shape: {}".format(y.shape))
print("---------------------------------")
for col_name in y.columns:
    print("Target: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(y[col_name].unique()), y[col_name].mean(), y[col_name].std(), y[col_name].min(), y[col_name].max()))
```

    Design matrix shape: (17379, 11)
    ---------------------------------
    Feature: season         , unique:    4, Mean:   2.50, Std:   1.11, Min:   1.00, Max:   4.00
    Feature: yr             , unique:    2, Mean:   0.50, Std:   0.50, Min:   0.00, Max:   1.00
    Feature: mnth           , unique:   12, Mean:   6.54, Std:   3.44, Min:   1.00, Max:  12.00
    Feature: hr             , unique:   24, Mean:  11.55, Std:   6.91, Min:   0.00, Max:  23.00
    Feature: holiday        , unique:    2, Mean:   0.03, Std:   0.17, Min:   0.00, Max:   1.00
    Feature: weekday        , unique:    7, Mean:   3.00, Std:   2.01, Min:   0.00, Max:   6.00
    Feature: workingday     , unique:    2, Mean:   0.68, Std:   0.47, Min:   0.00, Max:   1.00
    Feature: weathersit     , unique:    4, Mean:   1.43, Std:   0.64, Min:   1.00, Max:   4.00
    Feature: temp           , unique:   50, Mean:   0.50, Std:   0.19, Min:   0.02, Max:   1.00
    Feature: hum            , unique:   89, Mean:   0.63, Std:   0.19, Min:   0.00, Max:   1.00
    Feature: windspeed      , unique:   30, Mean:   0.19, Std:   0.12, Min:   0.00, Max:   0.85
    
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


    2025-02-10 14:25:33.003605: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2025-02-10 14:25:33.003630: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:137] retrieving CUDA diagnostic information for host: givasile-ubuntu-XPS-15-9500
    2025-02-10 14:25:33.003636: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:144] hostname: givasile-ubuntu-XPS-15-9500
    2025-02-10 14:25:33.003773: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:168] libcuda reported version is: 560.35.5
    2025-02-10 14:25:33.003793: I external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:172] kernel reported version is: 550.120.0
    2025-02-10 14:25:33.003798: E external/local_xla/xla/stream_executor/cuda/cuda_diagnostics.cc:262] kernel version 550.120.0 does not match DSO version 560.35.5 -- cannot find working devices in this configuration


    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - loss: 0.6231 - mae: 0.5745 - root_mean_squared_error: 0.7853
    Epoch 2/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.3870 - mae: 0.4506 - root_mean_squared_error: 0.6219
    Epoch 3/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.2976 - mae: 0.3851 - root_mean_squared_error: 0.5454
    Epoch 4/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step - loss: 0.2237 - mae: 0.3326 - root_mean_squared_error: 0.4728
    Epoch 5/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step - loss: 0.1619 - mae: 0.2836 - root_mean_squared_error: 0.4023
    Epoch 6/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.1193 - mae: 0.2386 - root_mean_squared_error: 0.3451
    Epoch 7/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.0906 - mae: 0.2075 - root_mean_squared_error: 0.3009
    Epoch 8/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0753 - mae: 0.1895 - root_mean_squared_error: 0.2745
    Epoch 9/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0669 - mae: 0.1784 - root_mean_squared_error: 0.2586
    Epoch 10/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0610 - mae: 0.1703 - root_mean_squared_error: 0.2469
    Epoch 11/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0554 - mae: 0.1614 - root_mean_squared_error: 0.2353 
    Epoch 12/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0500 - mae: 0.1524 - root_mean_squared_error: 0.2235
    Epoch 13/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0462 - mae: 0.1459 - root_mean_squared_error: 0.2149 
    Epoch 14/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0431 - mae: 0.1407 - root_mean_squared_error: 0.2075 
    Epoch 15/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0410 - mae: 0.1372 - root_mean_squared_error: 0.2026
    Epoch 16/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0399 - mae: 0.1360 - root_mean_squared_error: 0.1996 
    Epoch 17/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0381 - mae: 0.1325 - root_mean_squared_error: 0.1952 
    Epoch 18/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 0.0378 - mae: 0.1323 - root_mean_squared_error: 0.1945 
    Epoch 19/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0376 - mae: 0.1325 - root_mean_squared_error: 0.1940
    Epoch 20/20
    [1m28/28[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0378 - mae: 0.1331 - root_mean_squared_error: 0.1944
    [1m435/435[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 848us/step - loss: 0.0420 - mae: 0.1422 - root_mean_squared_error: 0.2048
    [1m109/109[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 856us/step - loss: 0.0701 - mae: 0.1725 - root_mean_squared_error: 0.2633





    [0.06317276507616043, 0.16644978523254395, 0.25134193897247314]



We train a deep fully-connected Neural Network with 3 hidden layers for \(20\) epochs. 
The model achieves a root mean squared error on the test of about $0.24$ units, that corresponds to approximately \(0.26 * 181 = 47\) counts.

## Explain

We will focus on the feature `temp` (temperature) because its global effect is quite heterogeneous and the heterogeneity can be further explained using regional effects.


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

## Global Effect  

**Overview of All Features**

We start by examining all relevant features. Feature effect methods are generally more meaningful for numerical features. To compare effects across features more easily, we standardize the `y_limits`.  

Relevant features:

- `month`  
- `hr`  
- `weekday`  
- `workingday`  
- `temp`  
- `humidity`  
- `windspeed` 

We observe that features: `hour`, `temperature` and `humidity` have an intersting structure. Out of them `hour` has by far the most influence on the output, so it makes sensce to fucse on it further.


```python
pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=2000)
for i in [2, 3, 8, 9, 10]:
    pdp.plot(feature=i, centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True, nof_ice=200, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_1.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_2.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_3.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_4.png)
    


### PDP 


```python
pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=5000)
pdp.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True, nof_ice=200)
```

    2025-02-08 22:02:10.574746: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-02-08 22:02:10.693614: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-02-08 22:02:10.805763: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-02-08 22:02:11.586071: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.
    2025-02-08 22:02:11.705140: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 614400000 exceeds 10% of free system memory.



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_20_1.png)
    


### RHALE


```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
rhale.plot(feature=3, heterogeneity="std", centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_22_0.png)
    



```python
shap_dp = effector.ShapDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=500)
shap_dp.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)
```

    PermutationExplainer explainer: 501it [02:32,  3.13it/s]                                                                                                                                                    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_23_1.png)
    


### Conclusion  

All methods agree that `hour` affects `bike-rentals` with two peaks, around 8:00 and 17:00, likely reflecting commute hours. However, the effect varies significantly, so regional effects may help in understanding the origin of this heterogeneity.

## Regional Effect

### RegionalPDP


```python
regional_pdp = effector.RegionalPDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, nof_instances=5_000)
regional_pdp.summary(features=3, scale_x_list=scale_x_list)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.08s/it]

    
    
    Feature 3 - Full partition tree:
    Node id: 0, name: hr, heter: 0.44 || nof_instances:  5000 || weight: 1.00
            Node id: 1, name: hr | workingday == 0.00, heter: 0.38 || nof_instances:  1588 || weight: 0.32
                    Node id: 3, name: hr | workingday == 0.00 and temp <= 6.81, heter: 0.19 || nof_instances:   785 || weight: 0.16
                    Node id: 4, name: hr | workingday == 0.00 and temp > 6.81, heter: 0.22 || nof_instances:   803 || weight: 0.16
            Node id: 2, name: hr | workingday != 0.00, heter: 0.30 || nof_instances:  3412 || weight: 0.68
                    Node id: 5, name: hr | workingday != 0.00 and temp <= 6.81, heter: 0.21 || nof_instances:  1467 || weight: 0.29
                    Node id: 6, name: hr | workingday != 0.00 and temp > 6.81, heter: 0.21 || nof_instances:  1945 || weight: 0.39
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    Level 0, heter: 0.44
            Level 1, heter: 0.32 || heter drop : 0.12 (units), 27.28% (pcg)
                    Level 2, heter: 0.21 || heter drop : 0.12 (units), 35.73% (pcg)
    
    


    



```python
for node_idx in [1, 2]:
    regional_pdp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_28_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_28_1.png)
    



```python
for node_idx in [3, 4, 5, 6]:
    regional_pdp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_1.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_2.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_3.png)
    


### RegionalRHALE


```python
regional_rhale = effector.RegionalRHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
regional_rhale.summary(features=3, scale_x_list=scale_x_list)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.19s/it]

    
    
    Feature 3 - Full partition tree:
    Node id: 0, name: hr, heter: 5.68 || nof_instances: 13903 || weight: 1.00
            Node id: 1, name: hr | workingday == 0.00, heter: 0.75 || nof_instances:  4385 || weight: 0.32
            Node id: 2, name: hr | workingday != 0.00, heter: 5.44 || nof_instances:  9518 || weight: 0.68
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    Level 0, heter: 5.68
            Level 1, heter: 3.96 || heter drop : 1.71 (units), 30.22% (pcg)
    
    


    



```python
regional_rhale.plot(feature=3, node_idx=1, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
regional_rhale.plot(feature=3, node_idx=2, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_32_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_32_1.png)
    


### Regional SHAP-DP


```python
regional_shap_dp = effector.RegionalShapDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, nof_instances=500)
```


```python
regional_shap_dp.summary(features=3, scale_x_list=scale_x_list)
```

    
    
    Feature 3 - Full partition tree:
    Node id: 0, name: hr, heter: 0.06 || nof_instances:   500 || weight: 1.00
            Node id: 1, name: hr | workingday == 0.00, heter: 0.02 || nof_instances:   155 || weight: 0.31
                    Node id: 3, name: hr | workingday == 0.00 and temp <= 6.81, heter: 0.01 || nof_instances:    83 || weight: 0.17
                    Node id: 4, name: hr | workingday == 0.00 and temp > 6.81, heter: 0.03 || nof_instances:    72 || weight: 0.14
            Node id: 2, name: hr | workingday != 0.00, heter: 0.03 || nof_instances:   345 || weight: 0.69
                    Node id: 5, name: hr | workingday != 0.00 and temp <= 6.81, heter: 0.02 || nof_instances:   156 || weight: 0.31
                    Node id: 6, name: hr | workingday != 0.00 and temp > 6.81, heter: 0.02 || nof_instances:   189 || weight: 0.38
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    Level 0, heter: 0.06
            Level 1, heter: 0.03 || heter drop : 0.03 (units), 56.04% (pcg)
                    Level 2, heter: 0.02 || heter drop : 0.01 (units), 20.57% (pcg)
    
    



```python
for node_idx in [1, 2]:
    regional_shap_dp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_36_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_36_1.png)
    



```python
for node_idx in [3, 4, 5, 6]:
    regional_shap_dp.plot(feature=3, node_idx=node_idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_0.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_1.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_2.png)
    



    
![png](01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_3.png)
    


### Conclusion  

Regional methods reveal two distinct patterns: one for working days and another for non-working days.  

- On working days, the effect mirrors the global trend, with peaks around 8:00 and 17:00, likely due to commuting.  
- On non-working days, the pattern shifts to a single peak around 13:00, likely driven by sightseeing and leisure.  

PDP and ShapDP push it more; that on non-working days, `hour` interacts with `temperature` — peaks between 12:00 and 14:00 are more pronounced in warmer conditions. Makes sense, right? If you go for sightseeing, you probably 
