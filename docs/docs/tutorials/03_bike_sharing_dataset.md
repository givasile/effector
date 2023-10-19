# Bike-Sharing Dataset

The Bike-Sharing Dataset contains the bike rentals for almost every hour over the period 2011 and 2012. 
The dataset contains 14 features and we select the 11 features that are relevant to the prediction task. 
The features contain information about the day, like the month, the hour, the day of the week, the day-type,
and the weather conditions. 

Lets take a closer look


```python
import effector
import pandas as pd
import tensorflow as tf
from tensorflow import keras
```

    2023-10-19 14:42:56.026054: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-19 14:42:56.190201: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-10-19 14:42:56.190219: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2023-10-19 14:42:56.219898: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2023-10-19 14:42:56.869564: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2023-10-19 14:42:56.869623: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2023-10-19 14:42:56.869630: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.


## Preprocess the data


```python
# load dataset
df = pd.read_csv("./../data/Bike-Sharing-Dataset/hour.csv")

# drop columns
df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)
```


```python
for col_name in df.columns:
    print("Feature: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(df[col_name].unique()), df[col_name].mean(), df[col_name].std(), df[col_name].min(), df[col_name].max()))
```

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
    Feature: cnt            , unique:  869, Mean: 189.46, Std: 181.39, Min:   1.00, Max: 977.00


Feature Table:

| Feature      | Description                            | Value Range                                         |
|--------------|----------------------------------------|-----------------------------------------------------|
| season       | season                                 | 1: winter, 2: spring, 3: summer, 4: fall            |
| yr           | year                                   | 0: 2011, 1: 2012                                    |
| mnth         | month                                  | 1 to 12                                             |
| hr           | hour                                   | 0 to 23                                             |
| holiday      | whether the day is a holiday or not    | 0: no, 1: yes                                       |
| weekday      | day of the week                        | 0: Sunday, 1: Monday, …, 6: Saturday                |
| workingday   | whether the day is a working day or not | 0: no, 1: yes                                      |
| weathersit   | weather situation                      | 1: clear, 2: mist, 3: light rain, 4: heavy rain     |
| temp         | temperature                            | normalized, [0.02, 1.00]                            |
| hum          | humidity                               | normalized, [0.00, 1.00]                            |
| windspeed    | wind speed                             | normalized, [0.00, 1.00]                            |


Target:

| Target       | Description                            | Value Range                                         |
|--------------|----------------------------------------|-----------------------------------------------------|
| cnt          | bike rentals per hour                  | [1, 977]                                            |



```python
def preprocess(df):
    # shuffle
    df.sample(frac=1).reset_index(drop=True)

    # Standarize X
    X_df = df.drop(["cnt"], axis=1)
    x_mean = X_df.mean()
    x_std = X_df.std()
    X_df = (X_df - X_df.mean()) / X_df.std()

    # Standarize Y
    Y_df = df["cnt"]
    y_mean = Y_df.mean()
    y_std = Y_df.std()
    Y_df = (Y_df - Y_df.mean()) / Y_df.std()
    return X_df, Y_df, x_mean, x_std, y_mean, y_std

# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = preprocess(df)
```


```python
def split(X_df, Y_df):
    # data split
    X_train = X_df[:int(0.8 * len(X_df))]
    Y_train = Y_df[:int(0.8 * len(Y_df))]
    X_test = X_df[int(0.8 * len(X_df)):]
    Y_test = Y_df[int(0.8 * len(Y_df)):]
    return X_train, Y_train, X_test, Y_test

# train/test split
X_train, Y_train, X_test, Y_test = split(X_df, Y_df)
```

## Fit a Neural Network

We train a deep fully-connected Neural Network with 3 hidden layers for \(20\) epochs. 
The model achieves a mean absolute error on the test of about \(38\) counts.


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


    2023-10-19 14:42:57.615092: E tensorflow/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
    2023-10-19 14:42:57.615115: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (givasile-ubuntu-XPS-15-9500): /proc/driver/nvidia/version does not exist
    2023-10-19 14:42:57.615360: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    28/28 [==============================] - 1s 9ms/step - loss: 0.4572 - mae: 0.4918 - root_mean_squared_error: 0.6762
    Epoch 2/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.3185 - mae: 0.4041 - root_mean_squared_error: 0.5644
    Epoch 3/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.2496 - mae: 0.3529 - root_mean_squared_error: 0.4996
    Epoch 4/20
    28/28 [==============================] - 0s 10ms/step - loss: 0.1986 - mae: 0.3125 - root_mean_squared_error: 0.4457
    Epoch 5/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.1424 - mae: 0.2617 - root_mean_squared_error: 0.3773
    Epoch 6/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.1000 - mae: 0.2161 - root_mean_squared_error: 0.3162
    Epoch 7/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0775 - mae: 0.1903 - root_mean_squared_error: 0.2784
    Epoch 8/20
    28/28 [==============================] - 0s 11ms/step - loss: 0.0731 - mae: 0.1877 - root_mean_squared_error: 0.2703
    Epoch 9/20
    28/28 [==============================] - 0s 10ms/step - loss: 0.0620 - mae: 0.1725 - root_mean_squared_error: 0.2490
    Epoch 10/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0634 - mae: 0.1800 - root_mean_squared_error: 0.2518
    Epoch 11/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0566 - mae: 0.1640 - root_mean_squared_error: 0.2379
    Epoch 12/20
    28/28 [==============================] - 0s 12ms/step - loss: 0.0487 - mae: 0.1540 - root_mean_squared_error: 0.2206
    Epoch 13/20
    28/28 [==============================] - 0s 11ms/step - loss: 0.0411 - mae: 0.1375 - root_mean_squared_error: 0.2026
    Epoch 14/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0401 - mae: 0.1367 - root_mean_squared_error: 0.2003
    Epoch 15/20
    28/28 [==============================] - 0s 11ms/step - loss: 0.0402 - mae: 0.1386 - root_mean_squared_error: 0.2006
    Epoch 16/20
    28/28 [==============================] - 0s 11ms/step - loss: 0.0390 - mae: 0.1334 - root_mean_squared_error: 0.1975
    Epoch 17/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0363 - mae: 0.1287 - root_mean_squared_error: 0.1905
    Epoch 18/20
    28/28 [==============================] - 0s 10ms/step - loss: 0.0352 - mae: 0.1277 - root_mean_squared_error: 0.1876
    Epoch 19/20
    28/28 [==============================] - 0s 11ms/step - loss: 0.0367 - mae: 0.1332 - root_mean_squared_error: 0.1915
    Epoch 20/20
    28/28 [==============================] - 0s 10ms/step - loss: 0.0342 - mae: 0.1274 - root_mean_squared_error: 0.1850
    435/435 [==============================] - 1s 1ms/step - loss: 0.0350 - mae: 0.1265 - root_mean_squared_error: 0.1870
    109/109 [==============================] - 0s 1ms/step - loss: 0.1852 - mae: 0.2996 - root_mean_squared_error: 0.4304





    [0.18520371615886688, 0.2996417284011841, 0.4303530156612396]



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
scale_x = {"mean": x_mean[3], "std": x_std[3]}
scale_y = {"mean": y_mean, "std": y_std}
scale_x_list =[{"mean": x_mean[i], "std": x_std[i]} for i in range(len(x_mean))]
col_names = X_df.columns.to_list()
```


```python

rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Greedy(init_nof_bins=200, min_points_per_bin=30, discount=20, cat_limit=10)
rhale.fit(features=3, binning_method=binning_method)
fig, ax1, ax2 = rhale.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_13_0.png)
    



```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.DynamicProgramming(max_nof_bins=24, min_points_per_bin=30, discount=0.)
rhale.fit(features=3, binning_method=binning_method)
fig, ax1, ax2 = rhale.plot(feature=3)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_14_0.png)
    



```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Fixed(nof_bins=100, min_points_per_bin=0, cat_limit=10)
rhale.fit(features=3, binning_method=binning_method)
fig, ax1, ax2 = rhale.plot(feature=3, scale_x=scale_x, scale_y=scale_y, centering=True)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_15_0.png)
    



```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Fixed(nof_bins=50, min_points_per_bin=0, cat_limit=10)
rhale.fit(features=3, binning_method=binning_method)
fig, ax1, ax2 = rhale.plot(feature=3, scale_x=scale_x, scale_y=scale_y, centering=True, confidence_interval=True)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_16_0.png)
    



```python
pdp_ice = effector.PDPwithICE(data=X_train.to_numpy(), model=model_forward)
fig, ax = pdp_ice.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_17_0.png)
    


# Regional Effects


```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac).plot(feature=3, centering=True, confidence_interval=True, scale_x=scale_x, scale_y=scale_y)
```

    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3757: RuntimeWarning: Degrees of freedom <= 0 for slice
      return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/_methods.py:222: RuntimeWarning: invalid value encountered in true_divide
      arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe',
    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/_methods.py:256: RuntimeWarning: invalid value encountered in true_divide
      ret = ret.dtype.type(ret / rcount)



    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_19_1.png)
    





```python
regional_rhale = effector.RegionalRHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac,
                                        cat_limit=10,
                                        feature_names=col_names)
regional_rhale.print_splits(features=3, only_important=True, scale_x=scale_x_list)

```

    Important splits for feature hr
    - On feature workingday (cat)
      - Candidate split positions: 0.00, 1.00
      - Position of split: 0.00
      - Heterogeneity before split: 5.47
      - Heterogeneity after split: 3.39
      - Heterogeneity drop: 2.08 (61.31 %)
      - Number of instances before split: 13903
      - Number of instances after split: [4387, 9516]
    - On feature temp (cont)
      - Candidate split positions: 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.39, 0.44, 0.49, 0.53, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83, 0.88, 0.93, 0.98
      - Position of split: 0.44
      - Heterogeneity before split: 3.39
      - Heterogeneity after split: 2.89
      - Heterogeneity drop: 0.50 (17.25 %)
      - Number of instances before split: [4387, 9516]
      - Number of instances after split: [1943, 2444, 3542, 5974]



```python
regional_rhale.plot_first_level(feature=3, confidence_interval=True, centering=True, scale_x=scale_x_list, scale_y=scale_y)
```

    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3757: RuntimeWarning: Degrees of freedom <= 0 for slice
      return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/_methods.py:222: RuntimeWarning: invalid value encountered in true_divide
      arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe',
    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/_methods.py:256: RuntimeWarning: invalid value encountered in true_divide
      ret = ret.dtype.type(ret / rcount)



    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_22_1.png)
    


    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3757: RuntimeWarning: Degrees of freedom <= 0 for slice
      return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/_methods.py:222: RuntimeWarning: invalid value encountered in true_divide
      arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe',
    /home/givasile/miniconda3/envs/effector/lib/python3.9/site-packages/numpy/core/_methods.py:256: RuntimeWarning: invalid value encountered in true_divide
      ret = ret.dtype.type(ret / rcount)



    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_22_3.png)
    



```python

```
