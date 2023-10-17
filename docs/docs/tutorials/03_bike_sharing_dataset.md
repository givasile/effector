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

    2023-10-17 13:09:43.963184: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-17 13:09:44.088048: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-10-17 13:09:44.088064: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2023-10-17 13:09:44.113370: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2023-10-17 13:09:44.693738: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2023-10-17 13:09:44.693799: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2023-10-17 13:09:44.693805: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.


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


    2023-10-17 13:09:45.331592: E tensorflow/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
    2023-10-17 13:09:45.331618: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (givasile-ubuntu-XPS-15-9500): /proc/driver/nvidia/version does not exist
    2023-10-17 13:09:45.331914: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    28/28 [==============================] - 1s 9ms/step - loss: 0.4389 - mae: 0.4817 - root_mean_squared_error: 0.6625
    Epoch 2/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.3147 - mae: 0.4033 - root_mean_squared_error: 0.5610
    Epoch 3/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.2549 - mae: 0.3576 - root_mean_squared_error: 0.5049
    Epoch 4/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.1975 - mae: 0.3115 - root_mean_squared_error: 0.4444
    Epoch 5/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.1396 - mae: 0.2585 - root_mean_squared_error: 0.3736
    Epoch 6/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.1080 - mae: 0.2272 - root_mean_squared_error: 0.3287
    Epoch 7/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0801 - mae: 0.1938 - root_mean_squared_error: 0.2830
    Epoch 8/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0664 - mae: 0.1797 - root_mean_squared_error: 0.2576
    Epoch 9/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0657 - mae: 0.1830 - root_mean_squared_error: 0.2563
    Epoch 10/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0596 - mae: 0.1722 - root_mean_squared_error: 0.2442
    Epoch 11/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0497 - mae: 0.1515 - root_mean_squared_error: 0.2228
    Epoch 12/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0487 - mae: 0.1549 - root_mean_squared_error: 0.2207
    Epoch 13/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0452 - mae: 0.1453 - root_mean_squared_error: 0.2125
    Epoch 14/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0437 - mae: 0.1464 - root_mean_squared_error: 0.2090
    Epoch 15/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0413 - mae: 0.1398 - root_mean_squared_error: 0.2033
    Epoch 16/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0428 - mae: 0.1404 - root_mean_squared_error: 0.2069
    Epoch 17/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0366 - mae: 0.1294 - root_mean_squared_error: 0.1912
    Epoch 18/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0357 - mae: 0.1280 - root_mean_squared_error: 0.1890
    Epoch 19/20
    28/28 [==============================] - 0s 9ms/step - loss: 0.0332 - mae: 0.1228 - root_mean_squared_error: 0.1823
    Epoch 20/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0332 - mae: 0.1241 - root_mean_squared_error: 0.1823
    435/435 [==============================] - 1s 1ms/step - loss: 0.0322 - mae: 0.1199 - root_mean_squared_error: 0.1794
    109/109 [==============================] - 0s 2ms/step - loss: 0.2048 - mae: 0.3111 - root_mean_squared_error: 0.4526





    [0.20484140515327454, 0.31109121441841125, 0.4525940716266632]



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
```


```python

rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Greedy(init_nof_bins=200, min_points_per_bin=30, discount=20, cat_limit=10)
rhale.fit(features=3, binning_method=binning_method)
rhale.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_13_0.png)
    



```python
ale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.DynamicProgramming(max_nof_bins=24, min_points_per_bin=30, discount=0.)
ale.fit(features=3, binning_method=binning_method)
ale.plot(feature=3)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_14_0.png)
    



```python
ale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Fixed(nof_bins=100, min_points_per_bin=0, cat_limit=10)
ale.fit(features=3, binning_method=binning_method)
ale.plot(feature=3, scale_x=scale_x, scale_y=scale_y, centering=True)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_15_0.png)
    



```python
ale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Fixed(nof_bins=50, min_points_per_bin=0, cat_limit=10)
ale.fit(features=3, binning_method=binning_method)
ale.plot(feature=3, scale_x=scale_x, scale_y=scale_y, centering=True, confidence_interval=True)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_16_0.png)
    



```python
pdp_ice = effector.PDPwithICE(data=X_train.to_numpy(), model=model_forward)
pdp_ice.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y)
```


    
![png](03_bike_sharing_dataset_files/03_bike_sharing_dataset_17_0.png)
    





    (<Figure size 432x288 with 1 Axes>,
     <AxesSubplot:title={'center':'PDP-ICE: feature 4'}, xlabel='feature 4', ylabel='y'>)




```python

```
