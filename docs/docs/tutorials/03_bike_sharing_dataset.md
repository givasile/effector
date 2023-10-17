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
    28/28 [==============================] - 1s 8ms/step - loss: 0.4500 - mae: 0.4897 - root_mean_squared_error: 0.6708
    Epoch 2/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.3131 - mae: 0.4006 - root_mean_squared_error: 0.5595
    Epoch 3/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.2470 - mae: 0.3518 - root_mean_squared_error: 0.4970
    Epoch 4/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.1986 - mae: 0.3128 - root_mean_squared_error: 0.4456
    Epoch 5/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.1434 - mae: 0.2615 - root_mean_squared_error: 0.3787
    Epoch 6/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.1088 - mae: 0.2255 - root_mean_squared_error: 0.3298
    Epoch 7/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0816 - mae: 0.1953 - root_mean_squared_error: 0.2856
    Epoch 8/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0781 - mae: 0.1953 - root_mean_squared_error: 0.2794
    Epoch 9/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0643 - mae: 0.1772 - root_mean_squared_error: 0.2536
    Epoch 10/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0504 - mae: 0.1541 - root_mean_squared_error: 0.2245
    Epoch 11/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0464 - mae: 0.1470 - root_mean_squared_error: 0.2155
    Epoch 12/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0463 - mae: 0.1470 - root_mean_squared_error: 0.2152
    Epoch 13/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0452 - mae: 0.1467 - root_mean_squared_error: 0.2126
    Epoch 14/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0386 - mae: 0.1341 - root_mean_squared_error: 0.1965
    Epoch 15/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0372 - mae: 0.1309 - root_mean_squared_error: 0.1928
    Epoch 16/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0362 - mae: 0.1284 - root_mean_squared_error: 0.1903
    Epoch 17/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0354 - mae: 0.1300 - root_mean_squared_error: 0.1882
    Epoch 18/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0361 - mae: 0.1321 - root_mean_squared_error: 0.1899
    Epoch 19/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0359 - mae: 0.1324 - root_mean_squared_error: 0.1896
    Epoch 20/20
    28/28 [==============================] - 0s 8ms/step - loss: 0.0316 - mae: 0.1202 - root_mean_squared_error: 0.1776
    435/435 [==============================] - 1s 1ms/step - loss: 0.0333 - mae: 0.1285 - root_mean_squared_error: 0.1825
    109/109 [==============================] - 0s 1ms/step - loss: 0.2344 - mae: 0.3406 - root_mean_squared_error: 0.4841





    [0.23436342179775238, 0.34056356549263, 0.484110951423645]



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
    



```python

```
