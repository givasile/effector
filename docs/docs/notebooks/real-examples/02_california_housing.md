# California Housing

- Author: [givasile](https://givasile.github.io/)
- Runtime: ~35 s
- Description: Global and regional RHALE effects on the California-housing
  dataset, explaining a neural network's predicted house values — with
  regional splits on income, latitude and longitude.


```python
import numpy as np
import keras
import tensorflow as tf
import effector
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
```

    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1783328874.489834   13166 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
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
      Statistics and Probability Letters, 33:291-297, 1997.
    



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


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - loss: 1.0810 - mae: 0.8121 - root_mean_squared_error: 1.0397

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.8814 - mae: 0.7174 - root_mean_squared_error: 0.9363

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 20ms/step - loss: 0.7734 - mae: 0.6637 - root_mean_squared_error: 0.8749

    [1m10/15[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 21ms/step - loss: 0.7058 - mae: 0.6289 - root_mean_squared_error: 0.8345

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 20ms/step - loss: 0.6591 - mae: 0.6038 - root_mean_squared_error: 0.8057

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 0.4796 - mae: 0.5071 - root_mean_squared_error: 0.6925


    Epoch 2/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 34ms/step - loss: 0.3748 - mae: 0.4371 - root_mean_squared_error: 0.6122

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.3450 - mae: 0.4224 - root_mean_squared_error: 0.5871

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.3348 - mae: 0.4169 - root_mean_squared_error: 0.5784

    [1m10/15[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.3300 - mae: 0.4138 - root_mean_squared_error: 0.5743

    [1m14/15[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 18ms/step - loss: 0.3263 - mae: 0.4113 - root_mean_squared_error: 0.5711

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.3183 - mae: 0.4046 - root_mean_squared_error: 0.5642


    Epoch 3/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 32ms/step - loss: 0.3153 - mae: 0.4023 - root_mean_squared_error: 0.5615

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.3082 - mae: 0.3963 - root_mean_squared_error: 0.5552

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3013 - mae: 0.3914 - root_mean_squared_error: 0.5488

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2981 - mae: 0.3894 - root_mean_squared_error: 0.5460

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2951 - mae: 0.3865 - root_mean_squared_error: 0.5432


    Epoch 4/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.3122 - mae: 0.3892 - root_mean_squared_error: 0.5587

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2949 - mae: 0.3821 - root_mean_squared_error: 0.5430

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2934 - mae: 0.3830 - root_mean_squared_error: 0.5417

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2921 - mae: 0.3826 - root_mean_squared_error: 0.5405

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2917 - mae: 0.3833 - root_mean_squared_error: 0.5401


    Epoch 5/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 33ms/step - loss: 0.2718 - mae: 0.3727 - root_mean_squared_error: 0.5214

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2842 - mae: 0.3760 - root_mean_squared_error: 0.5330

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2791 - mae: 0.3737 - root_mean_squared_error: 0.5282

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2783 - mae: 0.3729 - root_mean_squared_error: 0.5275

    [1m14/15[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 17ms/step - loss: 0.2781 - mae: 0.3727 - root_mean_squared_error: 0.5273

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2781 - mae: 0.3727 - root_mean_squared_error: 0.5274


    Epoch 6/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2487 - mae: 0.3674 - root_mean_squared_error: 0.4987

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2710 - mae: 0.3753 - root_mean_squared_error: 0.5204

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.2746 - mae: 0.3752 - root_mean_squared_error: 0.5239

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2744 - mae: 0.3738 - root_mean_squared_error: 0.5238

    [1m14/15[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 18ms/step - loss: 0.2737 - mae: 0.3726 - root_mean_squared_error: 0.5231

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2710 - mae: 0.3688 - root_mean_squared_error: 0.5205


    Epoch 7/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2613 - mae: 0.3526 - root_mean_squared_error: 0.5112

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2564 - mae: 0.3538 - root_mean_squared_error: 0.5063

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2576 - mae: 0.3574 - root_mean_squared_error: 0.5076

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 17ms/step - loss: 0.2580 - mae: 0.3580 - root_mean_squared_error: 0.5079

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2630 - mae: 0.3610 - root_mean_squared_error: 0.5129


    Epoch 8/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2449 - mae: 0.3659 - root_mean_squared_error: 0.4948

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2513 - mae: 0.3611 - root_mean_squared_error: 0.5012

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2523 - mae: 0.3594 - root_mean_squared_error: 0.5023

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2538 - mae: 0.3586 - root_mean_squared_error: 0.5038

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2598 - mae: 0.3581 - root_mean_squared_error: 0.5097


    Epoch 9/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2596 - mae: 0.3465 - root_mean_squared_error: 0.5095

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2666 - mae: 0.3524 - root_mean_squared_error: 0.5163

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2673 - mae: 0.3568 - root_mean_squared_error: 0.5170

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2645 - mae: 0.3568 - root_mean_squared_error: 0.5142

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2574 - mae: 0.3567 - root_mean_squared_error: 0.5073


    Epoch 10/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2083 - mae: 0.3201 - root_mean_squared_error: 0.4564

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2384 - mae: 0.3403 - root_mean_squared_error: 0.4880

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2420 - mae: 0.3434 - root_mean_squared_error: 0.4918

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2437 - mae: 0.3447 - root_mean_squared_error: 0.4935

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2453 - mae: 0.3458 - root_mean_squared_error: 0.4951

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2509 - mae: 0.3499 - root_mean_squared_error: 0.5009


    Epoch 11/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2373 - mae: 0.3415 - root_mean_squared_error: 0.4871

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2439 - mae: 0.3430 - root_mean_squared_error: 0.4938

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2461 - mae: 0.3440 - root_mean_squared_error: 0.4960

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2458 - mae: 0.3444 - root_mean_squared_error: 0.4958

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2443 - mae: 0.3454 - root_mean_squared_error: 0.4942


    Epoch 12/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2343 - mae: 0.3369 - root_mean_squared_error: 0.4841

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2337 - mae: 0.3384 - root_mean_squared_error: 0.4834

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2354 - mae: 0.3387 - root_mean_squared_error: 0.4852

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2386 - mae: 0.3403 - root_mean_squared_error: 0.4884

    [1m14/15[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 18ms/step - loss: 0.2397 - mae: 0.3410 - root_mean_squared_error: 0.4896

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2428 - mae: 0.3428 - root_mean_squared_error: 0.4927


    Epoch 13/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2447 - mae: 0.3610 - root_mean_squared_error: 0.4947

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2483 - mae: 0.3521 - root_mean_squared_error: 0.4982

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2449 - mae: 0.3493 - root_mean_squared_error: 0.4948

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2439 - mae: 0.3479 - root_mean_squared_error: 0.4939

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2384 - mae: 0.3412 - root_mean_squared_error: 0.4883


    Epoch 14/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2389 - mae: 0.3276 - root_mean_squared_error: 0.4888

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2263 - mae: 0.3259 - root_mean_squared_error: 0.4756

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2281 - mae: 0.3281 - root_mean_squared_error: 0.4775

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2290 - mae: 0.3298 - root_mean_squared_error: 0.4785

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2309 - mae: 0.3343 - root_mean_squared_error: 0.4805


    Epoch 15/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2047 - mae: 0.3183 - root_mean_squared_error: 0.4524

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2208 - mae: 0.3250 - root_mean_squared_error: 0.4698

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.2227 - mae: 0.3260 - root_mean_squared_error: 0.4719

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2236 - mae: 0.3268 - root_mean_squared_error: 0.4728

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2243 - mae: 0.3277 - root_mean_squared_error: 0.4736

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2257 - mae: 0.3299 - root_mean_squared_error: 0.4751


    Epoch 16/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2332 - mae: 0.3471 - root_mean_squared_error: 0.4829

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2329 - mae: 0.3444 - root_mean_squared_error: 0.4826

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2330 - mae: 0.3412 - root_mean_squared_error: 0.4827

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2317 - mae: 0.3391 - root_mean_squared_error: 0.4813

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2265 - mae: 0.3318 - root_mean_squared_error: 0.4759


    Epoch 17/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2268 - mae: 0.3355 - root_mean_squared_error: 0.4763

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2342 - mae: 0.3378 - root_mean_squared_error: 0.4839

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2314 - mae: 0.3353 - root_mean_squared_error: 0.4810

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2301 - mae: 0.3338 - root_mean_squared_error: 0.4796

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2239 - mae: 0.3284 - root_mean_squared_error: 0.4732


    Epoch 18/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 34ms/step - loss: 0.2154 - mae: 0.3158 - root_mean_squared_error: 0.4641

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2255 - mae: 0.3258 - root_mean_squared_error: 0.4748

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2271 - mae: 0.3295 - root_mean_squared_error: 0.4766

    [1m10/15[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2268 - mae: 0.3300 - root_mean_squared_error: 0.4762

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 18ms/step - loss: 0.2267 - mae: 0.3303 - root_mean_squared_error: 0.4761

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2259 - mae: 0.3309 - root_mean_squared_error: 0.4753


    Epoch 19/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2130 - mae: 0.3228 - root_mean_squared_error: 0.4616

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2170 - mae: 0.3220 - root_mean_squared_error: 0.4658

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2203 - mae: 0.3245 - root_mean_squared_error: 0.4693

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2199 - mae: 0.3246 - root_mean_squared_error: 0.4689

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2189 - mae: 0.3236 - root_mean_squared_error: 0.4679


    Epoch 20/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2228 - mae: 0.3324 - root_mean_squared_error: 0.4720

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2262 - mae: 0.3359 - root_mean_squared_error: 0.4756

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2243 - mae: 0.3329 - root_mean_squared_error: 0.4736

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2229 - mae: 0.3313 - root_mean_squared_error: 0.4721

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2146 - mae: 0.3237 - root_mean_squared_error: 0.4632


    [1m  1/456[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m46s[0m 102ms/step - loss: 0.1898 - mae: 0.3025 - root_mean_squared_error: 0.4357

    [1m 24/456[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.1996 - mae: 0.3101 - root_mean_squared_error: 0.4465   

    [1m 53/456[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2060 - mae: 0.3125 - root_mean_squared_error: 0.4537

    [1m 80/456[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2070 - mae: 0.3130 - root_mean_squared_error: 0.4548

    [1m109/456[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2080 - mae: 0.3139 - root_mean_squared_error: 0.4559

    [1m138/456[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2078 - mae: 0.3142 - root_mean_squared_error: 0.4558

    [1m168/456[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2085 - mae: 0.3147 - root_mean_squared_error: 0.4565

    [1m197/456[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2086 - mae: 0.3149 - root_mean_squared_error: 0.4567

    [1m224/456[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2090 - mae: 0.3151 - root_mean_squared_error: 0.4572

    [1m252/456[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2091 - mae: 0.3151 - root_mean_squared_error: 0.4573

    [1m278/456[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2089 - mae: 0.3151 - root_mean_squared_error: 0.4570

    [1m304/456[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2086 - mae: 0.3150 - root_mean_squared_error: 0.4567

    [1m319/456[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2085 - mae: 0.3150 - root_mean_squared_error: 0.4566

    [1m340/456[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2083 - mae: 0.3149 - root_mean_squared_error: 0.4564

    [1m361/456[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2081 - mae: 0.3147 - root_mean_squared_error: 0.4562

    [1m383/456[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 2ms/step - loss: 0.2079 - mae: 0.3146 - root_mean_squared_error: 0.4559

    [1m408/456[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 2ms/step - loss: 0.2076 - mae: 0.3145 - root_mean_squared_error: 0.4556

    [1m427/456[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 2ms/step - loss: 0.2075 - mae: 0.3144 - root_mean_squared_error: 0.4555

    [1m444/456[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 2ms/step - loss: 0.2074 - mae: 0.3144 - root_mean_squared_error: 0.4554

    [1m456/456[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2063 - mae: 0.3141 - root_mean_squared_error: 0.4542


    [1m  1/114[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 15ms/step - loss: 0.1734 - mae: 0.3073 - root_mean_squared_error: 0.4164

    [1m 21/114[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 0.3320 - mae: 0.3708 - root_mean_squared_error: 0.5744 

    [1m 37/114[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 0.3361 - mae: 0.3718 - root_mean_squared_error: 0.5787

    [1m 55/114[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 3ms/step - loss: 0.3264 - mae: 0.3681 - root_mean_squared_error: 0.5704

    [1m 81/114[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 3ms/step - loss: 0.3149 - mae: 0.3646 - root_mean_squared_error: 0.5604

    [1m105/114[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 2ms/step - loss: 0.3064 - mae: 0.3616 - root_mean_squared_error: 0.5528

    [1m114/114[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2771 - mae: 0.3509 - root_mean_squared_error: 0.5264





    [0.2771107852458954, 0.3509494364261627, 0.5264131426811218]




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
y_limits = [-0.5, 5]
dy_limits = [-3, 3]
```

## Global effects


```python
rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, schema={"feature_names": feature_names, "target_name": target_name}, nof_instances="all")
for i in range(len(feature_names)):
    rhale.plot(feature=i, centering=True, scale_x=scale_x_list[i], scale_y=scale_y, y_limits=y_limits, dy_limits=dy_limits)
```


    
![png](02_california_housing_files/02_california_housing_14_0.png)
    



    
![png](02_california_housing_files/02_california_housing_14_1.png)
    



    
![png](02_california_housing_files/02_california_housing_14_2.png)
    



    
![png](02_california_housing_files/02_california_housing_14_3.png)
    



    
![png](02_california_housing_files/02_california_housing_14_4.png)
    



    
![png](02_california_housing_files/02_california_housing_14_5.png)
    



    
![png](02_california_housing_files/02_california_housing_14_6.png)
    



    
![png](02_california_housing_files/02_california_housing_14_7.png)
    


## Regional Effects


```python
reg_rhale = effector.RegionalRHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, schema={"feature_names": feature_names, "target_name": target_name}, nof_instances="all")
reg_rhale.fit("all", space_partitioner=effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.25))
reg_rhale.summary(features="all", scale_x_list=scale_x_list)
```

      0%|          | 0/8 [00:00<?, ?it/s]

     12%|█▎        | 1/8 [00:01<00:10,  1.55s/it]

     25%|██▌       | 2/8 [00:03<00:09,  1.52s/it]

     38%|███▊      | 3/8 [00:06<00:11,  2.31s/it]

     50%|█████     | 4/8 [00:07<00:07,  1.96s/it]

     62%|██████▎   | 5/8 [00:09<00:05,  1.80s/it]

     75%|███████▌  | 6/8 [00:12<00:04,  2.20s/it]

     88%|████████▊ | 7/8 [00:15<00:02,  2.58s/it]

    100%|██████████| 8/8 [00:18<00:00,  2.78s/it]

    100%|██████████| 8/8 [00:18<00:00,  2.35s/it]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    MedInc 🔹 [id: 0 | heter: 0.06 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.06
    
    
    
    
    Feature 1 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    HouseAge 🔹 [id: 0 | heter: 0.06 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.06
    
    
    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveRooms 🔹 [id: 0 | heter: 0.04 | inst: 14576 | w: 1.00]
        MedInc ≤ 3.73 🔹 [id: 1 | heter: 0.03 | inst: 8289 | w: 0.57]
        MedInc > 3.73 🔹 [id: 2 | heter: 0.03 | inst: 6287 | w: 0.43]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.04
        Level 1🔹heter: 0.03 | 🔻0.01 (28.20%)
    
    
    
    
    Feature 3 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveBedrms 🔹 [id: 0 | heter: 0.02 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.02
    
    
    
    
    Feature 4 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Population 🔹 [id: 0 | heter: 0.03 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 4 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.03
    
    
    
    
    Feature 5 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveOccup 🔹 [id: 0 | heter: 0.05 | inst: 14576 | w: 1.00]
        MedInc ≤ 3.37 🔹 [id: 1 | heter: 0.02 | inst: 6892 | w: 0.47]
        MedInc > 3.37 🔹 [id: 2 | heter: 0.05 | inst: 7684 | w: 0.53]
            HouseAge ≤ 25.60 🔹 [id: 3 | heter: 0.04 | inst: 3243 | w: 0.22]
            HouseAge > 25.60 🔹 [id: 4 | heter: 0.03 | inst: 4441 | w: 0.30]
    --------------------------------------------------
    Feature 5 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.05
        Level 1🔹heter: 0.04 | 🔻0.02 (32.02%)
            Level 2🔹heter: 0.02 | 🔻0.02 (50.55%)
    
    
    
    
    Feature 6 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Latitude 🔹 [id: 0 | heter: 0.60 | inst: 14576 | w: 1.00]
        Longitude ≤ -121.55 🔹 [id: 1 | heter: 0.37 | inst: 3810 | w: 0.26]
        Longitude > -121.55 🔹 [id: 2 | heter: 0.24 | inst: 10766 | w: 0.74]
            AveOccup ≤ 2.61 🔹 [id: 3 | heter: 0.25 | inst: 3485 | w: 0.24]
            AveOccup > 2.61 🔹 [id: 4 | heter: 0.14 | inst: 7281 | w: 0.50]
    --------------------------------------------------
    Feature 6 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.60
        Level 1🔹heter: 0.27 | 🔻0.32 (54.10%)
            Level 2🔹heter: 0.13 | 🔻0.14 (52.36%)
    
    
    
    
    Feature 7 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Longitude 🔹 [id: 0 | heter: 0.49 | inst: 14576 | w: 1.00]
        Latitude ≤ 36.22 🔹 [id: 1 | heter: 0.20 | inst: 8566 | w: 0.59]
        Latitude > 36.22 🔹 [id: 2 | heter: 0.32 | inst: 6010 | w: 0.41]
            Latitude ≤ 38.43 🔹 [id: 3 | heter: 0.25 | inst: 4724 | w: 0.32]
            Latitude > 38.43 🔹 [id: 4 | heter: 0.08 | inst: 1286 | w: 0.09]
    --------------------------------------------------
    Feature 7 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.49
        Level 1🔹heter: 0.25 | 🔻0.24 (49.63%)
            Level 2🔹heter: 0.09 | 🔻0.16 (63.90%)
    
    


    


**AveOccup: average number of people residing in a house**


```python
reg_rhale.plot(feature=5, node_idx=0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_18_0.png)
    


**Global Trend:** House prices decrease as the average number of people residing in a house increases with the highest slop in the lowest average occupancy values


```python
# plot the level-1 subregions (node ids depend on the fitted tree)
for node in reg_rhale.tree["feature_5"].nodes:
    if node.info["level"] == 1:
        reg_rhale.plot(feature=5, node_idx=node.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_20_0.png)
    



    
![png](02_california_housing_files/02_california_housing_20_1.png)
    



```python
for node in reg_rhale.tree["feature_5"].nodes:
    if node.info["level"] == 2:
        reg_rhale.plot(feature=5, node_idx=node.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_21_0.png)
    



    
![png](02_california_housing_files/02_california_housing_21_1.png)
    


**Global Trend:** House prices decrease as the average number of people per household (AveOccup) increases, with the steepest drop at low occupancy levels. This suggests that even small increases in crowding can significantly reduce home values, especially in less crowded areas.

**Regional Trends:**  
- **Low-Income Areas (MedInc ≤ 3.73):** The initial slope (at low AveOccup) becomes smoother, indicating that house prices decrease more gradually with crowding in poorer regions.
- **High-Income Areas (MedInc > 3.73):** The initial slope becomes steeper, and starts from higher house values.
  - **Newer homes (HouseAge ≤ 18.40):** The slope remains smoother, starting from lower prices.
  - **Older homes (HouseAge > 18.40)** The slope becomes even steeper, and starts from higher house values, meaning older homes in high-income areas lose value rapidly as they become crowded.

## Latitude (south to north)


```python
reg_rhale.plot(feature=6, node_idx=0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_24_0.png)
    


**Global Trend:** House prices decrease as we move north.  


```python
for node in reg_rhale.tree["feature_6"].nodes:
    if node.info["level"] == 1:
        reg_rhale.plot(feature=6, node_idx=node.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_26_0.png)
    



    
![png](02_california_housing_files/02_california_housing_26_1.png)
    



```python
for node in reg_rhale.tree["feature_6"].nodes:
    if node.info["level"] == 2:
        reg_rhale.plot(feature=6, node_idx=node.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_27_0.png)
    



    
![png](02_california_housing_files/02_california_housing_27_1.png)
    


**Global Trend:** House prices decrease as we move north.  

**Regional Trends:** Moreorless the same, with minor different curves.

## Longitude (west to east)


```python
reg_rhale.plot(feature=7, node_idx=0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](02_california_housing_files/02_california_housing_30_0.png)
    


**Global Trend:** House prices decrease as we move east.  


```python
for node in reg_rhale.tree["feature_7"].nodes:
    if node.info["level"] == 1:
        reg_rhale.plot(feature=7, node_idx=node.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_32_0.png)
    



    
![png](02_california_housing_files/02_california_housing_32_1.png)
    



```python
for node in reg_rhale.tree["feature_7"].nodes:
    if node.info["level"] == 2:
        reg_rhale.plot(feature=7, node_idx=node.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_33_0.png)
    



    
![png](02_california_housing_files/02_california_housing_33_1.png)
    


**Global Trend:** House prices decrease as we move east.  

**Regional Trends:**  
- **South (latitude <= 35.85):** Prices drop more sharply in the second half from west to east.
  - **AveOccup <= 2.61:** Prices drop even more steeper, suggesting that in less crowded southern areas, housing demand or value drops off more quickly as you move east.
  - **AveOccup > 2.61:** Patterns resemble the broader subregion (latitude <= 35.85), with no significant change in trend.
- **North (latitude > 35.85):** The steepest price decline happens in the western half (closer to the coast).  
  - **Latitude <= 38.43:** The sharp west-to-east price drop remains the same
  - **Latitude > 38.43:** The decline flattens, since the eastern part of far-northern California starts from lower prices
