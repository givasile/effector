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
    I0000 00:00:1783341746.953924   31768 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
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


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m12s[0m 909ms/step - loss: 1.0927 - mae: 0.8379 - root_mean_squared_error: 1.0453

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.8830 - mae: 0.7317 - root_mean_squared_error: 0.9369  

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.7788 - mae: 0.6768 - root_mean_squared_error: 0.8781

    [1m10/15[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.7116 - mae: 0.6395 - root_mean_squared_error: 0.8381

    [1m14/15[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 17ms/step - loss: 0.6520 - mae: 0.6059 - root_mean_squared_error: 0.8013

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - loss: 0.4832 - mae: 0.5094 - root_mean_squared_error: 0.6951


    Epoch 2/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.3669 - mae: 0.4368 - root_mean_squared_error: 0.6057

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3526 - mae: 0.4288 - root_mean_squared_error: 0.5937

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3456 - mae: 0.4239 - root_mean_squared_error: 0.5878

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.3401 - mae: 0.4203 - root_mean_squared_error: 0.5831

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.3246 - mae: 0.4105 - root_mean_squared_error: 0.5697


    Epoch 3/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2973 - mae: 0.3906 - root_mean_squared_error: 0.5452

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.3031 - mae: 0.3908 - root_mean_squared_error: 0.5505

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.3009 - mae: 0.3898 - root_mean_squared_error: 0.5485

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 17ms/step - loss: 0.2988 - mae: 0.3889 - root_mean_squared_error: 0.5466

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2971 - mae: 0.3873 - root_mean_squared_error: 0.5451


    Epoch 4/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.3439 - mae: 0.4086 - root_mean_squared_error: 0.5864

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3220 - mae: 0.4061 - root_mean_squared_error: 0.5674

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3141 - mae: 0.4023 - root_mean_squared_error: 0.5603

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.3088 - mae: 0.3993 - root_mean_squared_error: 0.5555

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2948 - mae: 0.3907 - root_mean_squared_error: 0.5429


    Epoch 5/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2684 - mae: 0.3602 - root_mean_squared_error: 0.5180

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2763 - mae: 0.3723 - root_mean_squared_error: 0.5256

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2751 - mae: 0.3739 - root_mean_squared_error: 0.5245

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2757 - mae: 0.3739 - root_mean_squared_error: 0.5251

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2802 - mae: 0.3754 - root_mean_squared_error: 0.5293


    Epoch 6/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2623 - mae: 0.3751 - root_mean_squared_error: 0.5122

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2690 - mae: 0.3701 - root_mean_squared_error: 0.5187

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2691 - mae: 0.3687 - root_mean_squared_error: 0.5188

    [1m10/15[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2705 - mae: 0.3691 - root_mean_squared_error: 0.5201

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 17ms/step - loss: 0.2716 - mae: 0.3691 - root_mean_squared_error: 0.5211

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2731 - mae: 0.3686 - root_mean_squared_error: 0.5226


    Epoch 7/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 48ms/step - loss: 0.3016 - mae: 0.3750 - root_mean_squared_error: 0.5491

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2893 - mae: 0.3753 - root_mean_squared_error: 0.5378

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2848 - mae: 0.3734 - root_mean_squared_error: 0.5336

    [1m10/15[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 19ms/step - loss: 0.2807 - mae: 0.3713 - root_mean_squared_error: 0.5297

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 18ms/step - loss: 0.2785 - mae: 0.3700 - root_mean_squared_error: 0.5276

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2693 - mae: 0.3643 - root_mean_squared_error: 0.5190


    Epoch 8/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2965 - mae: 0.3861 - root_mean_squared_error: 0.5445

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2863 - mae: 0.3777 - root_mean_squared_error: 0.5350

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2815 - mae: 0.3736 - root_mean_squared_error: 0.5305

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2776 - mae: 0.3709 - root_mean_squared_error: 0.5268

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2750 - mae: 0.3696 - root_mean_squared_error: 0.5243

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2677 - mae: 0.3658 - root_mean_squared_error: 0.5174


    Epoch 9/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 34ms/step - loss: 0.2806 - mae: 0.3613 - root_mean_squared_error: 0.5297

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2662 - mae: 0.3608 - root_mean_squared_error: 0.5159

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2662 - mae: 0.3608 - root_mean_squared_error: 0.5159

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2655 - mae: 0.3606 - root_mean_squared_error: 0.5153

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2610 - mae: 0.3576 - root_mean_squared_error: 0.5108


    Epoch 10/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2514 - mae: 0.3437 - root_mean_squared_error: 0.5014

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.2513 - mae: 0.3465 - root_mean_squared_error: 0.5013

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2522 - mae: 0.3482 - root_mean_squared_error: 0.5022

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2525 - mae: 0.3487 - root_mean_squared_error: 0.5025

    [1m14/15[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 17ms/step - loss: 0.2525 - mae: 0.3491 - root_mean_squared_error: 0.5025

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.2488 - mae: 0.3476 - root_mean_squared_error: 0.4988


    Epoch 11/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 32ms/step - loss: 0.2133 - mae: 0.3199 - root_mean_squared_error: 0.4618

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2250 - mae: 0.3320 - root_mean_squared_error: 0.4742

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2301 - mae: 0.3356 - root_mean_squared_error: 0.4796

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2332 - mae: 0.3380 - root_mean_squared_error: 0.4828

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2421 - mae: 0.3439 - root_mean_squared_error: 0.4921


    Epoch 12/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2520 - mae: 0.3451 - root_mean_squared_error: 0.5020

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2444 - mae: 0.3464 - root_mean_squared_error: 0.4943

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2442 - mae: 0.3451 - root_mean_squared_error: 0.4942

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2435 - mae: 0.3443 - root_mean_squared_error: 0.4935

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2409 - mae: 0.3410 - root_mean_squared_error: 0.4908


    Epoch 13/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2042 - mae: 0.3265 - root_mean_squared_error: 0.4519

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2198 - mae: 0.3318 - root_mean_squared_error: 0.4687

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2243 - mae: 0.3339 - root_mean_squared_error: 0.4735

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2271 - mae: 0.3353 - root_mean_squared_error: 0.4764

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2366 - mae: 0.3392 - root_mean_squared_error: 0.4865


    Epoch 14/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.1908 - mae: 0.3163 - root_mean_squared_error: 0.4369

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2072 - mae: 0.3215 - root_mean_squared_error: 0.4551

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2112 - mae: 0.3231 - root_mean_squared_error: 0.4595

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2148 - mae: 0.3246 - root_mean_squared_error: 0.4633

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2296 - mae: 0.3318 - root_mean_squared_error: 0.4791


    Epoch 15/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 35ms/step - loss: 0.2177 - mae: 0.3334 - root_mean_squared_error: 0.4665

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2296 - mae: 0.3361 - root_mean_squared_error: 0.4792

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2324 - mae: 0.3363 - root_mean_squared_error: 0.4820

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 17ms/step - loss: 0.2329 - mae: 0.3361 - root_mean_squared_error: 0.4826

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2285 - mae: 0.3320 - root_mean_squared_error: 0.4780


    Epoch 16/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2220 - mae: 0.3314 - root_mean_squared_error: 0.4712

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2213 - mae: 0.3278 - root_mean_squared_error: 0.4704

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2230 - mae: 0.3290 - root_mean_squared_error: 0.4723

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2256 - mae: 0.3302 - root_mean_squared_error: 0.4750

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2306 - mae: 0.3338 - root_mean_squared_error: 0.4802


    Epoch 17/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2128 - mae: 0.3212 - root_mean_squared_error: 0.4614

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2243 - mae: 0.3316 - root_mean_squared_error: 0.4735

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2214 - mae: 0.3283 - root_mean_squared_error: 0.4705

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 17ms/step - loss: 0.2215 - mae: 0.3282 - root_mean_squared_error: 0.4706

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2244 - mae: 0.3295 - root_mean_squared_error: 0.4737


    Epoch 18/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2165 - mae: 0.3242 - root_mean_squared_error: 0.4653

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2116 - mae: 0.3213 - root_mean_squared_error: 0.4600

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2139 - mae: 0.3227 - root_mean_squared_error: 0.4624

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.2144 - mae: 0.3226 - root_mean_squared_error: 0.4630

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2172 - mae: 0.3237 - root_mean_squared_error: 0.4661


    Epoch 19/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2460 - mae: 0.3270 - root_mean_squared_error: 0.4960

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2266 - mae: 0.3286 - root_mean_squared_error: 0.4759

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2234 - mae: 0.3280 - root_mean_squared_error: 0.4726

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2227 - mae: 0.3275 - root_mean_squared_error: 0.4719

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2189 - mae: 0.3252 - root_mean_squared_error: 0.4678


    Epoch 20/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2125 - mae: 0.3158 - root_mean_squared_error: 0.4610

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2138 - mae: 0.3165 - root_mean_squared_error: 0.4623

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2139 - mae: 0.3180 - root_mean_squared_error: 0.4625

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2141 - mae: 0.3188 - root_mean_squared_error: 0.4627

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2144 - mae: 0.3194 - root_mean_squared_error: 0.4631

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2153 - mae: 0.3212 - root_mean_squared_error: 0.4640


    [1m  1/456[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m46s[0m 103ms/step - loss: 0.1437 - mae: 0.2698 - root_mean_squared_error: 0.3791

    [1m 28/456[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.1988 - mae: 0.3135 - root_mean_squared_error: 0.4454   

    [1m 56/456[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2045 - mae: 0.3164 - root_mean_squared_error: 0.4519

    [1m 84/456[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2055 - mae: 0.3172 - root_mean_squared_error: 0.4531

    [1m113/456[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2068 - mae: 0.3185 - root_mean_squared_error: 0.4546

    [1m142/456[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2067 - mae: 0.3188 - root_mean_squared_error: 0.4546

    [1m169/456[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2069 - mae: 0.3190 - root_mean_squared_error: 0.4548

    [1m194/456[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2069 - mae: 0.3191 - root_mean_squared_error: 0.4548

    [1m224/456[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2073 - mae: 0.3195 - root_mean_squared_error: 0.4553

    [1m253/456[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2075 - mae: 0.3197 - root_mean_squared_error: 0.4555

    [1m281/456[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2074 - mae: 0.3197 - root_mean_squared_error: 0.4553

    [1m311/456[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2072 - mae: 0.3198 - root_mean_squared_error: 0.4551

    [1m340/456[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2070 - mae: 0.3198 - root_mean_squared_error: 0.4550

    [1m370/456[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 2ms/step - loss: 0.2069 - mae: 0.3197 - root_mean_squared_error: 0.4548

    [1m399/456[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 2ms/step - loss: 0.2066 - mae: 0.3196 - root_mean_squared_error: 0.4545

    [1m428/456[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 2ms/step - loss: 0.2064 - mae: 0.3195 - root_mean_squared_error: 0.4543

    [1m456/456[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2064 - mae: 0.3195 - root_mean_squared_error: 0.4543


    [1m  1/114[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 14ms/step - loss: 0.1508 - mae: 0.3102 - root_mean_squared_error: 0.3884

    [1m 28/114[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.3304 - mae: 0.3786 - root_mean_squared_error: 0.5732 

    [1m 57/114[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.3235 - mae: 0.3744 - root_mean_squared_error: 0.5678

    [1m 85/114[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.3137 - mae: 0.3711 - root_mean_squared_error: 0.5593

    [1m114/114[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.3051 - mae: 0.3676 - root_mean_squared_error: 0.5516

    [1m114/114[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2803 - mae: 0.3578 - root_mean_squared_error: 0.5294





    [0.28030481934547424, 0.3578280210494995, 0.5294381976127625]




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

     12%|█▎        | 1/8 [00:01<00:11,  1.58s/it]

     25%|██▌       | 2/8 [00:03<00:09,  1.59s/it]

     38%|███▊      | 3/8 [00:06<00:12,  2.42s/it]

     50%|█████     | 4/8 [00:07<00:08,  2.00s/it]

     62%|██████▎   | 5/8 [00:09<00:05,  1.82s/it]

     75%|███████▌  | 6/8 [00:12<00:04,  2.18s/it]

     88%|████████▊ | 7/8 [00:15<00:02,  2.51s/it]

    100%|██████████| 8/8 [00:18<00:00,  2.70s/it]

    100%|██████████| 8/8 [00:18<00:00,  2.33s/it]

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    MedInc 🔹 [id: 0 | heter: 0.05 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.05
    
    
    
    
    Feature 1 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    HouseAge 🔹 [id: 0 | heter: 0.05 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.05
    
    
    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveRooms 🔹 [id: 0 | heter: 0.03 | inst: 14576 | w: 1.00]
        MedInc ≤ 3.37 🔹 [id: 1 | heter: 0.03 | inst: 6892 | w: 0.47]
        MedInc > 3.37 🔹 [id: 2 | heter: 0.02 | inst: 7684 | w: 0.53]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.03
        Level 1🔹heter: 0.02 | 🔻0.01 (26.26%)
    
    
    
    
    Feature 3 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveBedrms 🔹 [id: 0 | heter: 0.01 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.01
    
    
    
    
    Feature 4 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Population 🔹 [id: 0 | heter: 0.02 | inst: 14576 | w: 1.00]
    --------------------------------------------------
    Feature 4 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.02
    
    
    
    
    Feature 5 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveOccup 🔹 [id: 0 | heter: 0.05 | inst: 14576 | w: 1.00]
        HouseAge ≤ 28.00 🔹 [id: 1 | heter: 0.02 | inst: 6394 | w: 0.44]
        HouseAge > 28.00 🔹 [id: 2 | heter: 0.04 | inst: 8182 | w: 0.56]
            MedInc ≤ 3.01 🔹 [id: 3 | heter: 0.02 | inst: 3302 | w: 0.23]
            MedInc > 3.01 🔹 [id: 4 | heter: 0.03 | inst: 4880 | w: 0.33]
    --------------------------------------------------
    Feature 5 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.05
        Level 1🔹heter: 0.03 | 🔻0.02 (38.61%)
            Level 2🔹heter: 0.01 | 🔻0.01 (51.29%)
    
    
    
    
    Feature 6 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Latitude 🔹 [id: 0 | heter: 0.55 | inst: 14576 | w: 1.00]
        Longitude ≤ -121.55 🔹 [id: 1 | heter: 0.46 | inst: 3810 | w: 0.26]
        Longitude > -121.55 🔹 [id: 2 | heter: 0.23 | inst: 10766 | w: 0.74]
    --------------------------------------------------
    Feature 6 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.55
        Level 1🔹heter: 0.29 | 🔻0.26 (47.19%)
    
    
    
    
    Feature 7 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Longitude 🔹 [id: 0 | heter: 0.44 | inst: 14576 | w: 1.00]
        Latitude ≤ 36.22 🔹 [id: 1 | heter: 0.20 | inst: 8566 | w: 0.59]
        Latitude > 36.22 🔹 [id: 2 | heter: 0.28 | inst: 6010 | w: 0.41]
            Latitude ≤ 38.43 🔹 [id: 3 | heter: 0.23 | inst: 4724 | w: 0.32]
            Latitude > 38.43 🔹 [id: 4 | heter: 0.10 | inst: 1286 | w: 0.09]
    --------------------------------------------------
    Feature 7 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.44
        Level 1🔹heter: 0.24 | 🔻0.20 (45.67%)
            Level 2🔹heter: 0.08 | 🔻0.15 (64.64%)
    
    


    


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
