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
    I0000 00:00:1783614403.307923  224338 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
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


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m12s[0m 858ms/step - loss: 0.9423 - mae: 0.7796 - root_mean_squared_error: 0.9707

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.7971 - mae: 0.6966 - root_mean_squared_error: 0.8913  

    [1m 7/15[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 18ms/step - loss: 0.7241 - mae: 0.6553 - root_mean_squared_error: 0.8485

    [1m11/15[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.6583 - mae: 0.6177 - root_mean_squared_error: 0.8079

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.6148 - mae: 0.5915 - root_mean_squared_error: 0.7801

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - loss: 0.4849 - mae: 0.5136 - root_mean_squared_error: 0.6963


    Epoch 2/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.3256 - mae: 0.4100 - root_mean_squared_error: 0.5706

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.3279 - mae: 0.4141 - root_mean_squared_error: 0.5726

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3241 - mae: 0.4119 - root_mean_squared_error: 0.5692

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.3236 - mae: 0.4110 - root_mean_squared_error: 0.5688

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.3244 - mae: 0.4093 - root_mean_squared_error: 0.5696


    Epoch 3/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.3081 - mae: 0.4015 - root_mean_squared_error: 0.5551

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3077 - mae: 0.4025 - root_mean_squared_error: 0.5547

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3042 - mae: 0.3985 - root_mean_squared_error: 0.5515

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 16ms/step - loss: 0.3027 - mae: 0.3964 - root_mean_squared_error: 0.5502

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.3016 - mae: 0.3921 - root_mean_squared_error: 0.5492


    Epoch 4/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.3166 - mae: 0.3939 - root_mean_squared_error: 0.5627

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3050 - mae: 0.3898 - root_mean_squared_error: 0.5523

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.3000 - mae: 0.3871 - root_mean_squared_error: 0.5477

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2960 - mae: 0.3851 - root_mean_squared_error: 0.5440

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2889 - mae: 0.3820 - root_mean_squared_error: 0.5375


    Epoch 5/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2812 - mae: 0.3818 - root_mean_squared_error: 0.5303

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2844 - mae: 0.3785 - root_mean_squared_error: 0.5333

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2866 - mae: 0.3797 - root_mean_squared_error: 0.5353

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2858 - mae: 0.3791 - root_mean_squared_error: 0.5346

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2842 - mae: 0.3773 - root_mean_squared_error: 0.5331


    Epoch 6/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2716 - mae: 0.3704 - root_mean_squared_error: 0.5212

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2761 - mae: 0.3766 - root_mean_squared_error: 0.5254

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2743 - mae: 0.3751 - root_mean_squared_error: 0.5237

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2748 - mae: 0.3749 - root_mean_squared_error: 0.5242

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2753 - mae: 0.3712 - root_mean_squared_error: 0.5247


    Epoch 7/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2918 - mae: 0.3929 - root_mean_squared_error: 0.5401

    [1m 4/15[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 17ms/step - loss: 0.2844 - mae: 0.3873 - root_mean_squared_error: 0.5333

    [1m 8/15[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2817 - mae: 0.3815 - root_mean_squared_error: 0.5308

    [1m12/15[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 17ms/step - loss: 0.2796 - mae: 0.3781 - root_mean_squared_error: 0.5287

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2737 - mae: 0.3718 - root_mean_squared_error: 0.5232


    Epoch 8/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2738 - mae: 0.3667 - root_mean_squared_error: 0.5232

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2685 - mae: 0.3637 - root_mean_squared_error: 0.5181

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2688 - mae: 0.3652 - root_mean_squared_error: 0.5184

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2675 - mae: 0.3643 - root_mean_squared_error: 0.5171

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2624 - mae: 0.3609 - root_mean_squared_error: 0.5122


    Epoch 9/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2564 - mae: 0.3678 - root_mean_squared_error: 0.5064

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2619 - mae: 0.3617 - root_mean_squared_error: 0.5117

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2624 - mae: 0.3612 - root_mean_squared_error: 0.5122

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2615 - mae: 0.3603 - root_mean_squared_error: 0.5114

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2572 - mae: 0.3567 - root_mean_squared_error: 0.5072


    Epoch 10/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2807 - mae: 0.3632 - root_mean_squared_error: 0.5298

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2642 - mae: 0.3577 - root_mean_squared_error: 0.5140

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2597 - mae: 0.3560 - root_mean_squared_error: 0.5095

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2569 - mae: 0.3549 - root_mean_squared_error: 0.5068

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2532 - mae: 0.3535 - root_mean_squared_error: 0.5031


    Epoch 11/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2445 - mae: 0.3493 - root_mean_squared_error: 0.4944

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2527 - mae: 0.3506 - root_mean_squared_error: 0.5027

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2525 - mae: 0.3514 - root_mean_squared_error: 0.5025

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2520 - mae: 0.3508 - root_mean_squared_error: 0.5020

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2500 - mae: 0.3488 - root_mean_squared_error: 0.5000


    Epoch 12/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2509 - mae: 0.3593 - root_mean_squared_error: 0.5009

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2380 - mae: 0.3486 - root_mean_squared_error: 0.4879

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2382 - mae: 0.3463 - root_mean_squared_error: 0.4880

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2386 - mae: 0.3454 - root_mean_squared_error: 0.4884

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2409 - mae: 0.3441 - root_mean_squared_error: 0.4908


    Epoch 13/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.1988 - mae: 0.3148 - root_mean_squared_error: 0.4458

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2282 - mae: 0.3312 - root_mean_squared_error: 0.4774

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2299 - mae: 0.3317 - root_mean_squared_error: 0.4793

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2302 - mae: 0.3318 - root_mean_squared_error: 0.4797

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2341 - mae: 0.3350 - root_mean_squared_error: 0.4838


    Epoch 14/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2151 - mae: 0.3344 - root_mean_squared_error: 0.4638

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2198 - mae: 0.3302 - root_mean_squared_error: 0.4688

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2196 - mae: 0.3289 - root_mean_squared_error: 0.4686

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2217 - mae: 0.3294 - root_mean_squared_error: 0.4709

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2316 - mae: 0.3343 - root_mean_squared_error: 0.4813


    Epoch 15/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 30ms/step - loss: 0.2081 - mae: 0.3169 - root_mean_squared_error: 0.4562

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2105 - mae: 0.3205 - root_mean_squared_error: 0.4587

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 15ms/step - loss: 0.2119 - mae: 0.3203 - root_mean_squared_error: 0.4603

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2146 - mae: 0.3217 - root_mean_squared_error: 0.4632

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2254 - mae: 0.3282 - root_mean_squared_error: 0.4748


    Epoch 16/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2300 - mae: 0.3527 - root_mean_squared_error: 0.4796

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2280 - mae: 0.3446 - root_mean_squared_error: 0.4775

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2262 - mae: 0.3410 - root_mean_squared_error: 0.4756

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2253 - mae: 0.3384 - root_mean_squared_error: 0.4746

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2257 - mae: 0.3326 - root_mean_squared_error: 0.4751


    Epoch 17/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.2465 - mae: 0.3327 - root_mean_squared_error: 0.4965

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2300 - mae: 0.3318 - root_mean_squared_error: 0.4795

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2279 - mae: 0.3315 - root_mean_squared_error: 0.4773

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2269 - mae: 0.3310 - root_mean_squared_error: 0.4763

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2255 - mae: 0.3306 - root_mean_squared_error: 0.4749


    Epoch 18/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 28ms/step - loss: 0.2447 - mae: 0.3339 - root_mean_squared_error: 0.4947

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2277 - mae: 0.3268 - root_mean_squared_error: 0.4771

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2240 - mae: 0.3255 - root_mean_squared_error: 0.4732

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2224 - mae: 0.3245 - root_mean_squared_error: 0.4716

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2182 - mae: 0.3231 - root_mean_squared_error: 0.4671


    Epoch 19/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 31ms/step - loss: 0.2144 - mae: 0.3183 - root_mean_squared_error: 0.4631

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2221 - mae: 0.3254 - root_mean_squared_error: 0.4712

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2211 - mae: 0.3253 - root_mean_squared_error: 0.4702

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2199 - mae: 0.3247 - root_mean_squared_error: 0.4689

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.2161 - mae: 0.3228 - root_mean_squared_error: 0.4649


    Epoch 20/20


    [1m 1/15[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 29ms/step - loss: 0.1883 - mae: 0.2981 - root_mean_squared_error: 0.4339

    [1m 5/15[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2123 - mae: 0.3157 - root_mean_squared_error: 0.4606

    [1m 9/15[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 16ms/step - loss: 0.2124 - mae: 0.3162 - root_mean_squared_error: 0.4607

    [1m13/15[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 16ms/step - loss: 0.2116 - mae: 0.3165 - root_mean_squared_error: 0.4600

    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - loss: 0.2105 - mae: 0.3181 - root_mean_squared_error: 0.4588


    [1m  1/456[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m45s[0m 100ms/step - loss: 0.1386 - mae: 0.2494 - root_mean_squared_error: 0.3723

    [1m 25/456[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.1965 - mae: 0.3057 - root_mean_squared_error: 0.4428   

    [1m 53/456[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2066 - mae: 0.3112 - root_mean_squared_error: 0.4542

    [1m 82/456[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2090 - mae: 0.3127 - root_mean_squared_error: 0.4569

    [1m111/456[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2104 - mae: 0.3140 - root_mean_squared_error: 0.4584

    [1m140/456[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2101 - mae: 0.3143 - root_mean_squared_error: 0.4582

    [1m168/456[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2101 - mae: 0.3145 - root_mean_squared_error: 0.4582

    [1m198/456[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2100 - mae: 0.3146 - root_mean_squared_error: 0.4581

    [1m227/456[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2102 - mae: 0.3148 - root_mean_squared_error: 0.4584

    [1m256/456[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2102 - mae: 0.3149 - root_mean_squared_error: 0.4584

    [1m285/456[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2099 - mae: 0.3149 - root_mean_squared_error: 0.4581

    [1m314/456[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2096 - mae: 0.3148 - root_mean_squared_error: 0.4577

    [1m343/456[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 2ms/step - loss: 0.2093 - mae: 0.3147 - root_mean_squared_error: 0.4574

    [1m370/456[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 2ms/step - loss: 0.2090 - mae: 0.3146 - root_mean_squared_error: 0.4570

    [1m399/456[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 2ms/step - loss: 0.2086 - mae: 0.3145 - root_mean_squared_error: 0.4566

    [1m427/456[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 2ms/step - loss: 0.2083 - mae: 0.3144 - root_mean_squared_error: 0.4563

    [1m456/456[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2081 - mae: 0.3143 - root_mean_squared_error: 0.4561

    [1m456/456[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2067 - mae: 0.3144 - root_mean_squared_error: 0.4546


    [1m  1/114[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 14ms/step - loss: 0.1802 - mae: 0.3365 - root_mean_squared_error: 0.4245

    [1m 30/114[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.3429 - mae: 0.3816 - root_mean_squared_error: 0.5845 

    [1m 59/114[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 2ms/step - loss: 0.3318 - mae: 0.3751 - root_mean_squared_error: 0.5753

    [1m 89/114[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 2ms/step - loss: 0.3195 - mae: 0.3701 - root_mean_squared_error: 0.5646

    [1m114/114[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2799 - mae: 0.3523 - root_mean_squared_error: 0.5291





    [0.2799123227596283, 0.35226133465766907, 0.5290673971176147]




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
    


### Importance & one-click report

`importances()` ranks features by the dispersion of their mean effect (the
mu-twin of heterogeneity), and `effector.explain(...)` runs the whole
pipeline (fit -> rank -> regions) into a single serializable `Report`.


```python
# per-feature importance = dispersion of the mean effect (mu-twin of heterogeneity)
print("importances:", np.round(rhale.importances(), 3))

# one-click auto-explanation -> Report (serializable; self-contained HTML)
report = effector.explain(
    X_train.to_numpy(), model_forward, model_jac=model_jac,
    method="rhale",
    schema={"feature_names": feature_names, "target_name": target_name},
    nof_instances=2000,
)
report.show()
```

    importances: [0.409 0.019 0.102 0.024 0.044 0.338 0.966 0.813]


    
    RHALE report — target: MedHouseVal
    ============================================================
    feature                   importance     heter  #regions
    ------------------------------------------------------------
    Latitude                      0.9480    1.0971         7
    Longitude                     0.8178    0.9258         7
    MedInc                        0.4113    0.2828         1
    AveOccup                      0.3284    0.4070         7
    AveRooms                      0.1170    0.2153         1
    ============================================================
    
    
    Feature 6 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Latitude 🔹 [id: 0 | heter: 1.10 | inst: 2000 | w: 1.00]
        Longitude < -1.04 🔹 [id: 1 | heter: 1.30 | inst: 521 | w: 0.26]
            HouseAge < 0.07 🔹 [id: 2 | heter: 0.85 | inst: 245 | w: 0.12]
            HouseAge ≥ 0.07 🔹 [id: 3 | heter: 1.45 | inst: 276 | w: 0.14]
        Longitude ≥ -1.04 🔹 [id: 4 | heter: 0.77 | inst: 1479 | w: 0.74]
            AveOccup < -0.38 🔹 [id: 5 | heter: 0.79 | inst: 475 | w: 0.24]
            AveOccup ≥ -0.38 🔹 [id: 6 | heter: 0.53 | inst: 1004 | w: 0.50]
    --------------------------------------------------
    Feature 6 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 1.10
        Level 1🔹heter: 0.91 | 🔻0.19 (16.91%)
            Level 2🔹heter: 0.76 | 🔻0.16 (17.09%)
    
    
    
    
    Feature 7 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    Longitude 🔹 [id: 0 | heter: 0.93 | inst: 2000 | w: 1.00]
        AveOccup < -0.38 🔹 [id: 1 | heter: 1.02 | inst: 708 | w: 0.35]
            Latitude < 1.24 🔹 [id: 2 | heter: 0.95 | inst: 594 | w: 0.30]
            Latitude ≥ 1.24 🔹 [id: 3 | heter: 0.52 | inst: 114 | w: 0.06]
        AveOccup ≥ -0.38 🔹 [id: 4 | heter: 0.72 | inst: 1292 | w: 0.65]
            MedInc < 0.06 🔹 [id: 5 | heter: 0.55 | inst: 680 | w: 0.34]
            MedInc ≥ 0.06 🔹 [id: 6 | heter: 0.74 | inst: 612 | w: 0.31]
    --------------------------------------------------
    Feature 7 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.93
        Level 1🔹heter: 0.83 | 🔻0.10 (10.45%)
            Level 2🔹heter: 0.72 | 🔻0.11 (12.68%)
    
    
    
    
    Feature 5 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    AveOccup 🔹 [id: 0 | heter: 0.41 | inst: 2000 | w: 1.00]
        MedInc < -0.44 🔹 [id: 1 | heter: 0.25 | inst: 717 | w: 0.36]
            HouseAge < 0.66 🔹 [id: 2 | heter: 0.22 | inst: 481 | w: 0.24]
            HouseAge ≥ 0.66 🔹 [id: 3 | heter: 0.25 | inst: 236 | w: 0.12]
        MedInc ≥ -0.44 🔹 [id: 4 | heter: 0.44 | inst: 1283 | w: 0.64]
            HouseAge < -0.52 🔹 [id: 5 | heter: 0.27 | inst: 462 | w: 0.23]
            HouseAge ≥ -0.52 🔹 [id: 6 | heter: 0.39 | inst: 821 | w: 0.41]
    --------------------------------------------------
    Feature 5 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.41
        Level 1🔹heter: 0.37 | 🔻0.04 (8.61%)
            Level 2🔹heter: 0.31 | 🔻0.07 (17.93%)
    
    


## Regional Effects


```python
reg_rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, schema={"feature_names": feature_names, "target_name": target_name}, nof_instances="all")
reg_rhale.fit("all", centering=True)
finder = effector.space_partitioning.Best(min_heterogeneity_decrease_pcg=0.25)
partitions = {feat: reg_rhale.find_regions(feat, finder=finder) for feat in range(len(feature_names))}
for feat in range(len(feature_names)):
    partitions[feat].show(scale_x_list=scale_x_list)
```

    
    
    Feature 0 - Full partition tree:
    No splits found for feature 0
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    No splits found for feature 0
    
    
    
    
    Feature 1 - Full partition tree:
    No splits found for feature 1
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    No splits found for feature 1
    
    
    
    
    Feature 2 - Full partition tree:
    No splits found for feature 2
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    No splits found for feature 2
    
    
    
    
    Feature 3 - Full partition tree:
    No splits found for feature 3
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    No splits found for feature 3
    
    
    
    
    Feature 4 - Full partition tree:
    No splits found for feature 4
    --------------------------------------------------
    Feature 4 - Statistics per tree level:
    No splits found for feature 4
    
    
    
    
    Feature 5 - Full partition tree:
    No splits found for feature 5
    --------------------------------------------------
    Feature 5 - Statistics per tree level:
    No splits found for feature 5
    
    
    
    
    Feature 6 - Full partition tree:
    No splits found for feature 6
    --------------------------------------------------
    Feature 6 - Statistics per tree level:
    No splits found for feature 6
    
    
    
    
    Feature 7 - Full partition tree:
    No splits found for feature 7
    --------------------------------------------------
    Feature 7 - Statistics per tree level:
    No splits found for feature 7
    
    


**AveOccup: average number of people residing in a house**


```python
partitions[5].plot(0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_20_0.png)
    


**Global Trend:** House prices decrease as the average number of people residing in a house increases with the highest slop in the lowest average occupancy values


```python
# plot the level-1 subregions (region ids depend on the fitted tree)
for r in partitions[5]:
    if r.level == 1:
        partitions[5].plot(r.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


```python
for r in partitions[5]:
    if r.level == 2:
        partitions[5].plot(r.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

**Global Trend:** House prices decrease as the average number of people per household (AveOccup) increases, with the steepest drop at low occupancy levels. This suggests that even small increases in crowding can significantly reduce home values, especially in less crowded areas.

**Regional Trends:**  
- **Low-Income Areas (MedInc ≤ 3.73):** The initial slope (at low AveOccup) becomes smoother, indicating that house prices decrease more gradually with crowding in poorer regions.
- **High-Income Areas (MedInc > 3.73):** The initial slope becomes steeper, and starts from higher house values.
  - **Newer homes (HouseAge ≤ 18.40):** The slope remains smoother, starting from lower prices.
  - **Older homes (HouseAge > 18.40)** The slope becomes even steeper, and starts from higher house values, meaning older homes in high-income areas lose value rapidly as they become crowded.

## Latitude (south to north)


```python
partitions[6].plot(0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


    
![png](02_california_housing_files/02_california_housing_26_0.png)
    


**Global Trend:** House prices decrease as we move north.  


```python
for r in partitions[6]:
    if r.level == 1:
        partitions[6].plot(r.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


```python
for r in partitions[6]:
    if r.level == 2:
        partitions[6].plot(r.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

**Global Trend:** House prices decrease as we move north.  

**Regional Trends:** Moreorless the same, with minor different curves.

## Longitude (west to east)


```python
partitions[7].plot(0, centering=True, scale_x_list=scale_x_list, scale_y=scale_y)
```


    
![png](02_california_housing_files/02_california_housing_32_0.png)
    


**Global Trend:** House prices decrease as we move east.  


```python
for r in partitions[7]:
    if r.level == 1:
        partitions[7].plot(r.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```


```python
for r in partitions[7]:
    if r.level == 2:
        partitions[7].plot(r.idx, centering=True, scale_x_list=scale_x_list, scale_y=scale_y, y_limits=y_limits)
```

**Global Trend:** House prices decrease as we move east.  

**Regional Trends:**  
- **South (latitude <= 35.85):** Prices drop more sharply in the second half from west to east.
  - **AveOccup <= 2.61:** Prices drop even more steeper, suggesting that in less crowded southern areas, housing demand or value drops off more quickly as you move east.
  - **AveOccup > 2.61:** Patterns resemble the broader subregion (latitude <= 35.85), with no significant change in trend.
- **North (latitude > 35.85):** The steepest price decline happens in the western half (closer to the coast).  
  - **Latitude <= 38.43:** The sharp west-to-east price drop remains the same
  - **Latitude > 38.43:** The decline flattens, since the eastern part of far-northern California starts from lower prices
