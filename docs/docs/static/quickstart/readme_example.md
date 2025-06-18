```python
import effector
import keras
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
bike_sharing = effector.datasets.BikeSharing(pcg_train=0.8)
X_train, Y_train = bike_sharing.x_train, bike_sharing.y_train
X_test, Y_test = bike_sharing.x_test, bike_sharing.y_test
```

    2025-06-18 15:34:51.828372: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
# Define and train a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=20, verbose=0)
loss, mae, rmse = model.evaluate(X_test, Y_test, verbose=1)
```

    [1m109/109[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0704 - mae: 0.1704 - root_mean_squared_error: 0.2628



```python
def predict(x):
    return model(x).numpy().squeeze()
```


```python
# Initialize the Partial Dependence Plot (PDP)
pdp = effector.PDP(
    X_test,  # Use the test set as background data
    predict,  # Prediction function
    feature_names=bike_sharing.feature_names,  # (optional) Feature names
    target_name=bike_sharing.target_name  # (optional) Target variable name
)

# Plot the effect of a feature
pdp.plot(
    feature=3,  # Select the 3rd feature (temperature) for the PDP plot
    nof_ice=200,  # (optional) Number of Individual Conditional Expectation (ICE) curves
    scale_x={"mean": bike_sharing.x_test_mu[3], "std": bike_sharing.x_test_std[3]},  # (optional) scale x-axis
    scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},  # (optional) scale y-axis
    centering=True,  # (optional) Center PDP and ICE curves
    show_avg_output=True,  # (optional) Display the average prediction
    y_limits=[-200, 1000]  # (optional) Set y-axis limits
)
```


    
![png](readme_example_files/readme_example_3_0.png)
    



```python
# Initialize the Regional Partial Dependence Plot (RegionalPDP)
r_pdp = effector.RegionalPDP(
    X_test,  # Test set data
    predict,  # Prediction function
    feature_names=bike_sharing.feature_names,  # Feature names
    target_name=bike_sharing.target_name  # Target variable name
)

# Summarize the subregions of the 3rd feature (temperature)
r_pdp.summary(
    features=3,  # Select the 3rd feature for the summary
    scale_x_list=[  # scale each feature with mean and std
        {"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]}
        for i in range(X_test.shape[1])
    ]
)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.99s/it]

    
    
    Feature 3 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    hr 🔹 [id: 0 | heter: 0.24 | inst: 3476 | w: 1.00]
        workingday = 0.00 🔹 [id: 1 | heter: 0.12 | inst: 1129 | w: 0.32]
            temp ≤ 6.50 🔹 [id: 2 | heter: 0.06 | inst: 568 | w: 0.16]
            temp > 6.50 🔹 [id: 3 | heter: 0.08 | inst: 561 | w: 0.16]
        workingday ≠ 0.00 🔹 [id: 4 | heter: 0.12 | inst: 2347 | w: 0.68]
            temp ≤ 8.76 🔹 [id: 5 | heter: 0.08 | inst: 1083 | w: 0.31]
            temp > 8.76 🔹 [id: 6 | heter: 0.09 | inst: 1264 | w: 0.36]
    --------------------------------------------------
    Feature 3 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.24
        Level 1🔹heter: 0.12 | 🔻0.12 (48.89%)
            Level 2🔹heter: 0.08 | 🔻0.04 (31.11%)
    
    


    



```python
# Plot regional effects after the first-level split (workingday vs non-workingday)
for node_idx in [1, 4]:  # Iterate over the nodes of the first-level split
    r_pdp.plot(
        feature=3,  # Feature 3 (temperature)
        node_idx=node_idx,  # Node index (1: workingday, 2: non-workingday)
        nof_ice=200,  # Number of ICE curves
        scale_x_list=[  # Scale features by mean and std
            {"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]}
            for i in range(X_test.shape[1])
        ],
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},  # Scale the target
        y_limits=[-200, 1000]  # Set y-axis limits
    )
```


    
![png](readme_example_files/readme_example_5_0.png)
    



    
![png](readme_example_files/readme_example_5_1.png)
    



```python
# Plot regional effects after second-level splits (workingday vs non-workingday and hot vs cold temperature)
for node_idx in [2, 3, 5, 6]: # [3, 4, 5, 6]:  # Iterate over the nodes of the second-level splits
    r_pdp.plot(
        feature=3,  # Feature 3 (temperature)
        node_idx=node_idx,  # Node index (hot/cold temperature and workingday/non-workingday)
        nof_ice=200,  # Number of ICE curves
        scale_x_list=[  # Scale features by mean and std
            {"mean": bike_sharing.x_test_mu[i], "std": bike_sharing.x_test_std[i]}
            for i in range(X_test.shape[1])
        ],
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},  # Scale target
        y_limits=[-200, 1000]  # Set y-axis limits
    )
```


    
![png](readme_example_files/readme_example_6_0.png)
    



    
![png](readme_example_files/readme_example_6_1.png)
    



    
![png](readme_example_files/readme_example_6_2.png)
    



    
![png](readme_example_files/readme_example_6_3.png)
    



```python

```
