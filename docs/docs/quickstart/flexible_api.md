# Customize `.fit()`

## Dataset
     
```python
dist = effector.datasets.IndependentUniform(dim=3, low=-1, high=1)
X_test = dist.generate_data(n=200)
axis_limits = dist.axis_limits
```

## Black-box model and Jacobian

```python
model = effector.models.DoubleConditionalInteraction()
predict = model.predict
jacobian = model.jacobian
```


## Global Effect

### RHALE

=== "Simple API"

    ```python
    rhale = effector.RHALE(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
    rhale.plot(feature=0, y_limits=y_limits, dy_limits=dy_limits)
    ```
    ![Global-RHALE](./../static/quickstart/flexible_api_files/flexible_api_7_0.png){ align=center }

=== "Flexible API"

    ```python
    rhale = effector.RHALE(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
    rhale.fit(features=0, binning_method=effector.axis_partitioning.Fixed(nof_bins=5))
    rhale.plot(feature=0, y_limits=y_limits, dy_limits=dy_limits)
    ```
    
    ![Global-RHALE](./../static/quickstart/flexible_api_files/flexible_api_8_0.png){ align=center }

## Regional Effect

### RHALE

`init()`

=== "Simple API"

    ```python
    r_rhale = effector.RegionalRHALE(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
    r_rhale.summary(0)
    ```

=== "Flexible API"

    ```python
    rhale = effector.RHALE(X_test, predict, jacobian, axis_limits=axis_limits, nof_instances="all")
    rhale.fit(features=0, binning_method=effector.axis_partitioning.Fixed(nof_bins=5))
    rhale.plot(feature=0, y_limits=y_limits, dy_limits=dy_limits)
    ```

`.summary()` output

=== "Simple API"
    ```python
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 60.47 || nof_instances:   200 || weight: 1.00
            Node id: 1, name: x_0 | x_2 <= 0.0, heter: 2.36 || nof_instances:   105 || weight: 0.53
                    Node id: 3, name: x_0 | x_2 <= 0.0 and x_1 <= 0.0, heter: 0.06 || nof_instances:    45 || weight: 0.23
                    Node id: 4, name: x_0 | x_2 <= 0.0 and x_1  > 0.0, heter: 0.00 || nof_instances:    60 || weight: 0.30
            Node id: 2, name: x_0 | x_2  > 0.0, heter: 70.28 || nof_instances:    95 || weight: 0.47
                    Node id: 5, name: x_0 | x_2  > 0.0 and x_1 <= 0.0, heter: 0.00 || nof_instances:    45 || weight: 0.23
                    Node id: 6, name: x_0 | x_2  > 0.0 and x_1  > 0.0, heter: 8.08 || nof_instances:    50 || weight: 0.25
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 60.47
            Level 1, heter: 34.62 || heter drop : 25.85 (units), 42.75% (pcg)
                    Level 2, heter: 2.03 || heter drop : 32.59 (units), 94.12% (pcg)
    ```

=== "Flexible API "
    ```python
    Feature 0 - Full partition tree:
    Node id: 0, name: x_0, heter: 53.64 || nof_instances:   200 || weight: 1.00
            Node id: 1, name: x_0 | x_2 <= 0.0, heter: 2.42 || nof_instances:   105 || weight: 0.53
            Node id: 2, name: x_0 | x_2  > 0.0, heter: 61.95 || nof_instances:    95 || weight: 0.47
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    Level 0, heter: 53.64
            Level 1, heter: 30.70 || heter drop : 22.94 (units), 42.77% (pcg)
    ```

`.plot()` all

=== "Simple API"

     | `node_idx=3`: $x_0$ when $x_1 \leq 0$ and $x_2 \leq 0$ | `node_idx=4`: $x_0$ when $x_1 > 0$ and $x_2 \leq 0$|
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/flexible_api_files/flexible_api_16_0.png) | ![Alt text](./../static/quickstart/flexible_api_files/flexible_api_16_1.png) |
     | `node_idx=5`: $x_0$ when $x_1 \leq 0$ and $x_2 > 0$ | `node_idx=6`: $x_0$ when $x_1 > 0$ and $x_2 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/flexible_api_files/flexible_api_16_2.png) | ![Alt text](./../static/quickstart/flexible_api_files/flexible_api_16_3.png) |

=== "Flexible API"

    ```python
    rhale.plot(feature=0, node_idx=1, y_limits=y_limits, dy_limits=dy_limits)
    rhale.plot(feature=0, node_idx=2, y_limits=y_limits, dy_limits=dy_limits)
    ```

     | `node_idx=1`: $x_0$ when $x_1 \leq 0$ | `node_idx=2`: $x_0$ when $x_1 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/flexible_api_files/flexible_api_20_0.png) | ![Alt text](./../static/quickstart/flexible_api_files/flexible_api_20_1.png) |