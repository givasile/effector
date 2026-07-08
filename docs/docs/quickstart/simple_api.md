# An overview of `effector`'s API

- Author: [givasile](https://givasile.github.io/)
- Description: The simple entry point to `effector`: what inputs it needs
  (data, model, optionally the jacobian) and how to get a global or regional
  effect plot in a single line with `.plot()` and `.find_regions()`.

`effector` requires:

- a dataset, normally the test set
- a Machine Learning model
- (optionally) the jacobian of the black-box model

Then pick a global (1) or regional (2) Effect Method, to explain the ML model.
For the thinking behind the API — one engine, values, two entrances — see
[the mental model](../mental_model.md).
{ .annotate }

1.  :man_raising_hand: `effector` provides five global effect methods:
     - [`PDP`](./../../api_docs/api_global/#effector.global_effect_pdp.PDP)
     - [`RHALE`](./../../api_docs/api_global/#effector.global_effect_rhale.RHALE) 
     - [`ShapDP`](./../../api_docs/api_global/#effector.global_effect_shapdp.ShapDP)
     - [`ALE`](./../../api_docs/api_global/#effector.global_effect_ale.ALE)
     - [`DerPDP`](./../../api_docs/api_global/#effector.global_effect_derpdp.DerPDP)

2. :man_raising_hand: every global effect method also computes regional
   effects — call `.find_regions(feature)` on any of the five objects (`PDP`,
   `RHALE`, `ShapDP`, `ALE`, `DerPDP`) to get a
   [`Partition`](./../../api_docs/api_partition/#effector.partition.Partition)
   of subregions.

---
### Dataset

???+ note "A dataset, typically the test set"
     A `np.ndarray` with shape `(N, D)` — effector is numpy-only (R10).
     Started from a pandas DataFrame? Convert it once with
     `X, schema = effector.from_dataframe(df)`: it reads column names, dtypes
     (`float` → continuous, low-cardinality `int` → ordinal, `category`/strings
     → nominal, ordered `category` → ordinal), and category labels into a
     `Schema` you can inspect and tweak. It converts the data only — it never
     touches your model.

???+ note "Metadata travels in one `schema` argument"
     Everything else (names, types, target name, axis un-scaling) goes into a
     single optional `schema=` argument — an `effector.Schema` or a plain dict:

     ```python
     schema = {
         "feature_names": ["hour", "weekday", "temp"],
         "feature_types": ["ordinal", "nominal", "continuous"],
         "target_name": "bike-rentals",
     }
     effector.PDP(X_test, model, schema=schema)
     ```

     Every field is optional; explicit fields always win over inference.

=== "synthetic example"
     
     ```python
     N = 100
     D = 2
     X_test = np.random.uniform(-1, 1, (N, D))
     ```

=== "a real case"

    ```python
    from ucimlrepo import fetch_ucirepo 

    # fetch dataset 
    bike_sharing_dataset = fetch_ucirepo(id=275) 
  
    # data (as pandas dataframes) 
    X = bike_sharing_dataset.data.features 
    y = bike_sharing_dataset.data.targets 

    # split data (still pandas DataFrames)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # effector is numpy-only: convert the DataFrame to (numpy, schema)
    X_test_np, schema = effector.from_dataframe(X_test)
    ```
---
### ML model

???+ note "A trained black-box model"

     Must be a numpy→numpy `Callable`, signature
     `X: np.ndarray[N, D] -> np.ndarray[N]`. A model trained on a DataFrame, a
     PyTorch/TensorFlow tensor, or an sklearn `Pipeline` is yours to wrap into
     that shape — e.g. PyTorch:
     `model = lambda X: net(torch.as_tensor(X, dtype=torch.float32)).detach().numpy().ravel()`.
     See the [*effector is purely numpy based*](./pure_numpy.md) guide.


=== "synthetic example"

     ```python
     def predict(x):
        '''y = 10*x[0] if x[1] > 0 else -10*x[0] + noise'''
        y = np.zeros(x.shape[0])          
        ind = x[:, 1] > 0
        y[ind] = 10*x[ind, 0]
        y[~ind] = -10*x[~ind, 0]
        return y + np.random.normal(0, 1, x.shape[0])
     ```

=== "scikit-learn"

     The adapter wraps `model.predict` (and validates the output shape):

     ```python
     # model = sklearn.ensemble.RandomForestRegressor().fit(X, y)
     predict = effector.adapters.from_sklearn(model)

     # classifiers: explain a per-class probability instead
     predict = effector.adapters.classifier_proba(clf, class_=1)
     ```

=== "tensorflow"

     If you have a tensorflow model, use `model.predict`.

     ```python
     # X = ... (the training data)
     # y = ... (the training labels)
     # model = ... (a keras model, e.g., keras.Sequential)

     def predict(x):
        return model.predict(x)
     ```

=== "pytorch"

     The adapter handles eval mode, `no_grad`, device and dtype:

     ```python
     # model = ... (a pytorch model, e.g., torch.nn.Sequential)
     predict = effector.adapters.from_torch(model)
     ```

Whatever produced the callable, you can probe it on two rows of your data
before building an engine — `effector.adapters.check(predict, X)` — and get a
precise error message if the shapes are off.

---
### Jacobian (optional)

???+ note "Optional: The jacobian of the model's output w.r.t. the input"
    
    Must be a `Callable` with signature `X: np.ndarray[N, D]) -> np.ndarray[N, D]`.     
    It is not required, but for some methods (`RHALE` and `DerPDP`), it accelerates the computation.

=== "synthetic example"

     ```python
     def jacobian(x):
       '''dy/dx = 10 if x[1] > 0 else -10'''
       y = np.zeros_like(x)
       ind = x[:, 1] > 0
       y[ind, 0] = 10
       y[~ind, 0] = -10
       return y
     ```

=== "scikit-learn"
    
    Not available.

=== "tensorflow"

     ```python
     # X = ... (the training data)
     # y = ... (the training labels)
     # model = ... (a keras model, e.g., keras.Sequential)

     def jacobian(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x)
        return tape.jacobian(y, x)
     ```

=== "pytorch"

    ```python
     # model = ... (a pytorch model, e.g., torch.nn.Sequential)
     predict, jacobian = effector.adapters.from_torch(model, jacobian=True)
    ```

## Global Effect

???+ success "Global effect: how each feature affects the model's output **globally**, averaged over all instances."

    All `effector` global effect methods have the same API.
    They all share three main functions:

        - `.plot()`: visualizes the global effect
        - `.eval()`: evaluates the global effect at a grid of points
        - `.fit()`: allows for customizing the global method


### `.plot()`

`.plot()` is the most common method, as it visualizes the global effect.
Every verb takes the feature as an **index or a name** (names come from your
schema) — `pdp.plot(0)` and `pdp.plot("hour")` are the same call.
For example, to plot the effect of the first feature of the synthetic dataset, use:

=== "PDP"
    
    ```python
    pdp = effector.PDP(data=X, model=predict)
    pdp.plot(0)
    ```
    ![Global-PDP](./../static/quickstart/simple_api_files/simple_api_9_0.png){ align=center }

=== "RHALE"

    ```python
    rhale = effector.RHALE(data=X, model=predict, model_jac=jacobian)
    rhale.plot(0)
    ```

    ![Global-RHALE](./../static/quickstart/simple_api_files/simple_api_11_0.png){ align=center }

=== "ShapDP"

    ```python
    shap_dp = effector.ShapDP(data=X, model=predict)
    shap_dp.plot(0)
    ```
    ![Global-ShapDP](./../static/quickstart/simple_api_files/simple_api_13_0.png){ align=center }

=== "ALE"

    ```python
    ale = effector.ALE(data=X, model=predict)
    ale.plot(0)
    ```
    ![Global-ALE](./../static/quickstart/simple_api_files/simple_api_15_0.png){ align=center }

=== "derPDP"

    ```python
    d_pdp = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
    d_pdp.plot(0)
    ```

    ![Global-DerPDP](./../static/quickstart/simple_api_files/simple_api_17_0.png){ align=center }

???+ "Some important arguments of `.plot()`"
    
     - `heterogeneity`: whether to plot the heterogeneity of the global effect. The following options are available:
         - If `heterogeneity` is `True`, the heterogeneity is plot as a standard deviation around the global effect.
         - If `heterogeneity` is `False`, the global effect is plotted.
         - If `heterogeneity` is a string:
             - `"ice"` for `pdp` plots
             - `"shap_values"` for `shap_dp` plots

     - `centering`: whether to center the regional effect. The following options are available:
         - If `centering` is `False`, the regional effect is not centered
         - If `centering` is `True` or `zero_integral`, the regional effect is centered around the y axis.
         - If `centering` is `zero_start`, the regional effect starts from `y=0`

### `.eval()` 

`.eval()` evaluates the global effect at a grid of points.

=== "PDP"

    ```python
    pdp = effector.PDP(data=X, model=predict)
    y = pdp.eval(0, xs=np.linspace(-1, 1, 100))
    y_var = pdp.eval_heter(0, xs=np.linspace(-1, 1, 100))
    ```

=== "RHALE"

    ```python
    rhale = effector.RHALE(data=X, model=predict, model_jac=jacobian)
    y = rhale.eval(0, xs=np.linspace(-1, 1, 100))
    y_var = rhale.eval_heter(0, xs=np.linspace(-1, 1, 100))
    ```

=== "ShapDP"

    ```python
    shap_dp = effector.ShapDP(data=X, model=predict)
    y = shap_dp.eval(0, xs=np.linspace(-1, 1, 100))
    y_var = shap_dp.eval_heter(0, xs=np.linspace(-1, 1, 100))
    ```

=== "ALE"

    ```python
    ale = effector.ALE(data=X, model=predict)
    y = ale.eval(0, xs=np.linspace(-1, 1, 100))
    y_var = ale.eval_heter(0, xs=np.linspace(-1, 1, 100))
    ```

=== "derPDP"

    ```python
    d_pdp = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
    y = d_pdp.eval(0, xs=np.linspace(-1, 1, 100))
    y_var = d_pdp.eval_heter(0, xs=np.linspace(-1, 1, 100))
    ```

### `.fit()`

If you want to customize the global effect, use `.fit()` before `.plot()` or `.eval()`.
Check this [tutorial](./../flexible_api) for more details.

```python
global_effect = effector.<method_name>(data=X, model=predict)

# customize the global effect
global_effect.fit(features=[...], **kwargs)

global_effect.plot(0)
global_effect.eval(0, xs=np.linspace(-1, 1, 100))
```

## Regional Effect

???+ success "Regional Effect: How each feature affects the model's output **regionally**, averaged over instances **inside a subregion.**"
     
    Sometimes, global effects are very heterogeneous (local effects deviate from the global effect).
    Call `.find_regions(feature)` on any global effect object; it returns a
    `Partition`, whose functions are:

        - `.show()`: prints the partition tree
        - `.plot()`: visualizes a subregion's effect
        - `.eval()`: evaluates a subregion's effect at a grid of points
        - `.eval_heter()`: evaluates a subregion's heterogeneity


### `.find_regions()` and `.show()`

`find_regions()` is the first step in understanding the regional effect.  
Behind the scenes, it searches for a partitioning of the feature space into meaningful subregions (1)
and returns a `Partition` describing what, if anything, has been found.
Users should check `partition.show()` before plotting or evaluating a subregion.
To plot or evaluate a specific regional effect, they can use the region index `node_idx` from the partition tree.
{ .annotate }

1. meaningful in this context means that the regional effects in the respective subregions have lower heterogeneity than the global effect.

=== "PDP"
    
    ```python
    pdp = effector.PDP(data=X, model=predict)
    partition = pdp.find_regions(0)
    partition.show()
    ```

    ```python
     Feature 0 - Full partition tree:
     Node id: 0, name: x_0, heter: 34.79 || nof_instances:  1000 || weight: 1.00
             Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.09 || nof_instances:  1000 || weight: 1.00
             Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.09 || nof_instances:  1000 || weight: 1.00
     --------------------------------------------------
     Feature 0 - Statistics per tree level:
     Level 0, heter: 34.79
        Level 1, heter: 0.18 || heter drop : 34.61 (units), 99.48% (pcg)
    ```

=== "RHALE"

    ```python
    rhale = effector.RHALE(data=X, model=predict, model_jac=jacobian)
    partition = rhale.find_regions(0)
    partition.show()
    ```

    ```python
     Feature 0 - Full partition tree:
     Node id: 0, name: x_0, heter: 93.45 || nof_instances:  1000 || weight: 1.00
             Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
             Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
     --------------------------------------------------
     Feature 0 - Statistics per tree level:
     Level 0, heter: 93.45
             Level 1, heter: 0.00 || heter drop : 93.45 (units), 100.00% (pcg)
    ```

=== "ShapDP"

     ```python
     shap_dp = effector.ShapDP(data=X, model=predict, nof_instances=500)
     partition = shap_dp.find_regions(0)
     partition.show()
     ```

     ```python
     Feature 0 - Full partition tree:
     Node id: 0, name: x_0, heter: 8.33 || nof_instances:  1000 || weight: 1.00
             Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
             Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
     --------------------------------------------------
     Feature 0 - Statistics per tree level:
     Level 0, heter: 8.33
             Level 1, heter: 0.00 || heter drop : 8.33 (units), 99.94% (pcg)
     ```

=== "ALE"

    ```python
    ale = effector.ALE(data=X, model=predict)
    partition = ale.find_regions(0)
    partition.show()
    ```

    ```python
     Feature 0 - Full partition tree:
     Node id: 0, name: x_0, heter: 114.57 || nof_instances:  1000 || weight: 1.00
             Node id: 1, name: x_0 | x_1 <= 0.0, heter: 16.48 || nof_instances:  1000 || weight: 1.00
             Node id: 2, name: x_0 | x_1  > 0.0, heter: 17.41 || nof_instances:  1000 || weight: 1.00
     --------------------------------------------------
     Feature 0 - Statistics per tree level:
     Level 0, heter: 114.57
             Level 1, heter: 33.89 || heter drop : 80.68 (units), 70.42% (pcg)
    ```

<!-- === "DerPDP"

     ```python
     d_pdp = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
     partition = d_pdp.find_regions(0)
     partition.show()
     ```

    ```python
     Feature 0 - Full partition tree:
     Node id: 0, name: x_0, heter: 100.00 || nof_instances:  1000 || weight: 1.00
             Node id: 1, name: x_0 | x_1 <= 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
             Node id: 2, name: x_0 | x_1  > 0.0, heter: 0.00 || nof_instances:  1000 || weight: 1.00
     --------------------------------------------------
     Feature 0 - Statistics per tree level:
     Level 0, heter: 100.00
             Level 1, heter: 0.00 || heter drop : 100.00 (units), 100.00% (pcg)
    ``` -->

### `.plot()`

`partition.plot()` visualizes a subregion's effect. It takes the region index `node_idx` from the partition tree.
Apart from taking `node_idx` instead of the feature index, the API is the same as the global effect.

=== "PDP"

     ```python
     partition = effector.PDP(data=X, model=predict).find_regions(0)
     [partition.plot(node_idx) for node_idx in [1, 2]]
     ```

     | `node_idx=1`: $x_0$ when $x_1 \leq 0$ | `node_idx=2`: $x_0$ when $x_1 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/simple_api_files/simple_api_21_0.png) | ![Alt text](./../static/quickstart/simple_api_files/simple_api_21_1.png) |

=== "RHALE"

     ```python
     partition = effector.RHALE(data=X, model=predict, model_jac=jacobian).find_regions(0)
     [partition.plot(node_idx) for node_idx in [1, 2]]
     ```

     | `node_idx=1`: $x_0$ when $x_1 \leq 0$ | `node_idx=2`: $x_0$ when $x_1 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/simple_api_files/simple_api_24_0.png) | ![Alt text](./../static/quickstart/simple_api_files/simple_api_24_1.png) |

=== "ShapDP"

     ```python
     partition = effector.ShapDP(data=X, model=predict, nof_instances=500).find_regions(0)
     [partition.plot(node_idx) for node_idx in [1, 2]]
     ```

     | `node_idx=1`: $x_0$ when $x_1 \leq 0$ | `node_idx=2`: $x_0$ when $x_1 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/simple_api_files/simple_api_27_0.png) | ![Alt text](./../static/quickstart/simple_api_files/simple_api_27_1.png) |

=== "ALE"

     ```python
     partition = effector.ALE(data=X, model=predict).find_regions(0)
     [partition.plot(node_idx) for node_idx in [1, 2]]
     ```

     | `node_idx=1`: $x_0$ when $x_1 \leq 0$ | `node_idx=2`: $x_0$ when $x_1 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/simple_api_files/simple_api_30_0.png) | ![Alt text](./../static/quickstart/simple_api_files/simple_api_30_1.png) |


=== "derPDP"

     ```python
     partition = effector.DerPDP(data=X, model=predict, model_jac=jacobian).find_regions(0)
     [partition.plot(node_idx) for node_idx in [1, 2]]
     ```

     | `node_idx=1`: $x_0$ when $x_1 \leq 0$ | `node_idx=2`: $x_0$ when $x_1 > 0$ |
     |:---------:|:---------:|
     | ![Alt text](./../static/quickstart/simple_api_files/simple_api_33_0.png) | ![Alt text](./../static/quickstart/simple_api_files/simple_api_33_1.png) |

---

### `.eval()`

`partition.eval()` evaluates a subregion's effect at a grid of points, and
`partition.eval_heter()` its heterogeneity. Both take the region index `node_idx`.

=== "PDP"

    ```python
    partition = effector.PDP(data=X, model=predict).find_regions(0)
    y = partition.eval(1, xs=np.linspace(-1, 1, 100))         # region with node_idx=1
    y_heter = partition.eval_heter(1, xs=np.linspace(-1, 1, 100))
    ```

=== "RHALE"

    ```python
    partition = effector.RHALE(data=X, model=predict, model_jac=jacobian).find_regions(0)
    y = partition.eval(1, xs=np.linspace(-1, 1, 100))         # region with node_idx=1
    y_heter = partition.eval_heter(1, xs=np.linspace(-1, 1, 100))
    ```

=== "ShapDP"

    ```python
    partition = effector.ShapDP(data=X, model=predict, nof_instances=500).find_regions(0)
    y = partition.eval(1, xs=np.linspace(-1, 1, 100))         # region with node_idx=1
    y_heter = partition.eval_heter(1, xs=np.linspace(-1, 1, 100))
    ```

=== "ALE"

    ```python
    partition = effector.ALE(data=X, model=predict).find_regions(0)
    y = partition.eval(1, xs=np.linspace(-1, 1, 100))         # region with node_idx=1
    y_heter = partition.eval_heter(1, xs=np.linspace(-1, 1, 100))
    ```

=== "derPDP"

    ```python
    partition = effector.DerPDP(data=X, model=predict, model_jac=jacobian).find_regions(0)
    y = partition.eval(1, xs=np.linspace(-1, 1, 100))         # region with node_idx=1
    y_heter = partition.eval_heter(1, xs=np.linspace(-1, 1, 100))
    ```

### Customizing the search

If you want to customize the regional effect, pass a `finder` to `.find_regions()`.
Check this [tutorial](./../flexible_api) for more details. 
The `finder` controls the method that partitions the feature space. 

```python
effect = effector.<method_name>(data=X, model=predict)

# customize the region search
finder = effector.space_partitioning.Greedy(max_depth=2)
partition = effect.find_regions(feature=0, finder=finder)

partition.show()
partition.plot(1)                                   # region with node_idx=1
partition.eval(1, xs=np.linspace(-1, 1, 100))
```

