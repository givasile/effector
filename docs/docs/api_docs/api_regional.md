## Summary

All methods share a similar interface:

=== "PDP"
    
    ```python
    effector.RegionalPDP(data=X, model=predict)
    ```

=== "RHALE"

    ```python
    effector.RegionalRHALE(data=X, model=predict, model_jac=jacobian)
    ```

=== "ShapDP"

     ```python
        effector.RegionalShapDP(data=X, model=predict)
     ```

=== "ALE"

    ```python
    effector.RegionalALE(data=X, model=predict)
    ```

=== "DerPDP"

     ```python
     effector.DerPDP(data=X, model=predict, model_jac=jacobian)
     ```

They all have the four methods:

- `.fit()`
- `.summary()`
- `.plot()`
- `.eval()`

---

`.fit(features, centering, **method_specific_args)`

:   Fits the regional effect method to the data.

    ??? Tip "This is the place for customization"

        The `.fit()` step can be omitted if you are ok with the default settings.
        However, if you want more control over the fitting process, you can pass additional arguments to the `.fit()` method.
        Check some examples below:

    ??? Example "Usage"

        ```python
        features = [0, 1]
        
        # customize the space partitioning process
        space_partitioner = effector.space_partitioning.Regions(
            heter_pcg_drop_thres=0.3 # percentage drop threshold (default: 0.1),
            max_split_levels=1 # maximum number of split levels (default: 2)
        )
        ```

        === "PDP"

            ```python
            regional_method = effector.RegionalPDP(data=X, model=predict)
            regional_method.fit(
                features, 
                space_partitioner=space_partitioner,
                centering=True # center the data (default: False
            )
            ```

        === "RHALE"

            ```python
            regional_method = effector.RegionalRHALE(data=X, model=predict, model_jac=jacobian)
            regional_method.fit(features, space_partitioner=space_partitioner)
            ```

        === "ShapDP"

            ```python
            effector.RegionalShapDP(data=X, model=predict)
            regional_method.fit(features, space_partitioner=space_partitioner)
            ```

        === "ALE"

            ```python
            regional_method = effector.RegionalALE(data=X, model=predict)
            regional_method.fit(features, space_partitioner=space_partitioner)
            ```

        === "DerPDP"

            ```python
            regional_method = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
            regional_method.fit(features, space_partitioner=space_partitioner)
            ```

`.summary(feature)`

:   Prints a summary of the partition tree that is found for `feature`. 
  
    ??? Example "Usage"

        === "PDP"

            ```python
            effector.RegionalPDP(data=X, model=predict).summary(0)
            ```

        === "RHALE"

            ```python
            effector.RegionalRHALE(data=X, model=predict, model_jac=jacobian).summary(0)
            ```

        === "ShapDP"

             ```python
                effector.RegionalShapDP(data=X, model=predict).summary(0)
             ```

        === "ALE"

            ```python
            effector.RegionalALE(data=X, model=predict).summary(0)
            ```

        === "DerPDP"

             ```python
             effector.DerPDP(data=X, model=predict, model_jac=jacobian).summary(0)
             ```

    ??? Example "Output"

        === "PDP"
            
            ```python
            effector.RegionalPDP(data=X, model=predict).summary(0)
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
            effector.RegionalRHALE(data=X, model=predict, model_jac=jacobian).summary(0)
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
             effector.RegionalShapDP(data=X, model=predict).summary(0)
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
            effector.RegionalALE(data=X, model=predict).summary(0)
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
        
        === "DerPDP"
        
             ```python
             effector.DerPDP(data=X, model=predict, model_jac=jacobian).summary(0)
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
            ```

`.plot(feature, node_idx)`

:   Plots the regional effect of the feature `feature` at the node `node_idx`.

    ??? Example "Usage"

        === "PDP"

             ```python
             regional_effect = effector.RegionalPDP(data=X, model=predict)
             [regional_effect.plot(0, node_idx) for node_idx in [1, 2]]
             ```

        === "RHALE"

             ```python
             regional_effect = effector.RegionalRHALE(data=X, model=predict, model_jac=jacobian)
             [regional_effect.plot(0, node_idx) for node_idx in [1, 2]]
             ```

        === "ShapDP"

             ```python
             regional_effect = effector.RegionalShapDP(data=X, model=predict)
             [regional_effect.plot(0, node_idx) for node_idx in [1, 2]]
             ```

        === "ALE"

             ```python
             regional_effect = effector.RegionalALE(data=X, model=predict)
             [regional_effect.plot(0, node_idx) for node_idx in [1, 2]]
             ```

        === "DerPDP"

             ```python
             regional_effect = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
             [regional_effect.plot(0, node_idx) for node_idx in [1, 2]]
             ```

    ??? Example "Output"

        === "PDP"
        
             | `node_idx=1`: $x_1$ when $x_2 \leq 0$ | `node_idx=2`: $x_1$ when $x_2 > 0$ |
             |:---------:|:---------:|
             | ![Alt text](./static/homepage_example_20_3.png) | ![Alt text](./static/homepage_example_20_5.png) |
        
        === "RHALE"
        
             | `node_idx=1`: $x_1$ when $x_2 \leq 0$ | `node_idx=2`: $x_1$ when $x_2 > 0$ |
             |:---------:|:---------:|
             | ![Alt text](./static/homepage_example_26_3.png) | ![Alt text](./static/homepage_example_26_5.png) |

        === "ShapDP"

             | `node_idx=1`: $x_1$ when $x_2 \leq 0$ | `node_idx=2`: $x_1$ when $x_2 > 0$ |
             |:---------:|:---------:|
             | ![Alt text](./static/homepage_example_33_1.png) | ![Alt text](./static/homepage_example_33_2.png) |
        
        === "ALE"
        
             | `node_idx=1`: $x_1$ when $x_2 \leq 0$ | `node_idx=2`: $x_1$ when $x_2 > 0$ |
             |:---------:|:---------:|
             | ![Alt text](./static/homepage_example_29_3.png) | ![Alt text](./static/homepage_example_29_5.png) |
        
        
        === "derPDP"
        
             | `node_idx=1`: $x_1$ when $x_2 \leq 0$ | `node_idx=2`: $x_1$ when $x_2 > 0$ |
             |:---------:|:---------:|
             | ![Alt text](./static/homepage_example_23_3.png) | ![Alt text](./static/homepage_example_23_5.png) |


`.eval(feature, node_idx, xs)`

: Evaluate the regional effect at a specific grid of points.

    ??? Example "Usage"

        ```python
        # Example input
        feature = 0
        node_idx = 1
        xs = np.linspace(-1, 1, 100)
        ```

        === "PDP"

             ```python
             regional_effect = effector.RegionalPDP(data=X, model=predict)
             y, het = regional_effect.eval(feature, node_idx, xs)
             ```

        === "RHALE"

             ```python
             regional_effect = effector.RegionalRHALE(data=X, model=predict, model_jac=jacobian)
             y, het = regional_effect.eval(feature, node_idx, xs)
             ```

        === "ShapDP"

             ```python
             regional_effect = effector.RegionalShapDP(data=X, model=predict)
             y, het = regional_effect.eval(feature, node_idx, xs)
             ```

        === "ALE"

             ```python
             regional_effect = effector.RegionalALE(data=X, model=predict)
             y, het = regional_effect.eval(feature, node_idx, xs)
             ```

        === "DerPDP"

             ```python
             regional_effect = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
             y, het = regional_effect.eval(feature, node_idx, xs)
             ```

## API

### ::: effector.regional_effect.RegionalEffectBase
       options:
         show_root_heading: False
         show_symbol_type_toc: True
         inherited_members: True
         members:
           - eval
           - summary

### ::: effector.regional_effect_ale.RegionalALE
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         inherited_members: True
         members:
           - __init__
           - fit
           - plot

### ::: effector.regional_effect_ale.RegionalRHALE
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         inherited_members: True
         members:
           - __init__
           - fit
           - plot


### ::: effector.regional_effect_pdp.RegionalPDP
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         inherited_members: True
         members:
           - __init__
           - fit
           - plot

### ::: effector.regional_effect_pdp.RegionalDerPDP
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         inherited_members: True
         members:
           - __init__
           - fit
           - plot

### ::: effector.regional_effect_shap.RegionalShapDP
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         inherited_members: True
         members:
           - __init__
           - fit
           - plot
