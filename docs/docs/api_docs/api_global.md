#
## Summary

All global effect methods have a similar interface and workflow:

1. create an instance of the global effect method you want to use
2. (optional) `.fit()` to customize the method
4. `.plot()` to plot the global effect of a feature
5. `.eval()` to evaluate the global effect of a feature at a specific grid of points

---

## Usage

```python
# set up the input
X = ... # input data
predict = ... # model to be explained
jacobian = ... # jacobian of the model
```

1. **Create an instance of the global effect method you want to use**:

    === "PDP"
        
        ```python
        g_method = effector.PDP(data=X, model=predict)
        ```
    
    === "RHALE"
    
        ```python
        g_method = effector.RHALE(data=X, model=predict, model_jac=jacobian)
        ```
    
    === "ShapDP"
    
         ```python
         g_method = effector.ShapDP(data=X, model=predict)
         ```
    
    === "ALE"
    
        ```python
        g_method = effector.ALE(data=X, model=predict)
        ```
    
    === "DerPDP"
    
         ```python
         g_method = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
         ```

2. **Customize the global effect method (optional)**:

    `.fit(features, **method_specific_args)`
    
    ??? Tip "This is the place for customization"

        The `.fit()` step can be omitted if you are ok with the default settings; you can directly call the `.plot()`, or `.eval()` methods.
        However, if you want more control over the fitting process, you can pass additional arguments to the `.fit()` method.
        Check the Usage section below and the method-specific documentation for more information.

    ??? Example "Usage"


        ```python
        # customize the space partitioning algorithm
        axis_partitioner = effector.axis_partitioning.Greedy(
            init_nof_bins: int = 50, # start from 50 bins (default: 20)
            min_points_per_bin = 10, # minimum number of points per bin (default: 2)
            cat_limit = 20 # maximum number of categories for a feature to be considered categorical (default: 10)

        )
        g_method.fit(
            features=[0, 1], # list of features to be analyzed
            axis_partitioner=axis_partitioner, # custom axis partitioner
        )
        ```

3. **Plot the global effect of a feature**:

    `.plot(feature)`
    
    ??? Example "Usage"

        ```python
        feature = ...
        g_method.plot(feature, **plot_specific_args)
        ```

    ??? Example "Output"
   
        === "PDP"

             ![Alt text](./../static/quickstart/simple_api_files/simple_api_8_0.png)

        === "RHALE"

             ![Alt text](./../static/quickstart/simple_api_files/simple_api_10_0.png)        

        === "ShapDP"

             ![Alt text](./../static/quickstart/simple_api_files/simple_api_12_0.png)

        === "ALE"

              ![Alt text](./../static/quickstart/simple_api_files/simple_api_14_0.png)        
        
        === "derPDP"
        
              ![Alt text](./../static/quickstart/simple_api_files/simple_api_16_0.png)

4. **Evaluate the global effect of a feature at a specific grid of points**:
   
    `.eval(feature, xs)`

    ??? Example "Usage"

        ```python
        # Example input
        feature = ... # feature to be analyzed
        xs = ... # grid of points to evaluate the global effect, e.g., np.linspace(0, 1, 100)
        ```

         ```python
         y, het = r_method.eval(feature, xs)
         ```

## API

### ::: effector.global_effect_ale.ALE
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - __init__
          - fit
          - eval
          - plot

### ::: effector.global_effect_ale.RHALE
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - __init__
          - fit
          - eval
          - plot

### ::: effector.global_effect_pdp.PDP
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - __init__
          - fit
          - eval
          - plot

### ::: effector.global_effect_pdp.DerPDP
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - __init__
          - fit
          - eval
          - plot

### ::: effector.global_effect_shap.ShapDP
      options:
        show_root_heading: True
        show_symbol_type_toc: True
        inherited_members: True
        members:
          - __init__
          - fit
          - eval
          - plot
