# Measuring Runtime of Regional Effect Plots  

This notebook analyzes the runtime $T(\cdot)$ of Regional Effect plots, which depends on:  

- **$t_f$**: Time to evaluate the black-box function $f$.  
- **$N$**: Number of instances in $X$.  
- **$D$**: Number of features in $X$.  
- **$K$**: Number of points for centering the feature effect plot.  
- **$M$**: Number of evaluation points.  

The main factors affecting runtime are $t_f$, $N$, and $D$.  

### Runtime Breakdown  

1. **Global heterogeneity computation** ($T_{global}$):  
   - Done once for the entire dataset.  
   - Stores intermediate values for reuse.  
   - Runtime:  
     - $T_{global} = \mathcal{O}(N) + \mathcal{O}(t_f)$ for PDP and d-PDP.  
     - $T_{global} = \mathcal{O}(t_f)$ for RHALE.  

2. **Cart-based subregion heterogeneity** ($T_{cart}$):  
   - Iterates over $D-1$ features.  
   - Evaluates $P$ possible conditioning positions.  
   - Recursively splits the dataset up to depth $L$.  
   - Heterogeneity is computed without re-evaluating $f$, only splitting and indexing instances.  

   $$ T_{cart} = (D-1)PL \cdot T(N) $$  

### Total Runtime  

$$
T(t_f, N, D) \approx T_{global} + T_{cart} \approx \mathcal{O}(N) + \mathcal{O}(t_f) + \mathcal{O}(DPLN)
$$  

Runtime is **linear in all key variables**. When computing for all features, it scales as $D^2$.  

Now, let's test this in practice!


```python
import effector
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt
np.random.seed(21)
```

    /Users/dimitriskyriakopoulos/Documents/ath/Effector/Code/eff-env/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
def return_predict(t):
    def predict(x):
        time.sleep(t)
        model = effector.models.DoubleConditionalInteraction()
        return model.predict(x)
    return predict

def return_jacobian(t):
    def jacobian(x):
        time.sleep(t)
        model = effector.models.DoubleConditionalInteraction()
        return model.jacobian(x)
    return jacobian
```


```python
def measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=None,):
    fit_time_list, eval_time_list = [], []
    X = np.random.uniform(-1, 1, (N, D))
    xx = np.linspace(-1, 1, M)
    axis_limits = np.array([[-1] * D, [1] * D])

    method_map = {
        "pdp": effector.RegionalPDP,
        "d_pdp": effector.RegionalDerPDP,
        "ale": effector.RegionalALE,
        "rhale": effector.RegionalRHALE,
        "shap_dp": effector.RegionalShapDP
    }

    for _ in range(repetitions):
        # general kwargs
        method_kwargs = {"data": X, "model": model, "axis_limits": axis_limits, "nof_instances":"all"}
        fit_kwargs = {"features": features, "centering": True, "points_for_centering": K, "space_partitioner": effector.space_partitioning.Best(max_depth=2)}

        # specialize kwargs per method
        if method_name in ["d_pdp", "rhale"]:
            method_kwargs["model_jac"] = model_jac
        if method_name in ["rhale", "ale"]:
            fit_kwargs["binning_method"] = effector.axis_partitioning.Fixed(nof_bins=20, min_points_per_bin=0.)
            fit_kwargs.pop("centering")
            fit_kwargs.pop("points_for_centering")

        if method_name in ["pdp", "d_pdp"]:
            fit_kwargs.pop("centering")

        if method_name == "d_pdp":
            fit_kwargs.pop("points_for_centering")

        # init
        method = method_map[method_name](**method_kwargs)

        # fit
        tic = time.time()
        method.fit(**fit_kwargs)
        fit_time_list.append(time.time() - tic)

        # eval
        tic = time.time()
        for feat in features:
            eval_kwargs = {"feature": feat, "node_idx": 0, "xs": xx, "centering": True, "heterogeneity": True}
            method.eval(**eval_kwargs)
        eval_time_list.append(time.time() - tic)

    return {"fit": np.mean(fit_time_list), "eval": np.mean(eval_time_list), "total": (np.mean(fit_time_list) + np.mean(eval_time_list))}
```


```python
import matplotlib.pyplot as plt

def bar_plot(xs, time_dict, methods, metric, title, xlabel, ylabel, bar_width=0.02):

    bar_width = (np.max(xs) - np.min(xs)) / 40
    method_to_label = {"ale": "ALE", "rhale": "RHALE", "pdp": "PDP", "d_pdp": "d-pdp", "shap_dp": "SHAP DP"}
    plt.figure()
    
    # Calculate the offsets for each bar group
    offsets = np.linspace(-2*bar_width, 2*bar_width, len(methods))
    
    for i, method in enumerate(methods):
        label = method_to_label[method]
        plt.bar(
            xs + offsets[i],
            [tt[metric] for tt in time_dict[method]],
            label=label,
            width=bar_width
        )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
```

## Runtime vs $t_f$


```python
t = 0.001
N = 10_000
D = 5
K = 100
M = 100
repetitions = 2
features=[0]
```


```python
method_names = ["ale", "rhale", "pdp", "d_pdp"]
vec = np.array([.1, .5, 1.])
time_dict = {method_name: [] for method_name in method_names}
for t in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))
```

    100%|██████████| 1/1 [00:00<00:00,  2.60it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.57it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.75it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.90it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.06it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.13it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.67it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.71it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.19s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.23s/it]
    100%|██████████| 1/1 [00:00<00:00,  1.89it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.90it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.29s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.30s/it]
    100%|██████████| 1/1 [00:00<00:00,  1.30it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.29it/s]
    100%|██████████| 1/1 [00:02<00:00,  2.19s/it]
    100%|██████████| 1/1 [00:02<00:00,  2.20s/it]
    100%|██████████| 1/1 [00:00<00:00,  1.88it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.86it/s]
    100%|██████████| 1/1 [00:02<00:00,  2.31s/it]
    100%|██████████| 1/1 [00:02<00:00,  2.32s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.33s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.30s/it]



```python
for metric in ["total"]:
    if metric in ["fit", "eval"]:
        title = "Runtime: ." + metric + "() -- single feature"
    else:
        title = "Runtime: .fit() + .eval() -- single feature"
    
    bar_plot(
        vec, 
        time_dict, 
        method_names,
        metric=metric,
        title=title,
        xlabel="time (sec) to execute f(dataset)",
        ylabel="time (sec)"
)
```


    
![png](efficiency_regional_files/efficiency_regional_8_0.png)
    



```python
for metric in ["total"]:
    if metric in ["fit", "eval"]:
        title = "Runtime: ." + metric + "() -- single feature"
    else:
        title = "Runtime: .fit() + .eval() -- single feature"
    
    bar_plot(
        vec, 
        time_dict, 
        method_names,
        metric=metric,
        title=title,
        xlabel="time (sec) to execute f(dataset)",
        ylabel="time (sec)"
)
```


    
![png](efficiency_regional_files/efficiency_regional_9_0.png)
    


## Runtime vs. D


```python
t = 0.001
N = 10_000
D = 5
K = 100
M = 100
repetitions = 2
features=[0]
```


```python
method_names = ["ale", "rhale", "pdp", "d_pdp"]
vec = np.array([3, 4, 5, 6])
time_dict = {method_name: [] for method_name in method_names}
for D in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))
```

    100%|██████████| 1/1 [00:00<00:00,  8.26it/s]
    100%|██████████| 1/1 [00:00<00:00, 10.30it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.88it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.93it/s]
    100%|██████████| 1/1 [00:00<00:00,  6.10it/s]
    100%|██████████| 1/1 [00:00<00:00,  6.12it/s]
    100%|██████████| 1/1 [00:00<00:00,  7.38it/s]
    100%|██████████| 1/1 [00:00<00:00,  7.23it/s]
    100%|██████████| 1/1 [00:00<00:00,  7.39it/s]
    100%|██████████| 1/1 [00:00<00:00,  7.38it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.63it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.61it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.97it/s]
    100%|██████████| 1/1 [00:00<00:00,  4.47it/s]
    100%|██████████| 1/1 [00:00<00:00,  4.92it/s]
    100%|██████████| 1/1 [00:00<00:00,  4.83it/s]
    100%|██████████| 1/1 [00:00<00:00,  5.53it/s]
    100%|██████████| 1/1 [00:00<00:00,  5.54it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.88it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.88it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.59it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.36it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.52it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.69it/s]
    100%|██████████| 1/1 [00:00<00:00,  4.29it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.65it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.37it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.41it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.04it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.06it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.91it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.52it/s]



```python
for metric in ["total"]:
    if metric in ["fit", "eval"]:
        title = "Runtime: ." + metric + "() -- single feature"
    else:
        title = "Runtime: .fit() + .eval() -- single feature"
    
    bar_plot(
        vec, 
        time_dict, 
        method_names,
        metric=metric,
        title=title,
        xlabel="D: number of features",
        ylabel="time (sec)"
)
```


    
![png](efficiency_regional_files/efficiency_regional_13_0.png)
    



```python
for metric in ["total"]:
    if metric in ["fit", "eval"]:
        title = "Runtime: ." + metric + "() -- single feature"
    else:
        title = "Runtime: .fit() + .eval() -- single feature"
    
    bar_plot(
        vec, 
        time_dict, 
        method_names,
        metric=metric,
        title=title,
        xlabel="D: number of features",
        ylabel="time (sec)"
)
```


    
![png](efficiency_regional_files/efficiency_regional_14_0.png)
    


## Time vs N (number of features)


```python
t = 0.001
N = 100_000
D = 5
T = 100
K = 100
repetitions = 2
features=[0]
```


```python
method_names = ["ale", "rhale", "pdp", "d_pdp"]
vec = np.array([10_000, 20_000, 30_000])
time_dict = {method_name: [] for method_name in method_names}
for N in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, T, D, repetitions, K, model_jac=model_jac))
```

    100%|██████████| 1/1 [00:00<00:00,  4.31it/s]
    100%|██████████| 1/1 [00:00<00:00,  5.56it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.89it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.87it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.54it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.33it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.42it/s]
    100%|██████████| 1/1 [00:00<00:00,  3.50it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.81it/s]
    100%|██████████| 1/1 [00:00<00:00,  2.82it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.12s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.04s/it]
    100%|██████████| 1/1 [00:00<00:00,  1.65it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.47it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.17s/it]
    100%|██████████| 1/1 [00:00<00:00,  1.32it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.86it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.81it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.51s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.49s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.04s/it]
    100%|██████████| 1/1 [00:00<00:00,  1.18it/s]
    100%|██████████| 1/1 [00:01<00:00,  1.44s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.80s/it]



```python
for metric in ["total"]:
    if metric in ["fit", "eval"]:
        title = "Runtime: ." + metric + "() -- single feature"
    else:
        title = "Runtime: .fit() + .eval() -- single feature"
    
    bar_plot(
        vec, 
        time_dict, 
        method_names,
        metric=metric,
        title=title,
        xlabel="N: nof instances",
        ylabel="time (sec)"
)
```


    
![png](efficiency_regional_files/efficiency_regional_18_0.png)
    


## A demanding example


```python
t = 0.1
N = 50_000
D = 15
T = 100
K = 100
repetitions = 2
features=[0]
```


```python
method_names = ["ale", "rhale", "pdp", "d_pdp"]
time_dict = {method_name: [] for method_name in method_names}
model = return_predict(t)
model_jac = return_jacobian(t)
for method_name in method_names:
    time_dict[method_name].append(measure_time(method_name, features, model, N, T, D, repetitions, K, model_jac=model_jac))
```

    100%|██████████| 1/1 [00:03<00:00,  3.58s/it]
    100%|██████████| 1/1 [00:03<00:00,  3.55s/it]
    100%|██████████| 1/1 [00:12<00:00, 12.18s/it]
    100%|██████████| 1/1 [00:12<00:00, 12.08s/it]
    100%|██████████| 1/1 [00:05<00:00,  5.46s/it]
    100%|██████████| 1/1 [00:05<00:00,  5.44s/it]
    100%|██████████| 1/1 [00:08<00:00,  8.99s/it]
    100%|██████████| 1/1 [00:08<00:00,  8.92s/it]



```python
bar_plot(np.array([1, 2]), time_dict, method_names, metric="total", 
         title="a",
         xlabel="A difficult case",
         ylabel="time (sec)",
        )
```


    
![png](efficiency_regional_files/efficiency_regional_22_0.png)
    

