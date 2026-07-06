# Measuring Runtime of Regional Effect Plots

- Author: [givasile](https://givasile.github.io/)
- Runtime: ~7 min (deliberate benchmark)
- Description: Benchmarks how the runtime of each regional method
  (RegionalPDP, d-PDP, ALE, RHALE) scales with the model-call cost $t_f$,
  the number of features $D$ and the number of instances $N$, closing with
  a demanding large-scale example.

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

      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.89it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.89it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.94it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.94it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.02it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.02it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.05it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.05it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.30it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.30it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.19it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.19it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.11it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.11it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.13it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.13it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.30s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.30s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.34s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.34s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.07s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.07s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.03s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.03s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.62s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.62s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.67s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.67s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.33s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.33s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.28s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.28s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.33s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.33s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.33s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.33s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.02s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.02s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.11s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.11s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.61s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.61s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.61s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.61s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.77s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.77s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.74s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.74s/it]

    



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
    


## Runtime vs. D


```python
t = 1.
N = 10_000
D = 5
K = 100
M = 100
repetitions = 1
features=[0]
```


```python
method_names = ["ale", "rhale", "pdp", "d_pdp"]
vec = np.array([5, 8, 10])
time_dict = {method_name: [] for method_name in method_names}
for D in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))
```

      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.34s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.34s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.03s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.03s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.71s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.71s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.81s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.81s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.60s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.60s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.22s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.22s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:03<00:00,  3.38s/it]

    100%|██████████| 1/1 [00:03<00:00,  3.38s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.06s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.07s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.45s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.45s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.56s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.56s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.70s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.70s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.41s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.41s/it]

    



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


    
![png](efficiency_regional_files/efficiency_regional_12_0.png)
    


## Time vs N (number of instances)


```python
t = 1.0
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

      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.20s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.20s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.20s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.20s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.82it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.82it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.83it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.83it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.31s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.31s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.32s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.32s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.38s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.38s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.39s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.39s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.34s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.34s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.32s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.32s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.13it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.13it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.14it/s]

    100%|██████████| 1/1 [00:00<00:00,  1.14it/s]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.83s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.83s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.69s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.69s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.86s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.86s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.88s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.88s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.55s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.55s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.47s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.47s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.33s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.33s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:01<00:00,  1.26s/it]

    100%|██████████| 1/1 [00:01<00:00,  1.26s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:03<00:00,  3.07s/it]

    100%|██████████| 1/1 [00:03<00:00,  3.07s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:03<00:00,  3.05s/it]

    100%|██████████| 1/1 [00:03<00:00,  3.05s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.46s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.46s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:02<00:00,  2.67s/it]

    100%|██████████| 1/1 [00:02<00:00,  2.67s/it]

    



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


    
![png](efficiency_regional_files/efficiency_regional_16_0.png)
    


## A demanding example


```python
t = 3.0
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

      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:08<00:00,  8.52s/it]

    100%|██████████| 1/1 [00:08<00:00,  8.52s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:08<00:00,  8.53s/it]

    100%|██████████| 1/1 [00:08<00:00,  8.53s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:10<00:00, 10.93s/it]

    100%|██████████| 1/1 [00:10<00:00, 10.93s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:10<00:00, 10.92s/it]

    100%|██████████| 1/1 [00:10<00:00, 10.92s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:12<00:00, 12.27s/it]

    100%|██████████| 1/1 [00:12<00:00, 12.27s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:12<00:00, 12.19s/it]

    100%|██████████| 1/1 [00:12<00:00, 12.19s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:15<00:00, 15.74s/it]

    100%|██████████| 1/1 [00:15<00:00, 15.74s/it]

    


      0%|          | 0/1 [00:00<?, ?it/s]

    100%|██████████| 1/1 [00:15<00:00, 15.86s/it]

    100%|██████████| 1/1 [00:15<00:00, 15.87s/it]

    



```python
bar_plot(np.array([1, 2]), time_dict, method_names, metric="total", 
         title="a",
         xlabel="A difficult case",
         ylabel="time (sec)",
        )
```


    
![png](efficiency_regional_files/efficiency_regional_20_0.png)
    



```python

```


```python

```
