# Getting Started

`Effector` is a python package for feature effect plots; a collection of interpretability methods for visualizing the 
effect of individual features on the output of a machine learning model. 

---
### Feature effect plots in a single-line

`Effector` is designed to provide a simple API. In most cases, the user can get 
the plot with a single-line command.


```python
# for PDP
PDP(data=X, model=ml_model).plot(feature=0)

# for ALE
ALE(data=X, model=ml_model, model_jac=ml_model_jac).plot(feature=0)
```
--- 

### Heterogeneity of Feature Effect plots

`Effector` provides a measure of heterogeneity, i.e. measures how much individual effects deviate from 
the global effect. For all methods, the heterogeneity is plots by simply enabling 
the parameter `uncertainty=True`.

```python
# for PDP
PDP(data=X, model=ml_model).plot(feature=0, confidence_interval=True)

# for ALE
ALE(data=X, model=ml_model, model_jac=ml_model_jac).plot(feature=0, confidence_interval=True)
```

For more details, check out the [global effect tutorial](./tutorials/00_linear_global_effect/).

--- 

## Extended functionality around ALE

`Effector` provides additional functionality around ALE. For example, the user can



---

### Regional Effect plots in a single-line

