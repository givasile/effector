**Available quickstart tutorials:**

- [Simple API](./simple_api/): use `effector` with the default fitting arguments.
- [Fleible API](./flexible_api/): use `effector` with customized fitting arguments.
- [Global Effects](./global_effect_intro/): `effector` global effect usage on a real example
- [Regional Effects](./regional_effect_intro/): `effector` regional effect usage on a real example

---

 [Simple API](./simple_api/) means omit the `.fit()` step, which means **use the default fitting arguments**.

```python
# define the global effect method
global_method = effector.<global_method>(X, model, ...)

# use its functionalities
global_method.plot(feature=i, ...)
global_method.eval(feature=i, xs=xs, ...)
```

```python
# define the regional effect method
regional_method = effector.<regional_method>(X, model, ...)

# use its functionalities
regional_method.summary(features=..., ...)
regional_method.plot(feature=i, node_idx=j, ...)
regional_method.eval(feature=i, node_idx=j, xs=xs, ...)
```

--- 

[Flexible API](./flexible_api/) means use the `.fit()` step, which means **customize the fitting arguments**.

```python
# define the global effect method
global_method = effector.<global_method>(X, model, ...)

# fit the method
global_method.fit(features=[...], ...)

# use its functionalities
global_method.plot(feature=i, ...)
global_method.eval(feature=i, xs=xs, ...)
```

```python
# define the regional effect method
regional_method = effector.<regional_method>(X, model, ...)

# fit the method
regional_method.fit([features=...], ...)

# use its functionalities
regional_method.summary(features=..., ...)
regional_method.plot(feature=i, node_idx=j, ...)
regional_method.eval(feature=i, node_idx=j, xs=xs, ...)
```