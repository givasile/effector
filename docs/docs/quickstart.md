**Available quickstart tutorials:**


- [What are Global and Regional Effects](./global_and_regional_effects/)
- [An overview of `effector` API](./../notebooks/quickstart/general_api/) 
- [Use `effector` with default `.fit()` arguments](./simple_api/)
- [Use `effector` with custom `.fit()` arguments](./flexible_api/)

---

[//]: # ( [Simple API]&#40;./simple_api/&#41; means omit the `.fit&#40;&#41;` step, which means **use the default fitting arguments**.)

[//]: # ()
[//]: # (```python)

[//]: # (# define the global effect method)

[//]: # (global_method = effector.<global_method>&#40;X, model, ...&#41;)

[//]: # ()
[//]: # (# use its functionalities)

[//]: # (global_method.plot&#40;feature=i, ...&#41;)

[//]: # (global_method.eval&#40;feature=i, xs=xs, ...&#41;)

[//]: # (```)

[//]: # ()
[//]: # (```python)

[//]: # (# define the regional effect method)

[//]: # (regional_method = effector.<regional_method>&#40;X, model, ...&#41;)

[//]: # ()
[//]: # (# use its functionalities)

[//]: # (regional_method.summary&#40;features=..., ...&#41;)

[//]: # (regional_method.plot&#40;feature=i, node_idx=j, ...&#41;)

[//]: # (regional_method.eval&#40;feature=i, node_idx=j, xs=xs, ...&#41;)

[//]: # (```)

[//]: # ()
[//]: # (--- )

[//]: # ()
[//]: # ([Flexible API]&#40;./flexible_api/&#41; means use the `.fit&#40;&#41;` step, which means **customize the fitting arguments**.)

[//]: # ()
[//]: # (```python)

[//]: # (# define the global effect method)

[//]: # (global_method = effector.<global_method>&#40;X, model, ...&#41;)

[//]: # ()
[//]: # (# fit the method)

[//]: # (global_method.fit&#40;features=[...], ...&#41;)

[//]: # ()
[//]: # (# use its functionalities)

[//]: # (global_method.plot&#40;feature=i, ...&#41;)

[//]: # (global_method.eval&#40;feature=i, xs=xs, ...&#41;)

[//]: # (```)

[//]: # ()
[//]: # (```python)

[//]: # (# define the regional effect method)

[//]: # (regional_method = effector.<regional_method>&#40;X, model, ...&#41;)

[//]: # ()
[//]: # (# fit the method)

[//]: # (regional_method.fit&#40;[features=...], ...&#41;)

[//]: # ()
[//]: # (# use its functionalities)

[//]: # (regional_method.summary&#40;features=..., ...&#41;)

[//]: # (regional_method.plot&#40;feature=i, node_idx=j, ...&#41;)

[//]: # (regional_method.eval&#40;feature=i, node_idx=j, xs=xs, ...&#41;)
