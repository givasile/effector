# `Effector`s API

Effector is based on the idea of progressive disclosure of complexity; it starts simple but if you need more flexibility you can ask it.




## Global Effect

### Simple API

Global (1) plots can be made in a single line of code. The structure is:
{ .annotate }

1. global_methods are `PDP`, `RHALE`, `ShapDP`, `ALE`, `DerPDP`

```python
effector.<global_method>(data, predict, axis_limits).plot(feature=i)
```

### Regional Effect

Regional (2) plots can be made in a single line of code.
{ .annotate }

2. regional_methods are `RegionalPDP`, `RegionalRHALE`, `RegionalShapDP`, `RegionalALE`, `RegionalDerPDP`

```python
effector.<regional_method>(data, predict, axis_limits).plot(feature=i, node_id=j)
```



## Advanced API

### Global Effect
### Regional Effect
