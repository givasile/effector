# Getting Started

`Effector` is a python package for [global](./01_global_effect_intro/) 
and [regional](./02_regional_effect_intro/) effect analysis.

---
### Installation

`pip install effector`

---
### Global effect plots

`Effector` provides global effect plots in a single line of code. 
Given a dataset `X` (`np.ndarray`) and a machine learning model `model` (`callable`), 
the user can get the effect plot of `feature` by:

```python
# for Robust and Heterogeneity-aware ALE (RHALE)
RHALE(data=X, model=model).plot(feature)
```

![Feature effect plot](./../real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_19_1.png)


--- 

### Heterogeneity

`Effector` focuses on the heterogeneity of the effect, i.e., explaining how much the
instance-level effects deviate from the global effect. The user can interpret the 
heterogeneity through the `heterogeneity` argument:

```python
# for RHALE
RHALE(data=X, model=model).plot(feature, heterogeneity="std")
```

![Feature effect plot](./../real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_20_0.png)


For more details, check the [global effect tutorial](./01_global_effect_intro/).

--- 

### Regional Effect plots

High-heterogeneity is an indicator for regional effect analysis.
`Effector` provides the same simple API for regional effect plots. 

```python
RegionalRHALE(data=X, model=model).plot(feature=0, node_idx=1, heterogeneity=True)
RegionalRHALE(data=X, model=model).plot(feature=0, node_idx=2, heterogeneity=True)
```

![Feature effect plot](./../real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_26_0.png)
![Feature effect plot](./../real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_26_1.png)

---

## Methods supported

| Method   | Global Effect                                                                    | Regional Effect                                                                         |                                                                                                                                
|----------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| PDP      | [`PDP`](./../../03_API/#effector.global_effect_pdp.PDP)                          | [`RegionalPDP`](./../../03_API/#effector.regional_effect_pdp.RegionalPDP)               |
| d-PDP    | [`DerivativePDP`](./../../03_API/#effector.global_effect_pdp.DerivativePDP)      | [`RegionalDerivativePDP`](./../../03_API/#effector.regional_effect_pdp.RegionalDerivativePDP) |
| ALE      | [`ALE`](./../../03_API/#effector.global_effect_ale.ALE)                          | [`RegionalALE`](./../../03_API/#effector.regional_effect_ale.RegionalALE)               |
| RHALE    | [`RHALE`](./../../03_API/#effector.global_effect_ale.RHALE)                      | [`RegionalRHALE`](./../../03_API/#effector.regional_effect_ale.RegionalRHALE)           |
| SHAP-DP  | [`SHAPDependence`](./../../03_API/#effector.global_effect_shap.SHAPDependence)   | [`RegionalSHAP`](./../../03_API/#effector.regional_effect_shap.RegionalSHAPDependence)  |




