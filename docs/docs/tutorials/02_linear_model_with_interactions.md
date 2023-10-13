```python
# Define a model with interactions
def predict(x):
    y = 7 * x[:, 0] - 3 * x[:, 1] + 4 * x[:, 2] + 2 * x[:, 0] * x[:, 2]
    return y


def predict_grad(x):
    df_dx1 = 7 + 2 * x[:, 2]
    df_dx2 = -3 * np.ones([x.shape[0]])
    df_dx3 = 4 * np.ones([x.shape[0]]) + 2 * x[:, 0]
    return np.stack([df_dx1, df_dx2, df_dx3], axis=-1)


### PDP and ICE
effector.PDP(data=X, model=predict).plot(feature=0, centering=True, uncertainty=True)
effector.RHALE(data=X, model=predict, model_jac=predict_grad).plot(feature=0, uncertainty=True)
rhale = effector.RHALE(data=X, model=predict, model_jac=predict_grad)
rhale.fit(features="all")
rhale.plot(feature=2, centering=True, uncertainty=True)
### ICE
effector = importlib.reload(effector)
pdp_ice = effector.PDPwithICE(data=X, model=predict)
pdp_ice.fit(features="all")
# pdp_ice.eval(x=np.array([[0.5, 0.5, 0.5]]))
pdp_ice.plot(feature=0, centering=True)
pdp_ice.data[:, 0].max()
pdp_ice.axis_limits
pdp_ice.y_ice[2].plot(feature=0, centering=True)
pdp_ice.y_ice[2].data
pdp_ice.y_ice[2].eval(feature=0, xs=np.array([0.0, 0.1, 0.2, 0.3]))
```
