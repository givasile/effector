import effector
import numpy as np

# the black-box model
model = effector.models.DoubleConditionalInteraction()
predict = model.predict
jacobian = model.jacobian

# the dataset
dataset = effector.datasets.IndependentUniform(dim=3, low=-1, high = 1)
data = dataset.generate_data(100)
axis_limits = dataset.axis_limits
y_limits = [-2.5, 2.5]


# PDP effect
effector.PDP(data, predict, axis_limits=axis_limits).plot(feature=0, y_limits=y_limits)
effector.RHALE(data, predict, jacobian, axis_limits=axis_limits).plot(feature=0, y_limits=y_limits)
effector.ShapDP(data, predict, axis_limits=axis_limits).plot(feature=0, y_limits=y_limits)


# more control
pdp = effector.PDP(data, predict, axis_limits=axis_limits)
pdp.fit()
pdp.plot(feature=0, y_limits=y_limits)
pdp.eval(feature=0, xs=np.linspace(axis_limits[0, 0], axis_limits[1, 0], 30), centering=False, heterogeneity=True)

# REgional metholPds
effector.RegionalPDP(data, predict, axis_limits=axis_limits).summary(features=0)
effector.RegionalPDP(data, predict, axis_limits=axis_limits).plot(feature=0, y_limits=y_limits, node_idx=2, centering=True)
