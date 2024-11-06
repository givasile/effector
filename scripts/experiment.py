import numpy as np

import effector

model = effector.models.ConditionalInteraction()
dataset = effector.datasets.IndependentUniform(dim=3, low=-1, high=1)

x = dataset.generate_data(1_000)
y = model.predict(x)

pdp = effector.PDP(x, model.predict, dataset.axis_limits)
pdp.fit(features=0, centering=True)
pdp.plot(feature=0, centering=True, y_limits=[-1, 1])


yy = pdp.eval(
    feature=0,
    xs=np.linspace(-1, 1, 100),
    centering=True
)

