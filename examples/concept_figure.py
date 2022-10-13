import examples.utils as utils
path = utils.add_parent_path()
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import example_models.distributions as dist
import example_models.models as models

import numpy as np



def f(x):
    y = np.zeros_like(x[:,0])

    a = 1
    ind = np.logical_and(x[:, 0] >= 0, x[:, 0] < .5)
    y[ind] = np.sin(2*np.pi * a * x[ind, 0])


    # ind = np.logical_and(x[:, 0] > 0.25, x[:, 0] < 1)
    # y[ind] = - 2 * (x[ind, 0] - 0.25)

    # ind = np.logical_and(x[:, 0] > 0.625, x[:, 0] <= 1)
    # y[ind] = 3 * (x[ind, 0] - 0.625) - 0.75

    y = y + x[:,0]*x[:,1]
    return y


def dfdx(x):
    dydx = np.zeros_like(x)

    a = 1
    ind = np.logical_and(x[:, 0] >= 0, x[:, 0] <= .5)
    dydx[ind, 0] = a * 2 * np.pi * np.cos(2*np.pi * a * x[ind, 0])

    dydx[:,0] += x[:,1]
    return dydx

np.random.seed(seed=21)
axis_limits = np.array([[0,1], [0,1]]).T

x1 = np.linspace(0, 1, 1000)
x2 = np.zeros_like(x1)
x = np.stack([x1, x2], -1)

# low sigma
sigma = 1

x1 = np.random.uniform(0, 1, 1000)
x2 = np.random.normal(0, sigma, 1000)
x = np.stack([x1, x2], -1)

# dale1 = fe.DALE(data=x,
#                model=f,
#                model_jac=dfdx,
#                axis_limits=axis_limits)
# dale1.fit([0], alg_params = {"bin_method": "fixed", "nof_bins":50})
# dale1.plot(block=False)

dale2 = fe.DALE(data=x,
               model=f,
               model_jac=dfdx,
               axis_limits=axis_limits)
dale2.fit([0], alg_params = {"bin_method": "dp", "max_nof_bins":50})
dale2.plot(title="UALE - Feature Effect Plot", savefig= os.path.join(os.getcwd(), "examples/concept_figure/fig-1.pdf"))
