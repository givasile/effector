import numpy as np
import pythia
import pythia.regions as regions

class RepidSimpleDist:
    """
    x1 ~ U(-1, 1)
    x2 ~ U(-1, 1)
    x3 ~ Bernoulli(0.5)
    """

    def __init__(self):
        self.D = 2
        self.axis_limits = np.array([[-1, 1], [-1, 1], [0, 1]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x2 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x3 = np.random.choice([0, 1], int(N), p=[0.5, 0.5])

        x = np.stack((x1, x2, x3), axis=-1)
        return x


class RepidSimpleModel:
    def __init__(self, a1=0.2, a2=-8, a3=8, a4=16):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def predict(self, x):
        y = self.a1*x[:, 0] + self.a2*x[:, 1]

        cond = x[:, 0] > 0
        y[cond] += self.a3*x[cond, 1]

        cond = x[:, 2] == 0
        y[cond] += self.a4*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        y = np.stack([self.a1*np.ones(x.shape[0]), self.a2*np.ones(x.shape[0]), np.zeros(x.shape[0])], axis=-1)

        cond = x[:, 0] > 0
        y[cond, 1] += self.a3

        cond = x[:, 2] == 0
        y[cond, 1] += self.a4
        return y


np.random.seed(21)
dist = RepidSimpleDist()
model = RepidSimpleModel()

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)

def func(data):
    # if data is empty, return zero
    if data.shape[0] == 0:
        return 1000000
    feat = 1
    dpdp = pythia.pdp.dPDP(data, model.predict, model.jacobian, dist.axis_limits)
    dpdp.fit(features="all", normalize=False)
    start = dist.axis_limits[:, feat][0]
    stop = dist.axis_limits[:, feat][1]
    x = np.linspace(start, stop, 1000)
    x = 0.5 * (x[:-1] + x[1:])
    pdp_m, pdp_std, pdp_stderr = dpdp.eval(feature=feat, x=x, uncertainty=True)
    z = np.mean(pdp_std)
    return z

nof_levels = 2

# iterate to find nof_levels optimal splits
positions = []
features = []
list_of_X = [X]
for i in range(nof_levels):
    I_start, I, i, j, feature, position = regions.find_optimal_split(func, list_of_X, 1, [0, 2], 10, dist.axis_limits)

    positions.append(position)
    features.append(feature)

    new_list_of_X = []
    for x in list_of_X:
        # split X on the optimal feature and position
        X1 = x[x[:, feature] < position]
        X2 = x[x[:, feature] >= position]
        new_list_of_X.append(X1)
        new_list_of_X.append(X2)
    list_of_X = new_list_of_X





# I_start_0, I_0, i_0, j_0, feature_0, position_0 = regions.find_optimal_split(func, [X], 1, [0, 2], 10, dist.axis_limits)
#
# # split X on the optimal feature and position
# X1 = X[X[:, feature_0] < position_0]
# X2 = X[X[:, feature_0] >= position_0]
# I_start_1, I_1, i_1, j_1, feature_1, position_1 = regions.find_optimal_split(func, [X1], 1, [0, 2], 10, dist.axis_limits)
#
#
#
#
#
#
#
# # pdp_dice = pythia.pdp.PDPwithdICE(X, model.predict, model.jacobian, dist.axis_limits)
# # pdp_dice.fit(features="all", normalize=False)
# # pdp_dice.plot(feature=2, normalized=False)
# # pdp_dice.eval(feature=2, x=x, uncertainty=True)
