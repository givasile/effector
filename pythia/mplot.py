import numpy as np
import copy
import matplotlib.pyplot as plt
import typing
import pythia.helpers as helpers

class MPlot:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    @staticmethod
    def _mplot(x, points, f, s, tau):
        y = []
        for i in range(x.shape[0]):
            points1 = copy.deepcopy(points)
            points1 = points1[np.abs(points[:, s] - x[i]) < tau, :]
            points1[:, s] = x[i]
            y.append(np.mean(f(points1)))
        return np.array(y)

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all", normalize: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = {}
            if normalize:
                self.norm_const[s] = self._compute_norm_const(s)
            self.is_fitted[s] = True

    def eval(self, feature, x, tau):
        return self._mplot(x, self.data, self.model, feature, tau)

    def plot(self, feature, tau=0.5, step=1000):
        min_x = np.min(self.data[:, feature])
        max_x = np.max(self.data[:, feature])

        x = np.linspace(min_x, max_x, step)
        y = self.eval(x, feature, tau)

        plt.figure()
        plt.title("MPlot for feature %d" % (feature + 1))
        plt.plot(x, y, "b-")
        plt.show(block=False)
