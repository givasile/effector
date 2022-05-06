from feature_effect.dale import DALE
from feature_effect.ale import ALE
from feature_effect.pdp import PDP
from feature_effect.mplot import MPlot
from feature_effect import visualization as vis


class Estimator:
    def __init__(self, data, model, model_jac=None):
        self.data = data
        self.model = model
        self.model_jac = model_jac

        self.dale = None
        self.ale = None
        self.pdp = None
        self.mplot = None

    def fit(self, features: list, method: str = 'DALE', nof_bins=None):
        assert method in ["DALE", "ALE", "PDP", "MPlot", "all"]

        if method == "all":
            self.dale = DALE(self.data, self.model, self.model_jac)
            self.dale.fit(features, nof_bins)
            self.ale = ALE(self.data, self.model)
            self.ale.fit(features, nof_bins)
            self.pdp = PDP(self.data, self.model)
            self.mplot = MPlot(self.data, self.model)
        elif method == "DALE":
            self.dale = DALE(self.data, self.model, self.model_jac)
            self.dale.fit(features, nof_bins)
        elif method == "ALE":
            self.ale = ALE(self.data, self.model)
            self.ale.fit(features, nof_bins)
        elif method == "PDP":
            self.pdp = PDP(self.data, self.model)
        elif method == "MPlot":
            self.mplot = MPlot(self.data, self.model)

    def evaluate(self, x, feature: int, method: str = 'DALE'):
        assert method in ["DALE", "ALE", "PDP", "MPlot"]
        var = None
        if method == "DALE":
            y, var = self.dale.eval(x, feature)
            return y, var
        elif method == "ALE":
            y, var = self.ale.eval(x, feature)
            return y
        elif method == "PDP":
            return self.pdp.eval(x, feature)
        elif method == "MPlot":
            return self.mplot.eval(x, feature, tau=0.5)

    def plot(self, feature, method: str = 'DALE', ale_gt=None):
        assert method in ["DALE", "ALE", "PDP", "MPlot", "all"]
        if method == "all":
            vis.fe_all(self.dale, self.ale, self.pdp, self.mplot, feature, ale_gt)
        elif method == "DALE":
            self.dale.plot(feature, block=False)
        elif method == "ALE":
            self.ale.plot(feature)
        elif method == "PDP":
            self.pdp.plot(feature)
        elif method == "MPlot":
            self.mplot.plot(feature, tau=.1)
