import numpy as np
import copy
import matplotlib.pyplot as plt


class PDP:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    @staticmethod
    def _pdp(x, points, f, s):
        y = []
        for i in range(x.shape[0]):
            points1 = copy.deepcopy(points)
            points1[:, s] = x[i]
            y.append(np.mean(f(points1)))
        return np.array(y)

    def eval(self, x, feature):
        return self._pdp(x, self.data, self.model, feature)

    def plot(self, feature, step=1000):
        min_x = np.min(self.data[:, feature])
        max_x = np.max(self.data[:, feature])

        x = np.linspace(min_x, max_x, step)
        y = self.eval(x, feature)

        plt.figure()
        plt.title("PDP for feature %d" % (feature+1))
        plt.plot(x, y, "b-")
        plt.show(block=False)
