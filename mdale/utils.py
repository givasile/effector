import numpy as np
import pandas as pd
import PyALE
import copy


def pdp(f, X, xs, i):
    X1 = copy.deepcopy(X)
    X1[:,i] = xs
    return np.mean(f(X1))


def mplot(f, X, xs, i, tau):
    X1 = copy.deepcopy(X)
    X1 = X1[np.abs(X1[:,i] - xs) < tau, :]
    X1[:,i] = xs

    if X1.size == 0:
        return np.nan
    else:
        return np.mean(f(X1))


# ale1
def ale(X, f_bb, s, K, feature_type="auto"):
    X_df = pd.DataFrame(X, columns=["feat_" + str(i) for i in range(X.shape[-1])])

    class model():
        def __init__(self, f, X_df):
            self.predict = self.func
            self.f_bb = f

        def func(self, X_df):
            return self.f_bb(X_df.to_numpy())

    model_bb = model(f_bb, X_df)

    ale_computation = PyALE.ale(X_df, model_bb, feature= ["feat_" + str(s)],
                                feature_type=feature_type,
                                grid_size=K,
                                plot=False)

    x = ale_computation["eff"].index.to_numpy()
    y = ale_computation["eff"].to_numpy()
    return x, y
