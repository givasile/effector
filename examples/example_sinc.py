import copy
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe


def sinc(k):
    return np.sin(k)/k


def model(x):
    tau = 2.65
    w = np.pi

    ind = x[:, 0] > tau
    y = sinc(w * x[:, 0] ** 2) + x[:, 0] * x[:, 1] * 10
    y[ind] = sinc(w * tau ** 2) + x[:, 0][ind] * x[:, 1][ind] * 10
    return y


def model_jac(x_tmp):
    x = copy.deepcopy(x_tmp)
    tau = 2.65
    w = np.pi

    ind = x[:, 0] > tau

    y1 = (2 * np.cos(w * x[:, 0] ** 2) / x[:, 0] - np.sin(w * x[:, 0] ** 2) * 2 / w / x[:, 0]**3) + x[:, 1] * 10
    y1[ind] = x[:, 1][ind] * 10

    y2 = x[:, 0]

    grad = np.stack((y1, y2), axis=-1)
    return grad


def gen_data(N, noise_level):
    eps = 1e-03
    stop = 5
    x1 = np.concatenate((np.random.uniform(0. + eps, 2.65, size=int(N)),
                         np.array([2.7]),
                         np.random.normal(3.75, .3, size=int(N/12)),
                         np.array([4.99])
                         )
                        )

    x2 = np.random.normal(loc=0, scale=noise_level, size=(int(x1.shape[0])))
    x = np.stack((x1, x2), axis=-1)
    return x


def create_gt_effect_norm(data):
    def gt_effect_un(x_tmp):
        x = copy.deepcopy(x_tmp)
        tau = 2.65
        w = np.pi

        ind = x > tau

        y = sinc(w * x ** 2)
        y[ind] = sinc(w * tau ** 2)
        return y

    z = np.mean(gt_effect_un(data[:,0]))

    def tmp(x):
        return gt_effect_un(x) - z
    return tmp

example_dir = "./examples/example_1/"

np.random.seed(25)
# generate points
N = 1000
noise_level = .1
data = gen_data(N, noise_level)

gt_effect_norm = create_gt_effect_norm(data)
y = model(data)
y_grad = model_jac(data)

# bin estimation
bin_est = fe.bin_estimation.BinEstimator(data, y_grad, feature=0)
bin_est.plot()

bin_est_1 = fe.bin_estimation.BinEstimatorGreedy(data, y_grad, feature=0)
bin_est_1.solve()
bin_est_1.plot()


# ALE plots based on bin estimation
dale1 = fe.DALE(data, model, model_jac)
dale1.fit(alg_params={"nof_bins": 50})
dale1.plot(gt=gt_effect_norm, savefig=example_dir + "im_2.png")

dale1 = fe.DALE(data, model, model_jac)
dale1.fit(alg_params={"nof_bins": 5})
dale1.plot(gt=gt_effect_norm, savefig=example_dir + "im_3.png")

dale2 = fe.DALE(data, model, model_jac)
dale2.fit(method="variable-size", alg_params={"max_nof_bins": 50})
dale2.plot(gt=gt_effect_norm, savefig=example_dir + "im_4.png")


# limits = np.concatenate((np.linspace(0, 2.65, 99), np.array([5.01])))
# dale3 = fe.DALE(data, model, model_jac)
# dale3.fit(features=[0], method="variable-size", alg_params={"limits": limits})
# dale3.plot(gt=gt_effect_norm)
