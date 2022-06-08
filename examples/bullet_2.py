import numpy as np
import feature_effect as fe
import examples.example_utils as utils
import matplotlib.pyplot as plt

def gen_model_params():
    params = [{"b":0, "from": 0., "to": 20.},
              {"b":5., "from": 20, "to": 40},
              {"b":-5, "from": 40, "to": 60},
              {"b":1., "from": 60, "to": 80},
              {"b":-1., "from": 80, "to": 100}]
    x_start = -5
    utils.find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x = np.random.uniform(0, 100-eps, size=int(N))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x, np.array([100. - eps]))), axis=-1)
    return x

def compute_dale_for_N(N, seed):
    np.random.seed(seed)
    data = np.sort(generate_samples(N=N), axis=0)
    model = utils.create_model(model_params, data)
    dale_fixed = utils.fit_multiple_K(data, model, model_jac, K_max_fixed, min_points_per_bin, method="fixed-size")
    return dale_fixed


example_dir = "./examples/bullet_2/"
noise_level = 4.
K_max_fixed = 200
min_points_per_bin = 10
seed = 233
model_params = gen_model_params()
model_jac = utils.create_noisy_jacobian(model_params, noise_level, seed)


dale_list = []
# N = 1000
# dale_list.append(compute_dale_for_N(N, seed=seed))
# utils.plot_loss(dale_list[-1], savefig=example_dir + "im_3.png")

# for 50 points
N = 50
dale_list.append(compute_dale_for_N(N, seed=seed))
utils.plot_loss(dale_list[-1], savefig=example_dir + "im_1.png")

# for 100 points
N = 100
dale_list.append(compute_dale_for_N(N, seed=seed))
utils.plot_loss(dale_list[-1], savefig=example_dir + "im_2.png")

# for 1000 points
N = 1000
dale_list.append(compute_dale_for_N(N, seed=seed))
utils.plot_loss(dale_list[-1], savefig=example_dir + "im_3.png")

# for 10000 points
N = 10000
dale_list.append(compute_dale_for_N(N, seed=seed))
utils.plot_loss(dale_list[-1], savefig=example_dir + "im_4.png")
