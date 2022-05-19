import numpy as np
import matplotlib.pyplot as plt
import feature_effect as fe


params = [{"b":0.3, "from": 0., "to": 10.},
          {"b":7. , "from": 10, "to": 20},
          {"b":-1.5, "from": 20, "to": 30},
          {"b":0., "from": 30, "to": 40},
          {"b":-5., "from": 40, "to": 50},
          {"b":0.3, "from": 50, "to": 60},
          {"b":7. , "from": 60, "to": 70},
          {"b":-1.5, "from": 70, "to": 80},
          {"b":0., "from": 80, "to": 90},
          {"b":-5., "from": 90, "to": 100}]


def generate_samples(N):
    x = np.random.uniform(0, 100, size=int(N))
    x = np.expand_dims(np.concatenate((np.array([0.001]), x, np.array([99.9]))), axis=-1)
    return x


def f1(x):
    """Piece-wise linear"""

    def find_a(params, x_start):
        params[0]["a"] = x_start
        for i, param in enumerate(params):
            if i < len(params) - 1:
                a_next = param["a"] + (param["to"]-param["from"])*param["b"]
                params[i+1]["a"] = a_next

    limits = [param["from"] for param in params]
    limits.append(params[-1]["to"])


    x_start = -1
    find_a(params, x_start)

    ind = np.squeeze(np.digitize(x, limits))
    y = []
    for i, point in enumerate(x):
        res = params[ind[i] - 1]["a"] + \
            (point - params[ind[i] - 1]["from"])*params[ind[i] - 1]["b"]

        y.append(res)
    return np.array(y)

z = np.mean(f1(np.linspace(0.0001, 99.99, 10000)))

def f1_center(x):
    return f1(x) - z

def compute_data_effect(x):
    """Piece-wise linear"""

    def find_a(params, x_start):
        params[0]["a"] = x_start
        for i, param in enumerate(params):
            if i < len(params) - 1:
                a_next = param["a"] + (param["to"]-param["from"])*param["b"]
                params[i+1]["a"] = a_next

    limits = [param["from"] for param in params]
    limits.append(params[-1]["to"])


    x_start = -1
    find_a(params, x_start)

    x = np.squeeze(x)
    ind = np.squeeze(np.digitize(x, limits))
    res1 = np.array(params)[ind-1]
    y = np.array([r['b'] for r in res1])

    # add noise
    noise_level = 5.
    np.random.seed(2443534)
    noise = np.random.normal(0, noise_level, y.shape[0])
    return np.expand_dims(y+noise, -1)


gt_bins = {}
gt_bins["height"] = [par["b"] for par in params]
gt_bins["limits"] = [par["from"] for par in params]
gt_bins["limits"].append(params[-1]["to"])

# parameters
N = 10

# main part
seed = 4837571
np.random.seed(seed)

x = generate_samples(N=N)
y = f1_center(x)
data = x
data_effect = compute_data_effect(x)

plt.figure()
plt.plot(x, y, "ro")
# plt.plot(x, data_effect, "bo")
plt.show(block=False)

K = 343
dale = fe.DALE(data=x, model=f1, model_jac=compute_data_effect)
dale.fit(features=[0], k=K)
dale.plot(s=0, block=False, gt=f1_center, gt_bins=gt_bins)

K = 24
dale = fe.DALE(data=x, model=f1, model_jac=compute_data_effect)
dale.fit(features=[0], k=K)
dale.plot(s=0, block=False, gt=f1_center, gt_bins=gt_bins)



def compute_error(dale_func, gt_func):

    xx = np.linspace(0.001, 99.9, 20)
    y_pred = dale_func(xx, s=0)[0]
    y_gt = gt_func(xx)
    return np.mean(np.abs(y_pred - y_gt))



error = []
for k in range(1, 500):
    dale = fe.DALE(data=x, model=f1, model_jac=compute_data_effect)
    dale.fit(features=[0], k=k)
    dale_func = dale.eval
    gt_func = f1_center
    error.append(compute_error(dale_func, gt_func))

k_list = np.arange(1, 500)
plt.figure()
plt.plot(k_list, error, "bo")
plt.show(block=False)
