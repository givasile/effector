import numpy as np
import effector
import matplotlib.pyplot as plt


def test_plots():
    def generate_dataset(N, x1_min, x1_max, x2_sigma, x3_sigma):
        x1 = np.random.uniform(x1_min, x1_max, size=int(N))
        x2 = np.random.normal(loc=x1, scale=x2_sigma)
        x3 = np.random.uniform(x1_min, x1_max, size=int(N))
        return np.stack((x1, x2, x3), axis=-1)

    # generate the dataset
    np.random.seed(21)

    N = 1000
    x1_min = 0
    x1_max = 1
    x2_sigma = 0.1
    x3_sigma = 1.0
    X = generate_dataset(N, x1_min, x1_max, x2_sigma, x3_sigma)

    def predict(x):
        y = 7 * x[:, 0] - 3 * x[:, 1] + 4 * x[:, 2]
        return y

    def predict_grad(x):
        df_dx1 = 7 * np.ones([x.shape[0]])
        df_dx2 = -3 * np.ones([x.shape[0]])
        df_dx3 = 4 * np.ones([x.shape[0]])
        return np.stack([df_dx1, df_dx2, df_dx3], axis=-1)

    assert all(
        [
            effector.PDP(data=X, model=predict).plot(
                feature=i, y_limits=[-5, 5], show_plot=False
            )
            is not None
            for i in [0, 1, 2]
        ]
    )

    assert all(
        [
            effector.DerPDP(data=X, model=predict, model_jac=predict_grad).plot(
                feature=i, heterogeneity=True, dy_limits=[-10, 10], show_plot=False
            )
            for i in range(3)
        ]
    )

    assert all(
        [
            effector.ALE(data=X, model=predict).plot(
                feature=i, y_limits=[-5, 5], dy_limits=[-10, 10], show_plot=False
            )
            for i in range(3)
        ]
    )

    assert all(
        [
            effector.RHALE(data=X, model=predict, model_jac=predict_grad).plot(
                feature=i, y_limits=[-5, 5], dy_limits=[-10, 10], show_plot=False
            )
            for i in range(3)
        ]
    )

    assert all(
        [
            effector.ShapDP(data=X, model=predict).plot(feature=i, show_plot=False)
            for i in range(3)
        ]
    )

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X = (X - x_mean) / x_std
    y_mean = np.mean(predict(X))
    y_std = np.std(predict(X))

    scale_x_list = [{"mean": x_mean[i], "std": x_std[i]} for i in range(X.shape[1])]
    scale_y = {"mean": y_mean, "std": y_std}

    assert all(
        [
            effector.PDP(data=X, model=predict).plot(
                feature=i,
                y_limits=[-5, 5],
                show_plot=False,
                scale_x=scale_x_list[i],
                scale_y=scale_y,
                use_vectorized=False,
                nof_ice=200,
                nof_points=25,
            )
            is not None
            for i in [0, 1, 2]
        ]
    )
    plt.close("all")

    assert all(
        [
            effector.PDP(data=X, model=predict).plot(
                feature=i,
                y_limits=[-5, 5],
                heterogeneity=False,
                show_plot=False,
                scale_x=scale_x_list[i],
                scale_y=scale_y,
            )
            is not None
            for i in [0, 1, 2]
        ]
    )
    plt.close("all")
    assert all(
        [
            effector.DerPDP(data=X, model=predict, model_jac=predict_grad).plot(
                feature=i,
                heterogeneity=True,
                dy_limits=[-10, 10],
                show_plot=False,
                scale_x=scale_x_list[i],
                scale_y=scale_y,
                use_vectorized=False,
                nof_ice=200,
                nof_points=25,
            )
            for i in range(3)
        ]
    )
    plt.close("all")
    assert all(
        [
            effector.ALE(data=X, model=predict).plot(
                feature=i,
                y_limits=[-5, 5],
                dy_limits=[-10, 10],
                show_plot=False,
                scale_x=scale_x_list[i],
                scale_y=scale_y,
                centering=False,
                show_avg_output=True,
                show_only_aggregated=True,
            )
            for i in range(3)
        ]
    )
    plt.close("all")
    assert all(
        [
            effector.RHALE(data=X, model=predict, model_jac=predict_grad).plot(
                feature=i,
                y_limits=[-5, 5],
                dy_limits=[-10, 10],
                show_plot=False,
                scale_x=scale_x_list[i],
                scale_y=scale_y,
            )
            for i in range(3)
        ]
    )
    plt.close("all")
    assert all(
        [
            effector.ShapDP(data=X, model=predict).plot(
                feature=i,
                show_plot=False,
                scale_x=scale_x_list[i],
                scale_y=scale_y,
                nof_shap_values=20,
                nof_points=25,
                only_shap_values=True,
            )
            for i in range(3)
        ]
    )
    plt.close("all")
