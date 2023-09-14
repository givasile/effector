import effector
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def preprocess_bike(df):
    # shuffle
    df.sample(frac=1).reset_index(drop=True)

    # Standarize X
    X_df = df.drop(["cnt"], axis=1)
    x_mean = X_df.mean()
    x_std = X_df.std()
    X_df = (X_df - X_df.mean()) / X_df.std()

    # Standarize Y
    Y_df = df["cnt"]
    y_mean = Y_df.mean()
    y_std = Y_df.std()
    Y_df = (Y_df - Y_df.mean()) / Y_df.std()
    return X_df, Y_df, x_mean, x_std, y_mean, y_std


def preprocess_california(df):
    # shuffle
    df.sample(frac=1).reset_index(drop=True)

    df = df.dropna()
    X_df = df.iloc[:, :-2]
    Y_df = df.iloc[:, -2]

    # normalize
    X_mean = X_df.mean()
    X_std = X_df.std()
    X_df = (X_df - X_df.mean()) / X_df.std()

    Y_mean = Y_df.mean()
    Y_std = Y_df.std()
    Y_df = (Y_df - Y_df.mean()) / Y_df.std()

    # remove points 3 std away from the mean
    ind = (X_df.abs() > 3).sum(1) == 0
    X_df = X_df.loc[ind, :]
    Y_df = Y_df.loc[ind]

    return X_df, Y_df, X_mean, X_std, Y_mean, Y_std


def split(X_df, Y_df):
    # data split
    X_train = X_df[:int(0.8 * len(X_df))]
    Y_train = Y_df[:int(0.8 * len(Y_df))]
    X_test = X_df[int(0.8 * len(X_df)):]
    Y_test = Y_df[int(0.8 * len(Y_df)):]
    return X_train, Y_train, X_test, Y_test


# def plot_subregions_rhale(feat, feature, type, position, X_train, model, model_jac):
#     # Regional Plot
#     if type == "cat":
#         rhale = effector.RHALE(data=X_train.to_numpy(), model=model, model_jac=model_jac)
#         rhale_1 = effector.RHALE(data=X_train[X_train.iloc[:, feature] == position].to_numpy(), model=model, model_jac=model_jac)
#         rhale_2 = effector.RHALE(data=X_train[X_train.iloc[:, feature] != position].to_numpy(), model=model, model_jac=model_jac)
#     else:
#         rhale = effector.RHALE(data=X_train.to_numpy(), model=model, model_jac=model_jac)
#         rhale_1 = effector.RHALE(data=X_train[X_train.iloc[:, feature] <= position].to_numpy(), model=model, model_jac=model_jac)
#         rhale_2 = effector.RHALE(data=X_train[X_train.iloc[:, feature] > position].to_numpy(), model=model, model_jac=model_jac)
#
#     def plot(rhale):
#         binning_method = effector.binning_methods.Fixed(nof_bins=100)
#         rhale.fit(features=feat, binning_method=binning_method, centering="zero_integral")
#         scale_x = {"mean": x_mean.iloc[feat], "std": x_std.iloc[feat]}
#         scale_y = {"mean": 0, "std": y_std}
#         rhale.plot(feature=feat, uncertainty=None, scale_x=scale_x, scale_y=scale_y)
#         plt.show()
#
#     # plot global and regionals
#     plot(rhale)
#     plot(rhale_1)
#     plot(rhale_2)
