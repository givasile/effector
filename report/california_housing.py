import sys, os
import timeit
sys.path.append(os.path.dirname(os.getcwd()))
import effector
import numpy as np
import effector.interaction as interaction
import effector.regions as regions
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor
import sklearn.metrics as metrics
import importlib
from sklearn.utils import check_random_state
import random
pythia = importlib.reload(effector)
import report_utils


# make everything reproducible
np.random.seed(4452)
tf.random.set_seed(4542)
check_random_state(4452)
random.seed(4245)


def plot_subregions(splits, feat, feat_ram, X_train, model, model_jac, gam, ram):
    foc = splits["feat_{0}".format(feat)][0]["feature"]
    position = splits["feat_{0}".format(feat)][0]["position"]
    type = splits["feat_{0}".format(feat)][0]["type"]

    # Regional Plot
    if type == "cat":
        rhale = pythia.RHALE(data=X_train.to_numpy(), model=model, model_jac=model_jac)
        rhale_1 = pythia.RHALE(data=X_train[X_train.iloc[:, foc] == position].to_numpy(), model=model, model_jac=model_jac)
        rhale_2 = pythia.RHALE(data=X_train[X_train.iloc[:, foc] != position].to_numpy(), model=model, model_jac=model_jac)
    else:
        rhale = pythia.RHALE(data=X_train.to_numpy(), model=model, model_jac=model_jac)
        rhale_1 = pythia.RHALE(data=X_train[X_train.iloc[:, foc] <= position].to_numpy(), model=model, model_jac=model_jac)
        rhale_2 = pythia.RHALE(data=X_train[X_train.iloc[:, foc] > position].to_numpy(), model=model, model_jac=model_jac)

    rhale.fit(features=feat, binning_method=pythia.binning_methods.Greedy(), centering="zero_integral")
    rhale.plot(feature=feat, uncertainty=True)
    rhale_1.fit(features=feat, binning_method=pythia.binning_methods.Greedy(), centering="zero_integral")
    rhale_1.plot(feature=feat, uncertainty=True)
    rhale_2.fit(features=feat, binning_method=pythia.binning_methods.Greedy(), centering="zero_integral")
    rhale_2.plot(feature=feat, uncertainty=True)

    def get_effect_from_ebm(ebm_model, ii):
        explanation = ebm_model.explain_global()
        xx = explanation.data(ii)["names"][:-1]
        yy = explanation.data(ii)["scores"]
        return xx, yy

    def plot_fig(ebm, rhale, title, feat, feat_ram, xlabel, ylabel, save=False):
        # gam
        xx, y_ebm = get_effect_from_ebm(ebm, feat_ram)
        xx = np.array(xx)
        y_rhale = rhale.eval(feature=feat, x=xx, uncertainty=False, centering="zero_integral")

        plt.figure()
        plt.title(title)
        plt.plot(xx*x_std.iloc[feat] + x_mean.iloc[feat], y_ebm*y_std, "r--", label="EBM")
        plt.plot(xx*x_std.iloc[feat] + x_mean.iloc[feat], y_rhale*y_std, "b--", label="DALE")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if save is not False:
            plt.savefig(save)

        plt.show()

    plot_fig(gam, rhale, "GAM", feat, feat, "Longitude", "Median Price", "./figures/california_gam.pdf")
    plot_fig(ram, rhale_1, "RAM", feat, feat_ram[0], "Longitude (Latitude <= 34.89)", "Median Price", "./figures/california_ram_1.pdf")
    plot_fig(ram, rhale_2, "RAM", feat, feat_ram[1], "Longitude (Latitude > 34.89)", "Median Price", "./figures/california_ram_2.pdf")


# load dataset
df = pd.read_csv("./../data/California-Housing/housing.csv")

# # drop columns
# df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)
# print(df.columns)

# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = report_utils.preprocess_california(df)

# train/test split
X_train, Y_train, X_test, Y_test = report_utils.split(X_df, Y_df)

cols = df.columns

# Train - Evaluate - Explain a neural network
model = keras.Sequential([
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=60, verbose=1)
model.evaluate(X_train, Y_train, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)


def model_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()

def model_forward(x):
    return model(x).numpy().squeeze()

# Explain
i = 7
ale = pythia.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = pythia.binning_methods.Greedy(init_nof_bins=200, min_points_per_bin=10)
ale.fit(features=i, binning_method=binning_method)
ale.plot(feature=i)


# find regions
reg = pythia.regions.Regions(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, cat_limit=25)
reg.find_splits(nof_levels=2, nof_candidate_splits=10, method="rhale")
opt_splits = reg.choose_important_splits(0.2)

# check opt_splits for feature s
for s in range(X_train.shape[1]):
    print("\n")
    if bool(opt_splits["feat_%s" % s]) is False:
        print("Feature: %s, no splits" % cols[s])
    else:
        for i in range(len(opt_splits["feat_%s" % s])):
            print("Feature: %s, level: %s, opt_splits:" % (cols[s], i+1))
            print("---> Feature: %s" % (cols[opt_splits["feat_%s" % s][i]["feature"]]))
            print("---> Position: %s" % (opt_splits["feat_%s" % s][i]["position"]))
            print("---> Candidate Positions: %s" % (opt_splits["feat_%s" % s][i]["candidate_split_positions"]))
            print("---> Weighted Heterogeneity before: %s" % (opt_splits["feat_%s" % s][i]["weighted_heter"] + opt_splits["feat_%s" % s][i]["weighted_heter_drop"]))
            print("---> Weighted Heterogeneity after : %s" % (opt_splits["feat_%s" % s][i]["weighted_heter"]))
            print("---> Weighted Heterogeneity drop: %s" % (opt_splits["feat_%s" % s][i]["weighted_heter_drop"]))


# transform data
transf = pythia.regions.DataTransformer(splits=opt_splits)
X_train_new = transf.transform(X_train.to_numpy())
X_test_new = transf.transform(X_test.to_numpy())

def fit_eval_gam(title, model, X_train, Y_train, X_test, Y_test):
    print(title)
    model.fit(X_train, Y_train)
    y_train_pred = model.predict(X_train)
    print(model.score(X_test, Y_test))
    print("RMSE - TRAIN: ", metrics.mean_squared_error(Y_train, y_train_pred, squared=False))
    print("MAE - TRAIN", metrics.mean_absolute_error(Y_train, y_train_pred))
    print("RMSE - TEST: ", metrics.mean_squared_error(Y_test, model.predict(X_test), squared=False))
    print("MAE - TEST", metrics.mean_absolute_error(Y_test, model.predict(X_test)))


# fit a RAM (no interactions)
title = "\nRAM (no interactions)"
ram_no_int = ExplainableBoostingRegressor(interactions=0)
fit_eval_gam(title, ram_no_int, X_train_new, Y_train, X_test_new, Y_test)

# fit a GAM (no interactions)
title = "\nGAM (no interactions)"
gam_no_int = ExplainableBoostingRegressor(interactions=0)
fit_eval_gam(title, gam_no_int, X_train, Y_train, X_test, Y_test)

# fit RAM (with interactions)
title = "\nRAM (with interactions)"
ram_int = ExplainableBoostingRegressor()
fit_eval_gam(title, ram_int, X_train_new, Y_train, X_test_new, Y_test)

# fit a GAM (with interactions)
title = "\nGAM (with interactions)"
gam_int = ExplainableBoostingRegressor()
fit_eval_gam(title, gam_int, X_train, Y_train, X_test, Y_test)

plot_subregions(splits=reg.important_splits, feat=0, feat_ram=[0, 1], X_train=X_train, model=model, model_jac=model_jac, gam=gam_no_int, ram=ram_no_int)
