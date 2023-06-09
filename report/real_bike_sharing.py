import sys, os
import timeit
sys.path.append(os.path.dirname(os.getcwd()))
import pythia
import numpy as np
import pythia.interaction as interaction
import pythia.regions as regions
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor
import sklearn.metrics as metrics

def preprocess(df):
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


def split(X_df, Y_df):
    # data split
    X_train = X_df[:int(0.8 * len(X_df))]
    Y_train = Y_df[:int(0.8 * len(Y_df))]
    X_test = X_df[int(0.8 * len(X_df)):]
    Y_test = Y_df[int(0.8 * len(Y_df)):]
    return X_train, Y_train, X_test, Y_test


def plot_subregions_rhale(feat, feature, type, position, X_train, model, model_jac):
    # Regional Plot
    if type == "cat":
        rhale = pythia.RHALE(data=X_train.to_numpy(), model=model, model_jac=model_jac)
        rhale_1 = pythia.RHALE(data=X_train[X_train.iloc[:, feature] == position].to_numpy(), model=model, model_jac=model_jac)
        rhale_2 = pythia.RHALE(data=X_train[X_train.iloc[:, feature] != position].to_numpy(), model=model, model_jac=model_jac)
    else:
        rhale = pythia.RHALE(data=X_train.to_numpy(), model=model, model_jac=model_jac)
        rhale_1 = pythia.RHALE(data=X_train[X_train.iloc[:, feature] <= position].to_numpy(), model=model, model_jac=model_jac)
        rhale_2 = pythia.RHALE(data=X_train[X_train.iloc[:, feature] > position].to_numpy(), model=model, model_jac=model_jac)

    def plot(rhale):
        binning_method = pythia.binning_methods.Fixed(nof_bins=100)
        rhale.fit(features=feat, binning_method=binning_method, centering="zero_integral")
        scale_x = {"mean": x_mean.iloc[feat], "std": x_std.iloc[feat]}
        scale_y = {"mean": 0, "std": y_std}
        rhale.plot(feature=feat, uncertainty=None, scale_x=scale_x, scale_y=scale_y)
        plt.show()

    # plot global and regionals
    plot(rhale)
    plot(rhale_1)
    plot(rhale_2)


def plot_subregions_pdp_ice(feat, features, types, positions, X_train, model):
    # Regional Plot
    if types[0] == "categorical":
        pdp = pythia.pdp.PDPwithICE(data=X_train.to_numpy(), model=model)
        pdp_1 = pythia.pdp.PDPwithICE(data=X_train[X_train.iloc[:, features[0]] == positions[0]].to_numpy(), model=model)
        pdp_2 = pythia.pdp.PDPwithICE(data=X_train[X_train.iloc[:, features[0]] != positions[0]].to_numpy(), model=model)
    else:
        pdp = pythia.pdp.PDPwithICE(data=X_train.to_numpy(), model=model)
        pdp_1 = pythia.pdp.PDPwithICE(data=X_train[X_train.iloc[:, features[0]] <= positions[0]].to_numpy(), model=model)
        pdp_2 = pythia.pdp.PDPwithICE(data=X_train[X_train.iloc[:, features[0]] > positions[0]].to_numpy(), model=model)

    def plot(pdp):
        pdp.fit(features=feat, centering=False)
        scale_x = {"mean": x_mean.iloc[feat], "std": x_std.iloc[feat]}
        scale_y = {"mean": 0, "std": y_std}
        pdp.plot(feature=feat, scale_x=scale_x, scale_y=scale_y)
        plt.show()

    # plot global and regionals
    plot(pdp)
    plot(pdp_1)
    plot(pdp_2)

# load dataset
df = pd.read_csv("./../data/Bike-Sharing-Dataset/hour.csv")

# drop columns
df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)
print(df.columns)

# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = preprocess(df)

# train/test split
X_train, Y_train, X_test, Y_test = split(X_df, Y_df)

cols = df.columns

# # train model
# lin_model = linear_model.LinearRegression()
# lin_model.fit(X_train, Y_train)
# # root mean squared error
# print("RMSE: ", metrics.mean_squared_error(Y_train, lin_model.predict(X_train), squared=False)*y_std)
# print("R2 score: ", lin_model.score(X_train, Y_train))
#
# # Same on the test set
# print("RMSE: ", metrics.mean_squared_error(Y_test, lin_model.predict(X_test), squared=False)*y_std)
# print("R2 score: ", lin_model.score(X_test, Y_test))
#
# def lin_model_jac(x):
#     return np.ones_like(x) * lin_model.coef_
#
# # Explain
# ale = pythia.ALE(data=X_train.to_numpy(), model=lin_model.predict)
# binning_method = pythia.binning_methods.Fixed(nof_bins=30)
# ale.fit(features="all")
# ale.plot(feature=8)
#
# rhale = pythia.RHALE(data=X_train.to_numpy(), model=lin_model.predict, model_jac=lin_model_jac)
# binning_method = pythia.binning_methods.DynamicProgramming(max_nof_bins=30, min_points_per_bin=10)
# rhale.fit(features="all", binning_method=binning_method)
# rhale.plot(feature=8)
#
# # pdp = pythia.PDP(data=X_train.to_numpy(), model=lin_model.predict)
# # pdp.plot(feature=8)
#
#
#
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation="relu"),
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

model.compile(optimizer="adam", loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, epochs=3, verbose=1)
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

# find regions
reg = pythia.regions.Regions(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, cat_limit=25)
reg.find_splits(nof_levels=2, nof_candidate_splits=10, method="rhale")
opt_splits = reg.choose_important_splits(0.2)

transf = pythia.regions.DataTransformer(splits=opt_splits)
X_train_new = transf.transform(X_train.to_numpy())
X_test_new = transf.transform(X_test.to_numpy())

# train model
# fit a GAM to the transformed data
gam_subspaces = ExplainableBoostingRegressor(interactions=0)
gam_subspaces.fit(X_train_new, Y_train)
y_train_pred = gam_subspaces.predict(X_train_new)
print(gam_subspaces.score(X_test_new, Y_test))
print("RMSE: ", metrics.mean_squared_error(Y_train, y_train_pred, squared=False))
print("RMSE: ", metrics.mean_squared_error(Y_test, gam_subspaces.predict(X_test_new), squared=False))

# fit an EBM
ebm = ExplainableBoostingRegressor(interactions=0)
ebm.fit(X_train, Y_train)
y_train_pred = ebm.predict(X_train)
print(ebm.score(X_test, Y_test))
print("RMSE: ", metrics.mean_squared_error(Y_train, y_train_pred, squared=False))
print("RMSE: ", metrics.mean_squared_error(Y_test, ebm.predict(X_test), squared=False))


# fit GAM with interactions to the initial data
ebm_interactions = ExplainableBoostingRegressor()
ebm_interactions.fit(X_train, Y_train)
y_train_pred = ebm_interactions.predict(X_train)
print(ebm_interactions.score(X_test, Y_test))
print("RMSE: ", metrics.mean_squared_error(Y_train, y_train_pred, squared=False))
print("RMSE: ", metrics.mean_squared_error(Y_test, ebm_interactions.predict(X_test), squared=False))


# fit GAM with interactions to the transformed data
ebm_interactions = ExplainableBoostingRegressor()
ebm_interactions.fit(X_train_new, Y_train)
y_train_pred = ebm_interactions.predict(X_train_new)
print(ebm_interactions.score(X_test_new, Y_test))
print("RMSE: ", metrics.mean_squared_error(Y_train, y_train_pred, squared=False))
print("RMSE: ", metrics.mean_squared_error(Y_test, ebm_interactions.predict(X_test_new), squared=False))
