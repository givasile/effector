import effector
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

# ## Preprocess the data
# load dataset
df = pd.read_csv("./../data/Bike-Sharing-Dataset/hour.csv")

# drop columns
df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)

for col_name in df.columns:
    print(
        "Feature: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(
            col_name,
            len(df[col_name].unique()),
            df[col_name].mean(),
            df[col_name].std(),
            df[col_name].min(),
            df[col_name].max(),
        )
    )


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


# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = preprocess(df)
col_names = X_df.columns.to_list()
target_name = "bike-rentals"

def split(X_df, Y_df):
    # data split
    X_train = X_df[: int(0.8 * len(X_df))]
    Y_train = Y_df[: int(0.8 * len(Y_df))]
    X_test = X_df[int(0.8 * len(X_df)) :]
    Y_test = Y_df[int(0.8 * len(Y_df)) :]
    return X_train, Y_train, X_test, Y_test


# train/test split
X_train, Y_train, X_test, Y_test = split(X_df, Y_df)


# Train - Evaluate - Explain a neural network
model = keras.Sequential(
    [
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(1),
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae", keras.metrics.RootMeanSquaredError()],
)
model.fit(X_train, Y_train, batch_size=512, epochs=5, verbose=1)
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


scale_x = {"mean": x_mean[3], "std": x_std[3]}
scale_y = {"mean": y_mean, "std": y_std}
scale_x_list = [{"mean": x_mean[i], "std": x_std[i]} for i in range(len(x_mean))]


# pdp with ICE
pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, nof_instances=1000, feature_names=col_names, target_name=target_name)
pdp.plot(feature=3, heterogeneity="ice", centering=True, scale_x=scale_x, scale_y=scale_y, nof_ice=300)


# d-PDP with ICE
d_pdp = effector.DerivativePDP(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, nof_instances=1000, feature_names=col_names, target_name=target_name)
d_pdp.fit(features=3, centering=False)
d_pdp.plot(feature=3, heterogeneity="ice", centering=False, scale_x=scale_x, scale_y=scale_y, nof_ice=500)

# Regional RHALE
regional_rhale = effector.RegionalRHALEBase(
    data=X_train.to_numpy(),
    model=model_forward,
    model_jac=model_jac,
    cat_limit=10,
    feature_names=col_names,
)

regional_rhale.fit(
    features=3,
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.1,
    binning_method="greedy",
    max_depth=2,
    nof_candidate_splits_for_numerical=5,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)
regional_rhale.describe_subregions(features=3, only_important=True, scale_x=scale_x_list)
regional_rhale.plot_first_level(feature=3, heterogeneity=True, centering=True, scale_x_per_feature=scale_x_list, scale_y=scale_y)


# Regional PDP
regional_pdp = effector.RegionalPDPBase(
    data=X_train.to_numpy(),
    model=model_forward,
    cat_limit=10,
    feature_names=col_names,
)

regional_pdp.fit(
    features=3,
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.1,
    max_depth=2,
    nof_candidate_splits_for_numerical=5,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)

regional_pdp.describe_subregions(features=3, only_important=True, scale_x=scale_x_list)
regional_pdp.plot_first_level(feature=3, heterogeneity=True, centering=True, scale_x_per_feature=scale_x_list, scale_y=scale_y)
