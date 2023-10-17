#!/usr/bin/env python
# coding: utf-8

# # Bike-Sharing Dataset
# 
# The Bike-Sharing Dataset contains the bike rentals for almost every hour over the period 2011 and 2012. 
# The dataset contains 14 features and we select the 11 features that are relevant to the prediction task. 
# The features contain information about the day, like the month, the hour, the day of the week, the day-type,
# and the weather conditions. 
# 
# Lets take a closer look

# In[1]:


import effector
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ## Preprocess the data

# In[2]:


# load dataset
df = pd.read_csv("./../data/Bike-Sharing-Dataset/hour.csv")

# drop columns
df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)


# In[3]:


for col_name in df.columns:
    print("Feature: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(df[col_name].unique()), df[col_name].mean(), df[col_name].std(), df[col_name].min(), df[col_name].max()))


# Feature Table:
# 
# | Feature      | Description                            | Value Range                                         |
# |--------------|----------------------------------------|-----------------------------------------------------|
# | season       | season                                 | 1: winter, 2: spring, 3: summer, 4: fall            |
# | yr           | year                                   | 0: 2011, 1: 2012                                    |
# | mnth         | month                                  | 1 to 12                                             |
# | hr           | hour                                   | 0 to 23                                             |
# | holiday      | whether the day is a holiday or not    | 0: no, 1: yes                                       |
# | weekday      | day of the week                        | 0: Sunday, 1: Monday, …, 6: Saturday                |
# | workingday   | whether the day is a working day or not | 0: no, 1: yes                                      |
# | weathersit   | weather situation                      | 1: clear, 2: mist, 3: light rain, 4: heavy rain     |
# | temp         | temperature                            | normalized, [0.02, 1.00]                            |
# | hum          | humidity                               | normalized, [0.00, 1.00]                            |
# | windspeed    | wind speed                             | normalized, [0.00, 1.00]                            |
# 
# 
# Target:
# 
# | Target       | Description                            | Value Range                                         |
# |--------------|----------------------------------------|-----------------------------------------------------|
# | cnt          | bike rentals per hour                  | [1, 977]                                            |
# 

# In[4]:


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


# In[5]:


def split(X_df, Y_df):
    # data split
    X_train = X_df[:int(0.8 * len(X_df))]
    Y_train = Y_df[:int(0.8 * len(Y_df))]
    X_test = X_df[int(0.8 * len(X_df)):]
    Y_test = Y_df[int(0.8 * len(Y_df)):]
    return X_train, Y_train, X_test, Y_test

# train/test split
X_train, Y_train, X_test, Y_test = split(X_df, Y_df)


# ## Fit a Neural Network
# 
# We train a deep fully-connected Neural Network with 3 hidden layers for \(20\) epochs. 
# The model achieves a mean absolute error on the test of about \(38\) counts.

# In[6]:


# Train - Evaluate - Explain a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=20, verbose=1)
model.evaluate(X_train, Y_train, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)


# ## Explain

# In[7]:


def model_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()

def model_forward(x):
    return model(x).numpy().squeeze()


# In[8]:


# rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
# binning_method = effector.binning_methods.Greedy(init_nof_bins=200, min_points_per_bin=0)
# rhale.fit(features=3, binning_method=binning_method)
# rhale.plot(feature=3, centering=True)


# rhale.method_args
# rhale.feature_effect

# # In[9]:
#
#
# ale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
# binning_method = effector.binning_methods.DynamicProgramming(max_nof_bins=40, min_points_per_bin=10)
# ale.fit(features=3, binning_method=binning_method)
# ale.plot(feature=3)
#
#
# # In[10]:
#
#
ale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac)
binning_method = effector.binning_methods.Fixed(nof_bins=100, min_points_per_bin=0, cat_limit=10)
breakpoint()
ale.fit(features=3, binning_method=binning_method)
ale.plot(feature=3)
#
#
# # In[10]:
#
#
#
#
