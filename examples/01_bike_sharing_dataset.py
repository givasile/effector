#!/usr/bin/env python
# coding: utf-8

# # Bike-Sharing Dataset
# 
# The Bike-Sharing Dataset contains the bike rentals for almost every hour over the period 2011 and 2012. 
# The dataset contains 14 features and we select the 11 features that are relevant to the prediction task. 
# The features contain information about the day, like the month, the hour, the day of the week, the day-type,
# and the weather conditions. 
# 
# Lets take a closer look!

# In[1]:


import effector
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)


# ## Preprocess the data

# In[2]:


# load dataset
df = pd.read_csv("./data/Bike-Sharing-Dataset/hour.csv")

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


import numpy as np

def split(X_df, Y_df):
    # shuffle indices
    indices = X_df.index.tolist()
    np.random.shuffle(indices)
    
    # data split
    train_size = int(0.8 * len(X_df))
    
    X_train = X_df.iloc[indices[:train_size]]
    Y_train = Y_df.iloc[indices[:train_size]]
    X_test = X_df.iloc[indices[train_size:]]
    Y_test = Y_df.iloc[indices[train_size:]]
    
    return X_train, Y_train, X_test, Y_test

# train/test split
X_train, Y_train, X_test, Y_test = split(X_df, Y_df)


# In[6]:


Y_train.std()


# ## Fit a Neural Network
# 
# We train a deep fully-connected Neural Network with 3 hidden layers for \(20\) epochs. 
# The model achieves a mean absolute error on the test of about \(38\) counts.

# In[7]:


# Train - Evaluate - Explain a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=10, verbose=1)
model.evaluate(X_train, Y_train, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)


# ## Explain

# In[8]:


def model_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()

def model_forward(x):
    return model(x).numpy().squeeze()


# In[9]:


scale_x = {"mean": x_mean.iloc[3], "std": x_std.iloc[3]}
scale_y = {"mean": y_mean, "std": y_std}
scale_x_list =[{"mean": x_mean.iloc[i], "std": x_std.iloc[i]} for i in range(len(x_mean))]
feature_names = X_df.columns.to_list()
target_name = "bike-rentals"


# # ## Global Effect
#
# # ### PDP
#
# # In[10]:
#
#
# pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=5000)
# pdp.plot(feature=3, centering=False, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)
#
#
# # In[11]:
#
#
# pdp.plot(feature=3, heterogeneity="std", centering=True, scale_x=scale_x, scale_y=scale_y)
#
#
# # In[12]:
#
#
# pdp.plot(feature=3, heterogeneity="ice", centering=True, scale_x=scale_x, scale_y=scale_y, nof_ice=300)
#
#
# # ### RHALE
#
# # In[13]:
#
#
# rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
# binning_method = effector.binning_methods.Greedy(init_nof_bins=100, min_points_per_bin=10, discount=1., cat_limit=10)
# rhale.fit(features=3, binning_method=binning_method)
# rhale.plot(feature=3, centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)
#
#
# # In[14]:
#
#
# rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
# binning_method = effector.binning_methods.Greedy(init_nof_bins=100, min_points_per_bin=10, discount=1., cat_limit=10)
# rhale.fit(features=3, binning_method=binning_method)
# rhale.plot(feature=3, heterogeneity="std", centering=True, scale_x=scale_x, scale_y=scale_y, show_avg_output=True)


# # Regional Effects

# ### RegionalRHALE

# In[15]:


# Regional RHALE
regional_rhale = effector.RegionalRHALE(
    data=X_train.to_numpy(),
    model=model_forward,
    model_jac=model_jac,
    cat_limit=10,
    feature_names=feature_names,
    nof_instances="all"
)

binning_method = effector.binning_methods.Greedy(init_nof_bins=100, min_points_per_bin=10, discount=0., cat_limit=10)
regional_rhale.fit(
    features=3,
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.2,
    binning_method=binning_method,
    max_depth=2,
    nof_candidate_splits_for_numerical=10,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)


# In[16]:

regional_rhale.show_partitioning(features=3, only_important=True, scale_x_list=scale_x_list)
# regional_rhale.describe_subregions(features=3, scale_x_list=scale_x_list)

# In[ ]:


# regional_rhale.plot_first_level(feature=3, heterogeneity=True, centering=True, scale_x_per_feature=scale_x_list, scale_y=scale_y, show_avg_output=True)


# # ### RegionalPDP
#
# # In[ ]:
#
#
# regional_pdp = effector.RegionalPDP(
#     data=X_train.to_numpy(),
#     model=model_forward,
#     cat_limit=10,
#     feature_names=feature_names,
# )
#
# regional_pdp.fit(
#     features=3,
#     heter_small_enough=0.1,
#     heter_pcg_drop_thres=0.1,
#     max_split_levels=2,
#     nof_candidate_splits_for_numerical=5,
#     min_points_per_subregion=10,
#     candidate_conditioning_features="all",
#     split_categorical_features=True,
# )
#
#
# # In[ ]:
#
#
# regional_pdp.describe_subregions(features=3, only_important=True, scale_x=scale_x_list)
#
#
# # In[ ]:
#
#
# regional_pdp.plot_first_level(feature=3, heterogeneity=True, centering=True, scale_x_per_feature=scale_x_list, scale_y=scale_y, show_avg_output=True)
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
#
#
