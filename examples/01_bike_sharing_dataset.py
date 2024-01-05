#!/usr/bin/env python
# coding: utf-8

# # Bike-Sharing Dataset
# 
# The Bike-Sharing Dataset encompasses bike rentals recorded for nearly every hour between 2011 and 2012 in California, USA. The dataset contains 17,379 records, each with 14 features. The target variable is the number of bike rentals per hour. The dataset is available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).
# 
# Let's take a closer look! For running the code, you need to install `tensorflow`, `keras` and `pandas`.

# In[1]:


import effector
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ## Preprocess the data
# 
# From the original dataset, we drop the columns `instant`, `dteday`, `casual`, `registered`, and `atemp`:
#  
# - `instant` is a unique identifier for each record
# - `dteday` contains the date of the record
# - `casual` and `registered` contain the number of casual and registered users, respectively. 
# - `atemp` contains the "feels like" temperature in Celsius, which contains overlapping information with `temp`.

# In[2]:


# load dataset
df = pd.read_csv("./data/Bike-Sharing-Dataset/hour.csv")

# drop columns
df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)


# In[3]:


for col_name in df.columns:
    print("Feature: {:15}, unique: {:4d}, Mean: {:6.2f}, Std: {:6.2f}, Min: {:6.2f}, Max: {:6.2f}".format(col_name, len(df[col_name].unique()), df[col_name].mean(), df[col_name].std(), df[col_name].min(), df[col_name].max()))


# After dropping the redundant features we are left with the following design matrix. Please note that the features `temp`, `hum`, and `windspeed` are normalized to the range \([0, 1]\).
# 
# 
# | Feature       | Description                              | Values                                                |
# |---------------|------------------------------------------|-------------------------------------------------------|
# | `season`      | season                                   | 1: winter, 2: spring, 3: summer, 4: fall              |
# | `yr`          | year                                     | 0: 2011, 1: 2012                                      |
# | `mnth`        | month                                    | 1 to 12                                               |
# | `hr`          | hour                                     | 0 to 23                                               |
# | `holiday`     | whether the day is a holiday or not      | 0: no, 1: yes                                         |
# | `weekday`     | day of the week                          | 0: Sunday, 1: Monday, …, 6: Saturday                  |
# | `workingday`  | whether the day is a working day or not  | 0: no, 1: yes                                         |
# | `weathersit`  | weather situation                        | 1: clear, 2: mist, 3: light rain, 4: heavy rain       |
# | `temp`        | temperature                              | the quantity is normalized at the range: [0.02, 1.00] |
# | `hum`         | humidity                                 | the quantity is normalized at the range: [0.00, 1.00] |
# | `windspeed`   | wind speed                               | the quantity is normalized at the range: [0.00, 0.85]  |
# 
# 
# Target:
# 
# | Target        | Description                            | Value Range                                         |
# |---------------|----------------------------------------|-----------------------------------------------------|
# | `cnt`         | bike rentals per hour                  | [1, 977]                                            |
# 

# To fit a neural network, we standardize the features and the target variable. We split the data into a training and a test set. The training set contains \(80\%\) of the data, while the test set contains the remaining \(20\%\).

# In[4]:


def standardize(df):
    # shuffle
    df.sample(frac=1).reset_index(drop=True)

    # standardize X
    X_df = df.drop(["cnt"], axis=1)
    x_mean = X_df.mean()
    x_std = X_df.std()
    X_df = (X_df - X_df.mean()) / X_df.std()

    # standardize Y
    Y_df = df["cnt"]
    y_mean = Y_df.mean()
    y_std = Y_df.std()
    Y_df = (Y_df - Y_df.mean()) / Y_df.std()
    return X_df, Y_df, x_mean, x_std, y_mean, y_std

# shuffle and standarize all features
X_df, Y_df, x_mean, x_std, y_mean, y_std = standardize(df)


# In[5]:


from sklearn.model_selection import train_test_split

def split(X_df, Y_df, test_size=0.2, random_state=None):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=test_size, random_state=random_state)

    return X_train, Y_train, X_test, Y_test

split_seed = 42
X_train, Y_train, X_test, Y_test = split(X_df, Y_df, test_size=0.2, random_state=split_seed)


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

print("Evaluate on train data")
model.evaluate(X_train, Y_train, verbose=1)

print("Evaluate on test data")
model.evaluate(X_test, Y_test, verbose=1)


# The model achieves a root mean squared error of about \(0.25\) normalized units on the test set, which corresponds to about \(0.25 * 181 = 45.25\) counts.

# ## Explain

# Let's now use `effector` to explain the model. We first define the model's Jacobian and forward function. The Jacobian is not explicitly needed for any of `effector`'s methods, but it speeds the `RHALE` and `RegionalRHALE`. Therefore, we define it here using `tensorflow`'s automatic differentiation.

# In[7]:


def model_forward(x):
    return model(x).numpy().squeeze()

def model_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()


# In[8]:


feature_names = X_df.columns.to_list()
feature_names
target_name = Y_df.name
target_name
scale_x_list =[{"mean": x_mean.iloc[i], "std": x_std.iloc[i]} for i in range(len(x_mean))]
scale_y = {"mean": y_mean, "std": y_std}


# ## Global Effect (without heterogeneity)

# In[9]:


# feat = 3
# pdp = effector.PDP(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name)
# pdp.plot(feature=feat, centering=True, scale_x=scale_x_list[feat], scale_y=scale_y, show_avg_output=True)
#
#
# # In[10]:
#
#
# rhale = effector.RHALE(data=X_train.to_numpy(), model=model_forward, model_jac=model_jac, feature_names=feature_names, target_name=target_name)
# rhale.plot(feature=feat, centering=True, scale_x=scale_x_list[feat], scale_y=scale_y, show_avg_output=True)
#
#
# # In[11]:
#
#
# # shap = effector.SHAPDependence(data=X_train.to_numpy(), model=model_forward, feature_names=feature_names, target_name=target_name, nof_instances=100)
# # shap.fit(features=feat, smoothing_factor=5., centering=True)
# # shap.plot(feature=feat, centering=True, show_avg_output=True, scale_x=scale_x_list[feat], scale_y=scale_y)
#
#
# # ## Global Effect (with heterogeneity)
#
# # In[12]:
#
#
# feat = 3
# pdp.plot(feature=feat, centering=True, heterogeneity="ice", scale_x=scale_x_list[feat], scale_y=scale_y, show_avg_output=True)
#
#
# # In[13]:
#
#
# rhale.plot(feature=feat, centering=True, heterogeneity=True, scale_x=scale_x_list[feat], scale_y=scale_y, show_avg_output=True)
#
#
# # In[14]:
#
#
# # shap.plot(feature=feat, centering=True, heterogeneity="shap_values", show_avg_output=True, scale_x=scale_x_list[feat], scale_y=scale_y)
#
#
# # # Regional Effect
#
# # ### RegionalRHALE
#
# # In[15]:


# Regional RHALE
regional_rhale = effector.RegionalRHALE(
    data=X_train.to_numpy(),
    model=model_forward,
    model_jac=model_jac,
    nof_instances="all",
    feature_names=feature_names,
)

regional_rhale.fit(
    features=3,
    nof_candidate_splits_for_numerical=5
)


# # In[16]:
#
#
# regional_rhale.print_level_stats(features=3)
#
#
# # In[17]:
#
#
# scale_x_list
#
#
# # In[18]:
#
#
# regional_rhale.print_tree(features=3, scale_x_per_feature=scale_x_list)
#
#
# # In[19]:
#
#
# regional_rhale.describe_subregions(features=3, scale_x=scale_x_list, only_important=True)
#
#
# # In[20]:
#
#
# regional_rhale.plot(feature=3, node_idx=1, centering=True, scale_x=scale_x_list[feat], scale_y=scale_y)
#
#
# # In[21]:
#
#
# regional_rhale.plot(feature=3, node_idx=2, centering=True, scale_x=scale_x_list[feat], scale_y=scale_y)
#

# # ### RegionalPDP
#
# # In[44]:
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
#     max_depth=2,
#     nof_candidate_splits_for_numerical=5,
#     min_points_per_subregion=10,
#     candidate_conditioning_features="all",
#     split_categorical_features=True,
# )
#
#
# # In[45]:
#
#
# regional_pdp.describe_subregions(features=3, only_important=True, scale_x=scale_x_list)
#
#
# # In[46]:
#
#
# regional_pdp.print_tree(features=3)
#
#
# # In[50]:
#
#
# regional_pdp.plot(feature=3, node_idx=1, heterogeneity=True, centering=True, scale_x=scale_x_list[3], scale_y=scale_y)
#
#
# # In[51]:
#
#
# regional_pdp.plot(feature=3, node_idx=2, heterogeneity=True, centering=True, scale_x=scale_x_list[3], scale_y=scale_y)
#
#
# # In[ ]:
#
#
# ## Regional SHAP
#
#
# # In[55]:
#
#
# # regional_shap = effector.RegionalSHAP(
# #     data=X_train.to_numpy(),
# #     model=model_forward,
# #     cat_limit=10,
# #     feature_names=feature_names,
# #     nof_instances=50
# # )
# #
# # regional_shap.fit(
# #     features=3,
# #     heter_small_enough=0.1,
# #     heter_pcg_drop_thres=0.1,
# #     max_depth=1,
# #     nof_candidate_splits_for_numerical=5,
# #     min_points_per_subregion=10,
# #     candidate_conditioning_features=[6],
# #     split_categorical_features=True,
# # )
#
#
# # In[56]:
#
#
# # regional_shap.describe_subregions(features=3, only_important=True, scale_x=scale_x_list[3])
#
#
# # In[ ]:
#
#
#
#
