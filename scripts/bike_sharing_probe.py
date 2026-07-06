import timeit
import effector
import keras
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

bike_sharing = effector.datasets.BikeSharing(pcg_train=0.8)
X_train, Y_train = bike_sharing.x_train, bike_sharing.y_train
X_test, Y_test = bike_sharing.x_test, bike_sharing.y_test


#%%
# Define and train a neural network
model = keras.Sequential([
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae", keras.metrics.RootMeanSquaredError()])
model.fit(X_train, Y_train, batch_size=512, epochs=20, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)

#%%
def predict(x):
    return model(x).numpy().squeeze()

#%%
def run_shap(backend):
    tic = timeit.default_timer()
    g_method = effector.ShapDP(
        X_test,
        predict,
        nof_instances=200, # use only 200 instances for the sake of speed
        feature_names=bike_sharing.feature_names,
        target_name=bike_sharing.target_name,
        backend=backend
    )
    g_method.fit(features="all", budget=512)
    g_method.plot(
        feature=3,
        scale_x={"mean": bike_sharing.x_test_mu[3], "std": bike_sharing.x_test_std[3]},
        scale_y={"mean": bike_sharing.y_test_mu, "std": bike_sharing.y_test_std},
        centering=True,
        show_avg_output=True,
        y_limits=[-200, 1000]
    )
    toc = timeit.default_timer()
    print(backend, ": ", toc - tic)

# with shap
run_shap("shap")
# Output: PermutationExplainer explainer: 201it [00:52,  3.03it/s]
# Output: shap :  8.19535578200157

# with shapiq
run_shap("shapiq")
# Error log
# Traceback (most recent call last):
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3577, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-19-dccbeda9397d>", line 28, in <module>
#     run_shap("shapiq")
#   File "<ipython-input-19-dccbeda9397d>", line 11, in run_shap
#     g_method.fit(features="all", budget=512)
#   File "/home/givasile/github/packages/effector/effector/global_effect_shap.py", line 250, in fit
#     self.feature_effect["feature_" + str(s)] = self._fit_feature(
#   File "/home/givasile/github/packages/effector/effector/global_effect_shap.py", line 137, in _fit_feature
#     explanations = explainer.explain_X(data, budget=budget)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/explainer/_base.py", line 147, in explain_X
#     ivs.append(self.explain(X[i, :], **kwargs))
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/explainer/_base.py", line 100, in explain
#     explanation = self.explain_function(x=x, *args, **kwargs)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/explainer/tabular.py", line 208, in explain_function
#     interaction_values = self._approximator(budget=budget, game=imputer)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/approximator/_base.py", line 113, in __call__
#     return self.approximate(budget, game, *args, **kwargs)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/approximator/permutation/sv.py", line 58, in approximate
#     empty_val = float(game(np.zeros(self.n, dtype=bool))[0])
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/games/base.py", line 279, in __call__
#     values = self.value_function(coalitions)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/games/imputer/marginal_imputer.py", line 108, in value_function
#     predictions = self.predict(imputed_data)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/games/imputer/base.py", line 100, in predict
#     return self._predict_function(self.model, x)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/explainer/utils.py", line 175, in predict_callable
#     return model(data)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/explainer/_base.py", line 156, in predict
#     return self._shapiq_predict_function(self.model, x)
#   File "/home/givasile/miniconda3/envs/effector-dev/lib/python3.10/site-packages/shapiq/explainer/utils.py", line 167, in _predict_function_with_class_index
#     elif predictions.shape[1] == 1:
# IndexError: tuple index out of range


