import tensorflow as tf
import xgboost as xgb
from abc import abstractmethod, ABC
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class BlackBoxModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    def predict(self, X):
        raise NotImplementedError()

    def jac(self, X):
        raise NotImplementedError()


class KerasBlackBoxClassifier(BlackBoxModel):
    def __init__(
        self,
        input_dim=None,
        hidden_layers=[50, 50],
        activation="sigmoid",
        learning_rate=0.001,
    ):
        self.model = keras.Sequential()
        if input_dim is not None:
            self.model.add(keras.layers.Input(shape=(input_dim,)))
        for units in hidden_layers:
            self.model.add(keras.layers.Dense(units, activation=activation))
        self.model.add(keras.layers.Dense(1, activation="sigmoid"))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

    def fit(self, X, y, batch_size=200, epochs=200):
        self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
        )

    def forward(self, X):
        return self.model(X).numpy().squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def jac(self, X):
        x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            pred = self.model(x_tensor)
            grads = t.gradient(pred, x_tensor)
        return grads.numpy()


class KerasBlackBoxRegressor(BlackBoxModel):
    def __init__(
        self,
        input_dim=None,
        hidden_layers=[50, 50],
        activation="relu",
        learning_rate=0.001,
        verbose=1,
    ):
        self.model = keras.Sequential()

        if input_dim is not None:
            self.model.add(keras.layers.Input(shape=(input_dim,)))
        for units in hidden_layers:
            self.model.add(keras.layers.Dense(units, activation=activation))

        self.model.add(
            keras.layers.Dense(1, activation=None)
        )  # No activation for regression output

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        self.verbose = verbose

    def fit(self, X, y, batch_size=200, epochs=200):
        self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=self.verbose,
        )

    def forward(self, X):
        return self.model(X).numpy().squeeze()

    def predict(self, X):
        return self.forward(X)

    def jac(self, X):
        x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            pred = self.model(x_tensor)
            grads = t.gradient(pred, x_tensor)
        return grads.numpy()


class RFClassifier(BlackBoxModel):
    def __init__(
        self,
    ):
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X, y)

    def forward(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class RFRegressor(BlackBoxModel):
    def __init__(
        self,
    ):
        self.model = None

    def fit(self, X, y):
        self.model = self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )

        self.model.fit(X, y)

    def forward(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.forward(X)


class XGBClassifier(BlackBoxModel):
    def __init__(
        self,
    ):
        self.model = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=300,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def forward(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class XGBRegressor(BlackBoxModel):
    def __init__(
        self,
    ):
        self.model = xgb.XGBRegressor(
            learning_rate=0.1, n_estimators=300, n_jobs=-1, random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def forward(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.forward(X)


CLASSIFICATION_DATASETS_BLACKBOX = {
    "DNNClassifier": KerasBlackBoxClassifier,
    "RFClassifier": RFClassifier,
    "XGBClassifier": XGBClassifier,
}
REGRESSION_DATASETS_BLACKBOX = {
    "DNNRegressor": KerasBlackBoxRegressor,
    "RFRegressor": RFRegressor,
    "XGBRegressor": XGBRegressor,
}
