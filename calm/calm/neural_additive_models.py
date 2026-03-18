import tensorflow as tf
import keras


def simple_nam(dim, subnetwork, classification=False):
    """Create a simple NAM model for regression or classification."""

    class Slicer(keras.layers.Layer):
        """Slices the input features into individual components."""

        def __init__(self, dim):
            super(Slicer, self).__init__()
            self.dim = dim

        def call(self, inputs):
            sliced_inputs = [inputs[:, i : i + 1] for i in range(self.dim)]
            return sliced_inputs

    class MaskedModel(keras.models.Model):
        """Defines the NAM with separate subnetworks for each feature."""

        def __init__(self, dim, subnetwork=None):
            super(MaskedModel, self).__init__()
            self.submodels = []

            if subnetwork is None:
                for _ in range(dim):
                    self.submodels.append(
                        keras.models.Sequential(
                            [
                                keras.layers.Dense(
                                    100, activation="relu", input_shape=(1,)
                                ),
                                keras.layers.Dense(100, activation="relu"),
                                keras.layers.Dense(10, activation="relu"),
                                keras.layers.Dense(1),
                            ]
                        )
                    )
            else:
                for _ in range(dim):
                    self.submodels.append(keras.models.clone_model(subnetwork))

        def call(self, inputs):
            """Compute each feature effect and apply masking."""
            return [
                model_i(inputs[0][i]) * inputs[1][:, i : i + 1]
                for i, model_i in enumerate(self.submodels)
            ]

    # Inputs
    x_symb = keras.layers.Input(shape=(dim,))
    mask_symb = keras.layers.Input(shape=(dim,))

    # Feature-wise NAM processing
    yy = Slicer(dim)(x_symb)
    yy = MaskedModel(dim, subnetwork)([yy, mask_symb])
    yy = keras.layers.concatenate(yy, axis=-1)
    yy = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(yy)

    if classification:
        yy = keras.layers.Dense(1, activation="sigmoid")(yy)

    model = keras.models.Model(inputs=[x_symb, mask_symb], outputs=yy)
    return model
