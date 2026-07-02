import effector


def test_models():
    models_and_dims = [
        (effector.models.ConditionalInteraction(), 3),
        (effector.models.DoubleConditionalInteraction(), 3),
        (effector.models.ConditionalInteraction4Regions(), 4),
        (effector.models.GeneralInteraction(), 3),
    ]
    for model, dim in models_and_dims:
        x = effector.datasets.IndependentUniform(dim=dim, low=-1, high=1).generate_data(
            1000, seed=21
        )

        # Test predict method
        y_pred = model.predict(x)
        assert y_pred.shape == (1000,)

        # Test jacobian method
        jacobian = model.jacobian(x)
        assert jacobian.shape == (1000, dim)
