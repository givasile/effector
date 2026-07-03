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


def test_jacobian_matches_finite_differences():
    """§3.5: each model's analytic jacobian agrees with finite differences."""
    import numpy as np

    from effector import utils

    models_and_dims = [
        (effector.models.ConditionalInteraction(), 3),
        (effector.models.DoubleConditionalInteraction(), 3),
        (effector.models.ConditionalInteraction4Regions(), 4),
        (effector.models.GeneralInteraction(), 3),
    ]
    for model, dim in models_and_dims:
        x = effector.datasets.IndependentUniform(dim=dim, low=-1, high=1).generate_data(
            200, seed=21
        )
        jac_analytic = model.jacobian(x)
        jac_numeric = utils.compute_jacobian_numerically(model.predict, x)
        np.testing.assert_allclose(
            jac_analytic, jac_numeric, atol=1e-4, err_msg=type(model).__name__
        )
