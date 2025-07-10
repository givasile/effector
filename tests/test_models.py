import effector

def test_models():
    # Test ConditionalInteraction model
    for model in [
        effector.models.ConditionalInteraction(),
        effector.models.DoubleConditionalInteraction(),
        effector.models.ConditionalInteraction4Regions(),
        effector.models.GeneralInteraction()
    ]:
        x = effector.datasets.IndependentUniform(dim=3, low=-1, high=1).generate_data(1000, seed=21)

        # Test predict method
        y_pred = model.predict(x)
        assert y_pred.shape == (1000,)

        # Test jacobian method
        jacobian = model.jacobian(x)
        assert jacobian.shape == (1000, 3)
