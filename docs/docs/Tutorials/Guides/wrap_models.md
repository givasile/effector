# Wrap ML models into callables

Integration of `Effector` with common machine learning libraries is straightforward:

=== "scikit-learn"

    ``` python
    import effector
    from sklearn.ensemble import RandomForestRegressor

    # X = ... # input data
    # y = ... # target data

    model = RandomForestRegressor()
    model.fit(X, y)

    def model_callable(X):
      return model.predict(X)

    # global and regional pdp effect
    effector.PDP(X, model_callable).plot(feature=0)
    effector.RegionalPDP(X, model_callable).plot(feature=0, node_idx=0)

    # global and regional rhale effect
    effector.RHALE(X, model_callable).plot(feature=0)
    effector.RegionalRHALE(X, model_callable).plot(feature=0, node_idx=0)
    ```

=== "pytorch"

    ``` python
    import effector
    import torch
    import torch.nn as nn

    # X = ... # input data
    # y = ... # target data

    class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(D, 10)
        self.fc2 = nn.Linear(10, 1)

      def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    model = Net()

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(10):
      optimizer.zero_grad()
      y_pred = model(X)
      loss = criterion(y_pred, y)
      loss.backward()
      optimizer.step()

    def model_callable(X):
        return model(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    def model_jac_callable(X):
        X = torch.tensor(X, dtype=torch.float32)
        X.requires_grad = True
        y = model(X)
        return torch.autograd.grad(y, X)[0].numpy()

    # global and regional pdp effect
    effector.PDP(X, model_callable).plot(feature=0)
    effector.RegionalPDP(X, model_callable).plot(feature=0, node_idx=0)

    # global and regional rhale effect
    effector.RHALE(X, model_callable, model_jac_callable).plot(feature=0)
    effector.RegionalRHALE(X, model_callable, model_jac_callable).plot(feature=0, node_idx=0)
    ```

=== "tensorflow"

    ``` python
    import effector
    import tensorflow as tf

    # X = ... # input data
    # y = ... # target data
    
    # define model
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

    # train model
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10)

    def model_callable(X):
      return model.predict(X)

    def model_jac_callable(X):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            y = model(X)
        return tape.gradient(y, X).numpy()

    # global and regional pdp effect
    effector.PDP(X, model_callable).plot(feature=0)
    effector.RegionalPDP(X, model_callable).plot(feature=0, node_idx=0)

    # global and regional rhale effect
    effector.RHALE(X, model_callable, model_jac_callable).plot(feature=0)
    effector.RegionalRHALE(X, model_callable, model_jac_callable).plot(feature=0, node_idx=0)

    ```

=== "xgboost"

    ``` python
    import effector
    import xgboost as xgb

    # X = ... # input data
    # y = ... # target data

    model = xgb.XGBRegressor()
    model.fit(X, y)

    def model_callable(X):
      return model.predict(X)

    # Attention: Tree based models are not differentiable, so there is no way to compute the Jacobian.
    # global and regional pdp effect
    effector.PDP(X, model_callable).plot(feature=0)
    effector.RegionalPDP(X, model_callable).plot(feature=0, node_idx=0)

    # global and regional rhale effect
    effector.RHALE(X, model_callable).plot(feature=0)
    effector.RegionalRHALE(X, model_callable).plot(feature=0, node_idx=0)
    ```