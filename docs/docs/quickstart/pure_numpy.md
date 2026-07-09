# effector is purely numpy based

`effector` speaks one language: **numpy**. Both things you hand it live in numpy
and nowhere else:

- **`data`** — a 2-D numeric numpy array, shape `(N, D)`.
- **`model`** (and the optional **`model_jac`**) — a `numpy → numpy` callable:
  `X: (N, D) → (N,)` for the model, `→ (N, D)` for the Jacobian.
- **metadata** — everything else (feature names, types, target name, scaling)
  travels in a single `schema=` argument.

That's the whole contract. If your model was trained in PyTorch, TensorFlow, or
on a pandas DataFrame, **you** wrap it into a `numpy → numpy` function — and
that wrapper is exactly where the framework-specific concerns (dtype, device,
batching) belong, because only you know them.

For the common cases you don't have to write the wrapper by hand:
[`effector.adapters`](../api_docs/api_adapters.md) *returns* the wrapper for
sklearn estimators, classifiers, and torch modules. It is facilitation only —
the adapter hands you a plain callable, and **you** make the final pass into
the constructor. Nothing is auto-detected:

```python
model = effector.adapters.from_sklearn(est)      # estimator -> plain callable
effector.adapters.check(model, X)                # the handshake: probe on 2 rows
effector.PDP(X, model, schema=schema)            # the final pass is yours
```

!!! tip "Why numpy?"
    A numpy array is the one format PyTorch, TensorFlow, JAX, scikit-learn,
    XGBoost, and plain Python functions all speak. Standing on numpy makes
    `effector` **fast** (the model is called as-is, with zero per-call
    conversion) and lets it explain **any** model without shipping a single
    framework adapter.

---

## The base case: numpy in, numpy model

Nothing to wrap — just call it.

```python
import numpy as np
import effector

X = np.random.uniform(-1, 1, (500, 3))            # (N, D) numpy
model = lambda A: A[:, 0] + A[:, 1] * (A[:, 2] > 0)   # numpy -> numpy

effector.PDP(X, model).plot(feature=0)
```

Add a `schema` whenever you want readable axes:

```python
schema = effector.Schema(feature_names=["age", "income", "region"])
effector.PDP(X, model, schema=schema).plot(feature=0)
```

## PyTorch

The adapter builds the wrapper for you — `eval()` mode, `no_grad`, device and
dtype handled:

```python
model = effector.adapters.from_torch(net)
effector.PDP(X, model).plot(feature=0)
```

Which is exactly this hand-written function — write it yourself whenever your
forward pass needs anything special (a head selection, custom batching):

```python
import torch

net.eval()
device = next(net.parameters()).device

def model(X):                                  # numpy (N, D) -> numpy (N,)
    with torch.no_grad():
        t = torch.as_tensor(X, dtype=torch.float32, device=device)
        return net(t).cpu().numpy().ravel()

effector.PDP(X, model).plot(feature=0)
```

!!! note "Bonus: exact gradients for RHALE / DerPDP"
    The Jacobian is *also* just `numpy → numpy`, so you can hand `effector` the
    **exact** autograd gradient instead of its numerical fallback.
    `from_torch(net, jacobian=True)` returns the pair:

    ```python
    model, model_jac = effector.adapters.from_torch(net, jacobian=True)
    effector.RHALE(X, model, model_jac).plot(feature=0)
    ```

    or by hand:

    ```python
    def model_jac(X):                          # numpy (N, D) -> numpy (N, D)
        t = torch.as_tensor(X, dtype=torch.float32,
                            device=device).requires_grad_(True)
        net(t).sum().backward()
        return t.grad.cpu().numpy()
    ```

## Classifiers

Class labels are not a regression surface, so `effector` explains a
**per-class probability** instead — one explanation per class:

```python
model = effector.adapters.classifier_proba(clf, class_="yes")   # P(class="yes")
effector.PDP(X, model, schema=schema).plot(feature=0)
```

Loop over `clf.classes_` if you want every class explained.

## TensorFlow / Keras

Keras `.predict` already takes numpy in and returns numpy out, so the wrapper is
almost a no-op:

```python
model = lambda X: net.predict(X, verbose=0).ravel()
effector.PDP(X, model).plot(feature=0)
```

## Starting from a pandas DataFrame

Most workflows begin with a DataFrame. Convert it **once** with
`effector.from_dataframe`, which reads column names, dtypes, and category
labels into `(X, schema)` — and never touches your model:

```python
import dataclasses
import effector

X_np, schema = effector.from_dataframe(df)     # numpy matrix + a populated Schema
print(schema.feature_types)                    # inspect the guesses...
# Schema is a frozen dataclass — override anything you don't like with replace:
schema = dataclasses.replace(schema, feature_types=["ordinal", "continuous", "nominal"])

effector.PDP(X_np, model, schema=schema).plot(feature=0)
```

!!! warning "Inspect the returned schema"
    `from_dataframe` infers types from dtypes, but an **integer** column is
    ambiguous — ordinal, continuous, or a label-encoded nominal are all
    plausible. `effector` emits a `UserWarning` for those guesses; check them
    and set `feature_types` explicitly when needed.

## A model that consumes a DataFrame (e.g. an sklearn Pipeline)

If your model *needs* a DataFrame (a `ColumnTransformer`/`OneHotEncoder` pipeline
trained on string columns), reconstruct the frame **inside your wrapper**:

```python
levels = ["clear", "mist", "rain"]

def model(X):                                  # numpy grid -> numpy predictions
    frame = pd.DataFrame({
        "hour":    X[:, 0],
        "temp":    X[:, 1],
        "weather": pd.Categorical.from_codes(
            np.clip(np.round(X[:, 2]), 0, 2).astype(int), levels),
    })
    return pipeline.predict(frame)

X_np, schema = effector.from_dataframe(df)
effector.PDP(X_np, model, schema=schema).plot(feature=0)
```

You own the reconstruction, so it's visible and testable — `effector` never
guesses how to call your model.

---

## What changed

Passing a DataFrame straight into a constructor no longer works — it raises,
with the fix in the message:

```python
effector.PDP(df, model)
# TypeError: effector is numpy-only: `data` must be a 2-D numeric numpy array,
#   not a pandas DataFrame. Convert it first:
#       X, schema = effector.from_dataframe(df)
#   and pass a numpy->numpy `model` ...
```

The one-line fix is `X, schema = effector.from_dataframe(df)`.

---

**See also:** the [API overview](./simple_api.md) for the full constructor
signature, and the [design contract](../guides/design.md) (R8 constructor contract,
R10 input contract) for the authoritative spec.
