# Adapters

## Summary

effector engines take a plain numpy-in / numpy-out callable. The adapters are
**facilitation only**: each one *returns* such a callable for a common model
object — you call the adapter, look at what it gave you, and make the final
pass into the constructor yourself. Nothing in effector auto-detects or
silently wraps a model.

```python
import effector

# sklearn regressor / pipeline
model = effector.adapters.from_sklearn(estimator)

# classifier: explain P(class = k), one explanation per class
model = effector.adapters.classifier_proba(clf, class_="yes")

# torch module (jacobian=True also builds the autograd jacobian for RHALE/DerPDP)
model = effector.adapters.from_torch(net)
model, model_jac = effector.adapters.from_torch(net, jacobian=True)

# the explicit handshake: probe the wrapper on two rows of your data
effector.adapters.check(model, X)

pdp = effector.PDP(X, model, schema=schema)   # the final pass is yours
```

Every returned callable also validates its own output on each call (shape
`(N,)`, numeric) and raises a named, actionable error instead of letting a
wrong shape travel into a kernel.

---

## API

### ::: effector.adapters
      options:
        show_root_heading: True
        show_symbol_type_toc: True
