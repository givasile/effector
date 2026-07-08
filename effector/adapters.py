"""Model adapters — facilitation only, the final pass is yours.

effector engines take a plain numpy-in / numpy-out callable (R10). These
helpers *return* such callables for common model objects — they never run
automatically: you call the adapter, look at what it gave you, and pass it to
the constructor yourself. Nothing in effector auto-detects or silently wraps
a model.

Every returned callable validates its own output on each call (shape ``(N,)``,
numeric) and raises a named, actionable error instead of letting a wrong shape
travel into a kernel. `check` is the explicit handshake: probe a callable on
two rows of your data before building an engine.

sklearn / torch are never imported at module import time — only inside the
adapter that needs them.
"""

import typing

import numpy as np


def _validate_output(y, n: int, where: str) -> np.ndarray:
    """Coerce a model output to a numeric ``(n,)`` array or die loudly."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.shape != (n,):
        raise ValueError(
            f"{where}: model output has shape {y.shape}, expected ({n},). "
            f"effector explains a single scalar output per row; select one "
            f"column/head in your wrapper (for classifiers, use "
            f"effector.adapters.classifier_proba)."
        )
    if y.dtype.kind not in "fiub":
        raise ValueError(
            f"{where}: model output has non-numeric dtype {y.dtype}; "
            f"return a numeric array."
        )
    return y.astype(float, copy=False)


def from_sklearn(estimator) -> typing.Callable:
    """Wrap an sklearn-style regressor into a numpy->numpy callable.

    Returns a function ``f(X: (N, D) ndarray) -> (N,) ndarray`` that calls
    ``estimator.predict`` and validates the output shape on every call.
    Classifiers (anything with `predict_proba`) are rejected — class labels
    are not a regression surface; explain a probability instead:

        model = effector.adapters.classifier_proba(clf, class_=1)

    Args:
        estimator: a fitted object with ``.predict(X) -> (N,)`` (an sklearn
            regressor or pipeline; anything duck-typing it works).

    Returns:
        a plain callable — pass it to an engine yourself:
        ``effector.PDP(X, effector.adapters.from_sklearn(est), schema=schema)``
    """
    if not hasattr(estimator, "predict"):
        raise TypeError(
            "adapters.from_sklearn: object has no .predict method; "
            "pass a fitted sklearn-style estimator (or wrap your model as a "
            "numpy->numpy callable yourself)."
        )
    if hasattr(estimator, "predict_proba"):
        raise ValueError(
            "adapters.from_sklearn: this estimator is a classifier "
            "(has predict_proba). Explaining predicted labels is not "
            "meaningful; explain a per-class probability instead:\n"
            "    model = effector.adapters.classifier_proba(estimator, class_=...)"
        )

    def model(X: np.ndarray) -> np.ndarray:
        return _validate_output(
            estimator.predict(X), len(X), "adapters.from_sklearn"
        )

    return model


def classifier_proba(estimator, class_=1) -> typing.Callable:
    """Wrap a classifier into a numpy->numpy callable for one class' probability.

    The per-class probability is effector's classification story: the returned
    callable computes ``predict_proba(X)[:, k]`` — the surface P(class = k) —
    which every engine can explain like any regression output. One explanation
    per class; loop over classes yourself if you want them all.

    `class_` is resolved against ``estimator.classes_`` at wrap time:
    label-match first, positional-index fallback for plain ints.

    Args:
        estimator: a fitted object with ``.predict_proba(X) -> (N, C)`` and
            ``.classes_``.
        class_: the class to explain — a label from ``classes_`` or a
            positional column index.

    Returns:
        a plain callable ``f(X) -> (N,)`` of probabilities in [0, 1].
    """
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            "adapters.classifier_proba: object has no .predict_proba method; "
            "for a regressor use effector.adapters.from_sklearn."
        )
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.size == 0:
        raise TypeError(
            "adapters.classifier_proba: estimator has no .classes_; "
            "is it fitted?"
        )
    matches = np.flatnonzero(classes == class_)
    if matches.size == 1:
        col = int(matches[0])
    elif isinstance(class_, (int, np.integer)) and 0 <= class_ < classes.size:
        col = int(class_)
    else:
        raise ValueError(
            f"adapters.classifier_proba: class_={class_!r} is neither a label "
            f"in classes_={classes.tolist()} nor a valid column index."
        )

    def model(X: np.ndarray) -> np.ndarray:
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim != 2 or proba.shape[0] != len(X):
            raise ValueError(
                f"adapters.classifier_proba: predict_proba returned shape "
                f"{proba.shape}, expected ({len(X)}, n_classes)."
            )
        return _validate_output(
            proba[:, col], len(X), "adapters.classifier_proba"
        )

    return model


def from_torch(module, device=None, jacobian: bool = False):
    """Wrap a torch module into numpy->numpy callable(s).

    The forward wrapper puts the module in eval mode, runs under ``no_grad``,
    and moves tensors to/from `device`. With ``jacobian=True`` it also returns
    a jacobian callable built on autograd (for RHALE / DerPDP), computed via
    the sum-backward trick — valid because each row's output depends only on
    that row's input.

    torch is imported lazily; effector gains no torch dependency.

    Args:
        module: a ``torch.nn.Module`` whose forward maps ``(N, D)`` to
            ``(N,)`` or ``(N, 1)``.
        device: torch device for inference; default = the module's own.
        jacobian: also build ``jac(X) -> (N, D)`` via autograd.

    Returns:
        ``model`` — or ``(model, model_jac)`` when ``jacobian=True``. Pass
        them to an engine yourself.
    """
    import torch

    module.eval()
    if device is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    def model(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
            out = module(t)
        return _validate_output(
            out.detach().cpu().numpy(), len(X), "adapters.from_torch"
        )

    if not jacobian:
        return model

    def model_jac(X: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        t.requires_grad_(True)
        out = module(t)
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.ravel()
        if out.shape != (len(X),):
            raise ValueError(
                f"adapters.from_torch: module output has shape "
                f"{tuple(out.shape)}, expected ({len(X)},)."
            )
        out.sum().backward()
        return t.grad.detach().cpu().numpy().astype(float)

    return model, model_jac


def check(model, X, model_jac=None) -> None:
    """Probe a model wrapper on two rows of your data — the explicit handshake.

    Call this right before constructing an engine; it is the only place a
    model call happens outside the engines, and *you* trigger it. Raises with
    a precise message if the callable is not numpy-in / numpy-out with the
    shapes effector expects; returns None when everything checks out.

    Args:
        model: the callable you are about to pass to an engine.
        X: your data (2-D numpy array); only ``X[:2]`` is evaluated.
        model_jac: optional jacobian callable; probed for shape ``(2, D)``.
    """
    if not callable(model):
        raise TypeError(
            f"adapters.check: model is not callable (got "
            f"{type(model).__name__}); wrap it first — see "
            f"effector.adapters.from_sklearn / classifier_proba / from_torch."
        )
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"adapters.check: X must be 2-D, got {X.ndim}-D.")
    probe = X[:2]
    try:
        y = model(probe)
    except Exception as e:
        raise type(e)(
            f"adapters.check: model raised on a 2-row numpy probe — it is "
            f"not numpy-in/numpy-out as effector requires. Original error: {e}"
        ) from e
    _validate_output(y, len(probe), "adapters.check")
    if model_jac is not None:
        if not callable(model_jac):
            raise TypeError(
                f"adapters.check: model_jac is not callable (got "
                f"{type(model_jac).__name__})."
            )
        jac = np.asarray(model_jac(probe))
        if jac.shape != probe.shape:
            raise ValueError(
                f"adapters.check: model_jac output has shape {jac.shape}, "
                f"expected {probe.shape} (one gradient row per input row)."
            )
