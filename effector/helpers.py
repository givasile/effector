import re
import typing

import numpy as np

BIG_M = 1e8
EPS = 1e-8


def prep_features(feat: typing.Union[str, int, list], D: int) -> list:
    """Normalize the `features` argument to a list of valid feature indices.

    Accepts an `int`, a `list` of ints, or the string `"all"`; anything else
    raises `TypeError`, and out-of-range indices raise `ValueError` (R9).
    """
    if isinstance(feat, str):
        if feat != "all":
            raise ValueError(
                f"Invalid features argument: {feat!r}; the only valid string is 'all'"
            )
        return [i for i in range(D)]

    if isinstance(feat, bool):
        raise TypeError(f"Invalid features argument of type bool: {feat!r}")
    if isinstance(feat, int):
        feat = [feat]
    if not isinstance(feat, list):
        raise TypeError(
            f"Invalid features argument of type {type(feat).__name__}: {feat!r}; "
            "use an int, a list of ints, or 'all'"
        )

    for f in feat:
        if isinstance(f, bool) or not isinstance(f, int):
            raise TypeError(f"Feature indices must be ints, got {f!r}")
        if not 0 <= f < D:
            raise ValueError(
                f"Feature index {f} out of range for data with {D} features"
            )
    return feat


def prep_conditioning_features(
    ccf: typing.Union[str, list], feature: int, D: int
) -> list:
    """Normalize the `candidate_conditioning_features` argument: `"all"` means
    every feature except the feature of interest."""
    if isinstance(ccf, str):
        if ccf != "all":
            raise ValueError(
                f"Invalid candidate_conditioning_features argument: {ccf!r}; "
                "the only valid string is 'all'"
            )
        return [i for i in range(D) if i != feature]
    return ccf


def prep_centering(centering: typing.Union[bool, str]) -> typing.Union[bool, str]:
    """Normalize the `centering` argument to `False | "zero_integral" | "zero_start"`."""
    if isinstance(centering, bool):
        return "zero_integral" if centering else False
    if isinstance(centering, str):
        if centering not in ["zero_start", "zero_integral"]:
            raise ValueError(
                f"Invalid centering: {centering!r}; "
                "valid options are False, True, 'zero_integral', 'zero_start'"
            )
        return centering
    raise TypeError(
        f"Invalid centering of type {type(centering).__name__}: {centering!r}"
    )


def prep_confidence_interval(
    confidence_interval: typing.Union[bool, str],
) -> typing.Union[bool, str]:
    """Normalize the heterogeneity/confidence-interval plot argument."""
    if isinstance(confidence_interval, bool):
        return "std" if confidence_interval else False
    if isinstance(confidence_interval, str):
        if confidence_interval not in ["std", "std_err", "ice", "shap_values"]:
            raise ValueError(
                f"Invalid heterogeneity option: {confidence_interval!r}; "
                "valid options are False, True, 'std', 'std_err', 'ice', 'shap_values'"
            )
        return confidence_interval
    raise TypeError(
        f"Invalid heterogeneity option of type "
        f"{type(confidence_interval).__name__}: {confidence_interval!r}"
    )


def axis_limits_from_data(data: np.ndarray) -> np.ndarray:
    """Compute axis limits from data."""
    D = data.shape[1]
    axis_limits = np.zeros([2, D])
    for d in range(D):
        axis_limits[0, d] = data[:, d].min()
        axis_limits[1, d] = data[:, d].max()
    return axis_limits


def prep_nof_instances(
    nof_instances: typing.Union[int, str],
    N: int,
    random_state: typing.Optional[int] = 21,
) -> typing.Tuple[int, np.ndarray]:
    """Prepares the argument nof_instances

    Args
    ---
        nof_instances (int or str): The number of instances to use for the explanation
        N (int): The number of instances in the dataset
        random_state (int or None): seed for the subsampling draw; `None` for
            fresh randomness. Every sampling site creates its own
            `np.random.default_rng(random_state)` — if a future site draws the
            same shape from the same population, switch to spawned child seeds.

    Returns
    ---
        nof_instances (int): The number of instances to use for the explanation
        indices (np.ndarray): The indices of the instances to use for the explanation
    """
    if isinstance(nof_instances, str):
        if nof_instances != "all":
            raise ValueError(
                f"Invalid nof_instances: {nof_instances!r}; "
                "the only valid string is 'all'"
            )
        nof_instances = N
    elif isinstance(nof_instances, bool) or not isinstance(nof_instances, int):
        raise TypeError(
            f"Invalid nof_instances of type "
            f"{type(nof_instances).__name__}: {nof_instances!r}"
        )

    indices = (
        np.random.default_rng(random_state).choice(N, nof_instances, replace=False)
        if nof_instances < N
        else np.arange(N)
    )
    return nof_instances, indices


def prep_data(
    data: np.ndarray,
    axis_limits: typing.Optional[np.ndarray] = None,
    nof_instances: typing.Union[int, str] = 10_000,
    data_effect: typing.Optional[np.ndarray] = None,
    random_state: typing.Optional[int] = 21,
) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray], np.ndarray, int, np.ndarray]:
    """Shared data preprocessing for every effect class (global, regional, facade):

    (i) if `axis_limits` is given, validate it and drop the points outside;
        otherwise infer the limits from the data;
    (ii) subsample `nof_instances` from what remains, seeded by `random_state`
        (`None` for fresh randomness).

    `data_effect` (the Jacobian on `data`), when given, is kept row-aligned with
    `data` through both steps.

    Returns:
        (data, data_effect, axis_limits, nof_instances, indices)
    """
    if data.ndim != 2:
        raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")

    if axis_limits is not None:
        if axis_limits.shape != (2, data.shape[1]):
            raise ValueError(
                f"axis_limits must have shape (2, {data.shape[1]}), "
                f"got {axis_limits.shape}"
            )
        if not np.all(axis_limits[0, :] <= axis_limits[1, :]):
            raise ValueError(
                "axis_limits lower bounds must not exceed the upper bounds"
            )

        accept_indices = indices_within_limits(data, axis_limits)
        data = data[accept_indices, :]
        data_effect = (
            data_effect[accept_indices, :] if data_effect is not None else None
        )
    else:
        axis_limits = axis_limits_from_data(data)

    nof_instances, indices = prep_nof_instances(
        nof_instances, data.shape[0], random_state
    )
    data = data[indices, :]
    data_effect = data_effect[indices, :] if data_effect is not None else None

    return data, data_effect, axis_limits, nof_instances, indices


def get_feature_names(dim: int) -> list:
    """Returns the feature names for the given dimensionality"""
    return ["x_" + str(i) for i in range(dim)]


def prep_avg_output(data, model, avg_output, scale_y) -> float:
    avg_output = avg_output if avg_output is not None else np.mean(model(data))
    avg_output = (
        avg_output * scale_y["std"] + scale_y["mean"]
        if scale_y is not None
        else avg_output
    )
    return avg_output


def indices_within_limits(data: np.ndarray, axis_limits: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the points that lie within `axis_limits` on
    every feature."""
    accept_indices = np.ones([data.shape[0]]) > 0
    dim = data.shape[1]
    for feature in range(dim):
        accept_left = data[:, feature] >= axis_limits[0, feature]
        accept_right = data[:, feature] <= axis_limits[1, feature]
        accept_indices = np.logical_and.reduce(
            [accept_indices, accept_left, accept_right]
        )
    if np.sum(accept_indices) == 0:
        raise ValueError("axis_limits exclude every data point")
    return accept_indices


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    return "_".join(re.findall(r"[A-Z][a-z]*|\d+", name)).lower()
