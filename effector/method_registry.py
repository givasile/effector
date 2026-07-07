"""R5 — the one method registry.

Every piece of per-method knowledge that used to live in scattered if/elif
chains — which class implements a method, whether it consumes the model
jacobian, and how it is displayed — is written here once.
"""

from collections import namedtuple

from effector.global_effect_ale import ALE, RHALE
from effector.global_effect_pdp import PDP, DerPDP
from effector.global_effect_shap import ShapDP

MethodSpec = namedtuple(
    "MethodSpec",
    [
        "cls",
        "needs_jac",
        "display_name",
        "supported_feature_types",
        "cat_strategy",
    ],
)


def _spec(cls, needs_jac, display_name):
    # capability fields are read from the class attributes (single source of
    # truth — a contract test pins the agreement)
    return MethodSpec(
        cls,
        needs_jac,
        display_name,
        cls.SUPPORTED_FEATURE_TYPES,
        cls.CAT_STRATEGY,
    )


METHODS = {
    "pdp": _spec(PDP, False, "PDP"),
    "derpdp": _spec(DerPDP, True, "d-PDP"),
    "ale": _spec(ALE, False, "ALE"),
    "rhale": _spec(RHALE, True, "RHALE"),
    "shapdp": _spec(ShapDP, False, "SHAP-DP"),
}

ALIASES = {
    "d-pdp": "derpdp",
    "der-pdp": "derpdp",
    "shap": "shapdp",
    "shap_dp": "shapdp",
    "shap-dp": "shapdp",
}


def canonical(name: str) -> str:
    """The canonical registry key for `name` (case-insensitive, aliases allowed)."""
    key = name.lower()
    key = ALIASES.get(key, key)
    if key not in METHODS:
        raise ValueError(
            "Unknown method '{}'. Supported methods: {} (aliases: {}).".format(
                name, sorted(METHODS), sorted(ALIASES)
            )
        )
    return key


def resolve(name: str) -> MethodSpec:
    """The `MethodSpec` for `name` (case-insensitive, aliases allowed)."""
    return METHODS[canonical(name)]
