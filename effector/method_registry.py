"""R5 — the one method registry.

Every piece of per-method knowledge that used to live in scattered if/elif
chains — which class implements a method, whether it consumes the model
jacobian, whether it carries a precomputed `data_effect`, and how it is
displayed — is written here once.
"""

from collections import namedtuple

from effector.global_effect_ale import ALE, RHALE
from effector.global_effect_pdp import PDP, DerPDP
from effector.global_effect_shap import ShapDP

MethodSpec = namedtuple(
    "MethodSpec", ["cls", "needs_jac", "uses_data_effect", "display_name"]
)

METHODS = {
    "pdp": MethodSpec(PDP, False, False, "PDP"),
    "derpdp": MethodSpec(DerPDP, True, False, "d-PDP"),
    "ale": MethodSpec(ALE, False, False, "ALE"),
    "rhale": MethodSpec(RHALE, True, True, "RHALE"),
    "shapdp": MethodSpec(ShapDP, False, False, "SHAP-DP"),
}

ALIASES = {
    "d-pdp": "derpdp",
    "der-pdp": "derpdp",
    "shap": "shapdp",
    "shap_dp": "shapdp",
    "shap-dp": "shapdp",
}


def resolve(name: str) -> MethodSpec:
    """The `MethodSpec` for `name` (case-insensitive, aliases allowed)."""
    key = name.lower()
    key = ALIASES.get(key, key)
    if key not in METHODS:
        raise ValueError(
            "Unknown method '{}'. Supported methods: {} (aliases: {}).".format(
                name, sorted(METHODS), sorted(ALIASES)
            )
        )
    return METHODS[key]
