"""
CALM submodule of effector.

Usage:
    from effector.calm import CALMRegressor, CALMClassifier, ...
"""

from .calm import (
    CALM_CLASSIFICATION_MODELS,
    CALM_REGRESSION_MODELS,
    RegionalPDPDetector,
    RegionalRHALEDetector,
    CALMRegressor,
    CALMClassifier,
)

from .masked_fitting import (
    NoInteractionsEBMClassifier,
    NoInteractionsEBMRegressor,
    WithInteractionsEBMClassifier,
    WithInteractionsEBMRegressor,
    MaskedNAMClassifier,
    MaskedNAMRegressor,
    PyGAMClassifier,
    PyGAMRegressor,
)

__all__ = [
    "CALM_CLASSIFICATION_MODELS",
    "CALM_REGRESSION_MODELS",
    "RegionalPDPDetector",
    "RegionalRHALEDetector",
    "CALMRegressor",
    "CALMClassifier",
    "NoInteractionsEBMClassifier",
    "NoInteractionsEBMRegressor",
    "WithInteractionsEBMClassifier",
    "WithInteractionsEBMRegressor",
    "MaskedNAMClassifier",
    "MaskedNAMRegressor",
    "PyGAMClassifier",
    "PyGAMRegressor",
]
