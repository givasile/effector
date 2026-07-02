"""Synthetic (model, distribution) pairs with closed-form effect ground truths.

A feature-effect ground truth is a property of the *pair* (prediction function,
data distribution) — never of the model alone: the same model under a different
distribution has different PDP/ALE/SHAP curves. Each class below bundles one such
pair and exposes its closed-form effects, so the class name states the exact
scope of validity of every ``*_gt`` method.

The derivations live in ``notebooks/synthetic-examples/`` (05, 06, 07) and
``notebooks/synthetic-examples/02_global_effect_methods_comparison.ipynb``;
the notebooks and ``tests/test_functional_*.py`` both consume the functions
defined here, so the two can never disagree about the right answer.

All ``*_gt(feature, xs)`` methods return the effect centered to zero integral
over the feature's axis range (matching ``eval(..., centering=True)``), unless
stated otherwise in their docstring.
"""

import numpy as np

from effector import datasets, models


def _center(ff, start, stop, nof_points=1000):
    """Zero-integral centering constant of ff over [start, stop]."""
    return np.mean(ff(np.linspace(start, stop, nof_points)))


class ConditionalInteractionUniform:
    """``models.ConditionalInteraction`` under iid U[-1, 1]^3.

    $f(x) = -x_1^2 1_{x_2 < 0} + x_1^2 1_{x_2 \\geq 0} + e^{x_3}$

    Derivations: notebooks/synthetic-examples/05_conditional_interaction_*.ipynb
    """

    dim = 3

    def __init__(self):
        self.model = models.ConditionalInteraction()
        self.dataset = datasets.IndependentUniform(dim=self.dim, low=-1, high=1)
        self.axis_limits = self.dataset.axis_limits

    def generate_data(self, n: int, seed: int = 21) -> np.ndarray:
        return self.dataset.generate_data(n, seed=seed)

    def pdp_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        if feature == 0:
            ff = lambda x: np.zeros_like(x)
        elif feature == 1:
            ff = lambda x: -1 / 3 * (x < 0) + 1 / 3 * (x >= 0)
        elif feature == 2:
            ff = lambda x: np.exp(x)
        return ff(xs) - _center(ff, -1, 1)

    def ale_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        # ALE agrees with PDP everywhere for this pair
        return self.pdp_gt(feature, xs)

    def rhale_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        if feature == 1:
            # the derivative w.r.t. x2 is zero a.e., so RHALE misses the step
            ff = lambda x: np.zeros_like(x)
            return ff(xs) - _center(ff, -1, 1)
        return self.pdp_gt(feature, xs)

    def pdp_heter_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        """Variance of the centered ICE curves at each grid point."""
        if feature == 0:
            return xs**4 - 2 / 3 * xs**2 + 1 / 9  # = (x^2 - 1/3)^2
        elif feature == 1:
            return np.full_like(xs, 4 / 45)  # ~ 0.089
        elif feature == 2:
            return np.zeros_like(xs)

    def ale_bin_variance_gt(self, feature: int, nof_bins: int = 31) -> np.ndarray:
        """Per-bin variance of the ALE local effects, Fixed(nof_bins) on [-1, 1].

        The central bin of feature 1 (the one containing the jump at 0) is NaN:
        its variance is an artifact of the discontinuity, not a ground truth.
        """
        bin_centers = np.linspace(-1 + 1 / nof_bins, 1 - 1 / nof_bins, nof_bins)
        if feature == 0:
            return 4 * bin_centers**2
        elif feature == 1:
            y = np.zeros_like(bin_centers)
            y[nof_bins // 2] = np.nan
            return y
        elif feature == 2:
            return np.zeros_like(bin_centers)

    # --- regional ground truth (feature of interest: 0) ---

    regional_split_feature = 1
    regional_split_position = 0.0

    def regional_effect_gt(self, side: str, xs: np.ndarray) -> np.ndarray:
        """Centered effect of x1 inside each region of the optimal split.

        side: "left" (x2 < 0) -> -x1^2, "right" (x2 >= 0) -> +x1^2.
        Per-region heterogeneity is 0: the effect is deterministic within a region.
        """
        sign = -1.0 if side == "left" else 1.0
        ff = lambda x: sign * x**2
        return ff(xs) - _center(ff, -1, 1)


class GeneralInteractionUniform:
    """``models.GeneralInteraction`` under iid U[-1, 1]^3.

    $f(x) = x_1 x_2^2 + e^{x_3}$

    Derivations: notebooks/synthetic-examples/06_general_interaction_*.ipynb
    """

    dim = 3

    def __init__(self):
        self.model = models.GeneralInteraction()
        self.dataset = datasets.IndependentUniform(dim=self.dim, low=-1, high=1)
        self.axis_limits = self.dataset.axis_limits

    def generate_data(self, n: int, seed: int = 21) -> np.ndarray:
        return self.dataset.generate_data(n, seed=seed)

    def pdp_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        if feature == 0:
            ff = lambda x: x / 3  # x * E[x2^2], E[x2^2] = 1/3
        elif feature == 1:
            ff = lambda x: np.zeros_like(x)  # E[x1] * x2^2 = 0
        elif feature == 2:
            ff = lambda x: np.exp(x)
        return ff(xs) - _center(ff, -1, 1)

    def ale_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        return self.pdp_gt(feature, xs)

    def rhale_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        return self.pdp_gt(feature, xs)

    def pdp_heter_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        """Variance of the centered ICE curves at each grid point.

        Feature 1 is the reason this pair exists: the x1*x2^2 interaction is
        invisible in the mean effect (E[x1] = 0) but not in the heterogeneity.
        """
        if feature == 0:
            return 4 / 45 * xs**2  # x^2 * Var[x2^2]
        elif feature == 1:
            return (xs**2 - 1 / 3) ** 2 / 3  # (x2^2 - E[x2^2])^2 * E[x1^2]
        elif feature == 2:
            return np.zeros_like(xs)


class ConditionalInteraction4RegionsUniform:
    """``models.ConditionalInteraction4Regions`` under iid U[-1, 1]^4.

    $f(x) = \\pm x_1^2 / \\pm x_1^4$ gated by the signs of $x_2, x_3$, plus $e^{x_4}$.

    Derivations: notebooks/synthetic-examples/07_conditional_interaction_4_regions_*.ipynb
    """

    dim = 4

    def __init__(self):
        self.model = models.ConditionalInteraction4Regions()
        self.dataset = datasets.IndependentUniform(dim=self.dim, low=-1, high=1)
        self.axis_limits = self.dataset.axis_limits

    def generate_data(self, n: int, seed: int = 21) -> np.ndarray:
        return self.dataset.generate_data(n, seed=seed)

    def pdp_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        if feature in (0, 1):
            ff = lambda x: np.zeros_like(x)
        elif feature == 2:
            # 0.5*E[x1^2] + 0.5*E[x1^4] = 0.5/3 + 0.5/5 = 4/15, sign flips at 0
            ff = lambda x: -4 / 15 * (x < 0) + 4 / 15 * (x >= 0)
        elif feature == 3:
            ff = lambda x: np.exp(x)
        return ff(xs) - _center(ff, -1, 1)

    def ale_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        return self.pdp_gt(feature, xs)

    def rhale_gt(self, feature: int, xs: np.ndarray) -> np.ndarray:
        if feature == 2:
            # zero derivative a.e.: RHALE misses the step (cf. rhale of feature 1
            # in ConditionalInteractionUniform)
            ff = lambda x: np.zeros_like(x)
            return ff(xs) - _center(ff, -1, 1)
        return self.pdp_gt(feature, xs)

    # --- regional ground truth (feature of interest: 0) ---
    # The optimal partition is two-level and its order is determined by the
    # math: level 1 splits on x3 (the SIGN gate — separating -x1^{2|4} from
    # +x1^{2|4} explains nearly all the heterogeneity), level 2 on x2 (the
    # power gate). 4 leaf regions with deterministic effects.

    regional_level1_split_feature = 2
    regional_level2_split_feature = 1
    regional_split_position = 0.0

    def regional_effect_gt(self, x2_side: str, x3_side: str, xs: np.ndarray) -> np.ndarray:
        """Centered effect of x1 inside each of the 4 leaf regions."""
        sign = -1.0 if x3_side == "left" else 1.0
        power = 2 if x2_side == "left" else 4
        ff = lambda x: sign * x**power
        return ff(xs) - _center(ff, -1, 1)


class CorrelatedInteraction:
    """The Gkolemis et al. 2023 (RHALE paper) example — correlated features.

    $f(x) = \\sin(2\\pi x_1) (1_{x_1 < 0} - 2 \\cdot 1_{x_3 < 0}) + x_1 x_2 + x_2$

    with $x_1$ mixture-uniform on [-0.5, 0.5] (5/6 of the mass below 0),
    $x_2 \\sim N(0, 2)$ and $x_3 = x_1 + N(0, 0.01)$ — strongly correlated.

    The only pair where PDP, ALE, RHALE and SHAP provably *differ*; each
    ``*_gt`` is the correct answer for that method's own definition.

    Derivations: notebooks/synthetic-examples/02_global_effect_methods_comparison.ipynb
    """

    dim = 3
    sigma_2 = 2.0
    sigma_3 = 0.01

    def __init__(self):
        self.axis_limits = np.array([[-0.5, 0.5], [-5.0, 5.0], [-0.5, 0.5]]).T

    def generate_data(self, n: int, seed: int = 21) -> np.ndarray:
        """Mixture sampler: 5/6 of x1 mass on [-0.5, 0], 1/6 on [0, 0.5]."""
        rng = np.random.default_rng(seed)
        n1 = int(5 * n / 6)
        n2 = n - n1
        x1 = np.concatenate(
            [
                np.array([-0.5]),
                rng.uniform(-0.5, 0, size=n1 - 2),
                np.array([-1e-5]),
                np.array([0.0]),
                rng.uniform(0, 0.5, size=n2 - 2),
                np.array([0.5]),
            ]
        )
        x2 = rng.normal(0, self.sigma_2, n)
        x3 = x1 + rng.normal(0, self.sigma_3, n)
        return np.stack([x1, x2, x3], -1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x[:, 0])
        ind = x[:, 0] < 0
        y[ind] = np.sin(2 * np.pi * x[ind, 0])
        ind = x[:, 2] < 0
        y[ind] -= 2 * np.sin(2 * np.pi * x[ind, 0])
        y += x[:, 0] * x[:, 1] + x[:, 1]
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        dydx = np.zeros_like(x)
        ind = x[:, 0] <= 0
        dydx[ind, 0] = 2 * np.pi * np.cos(2 * np.pi * x[ind, 0])
        ind = x[:, 2] <= 0
        dydx[ind, 0] -= 4 * np.pi * np.cos(2 * np.pi * x[ind, 0])
        dydx[:, 0] += x[:, 1]
        dydx[:, 1] = x[:, 0] + 1
        return dydx

    # --- ground truths for the feature of interest, x1 ---

    def pdp_gt(self, xs: np.ndarray) -> np.ndarray:
        """PDP averages over the *marginal* of x3, ignoring its correlation
        with x1: the -2 sin term contributes with probability 5/6 everywhere."""
        ff = lambda x: np.sin(2 * np.pi * x) * (x < 0) - 5 / 3 * np.sin(2 * np.pi * x)
        return ff(xs) - _center(ff, -0.5, 0.5)

    def d_pdp_gt(self, xs: np.ndarray) -> np.ndarray:
        """Derivative-PDP (uncentered): d/dx1 of the PDP integrand + E[x2]-terms."""
        return (
            2 * np.pi * np.cos(2 * np.pi * xs) * (xs < 0)
            - 10 * np.pi / 3 * np.cos(2 * np.pi * xs)
            + 1
        )

    def ale_gt(self, xs: np.ndarray) -> np.ndarray:
        """ALE conditions on x3 ~ x1, so the -2 sin term only acts where x1 < 0
        ... and there it flips the sign: sin - 2 sin = -sin."""
        ff = lambda x: -np.sin(2 * np.pi * x) * (x < 0)
        return ff(xs) - _center(ff, -0.5, 0.5)

    def rhale_gt(self, xs: np.ndarray) -> np.ndarray:
        return self.ale_gt(xs)

    def rhale_heter_gt(self, xs: np.ndarray) -> np.ndarray:
        """RHALE heterogeneity (std) accumulated from the x2 noise in df/dx1."""
        return (xs + 0.5) * self.sigma_2

    def shap_gt(self, xs: np.ndarray) -> np.ndarray:
        """Mean SHAP curve for x1: splits the interaction terms with x3."""
        ff = lambda x: -5 / 6 * np.sin(2 * np.pi * x)
        return ff(xs) - _center(ff, -0.5, 0.5)
