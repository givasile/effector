"""ToyEffect — the reference implementation of the R14 subclass contract.

A deliberately minimal effect method: the *local effect* of every instance is
just its model prediction f(x^i); the summary is the per-bin mean/variance of
those predictions along the feature axis; eval reads the step function.
Scientific value: none. Pedagogical value: the whole two-block lifecycle
(docs/design.md R14) in ~60 lines of method code —

1. answer the frame question: which config params invalidate the local
   effects? (here: none — predictions are instance-anchored; `nof_bins` is a
   summary-stage knob, like RHALE's binning)
2. implement the three pure kernels: `_compute_local` (the only one that may
   touch the model), `_summarize` (numpy in, payload out), `_eval_payload`
   (payload + xs in, numbers out)
3. write ZERO cache/retrigger/mask logic — the base owns all of it: masking,
   memoization, centering, heter_score, importance, find_regions and the
   Partition sugar all work on this class for free.

Copy this file as the starting point of a real new method.
"""

import numpy as np

from effector import ingestion, utils
from effector.global_effect import GlobalEffectBase


class ToyEffect(GlobalEffectBase):
    DEFAULT_CENTERING = False
    SUPPORTED_FEATURE_TYPES = frozenset({ingestion.CONTINUOUS})
    CAT_STRATEGY = None

    def __init__(self, data, model, **kwargs):
        super().__init__("toy", data, model, None, **kwargs)

    # -- 1. the frame declaration ------------------------------------------
    def _frame_from_config(self, feature):
        return ()  # instance-anchored: the cached predictions never go stale

    # -- 2. the model-touching kernel (the ONLY one) ------------------------
    def _compute_local(self, feature, frame):
        # feature-independent raw material is computed once per OBJECT and
        # shared across features (the RHALE-jacobian / ShapDP-table pattern);
        # here it coincides with the base's `_y_pred` slot
        if self._y_pred is None:
            self._y_pred = np.asarray(self.model(self.data))
        return {"frame": frame, "pred": self._y_pred}

    # -- 3. the pure summary kernel -----------------------------------------
    def _summarize(self, feature, mask=None, nof_bins=10):
        pred = self._local[feature]["pred"]
        col = self.data[:, feature]
        if mask is not None:
            pred, col = pred[mask], col[mask]
        limits = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_bins + 1
        )
        idx = np.clip(np.digitize(col, limits) - 1, 0, nof_bins - 1)
        counts = np.bincount(idx, minlength=nof_bins)
        mean = np.bincount(idx, weights=pred, minlength=nof_bins)
        sq = np.bincount(idx, weights=pred**2, minlength=nof_bins)
        with np.errstate(invalid="ignore"):
            mean = np.where(counts > 0, mean / np.maximum(counts, 1), np.nan)
            var = np.where(counts > 0, sq / np.maximum(counts, 1) - mean**2, np.nan)
        return {
            "limits": limits,
            "mean": utils.fill_nans(mean),
            "var": np.maximum(utils.fill_nans(var), 0.0),
        }

    # -- 4. the pure reader kernel ------------------------------------------
    def _eval_payload(self, feature, params, x, heterogeneity=False):
        idx = np.clip(
            np.digitize(x, params["limits"]) - 1, 0, len(params["mean"]) - 1
        )
        y = params["mean"][idx]
        return (y, params["var"][idx]) if heterogeneity else y

    # -- public surface: declare config, delegate everything to the base ----
    def fit(self, features="all", *, centering=False, nof_bins=10):
        self._fit_loop(features, centering, nof_bins=nof_bins)

    def plot(
        self,
        feature,
        heterogeneity=False,
        centering=False,
        show_plot=True,
        mask=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        mask = self._prep_mask(mask)
        params = self._summary(feature, mask)
        centers = (params["limits"][:-1] + params["limits"][1:]) / 2
        y = self.eval(feature, centers, centering=centering, mask=mask)
        fig, ax = plt.subplots()
        ax.step(centers, y, where="mid")
        if heterogeneity is not False:
            band = np.sqrt(self._eval_payload(feature, params, centers, True)[1])
            ax.fill_between(centers, y - band, y + band, alpha=0.2, step="mid")
        ax.set_xlabel(self.feature_names[feature])
        ax.set_ylabel(self.target_name)
        if show_plot:
            plt.show(block=False)
            return None
        return fig, ax
