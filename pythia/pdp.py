import typing
import copy
import numpy as np
import pythia.visualization as vis
import pythia.helpers as helpers
from pythia.fe_base import FeatureEffectBase


class PDP(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """

        Parameters
        ----------
        data: np.array (N, D), the design matrix
        model: Callable (N, D) -> (N,)
        axis_limits: Union[None, np.ndarray(2, D)]


        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        self.D = data.shape[1]
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(PDP, self).__init__(axis_limits)

    def _fit_feature(self, feature: int) -> typing.Dict:
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all", normalize: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(feature=s)
            if normalize is not False:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_normalize(normalize))
            self.is_fitted[s] = True

    def _eval_unnorm(
        self, feature: int, x: np.ndarray, uncertainty: bool = False
    ) -> np.ndarray:

        # compute main PDP effect
        mean_pdp = []
        sigma_pdp = []
        stderr = []
        for i in range(x.shape[0]):
            x_new = copy.deepcopy(self.data)
            x_new[:, feature] = x[i]
            mean_pdp.append(np.mean(self.model(x_new)))
            if uncertainty:
                std = np.std(self.model(x_new))
                sigma_pdp.append(std)
                stderr.append(std / np.sqrt(self.data.shape[0]))

        if not uncertainty:
            return np.array(mean_pdp)
        else:
            return np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr)

    def plot(self,
             feature: int,
             normalized: bool = True,
             confidence_interval: typing.Union[None, str] = None,
             nof_points: int = 30) -> None:
        title = "PDP: feature %d" % (feature + 1)
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        if normalized:
            vis.plot_1d(x, feature, self.eval, confidence=confidence_interval, title=title)
        else:
            vis.plot_1d(x, feature, self._eval_unnorm, confidence=confidence_interval, title=title)


class ICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
        """
        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)
        :param axis_limits: Union[None, np.ndarray(2, D)]
        :param instance: int, index of the instance
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        self.D = data.shape[1]
        self.instance = instance
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(ICE, self).__init__(axis_limits)

    def _fit_feature(self, feat: int):
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all", normalize: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s)
            if normalize is not False:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_normalize(normalize))
            self.is_fitted[s] = True

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False) -> np.ndarray:
        if uncertainty:
            raise NotImplementedError

        i = self.instance
        xi = copy.deepcopy(self.data[i, :])
        xi_repeat = np.tile(xi, (x.shape[0], 1))
        xi_repeat[:, feature] = x
        y = self.model(xi_repeat)
        return y

    def plot(self, feature: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "ICE: feature %d, instance %d" % (feature + 1, self.instance)
        if normalized:
            vis.plot_1d(x, feature, self.eval, confidence=None, title=title)
        else:
            vis.plot_1d(x, feature, self._eval_unnorm, confidence=None, title=title)


class PDPwithICE:
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: int = 100,
    ):
        """

        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)
        :param axis_limits: Union[None, np.ndarray(2, D)]
        :param nof_instances: int, number of instances to be used for the ICE plots
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        self.dim = data.shape[1]
        self.axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        self.nof_instances = nof_instances

        assert nof_instances <= data.shape[0], "Number of instances must be smaller than the number of data points."
        self.y_pdp = PDP(data=data, model=model, axis_limits=axis_limits)

        # choose nof_instances from the data with replacement
        self.indices = np.random.choice(data.shape[0], size=nof_instances, replace=False)

        self.y_ice = [ICE(data=data, model=model, axis_limits=axis_limits, instance=i) for i in self.indices]

        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def fit(self, features: typing.Union[int, str, list], normalize: typing.Union[bool, str] = True):
        # pdp
        self.y_pdp.fit(features, normalize)

        # ice curves
        for i in range(self.nof_instances):
            self.y_ice[i].fit(features, normalize)

    def plot(
        self,
        feature: int,
        normalized: bool = True,
        nof_points: int = 30,
        scale_x=None,
        scale_y=None,
        savefig=None
    ) -> None:

        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "PDP-ICE: feature %d" % (feature + 1)
        vis.plot_pdp_ice(x, feature, self.y_pdp, self.y_ice, title=title, normalize=normalized,
                         scale_x=scale_x, scale_y=scale_y, savefig=savefig)


class dPDP(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = "all",
    ):
        """

        Parameters
        ----------
        data: np.array (N, D), the design matrix
        model: Callable (N, D) -> (N,)
        axis_limits: Union[None, np.ndarray(2, D)]


        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.model_jac = model_jac

        assert nof_instances <= data.shape[0], "Number of instances must be smaller than the number of data points."
        if nof_instances == "all":
            self.data = data
        else:
            self.indices = np.random.choice(data.shape[0], size=nof_instances, replace=False)
            self.data = data[self.indices]

        self.D = data.shape[1]
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(dPDP, self).__init__(axis_limits)

    def _fit_feature(self, feature: int) -> typing.Dict:
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all", normalize: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(feature=s)
            if normalize is not False:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_normalize(normalize))
            self.is_fitted[s] = True

    def _eval_unnorm(
        self, feature: int, x: np.ndarray, uncertainty: bool = False
    ) -> np.ndarray:

        # compute main PDP effect
        mean_pdp = []
        sigma_pdp = []
        stderr = []
        for i in range(x.shape[0]):
            x_new = copy.deepcopy(self.data)
            x_new[:, feature] = x[i]
            mean_pdp.append(np.mean(self.model_jac(x_new)[:, feature]))
            if uncertainty:
                std = np.std(self.model_jac(x_new)[:, feature])
                sigma_pdp.append(std)
                stderr.append(std / np.sqrt(self.data.shape[0]))

        if not uncertainty:
            return np.array(mean_pdp)
        else:
            return np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr)

    def plot(self,
             feature: int,
             normalized: bool = True,
             confidence_interval: typing.Union[None, str] = None,
             nof_points: int = 30) -> None:
        title = "d-PDP: feature %d" % (feature + 1)
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        if normalized:
            vis.plot_1d(x, feature, self.eval, confidence=confidence_interval, title=title)
        else:
            vis.plot_1d(x, feature, self._eval_unnorm, confidence=confidence_interval, title=title)


class dICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
        """
        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)
        :param axis_limits: Union[None, np.ndarray(2, D)]
        :param instance: int, index of the instance
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.model_jac = model_jac
        self.data = data
        self.D = data.shape[1]
        self.instance = instance
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(dICE, self).__init__(axis_limits)

    def _fit_feature(self, feat: int):
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all", normalize: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s)
            if normalize is not False:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_normalize(normalize))
            self.is_fitted[s] = True

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False) -> np.ndarray:
        if uncertainty:
            raise NotImplementedError

        i = self.instance
        xi = copy.deepcopy(self.data[i, :])
        xi_repeat = np.tile(xi, (x.shape[0], 1))
        xi_repeat[:, feature] = x
        y = self.model_jac(xi_repeat)[:, feature]
        return y

    def plot(self, feature: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "ICE: feature %d, instance %d" % (feature + 1, self.instance)
        if normalized:
            vis.plot_1d(x, feature, self.eval, confidence=None, title=title)
        else:
            vis.plot_1d(x, feature, self._eval_unnorm, confidence=None, title=title)


class PDPwithdICE:
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            model_jac: callable,
            axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.model_jac = model_jac
        self.data = data
        self.dim = data.shape[1]
        self.axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )

        self.y_pdp = dPDP(data=data, model=model, model_jac=model_jac, axis_limits=axis_limits)
        self.y_dice = [dICE(data=data, model=model, model_jac=model_jac, axis_limits=axis_limits, instance=i) for i in
                       range(data.shape[0])]

        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def fit(self, features: typing.Union[int, str, list], normalize: typing.Union[bool, str] = True):
        # pdp
        self.y_pdp.fit(features, normalize)

        # ice curves
        for i in range(self.data.shape[0]):
            self.y_dice[i].fit(features, normalize)

    def plot(
            self,
            feature: int,
            normalized: bool = True,
            nof_points: int = 30,
            scale_x=None,
            scale_y=None,
            savefig=None
    ) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "PDP-ICE: feature %d" % (feature + 1)
        vis.plot_pdp_ice(x, feature, self.y_pdp, self.y_dice, title=title, normalize=normalized,
                         scale_x=scale_x, scale_y=scale_y, savefig=savefig)


class PDPGroundTruth(FeatureEffectBase):
    def __init__(self, func: np.ndarray, axis_limits: np.ndarray):
        self.func = func
        super(PDPGroundTruth, self).__init__(axis_limits)

    def _fit_feature(self, feat: int) -> typing.Dict:
        return {}

    def fit(self, features: typing.Union[int, str, list] = "all", normalize: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = {}
            if normalize:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_normalize(normalize))
            self.is_fitted[s] = True

    def _eval_unnorm(
        self, feature: int, x: np.ndarray, uncertainty: bool = False
    ) -> np.ndarray:
        """

        :param x: np.ndarray (N, D)
        :returns: np.ndarray (N,)

        """
        return self.func(x)

    def plot(self, feature: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        if normalized:
            y = self.eval(x, feature)
        else:
            y = self._eval_unnorm(x, feature)
        vis.plot_1d(x, y, title="Ground-truth PDP for feature %d" % (feature + 1))
