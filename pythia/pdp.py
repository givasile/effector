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
        nof_instances: typing.Union[int, str] = "all",
    ):
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        if nof_instances == "all":
            nof_instances = data.shape[0]
        self.nof_instances = nof_instances
        self.indices = np.random.choice(data.shape[0], self.nof_instances, replace=False)
        self.data = data[self.indices, :]
        self.D = data.shape[1]

        axis_limits = (helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits)
        super(PDP, self).__init__(axis_limits)

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            centering: typing.Union[bool, str] = False,
            ) -> None:
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self.norm_const[s] = self._compute_norm_const(s, centering) if centering is not False else self.norm_const[s]
            self.is_fitted[s] = True

    def _eval_unnorm(self,
                     feature: int,
                     x: np.ndarray,
                     uncertainty: bool = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # compute main PDP effect
        mean_pdp = []
        sigma_pdp = []
        stderr = []
        for i in range(x.shape[0]):
            x_new = copy.deepcopy(self.data)
            x_new[:, feature] = x[i]
            y = self.model(x_new)
            mean_pdp.append(np.mean(y))
            if uncertainty:
                std = np.std(y)
                sigma_pdp.append(std)
                stderr.append(std / np.sqrt(self.data.shape[0]))
        y = (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr)) if uncertainty else np.array(mean_pdp)
        return y

    def plot(self,
             feature: int,
             uncertainty: typing.Union[bool, str] = False,
             centering: typing.Union[bool, str] = False,
             nof_points: int = 30) -> None:
        title = "PDP: feature %d" % (feature + 1)
        uncertainty = helpers.prep_uncertainty(uncertainty)
        centering = helpers.prep_centering(centering)
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        func = self.eval
        vis.plot_1d(x, feature, func, confidence=uncertainty, centering=centering, title=title)


class dPDP(FeatureEffectBase):
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            model_jac: callable,
            axis_limits: typing.Union[None, np.ndarray] = None,
            nof_instances: typing.Union[int, str] = "all",
    ):
        # assertions
        assert data.ndim == 2

        # setters
        if nof_instances == "all":
            nof_instances = data.shape[0]
        self.nof_instances = nof_instances
        self.indices = np.random.choice(data.shape[0], self.nof_instances, replace=False)
        self.data = data[self.indices, :]
        self.model = model
        self.model_jac = model_jac
        self.D = self.data.shape[1]
        axis_limits = (helpers.axis_limits_from_data(self.data) if axis_limits is None else axis_limits)
        super(dPDP, self).__init__(axis_limits)

    def fit(self, features: typing.Union[int, str, list] = "all") -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.is_fitted[s] = True

    def _eval_unnorm(self,
                     feature: int,
                     x: np.ndarray,
                     uncertainty: bool = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # compute main PDP effect
        mean_pdp = []
        sigma_pdp = []
        stderr = []
        for i in range(x.shape[0]):
            x_new = copy.deepcopy(self.data)
            x_new[:, feature] = x[i]
            y = self.model_jac(x_new)[:, feature]
            mean_pdp.append(np.mean(y))
            if uncertainty:
                std = np.std(y)
                sigma_pdp.append(std)
                stderr.append(std / np.sqrt(self.data.shape[0]))

        y = (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr)) if uncertainty else np.array(mean_pdp)
        return y

    def plot(self,
             feature: int,
             uncertainty: typing.Union[bool, str] = False,
             nof_points: int = 30) -> None:
        title = "d-PDP for feature %d" % (feature + 1)
        uncertainty = helpers.prep_uncertainty(uncertainty)
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        func = self.eval
        vis.plot_1d(x, feature, func, confidence=uncertainty, centering=False, title=title)


class ICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
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

    def fit(self, features: typing.Union[int, str, list] = "all", centering: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self.norm_const[s] = self._compute_norm_const(s, centering) if centering is not False else self.norm_const[s]
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

    def plot(self, feature: int, centering: bool = True, nof_points: int = 30) -> None:
        # getters
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "ICE: feature %d, instance %d" % (feature + 1, self.instance)
        vis.plot_1d(x, feature, self.eval, confidence=False, centering=centering, title=title)


class PDPwithICE:
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 100,
    ):
        # assertions
        assert data.ndim == 2

        self.model = model
        self.dim = data.shape[1]
        if nof_instances == 'all':
            nof_instances = data.shape[0]
        assert self.nof_instances <= data.shape[0], "Number of instances must be smaller than the number of data points."
        self.nof_instances = nof_instances
        self.indices = np.random.choice(data.shape[0], size=self.nof_instances, replace=False)
        self.data = data[self.indices, :]
        self.axis_limits = (helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits)

        self.y_pdp = PDP(data=self.data, model=model, axis_limits=axis_limits)
        self.y_ice = [ICE(data=self.data, model=model, axis_limits=axis_limits, instance=i) for i in range(nof_instances)]

        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def fit(self, features: typing.Union[int, str, list], centering: typing.Union[bool, str] = True):
        centering = helpers.prep_centering(centering)
        self.y_pdp.fit(features, centering)
        [self.y_ice[i].fit(features, centering) for i in range(len(self.y_ice))]

    def plot(
        self,
        feature: int,
        centering: bool = True,
        nof_points: int = 30,
        scale_x=None,
        scale_y=None,
        savefig=None
    ) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "PDP-ICE: feature %d" % (feature + 1)
        vis.plot_pdp_ice(x, feature, self.y_pdp, self.y_ice, title=title, centering=centering,
                         scale_x=scale_x, scale_y=scale_y, savefig=savefig)


class dICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
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

    def fit(self, features: typing.Union[int, str, list] = "all", centering: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            if centering is not False:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_centering(centering))
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

    def plot(self, feature: int, centering: bool = False, nof_points: int = 30) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "d-ICE: feature %d, instance %d" % (feature + 1, self.instance)
        vis.plot_1d(x, feature, self.eval, confidence=False, centering=centering, title=title)


class PDPwithdICE:
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            model_jac: callable,
            axis_limits: typing.Union[None, np.ndarray] = None,
            nof_instances: typing.Union[int, str] = 100,
    ):
        # assertions
        assert data.ndim == 2

        self.model = model
        self.model_jac = model_jac
        self.nof_instances = nof_instances
        if self.nof_instances == 'all':
            self.nof_instances = data.shape[0]
        assert self.nof_instances <= data.shape[0], "Number of instances must be smaller than the number of data points."
        self.indices = np.random.choice(data.shape[0], size=self.nof_instances, replace=False)
        self.data = data[self.indices, :]
        self.axis_limits = (helpers.axis_limits_from_data(self.data) if axis_limits is None else axis_limits)
        self.y_pdp = dPDP(data=data, model=model, model_jac=model_jac, axis_limits=axis_limits, nof_instances=nof_instances)
        self.y_dice = [dICE(data=data, model=model, model_jac=model_jac, axis_limits=axis_limits, instance=i) for i in self.indices]
        self.dim = data.shape[1]

        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def fit(self, features: typing.Union[int, str, list]):
        self.y_pdp.fit(features)
        [self.y_dice[i].fit(features) for i in range(len(self.y_dice))]


    def plot(
            self,
            feature: int,
            nof_points: int = 30,
            scale_x=None,
            scale_y=None,
            savefig=None
    ) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "PDP with d-ICE: feature %d" % (feature + 1)
        vis.plot_pdp_ice(x, feature, self.y_pdp, self.y_dice, title=title, centering=False,
                         scale_x=scale_x, scale_y=scale_y, savefig=savefig)
