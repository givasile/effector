"""Data generators and real datasets used in the examples and tests.

Synthetic generators (`IndependentUniform`) produce numpy arrays with known
distributions — pair them with `effector.models` for ground-truth checks.
Real datasets (`BikeSharing`) fetch, split, and standardize a public dataset
into ready-to-use `x_train / y_train / x_test / y_test` numpy arrays.
"""

import numpy as np

from effector import helpers


class Base:
    def __init__(self, name: str, dim: int, axis_limits: np.ndarray):
        self.name = helpers.camel_to_snake(name)
        self.dim = dim
        self.axis_limits = axis_limits

    def generate_data(self, n: int, seed: int = 21) -> np.ndarray:
        """Generate `n` samples.

        Args:
            n: Number of samples.
            seed: Random seed, for reproducibility.

        Returns:
            The samples, shape `(n, dim)`.
        """
        raise NotImplementedError


class IndependentUniform(Base):
    """`dim` independent features, each uniform on `[low, high]`.

    The simplest possible distribution — no correlations, flat marginals — so
    any structure in an effect plot comes from the model alone. The default
    data source of the synthetic benchmarks.
    """

    def __init__(self, dim: int = 2, low: float = 0, high: float = 1):
        """Initialize the generator.

        Args:
            dim: Number of features.
            low: Lower bound of every feature.
            high: Upper bound of every feature.
        """
        axis_limits = np.array([[low, high] for _ in range(dim)]).T
        super().__init__(name=self.__class__.__name__, dim=dim, axis_limits=axis_limits)

    def generate_data(self, n: int, seed: int = 21) -> np.ndarray:
        """Generate `n` samples.

        Args:
            n: Number of samples.
            seed: Random seed, for reproducibility.

        Returns:
            The samples, shape `(n, dim)`.
        """
        np.random.seed(seed)
        x = np.random.uniform(
            self.axis_limits[0, :], self.axis_limits[1, :], (n, self.dim)
        )
        np.random.shuffle(x)
        return x


class RealDatasetBase:
    """Shared plumbing for real datasets: fetch, seeded train/test split,
    optional standardization (storing the per-feature mu/std for mapping plots
    back to natural units)."""

    def __init__(self, name: str, pcg_train, standardize, seed: int = 21):
        self.name = helpers.camel_to_snake(name)

        self.dataset: np.ndarray = None

        self.feature_names = None
        self.target_name = None

        # train set
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_train_mu = None
        self.x_train_std = None
        self.y_train_mu = None
        self.y_train_std = None

        # test set
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.x_test_mu = None
        self.x_test_std = None
        self.y_test_mu = None
        self.y_test_std = None

        # main logic
        self.fetch_and_preprocess()

        self.x_train, self.x_test, self.y_train, self.y_test = self.split(
            self.dataset[:, :-1], self.dataset[:, -1], pcg_train, seed
        )

        if standardize:
            self.x_train, self.x_train_mu, self.x_train_std = self.standardize(
                self.x_train
            )
            self.x_test, self.x_test_mu, self.x_test_std = self.standardize(self.x_test)
            self.y_train, self.y_train_mu, self.y_train_std = self.standardize(
                self.y_train
            )
            self.y_test, self.y_test_mu, self.y_test_std = self.standardize(self.y_test)

        self.postprocess()

    def fetch_and_preprocess(self):
        # self.dataset = ...
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    @staticmethod
    def standardize(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_standardized = (x - x_mean) / x_std
        return x_standardized, x_mean, x_std

    @staticmethod
    def split(x, y, pcg_train, seed: int = 21):
        n_train = int(x.shape[0] * pcg_train)

        # seeded shuffle: the train/test split is reproducible between runs
        rng = np.random.default_rng(seed)
        idx = rng.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]

        # split
        x_train = x[:n_train]
        x_test = x[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        return x_train, x_test, y_train, y_test


class BikeSharing(RealDatasetBase):
    """The UCI Bike Sharing dataset (hourly) — effector's canonical real example.

    17,379 hourly records of the Capital Bikeshare system; the target is the
    hourly rental count. Fetched via `ucimlrepo` (UCI id 275), with `dteday`
    and `atemp` dropped — 11 features remain (season, yr, mnth, hr, holiday,
    weekday, workingday, weathersit, temp, hum, windspeed). Data is split into
    train/test with a seeded shuffle and (by default) standardized; the UCI
    normalization constants of temp/hum/windspeed are folded into the stored
    mu/std, so `scale_x`/`scale_y` map plots back to natural units.

    ```python
    data = effector.datasets.BikeSharing()
    data.x_train, data.y_train    # (13903, 11), (13903,)
    data.feature_names, data.target_name
    ```
    """

    def __init__(self, pcg_train=0.8, standardize=True, seed: int = 21):
        """Fetch and prepare the dataset.

        Args:
            pcg_train: Fraction of samples in the train split.
            standardize: Standardize features and target to zero mean, unit
                variance (the mu/std used are stored on the object).
            seed: Random seed of the train/test shuffle.
        """
        super().__init__(
            name="BikeSharing", pcg_train=pcg_train, standardize=standardize, seed=seed
        )

    def fetch_and_preprocess(self):
        from ucimlrepo import fetch_ucirepo

        bike_sharing_dataset = fetch_ucirepo(id=275)

        # bike_sharing_dataset.feature_names
        X = bike_sharing_dataset.data.features
        X = X.drop(["dteday", "atemp"], axis=1)
        self.feature_names = X.columns.to_list()
        X = X.to_numpy()

        y = bike_sharing_dataset.data.targets
        self.target_name = y.columns.item()
        y = y.to_numpy()
        self.dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    def postprocess(self):
        # UCI (id=275) ships these features pre-normalized — temp: (t+8)/47,
        # hum: h/100, windspeed: w/67. Folding the constants into the stored
        # mu/std makes scale_x/scale_y map plots back to natural units.
        # After dropping dteday/atemp: 8 = temp, 9 = hum, 10 = windspeed.
        self.x_train_mu[8] += 8
        self.x_train_std[8] *= 47
        self.x_test_mu[8] += 8
        self.x_test_std[8] *= 47

        self.x_train_std[9] *= 100
        self.x_test_std[9] *= 100

        self.x_train_std[10] *= 67
        self.x_test_std[10] *= 67
