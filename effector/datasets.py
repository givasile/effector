"""Data generators and real datasets used in the examples and tests.

Synthetic generators (`IndependentUniform`) produce numpy arrays with known
distributions — pair them with `effector.models` for ground-truth checks.
Real datasets (`BikeSharing`, `MedicalCosts`, `AirfoilSelfNoise`,
`AdultIncome`) fetch, split, and optionally standardize a public
dataset into ready-to-use `x_train / y_train / x_test / y_test` numpy
arrays. Datasets with categorical columns also expose `feature_types` and
`category_names`, ready to feed `effector.Schema`.
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


class MedicalCosts(RealDatasetBase):
    """The Medical Cost Personal dataset — annual medical charges billed by
    an insurer, the textbook smoker × bmi interaction.

    1,338 policyholders with 6 features (age, sex, bmi, children, smoker,
    region); the target is the individual's yearly medical charges in USD.
    Fetched from the dataset's canonical mirror (stedy/Machine-Learning-with-R-datasets).
    Categorical columns are encoded to integer codes; `feature_types` and
    `category_names` are populated for `effector.Schema`. Kept in natural
    units by default (`standardize=False`) so partition rules read directly
    (e.g. `bmi < 30`).

    ```python
    data = effector.datasets.MedicalCosts()
    data.x_train, data.y_train        # (1070, 6), (1070,)
    schema = effector.Schema(
        feature_names=data.feature_names,
        feature_types=data.feature_types,
        category_names=data.category_names,
        target_name=data.target_name,
    )
    ```
    """

    URL = (
        "https://raw.githubusercontent.com/stedy/"
        "Machine-Learning-with-R-datasets/master/insurance.csv"
    )

    def __init__(self, pcg_train=0.8, standardize=False, seed: int = 21):
        """Fetch and prepare the dataset.

        Args:
            pcg_train: Fraction of samples in the train split.
            standardize: Standardize features and target (default False —
                natural units keep the partition rules readable).
            seed: Random seed of the train/test shuffle.
        """
        super().__init__(
            name="MedicalCosts", pcg_train=pcg_train, standardize=standardize, seed=seed
        )

    def fetch_and_preprocess(self):
        import pandas as pd

        raw = pd.read_csv(self.URL)

        levels = {
            "sex": ["female", "male"],
            "smoker": ["no", "yes"],
            "region": ["northeast", "northwest", "southeast", "southwest"],
        }
        for col, lv in levels.items():
            raw[col] = raw[col].map({name: i for i, name in enumerate(lv)})

        self.feature_names = ["age", "sex", "bmi", "children", "smoker", "region"]
        self.target_name = "charges"
        self.feature_types = [
            "continuous",  # age
            "nominal",  # sex
            "continuous",  # bmi
            "ordinal",  # children
            "nominal",  # smoker
            "nominal",  # region
        ]
        self.category_names = [
            None,
            levels["sex"],
            None,
            None,
            levels["smoker"],
            levels["region"],
        ]

        X = raw[self.feature_names].to_numpy(dtype=float)
        y = raw[self.target_name].to_numpy(dtype=float)
        self.dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    def postprocess(self):
        pass


class AirfoilSelfNoise(RealDatasetBase):
    """The NASA Airfoil Self-Noise dataset (UCI id 291).

    1,503 wind-tunnel measurements of NACA 0012 airfoil sections; the target
    is the scaled sound pressure level in dB. Five continuous features:
    frequency (Hz), angle of attack (deg), chord length (m), free-stream
    velocity (m/s), suction-side displacement thickness (m). Kept in natural
    units by default (`standardize=False`).

    ```python
    data = effector.datasets.AirfoilSelfNoise()
    data.x_train, data.y_train    # (1202, 5), (1202,)
    ```
    """

    def __init__(self, pcg_train=0.8, standardize=False, seed: int = 21):
        """Fetch and prepare the dataset.

        Args:
            pcg_train: Fraction of samples in the train split.
            standardize: Standardize features and target (default False —
                natural units keep the partition rules readable).
            seed: Random seed of the train/test shuffle.
        """
        super().__init__(
            name="AirfoilSelfNoise",
            pcg_train=pcg_train,
            standardize=standardize,
            seed=seed,
        )

    def fetch_and_preprocess(self):
        from ucimlrepo import fetch_ucirepo

        airfoil = fetch_ucirepo(id=291)

        X = airfoil.data.features
        self.feature_names = X.columns.to_list()
        y = airfoil.data.targets
        self.target_name = y.columns.item()
        self.dataset = np.concatenate(
            (X.to_numpy(dtype=float), y.to_numpy(dtype=float).reshape(-1, 1)), axis=1
        )

    def postprocess(self):
        pass


class AdultIncome(RealDatasetBase):
    """The Adult (census income) dataset (UCI id 2) — a classification
    example: explain `predict_proba` of the positive class.

    45,222 census records after dropping rows with missing values; the
    target is binary — whether yearly income exceeds $50K. 12 features
    remain after dropping `fnlwgt` (a sampling weight) and `education`
    (duplicated by `education-num`). Categorical columns are encoded to
    integer codes with the level names recorded in `category_names`; levels
    rarer than 50 rows are bucketed into `"Other"` (a level that rare can
    vanish from a train split, invalidating the schema). Kept in natural
    units by default (`standardize=False`); the 0/1 target is never a
    regression target — pair it with a classifier and explain the predicted
    probability.

    ```python
    data = effector.datasets.AdultIncome()
    data.x_train, data.y_train    # (36177, 12), (36177,) with y in {0, 1}
    ```
    """

    RARE_LEVEL_MIN_ROWS = 50

    def __init__(self, pcg_train=0.8, standardize=False, seed: int = 21):
        """Fetch and prepare the dataset.

        Args:
            pcg_train: Fraction of samples in the train split.
            standardize: Standardize the features (default False — natural
                units keep the partition rules readable). The 0/1 target is
                standardized too when True; leave False for classification.
            seed: Random seed of the train/test shuffle.
        """
        super().__init__(
            name="AdultIncome", pcg_train=pcg_train, standardize=standardize, seed=seed
        )

    def fetch_and_preprocess(self):
        from ucimlrepo import fetch_ucirepo

        adult = fetch_ucirepo(id=2)

        X = adult.data.features.drop(["fnlwgt", "education"], axis=1)
        y = adult.data.targets.iloc[:, 0].astype(str).str.strip()
        y = y.str.startswith(">50K").to_numpy().astype(float)

        X = X.replace("?", np.nan)
        keep = X.notna().all(axis=1).to_numpy()
        X, y = X.loc[keep].reset_index(drop=True), y[keep]

        self.feature_names = X.columns.to_list()
        self.target_name = "income>50K"
        self.feature_types = []
        self.category_names = []
        encoded = np.empty(X.shape, dtype=float)
        for j, col in enumerate(self.feature_names):
            vals = X[col]
            if vals.dtype == object:
                vals = vals.astype(str).str.strip()
                counts = vals.value_counts()
                rare = counts[counts < self.RARE_LEVEL_MIN_ROWS].index
                if len(rare):
                    vals = vals.where(~vals.isin(rare), "Other")
                levels = sorted(vals.unique())
                encoded[:, j] = vals.map({lv: i for i, lv in enumerate(levels)})
                self.feature_types.append("nominal")
                self.category_names.append(levels)
            else:
                encoded[:, j] = vals.to_numpy(dtype=float)
                self.feature_types.append(
                    "ordinal" if col == "education-num" else "continuous"
                )
                self.category_names.append(None)

        self.dataset = np.concatenate((encoded, y.reshape(-1, 1)), axis=1)

    def postprocess(self):
        pass
