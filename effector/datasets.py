import numpy as np
from effector import helpers

class Base:
    def __init__(self, name: str, dim: int, axis_limits: np.array):
        self.name = helpers.camel_to_snake(name)
        self.dim = dim
        self.axis_limits = axis_limits

    def generate_data(self, n: int, seed: int = 21) -> np.array:
        """Generate N samples
        Args:
            n : int
                Number of samples
            seed : int
                Seed for generating samples

        Returns:
            ndarray, shape: [n,2]
                The samples
        """
        raise NotImplementedError


class IndependentUniform(Base):
    def __init__(self, dim: int =2, low: float = 0, high: float = 1):
        axis_limits = np.array([[low, high] for _ in range(dim)]).T
        super().__init__(name=self.__class__.__name__, dim=dim, axis_limits=axis_limits)


    def generate_data(self, n: int, seed: int = 21) -> np.array:
        """Generate N samples

        Args:
            n : int
                Number of samples
            seed : int
                Seed for generating samples

        Returns:
            ndarray, shape: [n,2]
                The samples

        """
        np.random.seed(seed)
        x = np.random.uniform(self.axis_limits[0, :], self.axis_limits[1, :], (n, self.dim))
        np.random.shuffle(x)
        return x


class RealDatasetBase:
    def __init__(self, name: str, pcg_train, standardize):
        self.name = helpers.camel_to_snake(name)

        self.dataset: np.array = None

        self.feature_names = None
        self.target_name = None

        # train set
        self.x_train: np.array = None
        self.y_train: np.array = None
        self.x_train_mu = None
        self.x_train_std = None
        self.y_train_mu = None
        self.y_train_std = None

        # test set
        self.x_test: np.array = None
        self.y_test: np.array = None
        self.x_test_mu = None
        self.x_test_std = None
        self.y_test_mu = None
        self.y_test_std = None

        # main logic
        self.fetch_and_preprocess()

        self.x_train, self.x_test, self.y_train, self.y_test = self.split(
            self.dataset[:, :-1], self.dataset[:, -1], pcg_train)

        if standardize:
            self.x_train, self.x_train_mu, self.x_train_std = self.standarize(self.x_train)
            self.x_test, self.x_test_mu, self.x_test_std = self.standarize(self.x_test)
            self.y_train, self.y_train_mu, self.y_train_std = self.standarize(self.y_train)
            self.y_test, self.y_test_mu, self.y_test_std = self.standarize(self.y_test)

        self.postprocess()


    def fetch_and_preprocess(self):
        # self.dataset = ...
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError


    @staticmethod
    def standarize(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_standarized = (x - x_mean) / x_std
        return x_standarized, x_mean, x_std

    @staticmethod
    def split(x, y, pcg_train):
        n_train = int(x.shape[0] * pcg_train)

        # shuffle
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        # spit
        x_train = x[:n_train]
        x_test = x[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        return x_train, x_test, y_train, y_test


class BikeSharing(RealDatasetBase):
    def __init__(self, pcg_train=0.8, standardize=True):
        super().__init__(name="BikeSharing", pcg_train=pcg_train, standardize=standardize)

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
        self.x_train_mu[8] += 8
        self.x_train_std[8] *= 47
        self.x_test_mu[8] += 8
        self.x_test_std[8] *= 47


        self.x_train_std[9] *= 100
        self.x_test_std[9] *= 100

        self.x_train_std[10] *= 67
        self.x_test_std[10] *= 67


