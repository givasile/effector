import typing

import numpy as np
import scipy as sp
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
