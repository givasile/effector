from effector import helpers
import re
import numpy as np

class Base:
    def __init__(self, name):
        # CamelCase to snake_case
        self.name = helpers.camel_to_snake(name)

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConditionalInteraction(Base):
    def __init__(self):
        """Define a simple model.

        $f(x_1, x_2, x_3) = -x_1^2\mathbb{1}_{x_2 < 0} + x_1^2\mathbb{1}_{x_2 \geq 0} + e^{x_3}$

        """
        super().__init__(name=self.__class__.__name__)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Output of the model, shape (N,)
        """
        y = np.exp(x[:, 2])
        ind = x[:, 1] < 0
        y[ind] += -x[ind, 0]**2
        y[~ind] += x[~ind, 0]**2
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of the model.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Jacobian of the model, shape (N, 3)
        """
        y = np.zeros_like(x)
        y[:, 2] = np.exp(x[:, 2])
        ind = x[:, 1] < 0
        y[ind, 0] = -2*x[ind, 0]
        y[~ind, 0] = 2*x[~ind, 0]
        return y

class DoubleConditionalInteraction(Base):
    def __init__(self):
        """Define a simple model.

        $f(x_1, x_2, x_3) = -3x_1^2\mathbb{1}_{x_2 < 0}\mathbb{1}_{x_3 < 0} +
                            +x_1^2\mathbb{1}_{x_2 < 0}\mathbb{1}_{x_3 \geq 0}
                            -e^{x_1}\mathbb{1}_{x_2 \geq 0}\mathbb{1}_{x_3 < 0}
                            +e^{3x_1}\mathbb{1}_{x_2 \geq 0}\mathbb{1}_{x_3 \geq 0}$
                            $

        """
        super().__init__(name=self.__class__.__name__)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Output of the model, shape (N,)
        """
        y = np.zeros(x.shape[0])
        ind1 = x[:, 1] < 0
        ind2 = x[:, 2] < 0
        y[ind1 & ind2] = -3*x[ind1 & ind2, 0]**2
        y[ind1 & ~ind2] = x[ind1 & ~ind2, 0]**2
        y[~ind1 & ind2] = -np.exp(x[~ind1 & ind2, 0])
        y[~ind1 & ~ind2] = np.exp(3*x[~ind1 & ~ind2, 0])
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of the model.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Jacobian of the model, shape (N, 3)
        """
        y = np.zeros_like(x)
        ind1 = x[:, 1] < 0
        ind2 = x[:, 2] < 0
        y[ind1 & ind2, 0] = -2*3*x[ind1 & ind2, 0]
        y[ind1 & ~ind2, 0] = 2*x[ind1 & ~ind2, 0]
        y[~ind1 & ind2, 0] = -np.exp(x[~ind1 & ind2, 0])
        y[~ind1 & ~ind2, 0] = 3*np.exp(3*x[~ind1 & ~ind2, 0])
        return y
        
class ConditionalInteraction4Regions(Base):
    def __init__(self):
        """
        $f(x_1, x_2, x_3) = 
        \begin{cases} 
        -x_1^2 + e^{x_3}, & \text{if } x_2 < 0 \text{ and } x_3 < 0 \\
        x_1^2 + e^{x_3}, & \text{if } x_2 < 0 \text{ and } x_3 \geq 0 \\
        -x_1^4 + e^{x_3}, & \text{if } x_2 \geq 0 \text{ and } x_3 < 0 \\
        x_1^4 + e^{x_3}, & \text{if } x_2 \geq 0 \text{ and } x_3 \geq 0
        \end{cases}
        """
        super().__init__(name=self.__class__.__name__)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            x : Input data, shape (N, 4)

        Returns:
            Output of the model, shape (N,)
        """
        y = np.exp(x[:, 2])

        mask1 = (x[:, 1] < 0) & (x[:, 2] < 0)
        mask2 = (x[:, 1] < 0) & (x[:, 2] >= 0)
        mask3 = (x[:, 1] >= 0) & (x[:, 2] < 0)
        mask4 = (x[:, 1] >= 0) & (x[:, 2] >= 0)

        y[mask1] += -x[mask1, 0] ** 2
        y[mask2] += x[mask2, 0] ** 2
        y[mask3] += -x[mask3, 0] ** 4
        y[mask4] += x[mask4, 0] ** 4

        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of the model.

        Args:
            x : Input data, shape (N, 4)

        Returns:
            Jacobian of the model, shape (N, 4)
        """
        y = np.zeros(x.shape)

        y[:, 2] = np.exp(x[:, 2])

        mask1 = (x[:, 1] < 0) & (x[:, 2] < 0)
        mask2 = (x[:, 1] < 0) & (x[:, 2] >= 0)
        mask3 = (x[:, 1] >= 0) & (x[:, 2] < 0)
        mask4 = (x[:, 1] >= 0) & (x[:, 2] >= 0)

        y[mask1, 0] = -2 * x[mask1, 0]
        y[mask2, 0] = 2 * x[mask2, 0]
        y[mask3, 0] = -4 * x[mask3, 0] ** 3
        y[mask4, 0] = 4 * x[mask4, 0] ** 3

        return y

class GeneralInteraction(Base):
    def __init__(self):
        """Define a simple model.

        $f(x_1, x_2, x_3) = x_1 x_2^2 + e^{x_3}$

        """
        super().__init__(name=self.__class__.__name__)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Output of the model, shape (N,)
        """
        y = x[:, 0] * x[:, 1] ** 2 + np.exp(x[:, 2])
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of the model.

        Args:
            x : Input data, shape (N, 3)

        Returns:
            Jacobian of the model, shape (N, 3)
        """
        y = np.zeros_like(x)
        y[:, 0] = x[:, 1] ** 2
        y[:, 1] = 2 * x[:, 0] * x[:, 1]
        y[:, 2] = np.exp(x[:, 2])
        return y