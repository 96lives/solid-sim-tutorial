import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from potential.potential_args import PotentialArgs


class Potential:
    @staticmethod
    def val(p_args: PotentialArgs) -> float:
        """
        Custom implementation of the time integration step.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def grad(
        p_args: PotentialArgs,
    ) -> np.ndarray:
        """
        :return:
            Gradient of the potential with respect to the node positions
            :rtype: np.ndarray of shape 2*n
        """
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def hess(
        p_args: PotentialArgs,
    ) -> coo_matrix:
        raise NotImplementedError("Subclasses should implement this method")
