import numpy as np
from scipy.sparse import coo_matrix

from mass_spring_potential import MassSpringPotential
from inertia_potential import InertiaPotential


class IncrementalPotential:

    @staticmethod
    def val(
        x: np.ndarray,
        e: np.ndarray,
        x_tilde: np.ndarray,
        m: np.ndarray,
        l2: float,
        k: np.ndarray,
        h: float,
    ) -> float:

        pass

    @staticmethod
    def grad(
        x: np.ndarray,
        e: np.ndarray,
        x_tilde: np.ndarray,
        m: np.ndarray,
        l2: np.ndarray,
        k: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        :param x:
        :param e:
        :param x_tilde:
        :param m:
        :param l2:
        :param k:
        :param h:
        :return:
            array of N x 2
        """
        inertia_sparse = InertiaPotential.grad(x, e, x_tilde, m, l2, k, h)
        mass_spring_sparse = MassSpringPotential.grad(x, e, x_tilde, m, l2, k, h)
        grad = inertia_sparse + (h**2) * mass_spring_sparse
        return grad

    @staticmethod
    def hessian(
        x: np.ndarray,
        e: np.ndarray,
        x_tilde: np.ndarray,
        m: np.ndarray,
        l2: np.ndarray,
        k: np.ndarray,
        h: float,
    ) -> coo_matrix:
        inertia_sparse = InertiaPotential.hessian(x, e, x_tilde, m, l2, k, h)
        mass_spring_sparse = MassSpringPotential.hessian(x, e, x_tilde, m, l2, k, h)
        hessian = inertia_sparse + (h**2) * mass_spring_sparse
        return hessian
