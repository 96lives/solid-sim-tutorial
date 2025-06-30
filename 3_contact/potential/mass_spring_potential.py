from potential.base_potential import Potential
import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from utils import make_PSD


class MassSpringPotential(Potential):
    @staticmethod
    def val(
        x: np.ndarray,
        e: List[Tuple[int, int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        y_ground: float,
        contact_area: List[float],
        is_DBC: List[bool],
        h: float,
    ) -> float:
        """
        Custom implementation of the time integration step.

        Parameters:
        -----------
        x : numpy.ndarray of shape (n, 2)
            Current positions of nodes
        e : list of shape (m, 2)
            List of edges (pairs of node indices)
        x_tilde : numpy.ndarray of shape (n, 2)
            Current velocities of nodes
        m : list of shape (n,)
            Mass of each node
        l2 : list of shape (n,)
            Rest length squared for each spring
        k : list of shape (m,)
            Spring stiffness for each spring
        y_ground : float
            Y-coordinate of the ground
        contact_area : list of shape (n,)
            Contact area for each node
        is_DBC : list
            Boolean flags for Dirichlet boundary conditions
        h : float
            Time step size
        Returns:
        --------
        tuple
            Updated positions and velocities (x_new, v_new)
        """
        e_arr, l2_arr, k_arr = np.array(e), np.array(l2), np.array(k)
        x0, x1 = x[e_arr[:, 0]], x[e_arr[:, 1]]
        diff = np.sum((x1 - x0) ** 2, axis=1)
        energies = 0.5 * l2_arr * k_arr * (diff / l2_arr - 1) ** 2
        energy = np.sum(energies)
        return energy

    @staticmethod
    def grad(
        x: np.ndarray,
        e: List[Tuple[int, int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        y_ground: float,
        contact_area: List[float],
        is_DBC: List[bool],
        h: float,
    ) -> np.ndarray:
        """
        :return:
            Gradient of the potential with respect to the node positions
            :rtype: np.ndarray of shape 2*n
        """
        grad = np.zeros_like(x)

        for i in range(len(e)):
            e0, e1 = e[i][0], e[i][1]
            diff = x[e0] - x[e1]  # n x 2
            grad[e0] += 2 * k[i] * (np.sum(diff**2) / l2[i] - 1) * diff
            grad[e1] -= 2 * k[i] * (np.sum(diff**2) / l2[i] - 1) * diff
        grad = grad.reshape(-1)
        return grad

    @staticmethod
    def hess(
        x: np.ndarray,
        e: List[Tuple[int, int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        y_ground: float,
        contact_area: List[float],
        is_DBC: List[bool],
        h: float,
    ) -> coo_matrix:
        hess = np.zeros((len(x), 2, len(x), 2))  # dxdx, dxdy, dydx, dydy
        for i in range(len(e)):
            e0, e1 = e[i][0], e[i][1]
            x0, x1 = x[e0], x[e1]
            diff = x1 - x0
            hess_block_local_term0 = 4 * k[i] / l2[i] * np.outer(diff, diff)
            hess_block_local_term1 = (
                2 * k[i] * (np.sum(diff**2) / l2[i] - 1) * np.eye(2)
            )
            # 2 x 2 matrix
            hess_local_block = hess_block_local_term0 + hess_block_local_term1
            # 4 x 4 matrix
            hess_local = np.block(
                [
                    [hess_local_block, -hess_local_block],
                    [-hess_local_block, hess_local_block],
                ]
            )
            hess_local = make_PSD(hess_local)

            hess[e0, :, e0] += hess_local[:2, :2]
            hess[e0, :, e1] += hess_local[:2, 2:]
            hess[e1, :, e0] += hess_local[2:, :2]
            hess[e1, :, e1] += hess_local[2:, 2:]
        hess = hess.reshape(2 * len(x), 2 * len(x))
        hess_sparse = coo_matrix(hess)
        return hess_sparse
