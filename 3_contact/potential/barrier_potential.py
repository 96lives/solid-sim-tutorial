import numpy as np
from potential.base_potential import Potential
from typing import List, Tuple
from scipy.sparse import coo_matrix


class BarrierPotential(Potential):
    dhat = 0.01
    kappa = 1e5

    @staticmethod
    def barrier(d: np.ndarray) -> np.ndarray:
        ret = (
            0.5
            * BarrierPotential.kappa
            * ((d / BarrierPotential.dhat) - 1)
            * np.log(d / BarrierPotential.dhat)
        )
        ret[d > BarrierPotential.dhat] = 0.0
        return ret

    @staticmethod
    def barrier_grad(d: np.ndarray) -> np.ndarray:
        """
        db / dd
        :param d:
        :return:
        """
        d_norm = d / BarrierPotential.dhat
        ret = np.log(d_norm) + 1 - 1 / d_norm
        ret *= 0.5 * BarrierPotential.kappa / BarrierPotential.dhat
        ret[d > BarrierPotential.dhat] = 0.0
        return ret

    def barrier_hess(self, d: np.ndarray) -> np.ndarray:
        d_norm = d / BarrierPotential.dhat
        val = 1 / d_norm - 1 / d_norm**2
        val *= 0.5 * BarrierPotential.kappa / BarrierPotential.dhat**2
        ret = val * np.array([[0.0, 0.0], [0.0, 1.0]])

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
        l2 : list of shape (m,)
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
        dist = x[:, 1] - y_ground
        volume_weight = np.array(contact_area) * BarrierPotential.dhat
        ret = np.sum(volume_weight * BarrierPotential.barrier(dist))
        return ret

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
        volume_weight = np.array(contact_area) * BarrierPotential.dhat
        volume_weight = volume_weight.reshape(-1, 1)
        dist = x[:, 1] - y_ground
        partial_barrier = BarrierPotential.barrier_grad(dist).reshape(-1, 1)

        d_grad = np.zeros_like(x)
        d_grad[:, 1] = 1.0
        ret = volume_weight * partial_barrier * d_grad
        ret = ret.reshape(-1)
        return ret

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
        raise NotImplementedError("Subclasses should implement this method")
