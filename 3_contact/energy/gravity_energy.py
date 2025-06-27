import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from energy.base_energy import Energy


class GravityEnergy(Energy):
    g = 9.81  # Gravitational acceleration in m/s^2

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        m_array = np.array(m)
        energy = GravityEnergy.g * sum(m_array * x[:, 1])
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
        grad = np.zeros_like(x)
        grad[:, 1] = -GravityEnergy.g * np.array(m)
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
        # hess_zero = coo_matrix(([], ([], [])), shape=(len(x) * 2, len(x) * 2))
        hess = coo_matrix(np.eye(len(x) * 2))
        return hess
