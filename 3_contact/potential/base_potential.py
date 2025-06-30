import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix


class Potential:

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
        raise NotImplementedError("Subclasses should implement this method")

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
        raise NotImplementedError("Subclasses should implement this method")

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
