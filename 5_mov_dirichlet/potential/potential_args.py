import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PotentialArgs:
    x: np.ndarray
    e: List[Tuple[int, int]]
    x_tilde: np.ndarray
    m: List[float]
    l2: List[float]
    k: List[float]
    # y_ground: float,
    ground_n: np.ndarray
    ground_o: np.ndarray
    contact_area: List[float]
    mu: float
    is_DBC: List[bool]
    h: float
    mu_lambda: np.ndarray
    x_n: np.ndarray
    """
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
    # y_ground : float
    #     Y-coordinate of the ground
    ground_n: np array
        Normal of the slope
    ground_o: np array
        A point on the slope
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
