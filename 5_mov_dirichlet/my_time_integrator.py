from typing import List, Tuple
import numpy as np


def step_forward(
    x: np.ndarray,
    e: List[Tuple[int, int]],
    v: np.ndarray,
    m: List[float],
    l2: List[float],
    k: List[float],
    n: np.ndarray,
    o: np.ndarray,
    contact_area: List[float],
    mu: float,
    is_DBC: List[bool],
    DBC: List[int],  # dirichlet node index
    DBC_v: List[np.ndarray],  # dirichlet node velocity
    DBC_limit,  # dirichlet node limit position
    DBC_stiff: List[
        float
    ],  # DBC stiffness, adjusted and warm-started across time steps
    h: float,
    tol: float,
):
    pass
