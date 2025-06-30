from potential.incremental_potential import IncrementalPotential
import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from typing import List, Tuple


def step_forward(
    x: np.ndarray,
    e: List[Tuple[int, int]],
    v: np.ndarray,
    m: List[float],
    l2: List[float],
    k: List[float],
    y_ground: float,
    contact_area: List[float],
    is_DBC: List[bool],
    h: float,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom implementation of the time integration step.
    Returns:
    --------
    tuple
        Updated positions and velocities (x_new, v_new)
    """
    # Function body unchanged
    x_tilde = x + h * v
    x_i = x.copy()
    prev_energy = IncrementalPotential.val(
        x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
    )

    iter = 0
    while True:
        search_dir = get_search_dir(
            x_i, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        if norm(search_dir, np.inf) < tol:
            break

        alpha = 1.0
        x_i_next = x_i + alpha * search_dir
        curr_energy = IncrementalPotential.val(
            x_i_next, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        while curr_energy > prev_energy:
            alpha *= 0.5
            x_i_next = x_i + alpha * search_dir
            curr_energy = IncrementalPotential.val(
                x_i_next, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
            )

        prev_energy = curr_energy
        x_i = x_i_next
        iter += 1
        print(f"Iteration: {iter}, Energy: {curr_energy}")

    v_new = (x_i - x) / h
    return x_i, v_new


def get_search_dir(
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
    Use Newton's method to solve the time integration step.
    """
    grad = IncrementalPotential.grad(
        x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
    )
    hess = IncrementalPotential.hess(
        x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
    )
    x_next = spsolve(hess.toarray(), -grad)
    x_next = x_next.reshape(x.shape[0], 2)
    search_dir = x_next - x
    return search_dir
