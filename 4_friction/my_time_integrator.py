from potential.incremental_potential import IncrementalPotential
import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from typing import List, Tuple
from potential.barrier_potential import BarrierPotential
from potential.potential_args import PotentialArgs


def step_forward(
    x: np.ndarray,
    e: List[Tuple[int, int]],
    v: np.ndarray,
    m: List[float],
    l2: List[float],
    k: List[float],
    ground_n: np.ndarray,
    ground_o: np.ndarray,
    contact_area: List[float],
    mu: float,
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

    p_args = PotentialArgs(
        x=x_i,
        e=e,
        x_tilde=x_tilde,
        m=m,
        l2=l2,
        k=k,
        ground_n=ground_n,
        ground_o=ground_o,
        contact_area=contact_area,
        mu=mu,
        is_DBC=is_DBC,
        h=h,
    )

    prev_energy = IncrementalPotential.val(p_args)

    iter = 0
    while True:
        p_args.x = x_i
        search_dir = get_search_dir(p_args)
        if norm(search_dir, np.inf) / h < tol:
            break

        alpha = BarrierPotential.init_step_size(x_i, ground_n, ground_o, search_dir)
        x_i_next = x_i + alpha * search_dir
        p_args.x = x_i_next
        curr_energy = IncrementalPotential.val(p_args)

        while curr_energy > prev_energy:
            alpha *= 0.5
            x_i_next = x_i + alpha * search_dir
            p_args.x = x_i_next
            curr_energy = IncrementalPotential.val(p_args)

        prev_energy = curr_energy
        x_i = x_i_next
        iter += 1
        print(f"Iteration: {iter}, Energy: {curr_energy}, Alpha: {alpha}")

    v_new = (x_i - x) / h
    return x_i, v_new


def get_search_dir(p_args: PotentialArgs) -> np.ndarray:
    """
    Use Newton's method to solve the time integration step.
    """
    grad = IncrementalPotential.grad(p_args)
    hess = IncrementalPotential.hess(p_args)
    search_dir = spsolve(hess.toarray(), -grad)
    search_dir = search_dir.reshape(-1, 2)

    return search_dir
