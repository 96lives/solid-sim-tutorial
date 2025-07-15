from typing import List, Tuple
import numpy as np
from scipy.sparse.linalg import spsolve
from potential.incremental_potential import IncrementalPotential
from potential.potential_args import PotentialArgs
from potential.barrier_potential import BarrierPotential


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
    DBC_limit: List[np.ndarray],  # dirichlet node limit position
    DBC_stiff: List[
        float
    ],  # DBC stiffness, adjusted and warm-started across time steps
    h: float,
    tol: float,
):
    x_tilde = x + v * h
    x_i = x.copy()  # iterate x_i
    mu_lambda = BarrierPotential.compute_mu_lambda(x, n, o, contact_area, mu)
    p_args = PotentialArgs(
        x=x_i,
        e=e,
        x_tilde=x_tilde,
        m=m,
        l2=l2,
        k=k,
        ground_n=n,
        ground_o=o,
        contact_area=contact_area,
        mu=mu,
        is_DBC=is_DBC,
        DBC=DBC,
        DBC_v=DBC_v,
        DBC_limit=DBC_limit,
        DBC_stiff=DBC_stiff,
        h=h,
        mu_lambda=mu_lambda,
        x_n=x.copy(),
    )

    iter = 0
    while True:
        p_args.x = x_i
        p = get_search_dir(p_args)
        if np.linalg.norm(p, np.inf) / h < tol:
            break
        curr_energy = IncrementalPotential.val(p_args)

        # set alpha so that energy decreases
        alpha = BarrierPotential.init_step_size(x_i, n, o, p)
        p_args.x = x_i + alpha * p
        if curr_energy > IncrementalPotential.val(p_args):
            alpha *= 0.5
            p_args.x = x_i + alpha * p

        x_i = x_i + alpha * p
        iter += 1
        print(f"Iteration: {iter}, Alpha: {alpha}")

    v_i = (x_i - x) / h
    return x_i, v_i


def get_search_dir(p_args: PotentialArgs):
    grad = IncrementalPotential.grad(p_args)
    hess = IncrementalPotential.hess(p_args)

    p = spsolve(hess, -grad)
    p = p.reshape(-1, 2)
    return p
