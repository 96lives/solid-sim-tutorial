import numpy as np
from typing import List
from incremental_potential import IncrementalPotential
import scipy.linalg as LA
from scipy.sparse.linalg import spsolve
from utils import make_PSD
from cmath import inf


def step_forward(
    x: np.ndarray,
    e: List[List[int]],
    v: np.ndarray,
    m: List[float],
    l2: List[float],  # l2 distance of each edge
    k: List[float],  # k of each edge
    is_DBC: List[bool],  # whether each edge is Dirichlet
    h: float,  # time step size
    tol: float,
) -> [np.ndarray, np.ndarray]:
    x_tilde = x + v * h
    x_i = x.copy()

    ip_val_prev = IncrementalPotential.val(x, e, x_tilde, m, l2, k, is_DBC, h)
    # use implicit euler method
    iter = 0
    p = search_dir(x_i, e, x_tilde, m, l2, k, is_DBC, h)
    while LA.norm(p, inf) / h > tol:
        alpha = 1
        p = search_dir(x_i, e, x_tilde, m, l2, k, is_DBC, h)
        x_i = x_i + alpha * p
        iter += 1

    v = (x_i - x) / h
    return x_i, v


def search_dir(
    x: np.ndarray,
    e: List[List[int]],
    x_tilde: np.ndarray,
    m: List[float],
    l2: List[float],
    k: List[float],
    is_DBC: List[bool],  # whether each edge is Dirichlet
    h: float,
) -> np.ndarray:
    hess = IncrementalPotential.hessian(x, e, x_tilde, m, l2, k, is_DBC, h)
    grad = IncrementalPotential.grad(x, e, x_tilde, m, l2, k, is_DBC, h)

    # eliminate DOF by modifying gradient and Hessian for DBC
    for i, j in zip(*hess.nonzero()):
        if is_DBC[i // 2] or is_DBC[j // 2]:
            hess[i, j] = i == j

    for i in range(0, len(x)):
        if is_DBC[i]:
            grad[i * 2] = grad[i * 2 + 1] = 0.0

    p = spsolve(hess, -grad)
    p = p.reshape(-1, 2)
    return p
