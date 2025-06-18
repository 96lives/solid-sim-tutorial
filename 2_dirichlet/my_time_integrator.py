import numpy as np
from typing import List
from incremental_potential import IncrementalPotential
import scipy.linalg as LA
from scipy.sparse.linalg import spsolve
from utils import make_PSD


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

    ip_val = IncrementalPotential.val(x, e, x_tilde, m, l2, k, is_DBC, h)
    # use implicit euler method
    iter = 0
    p = search_dir(x_i, e, x_tilde, m, l2, k, is_DBC, h)
    while LA.norm(p / h, "inf") > tol:
        alpha = 1
        x_i = x + alpha * p
        p = search_dir(x_i, e, x_tilde, m, l2, k, is_DBC, h)
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
    x_new = spsolve(hess, -grad)
    return x_new
