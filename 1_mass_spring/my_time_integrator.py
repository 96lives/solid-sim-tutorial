import numpy as np
from cmath import inf
from numpy.linalg import norm
from typing import Tuple
from incremental_potential import IncrementalPotential
from scipy.sparse.linalg import spsolve


def step_forward(
    x: np.ndarray,  # N x 2
    e: np.ndarray,  # E x 2
    v: np.ndarray,  # N x 2
    m: np.ndarray,  # N
    l2: np.ndarray,  # E
    k: np.ndarray,
    h: float,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_tilde = x + v * h
    x_iter = x.copy()

    # newton loop
    iter = 0
    p = search_dir(x_iter, e, x_tilde, m, l2, k, h)
    while norm(p, inf) / h > tol:
        alpha = 1.0

        # Line search
        x_iter = x_iter + alpha * p
        p = search_dir(x_iter, e, x_tilde, m, l2, k, h)

    v_iter = (x_iter - x) / h
    return x_iter, v_iter


def search_dir(
    x: np.ndarray,
    e: np.ndarray,
    x_tilde: np.ndarray,
    m: np.ndarray,
    l2: np.ndarray,
    k: np.ndarray,
    h: float,
) -> np.ndarray:
    """
    :return:
        p: np array of N x2
    """
    hess = IncrementalPotential.hessian(x, e, x_tilde, m, l2, k, h)
    grad = IncrementalPotential.grad(x, e, x_tilde, m, l2, k, h)
    grad = grad.reshape(-1)
    search_dir = -spsolve(hess, grad)
    search_dir = search_dir.reshape(-1, 2)
    return search_dir
