import numpy as np
from scipy.sparse import coo_matrix
from typing import List
from utils import make_PSD


class InertiaPotential:
    @staticmethod
    def val(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> float:
        pass

    @staticmethod
    def grad(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> np.ndarray:
        grad = np.diag(m * 2) @ (x - x_tilde).reshape(-1, 1)
        return grad.reshape(-1)

    @staticmethod
    def hessian(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> coo_matrix:
        i_coords, j_coords, vals = [], [], []
        for x_idx in range(0, len(x)):
            for axis in range(2):
                i_coords.append(2 * x_idx + axis)
                j_coords.append(2 * x_idx + axis)
                vals.append(m[x_idx])
        ret = coo_matrix((vals, (i_coords, j_coords)))
        return ret


class MassSpringPotential:
    @staticmethod
    def val(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> float:
        pass

    @staticmethod
    def grad(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> np.ndarray:
        grad = np.zeros_like(x)
        for e_idx in range(len(e)):
            x1, x2 = x[e[e_idx][0]], x[e[e_idx][1]]
            diff = x1 - x2
            x1_grad = 2 * k[e_idx] * (diff.dot(diff) / l2[e_idx] - 1) * diff
            grad[e[e_idx][0]] += x1_grad
            grad[e[e_idx][1]] -= x1_grad
        return grad.reshape(-1)

    @staticmethod
    def hessian(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> coo_matrix:
        i_coords, j_coords, vals = [], [], []
        for e_idx in range(len(e)):
            diff = x[e[e_idx][0]] - x[e[e_idx][1]]
            hess_x1 = (
                2
                * k[e_idx]
                / l2[e_idx]
                * (
                    2 * np.outer(diff, diff)
                    + (np.dot(diff, diff) - l2[e_idx]) * np.eye(2)
                )
            )  # 2 x 2 hessian
            hess_local = make_PSD(
                np.block([[hess_x1, -hess_x1], [-hess_x1, hess_x1]]),
            )

            for e_idx1 in range(2):
                for e_idx2 in range(2):
                    for axis1 in range(2):
                        for axis2 in range(2):
                            i_coords.append(2 * e[e_idx][e_idx1] + axis1)
                            j_coords.append(2 * e[e_idx][e_idx2] + axis2)
                            vals.append(
                                hess_local[2 * e_idx1 + axis1, 2 * e_idx2 + axis2]
                            )
        ret = coo_matrix((vals, (i_coords, j_coords)))
        return ret


class GravityPotential:
    g = np.array([0.0, -9.81])
    # g = np.array([0.0, -9.81])

    @staticmethod
    def val(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> float:
        pass

    @staticmethod
    def grad(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> np.ndarray:
        grad = -np.array([m, m]).T * GravityPotential.g.reshape(-1, 2)
        grad = grad.reshape(-1)
        return grad

    @staticmethod
    def hessian(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> coo_matrix:
        # does not have hessian
        return coo_matrix(([], ([], [])), shape=(2 * len(x), 2 * len(x)))


class IncrementalPotential:
    @staticmethod
    def val(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> float:
        pass

    @staticmethod
    def grad(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> np.ndarray:
        inertia = InertiaPotential.grad(x, e, x_tilde, m, l2, k, is_DBC, h)
        mass_spring = MassSpringPotential.grad(x, e, x_tilde, m, l2, k, is_DBC, h)
        gravity = GravityPotential.grad(x, e, x_tilde, m, l2, k, is_DBC, h)
        # grad = inertia + (h**2) * (mass_spring + gravity)
        grad = inertia + (h**2) * gravity
        return grad

    @staticmethod
    def hessian(
        x: np.ndarray,
        e: List[List[int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        is_DBC: List[bool],  # whether each edge is Dirichlet
        h: float,
    ) -> coo_matrix:
        inertia_sparse = InertiaPotential.hessian(x, e, x_tilde, m, l2, k, is_DBC, h)
        mass_spring_sparse = MassSpringPotential.hessian(
            x, e, x_tilde, m, l2, k, is_DBC, h
        )
        gravity_sparse = GravityPotential.hessian(x, e, x_tilde, m, l2, k, is_DBC, h)
        hessian = inertia_sparse + (h**2) * (mass_spring_sparse + gravity_sparse)
        hessian = inertia_sparse + (h**2) * gravity_sparse
        return hessian
