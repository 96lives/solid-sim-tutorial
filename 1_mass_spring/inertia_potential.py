import numpy as np
from scipy.sparse import coo_matrix


class InertiaPotential:
    @staticmethod
    def val(
        x: np.ndarray,
        e: np.ndarray,
        x_tilde: np.ndarray,
        m: np.ndarray,
        l2: np.ndarray,
        k: np.ndarray,
        h: float,
    ) -> float:
        x_diff = x - x_tilde
        x_diff = x_diff[np.newaxis, ...]
        potential = 0.5 * x_diff.T @ m @ x_diff  # 1 x 1
        return potential.item()

    @staticmethod
    def grad(
        x: np.ndarray,
        e: np.ndarray,
        x_tilde: np.ndarray,
        m: np.ndarray,
        l2: np.ndarray,
        k: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        :param x: N x 2
        :param e:
        :param x_tilde:
        :param m: N
        :param l2:
        :param k:
        :param h:
        :return:
            grad of N x 2
        """
        ret = m[..., np.newaxis] * (x - x_tilde)
        return ret

    @staticmethod
    def hessian(
        x: np.ndarray,
        e: np.ndarray,
        x_tilde: np.ndarray,
        m: np.ndarray,
        l2: np.ndarray,
        k: np.ndarray,
        h: float,
    ) -> coo_matrix:
        i_coords, j_coords, vals = [], [], []
        for x_idx in range(0, len(x)):
            for x_coord in range(0, 2):
                i_coords.append(2 * x_idx + x_coord)
                j_coords.append(2 * x_idx + x_coord)
                vals.append(m[x_idx])
        ret = coo_matrix((vals, (i_coords, j_coords)))
        return ret
