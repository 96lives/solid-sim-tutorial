import numpy as np
from scipy.sparse import coo_matrix
import utils


class MassSpringPotential:
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
        pass

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

        :param x: array of N x 2
        :param e:
        :param x_tilde:
        :param m: array of N
        :param l2:
        :param k:
        :param h:
        :return:
            array of N x 2
        """
        ret = np.zeros_like(x)
        for i in range(len(e)):
            x1, x2 = x[e[i][0]], x[e[i][1]]
            diff = x1 - x2
            x1_grad = 2 * k[i] * (diff.dot(diff) / l2[i] - 1) * diff
            ret[e[i][0]] += x1_grad
            ret[e[i][1]] += -x1_grad
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
        for e_idx in range(0, len(e)):
            diff = x[e[e_idx][0]] - x[e[e_idx][1]]  # 2
            hessian_diff = (
                2
                * k[e_idx]
                / l2[e_idx]
                * (2 * np.outer(diff, diff) + (diff.dot(diff) - l2[e_idx]) * np.eye(2))
            )
            hessian_local = utils.make_PSD(
                np.block(
                    [[hessian_diff, -hessian_diff], [-hessian_diff, hessian_diff]]
                ),
            )
            # iterate through x_i
            for i in range(2):
                # iterate through x_j
                for j in range(2):
                    for i_coord in range(2):
                        for j_coord in range(2):
                            i_coords.append(2 * e[e_idx][i] + i_coord)
                            j_coords.append(2 * e[e_idx][j] + j_coord)
                            vals.append(hessian_local[2 * i + i_coord, 2 * j + j_coord])
        ret = coo_matrix((vals, (i_coords, j_coords)))
        return ret
