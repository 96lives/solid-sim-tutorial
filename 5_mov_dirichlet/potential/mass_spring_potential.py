from potential.base_potential import Potential
import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from utils import make_PSD
from potential.potential_args import PotentialArgs


class MassSpringPotential(Potential):
    @staticmethod
    def val(
        p_args: PotentialArgs,
    ) -> float:

        e_arr, l2_arr, k_arr = (
            np.array(p_args.e),
            np.array(p_args.l2),
            np.array(p_args.k),
        )
        x0, x1 = p_args.x[e_arr[:, 0]], p_args.x[e_arr[:, 1]]
        diff = np.sum((x1 - x0) ** 2, axis=1)
        energies = 0.5 * l2_arr * k_arr * (diff / l2_arr - 1) ** 2
        energy = np.sum(energies)
        return energy

    @staticmethod
    def grad(
        p_args: PotentialArgs,
    ) -> np.ndarray:
        """
        :return:
            Gradient of the potential with respect to the node positions
            :rtype: np.ndarray of shape 2*n
        """
        grad = np.zeros_like(p_args.x)

        for i in range(len(p_args.e)):
            e0, e1 = p_args.e[i][0], p_args.e[i][1]
            diff = p_args.x[e0] - p_args.x[e1]  # n x 2
            grad[e0] += 2 * p_args.k[i] * (np.sum(diff**2) / p_args.l2[i] - 1) * diff
            grad[e1] -= 2 * p_args.k[i] * (np.sum(diff**2) / p_args.l2[i] - 1) * diff
        grad = grad.reshape(-1)
        return grad

    @staticmethod
    def hess(
        p_args: PotentialArgs,
    ) -> coo_matrix:
        hess = np.zeros((len(p_args.x), 2, len(p_args.x), 2))  # dxdx, dxdy, dydx, dydy
        for i in range(len(p_args.e)):
            e0, e1 = p_args.e[i][0], p_args.e[i][1]
            x0, x1 = p_args.x[e0], p_args.x[e1]
            diff = x1 - x0
            hess_block_local_term0 = (
                4 * p_args.k[i] / p_args.l2[i] * np.outer(diff, diff)
            )
            hess_block_local_term1 = (
                2 * p_args.k[i] * (np.sum(diff**2) / p_args.l2[i] - 1) * np.eye(2)
            )
            # 2 x 2 matrix
            hess_local_block = hess_block_local_term0 + hess_block_local_term1
            # 4 x 4 matrix
            hess_local = np.block(
                [
                    [hess_local_block, -hess_local_block],
                    [-hess_local_block, hess_local_block],
                ]
            )
            hess_local = make_PSD(hess_local)

            hess[e0, :, e0] += hess_local[:2, :2]
            hess[e0, :, e1] += hess_local[:2, 2:]
            hess[e1, :, e0] += hess_local[2:, :2]
            hess[e1, :, e1] += hess_local[2:, 2:]
        hess = hess.reshape(2 * len(p_args.x), 2 * len(p_args.x))
        hess_sparse = coo_matrix(hess)
        return hess_sparse
