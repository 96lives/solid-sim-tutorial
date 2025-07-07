import numpy as np
from potential.base_potential import Potential
from typing import List, Tuple
from scipy.sparse import coo_matrix
from potential.potential_args import PotentialArgs


class BarrierPotential(Potential):
    dhat = 0.01
    kappa = 1e5

    @staticmethod
    def barrier(d: np.ndarray) -> np.ndarray:
        """
        :param d: np array of shape (n,), where n is the number of nodes.
        :return:
            np array of shape (n,) containing the values of the barrier potential at each node.
        """
        s = d / BarrierPotential.dhat
        ret = 0.5 * BarrierPotential.kappa * (s - 1) * np.log(s)
        ret[d >= BarrierPotential.dhat] = 0.0
        return ret

    @staticmethod
    def barrier_partial(d: np.ndarray) -> np.ndarray:
        """
        Partial derivative of barrier potential w.r.t. d
        :param d: np array of shape (n,), where n is the number of nodes.
        :return:
            np array of shape (n,) containing the values of the barrier potential at each node.
        """
        s = d / BarrierPotential.dhat
        ret = np.log(s) + 1 - (1 / s)
        ret *= 0.5 * BarrierPotential.kappa / BarrierPotential.dhat
        ret[d >= BarrierPotential.dhat] = 0.0
        return ret

    @staticmethod
    def barrier_second_partial(d: np.ndarray) -> np.ndarray:
        """
        Second partial derivative of barrier potential w.r.t. d
        :param d: np array of shape (n,), where n is the number of nodes.
        :return:
            np array of shape (n,) containing the values of the barrier potential at each node.
        """
        d_norm = d / BarrierPotential.dhat
        ret = 1 / d_norm + (1 / d_norm) ** 2
        ret *= 0.5 * BarrierPotential.kappa / (BarrierPotential.dhat**2)
        ret[d >= BarrierPotential.dhat] = 0.0
        return ret

    @staticmethod
    def val(
        p_args: PotentialArgs,
    ) -> float:
        dist = np.sum(
            p_args.ground_n.reshape(1, 2) * (p_args.x - p_args.ground_o.reshape(1, 2)),
            axis=1,
        )
        volume_weight = np.array(p_args.contact_area) * BarrierPotential.dhat
        ret = np.sum(volume_weight * BarrierPotential.barrier(dist))
        return ret

    @staticmethod
    def grad(
        p_args: PotentialArgs,
    ) -> np.ndarray:
        """
        :return:
            Gradient of the potential with respect to the node positions
            :rtype: np.ndarray of shape 2*n
        """
        volume_weight = np.array(p_args.contact_area) * BarrierPotential.dhat
        volume_weight = volume_weight.reshape(-1, 1)  # n x 1
        dist = np.sum(
            p_args.ground_n.reshape(1, 2) * (p_args.x - p_args.ground_o.reshape(1, 2)),
            axis=1,
        )
        barrier_partial = BarrierPotential.barrier_partial(dist).reshape(-1, 1)  # n x 1

        d_grad = np.repeat(p_args.ground_n.reshape(1, 2), p_args.x.shape[0], axis=0)
        ret = volume_weight * barrier_partial * d_grad
        ret = ret.reshape(-1)
        return ret

    @staticmethod
    def hess(
        p_args: PotentialArgs,
    ) -> coo_matrix:
        volume_weight = np.array(p_args.contact_area) * BarrierPotential.dhat
        dist = np.sum(
            p_args.ground_n.reshape(1, 2) * (p_args.x - p_args.ground_o.reshape(1, 2)),
            axis=1,
        )
        barrier_second_partial = BarrierPotential.barrier_second_partial(dist)

        # create sparse hessian matrix
        row_idxs, col_idxs, vals = [], [], []
        for x_idx in range(len(p_args.x)):
            row_idxs.append(2 * x_idx + 1)
            col_idxs.append(2 * x_idx + 1)

            d = dist[x_idx]
            dhat = BarrierPotential.dhat
            if d > dhat:
                val = 0.0
            else:
                val = (
                    volume_weight[x_idx]
                    * BarrierPotential.kappa
                    / (2 * d * d * dhat)
                    * (d + dhat)
                )
            vals.append(val)
        ret = coo_matrix(
            (vals, (row_idxs, col_idxs)), shape=(2 * len(p_args.x), 2 * len(p_args.x))
        )
        return ret

    @staticmethod
    def init_step_size(
        x: np.ndarray, ground_n: np.ndarray, ground_o: np.ndarray, p: np.ndarray
    ) -> float:
        """
        :param x: numpy.ndarray of shape (n, 2)
        :param p: search direction. numpy.ndarray of shape (n, 2)
        :return:
        """
        alpha = 1.0
        for i in range(len(x)):
            alpha_i = np.dot(ground_n, ground_o - x[i]) / np.dot(ground_n, p[i])
            if np.dot(ground_n, p[i]) < 0:
                alpha = min(alpha, 0.9 * alpha_i)
        return alpha
