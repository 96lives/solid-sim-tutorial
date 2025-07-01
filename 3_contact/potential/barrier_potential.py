import numpy as np
from potential.base_potential import Potential
from typing import List, Tuple
from scipy.sparse import coo_matrix


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
        x: np.ndarray,
        e: List[Tuple[int, int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        y_ground: float,
        contact_area: List[float],
        is_DBC: List[bool],
        h: float,
    ) -> float:
        """
        Custom implementation of the time integration step.

        Parameters:
        -----------
        x : numpy.ndarray of shape (n, 2)
            Current positions of nodes
        e : list of shape (m, 2)
            List of edges (pairs of node indices)
        x_tilde : numpy.ndarray of shape (n, 2)
            Current velocities of nodes
        m : list of shape (n,)
            Mass of each node
        l2 : list of shape (m,)
            Rest length squared for each spring
        k : list of shape (m,)
            Spring stiffness for each spring
        y_ground : float
            Y-coordinate of the ground
        contact_area : list of shape (n,)
            Contact area for each node
        is_DBC : list
            Boolean flags for Dirichlet boundary conditions
        h : float
            Time step size
        Returns:
        --------
        tuple
            Updated positions and velocities (x_new, v_new)
        """
        dist = x[:, 1] - y_ground
        volume_weight = np.array(contact_area) * BarrierPotential.dhat
        ret = np.sum(volume_weight * BarrierPotential.barrier(dist))
        return ret

    @staticmethod
    def grad(
        x: np.ndarray,
        e: List[Tuple[int, int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        y_ground: float,
        contact_area: List[float],
        is_DBC: List[bool],
        h: float,
    ) -> np.ndarray:
        """
        :return:
            Gradient of the potential with respect to the node positions
            :rtype: np.ndarray of shape 2*n
        """
        volume_weight = np.array(contact_area) * BarrierPotential.dhat
        volume_weight = volume_weight.reshape(-1, 1)  # n x 1
        dist = x[:, 1] - y_ground  # n
        barrier_partial = BarrierPotential.barrier_partial(dist).reshape(-1, 1)  # n x 1

        d_grad = np.zeros_like(x)
        d_grad[:, 1] = 1.0
        ret = volume_weight * barrier_partial * d_grad
        ret = ret.reshape(-1)
        return ret

    @staticmethod
    def hess(
        x: np.ndarray,
        e: List[Tuple[int, int]],
        x_tilde: np.ndarray,
        m: List[float],
        l2: List[float],
        k: List[float],
        y_ground: float,
        contact_area: List[float],
        is_DBC: List[bool],
        h: float,
    ) -> coo_matrix:
        volume_weight = np.array(contact_area) * BarrierPotential.dhat
        dist = x[:, 1] - y_ground  # n
        barrier_second_partial = BarrierPotential.barrier_second_partial(dist)

        # create sparse hessian matrix
        row_idxs, col_idxs, vals = [], [], []
        for x_idx in range(len(x)):
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

            # local_d_grad = np.zeros(2)
            # local_d_grad[1] = 1.0
            # local_hess = barrier_second_partial[x_idx] * np.outer(
            #     local_d_grad, local_d_grad
            # )
            # local_hess *= volume_weight[x_idx]
            # for local_row_idx in range(2):
            #     for local_col_idx in range(2):
            #         row_idxs.append(2 * x_idx + local_row_idx)
            #         col_idxs.append(2 * x_idx + local_col_idx)
            #         vals.append(local_hess[local_row_idx][local_col_idx])
        ret = coo_matrix((vals, (row_idxs, col_idxs)), shape=(2 * len(x), 2 * len(x)))
        return ret

    @staticmethod
    def init_step_size(x: np.ndarray, y_ground: float, p: np.ndarray) -> float:
        """

        :param x: numpy.ndarray of shape (n, 2)
        :param y_ground: float
        :param p: search direction. numpy.ndarray of shape (n, 2)
        :return:
        """
        alpha = (y_ground - x[:, 1]) / p[:, 1]
        alpha = min(0.9 * min(alpha), 1.0)
        return alpha
