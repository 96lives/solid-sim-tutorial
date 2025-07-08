from potential.base_potential import Potential
from potential.potential_args import PotentialArgs
from scipy.sparse import coo_matrix
from utils import make_PSD
import numpy as np


class FrictionPotential(Potential):
    epsv = 1e-3

    @staticmethod
    def f0(y: float, h_hat: float) -> float:
        if y > FrictionPotential.epsv * h_hat:
            return y
        elif y < 0:
            raise ValueError("y must be positive")
        else:
            val = (
                -(y**3) / (3 * FrictionPotential.epsv**2 * h_hat**2)
                + y**2 / (FrictionPotential.epsv * h_hat)
                + FrictionPotential.epsv * h_hat / 3
            )
            return val

    @staticmethod
    def f1_div_v_bar_norm(v_bar_norm: float) -> float:
        if v_bar_norm >= FrictionPotential.epsv:
            return 1 / v_bar_norm
        elif v_bar_norm < 0:
            raise ValueError("y must be positive")
        else:
            val = -(v_bar_norm) / FrictionPotential.epsv**2 + 2 / FrictionPotential.epsv
            return val

    @staticmethod
    def f1_hess_local_inner(v_bar: np.ndarray[2, 2]) -> np.ndarray[2, 2]:
        v_bar_norm = np.linalg.norm(v_bar)
        # first term
        if v_bar_norm > FrictionPotential.epsv:
            first_term_coeff = -1 / np.linalg.norm(v_bar_norm) ** 2
            second_term_coeff = 1 / v_bar_norm
        elif v_bar_norm < 0:
            raise ValueError("v_bar_norm must be positive")
        else:
            first_term_coeff = -1 / FrictionPotential.epsv**2
            second_term_coeff = (
                -v_bar_norm / FrictionPotential.epsv**2 + 2 / FrictionPotential.epsv
            )

        first_term = first_term_coeff * np.outer(v_bar, v_bar)
        second_term = second_term_coeff * np.eye(2)
        if v_bar_norm != 0:
            ret = first_term + second_term
        else:
            ret = second_term
        ret = make_PSD(ret)
        return ret

    @staticmethod
    def val(p_args: PotentialArgs) -> float:
        v = (p_args.x - p_args.x_n) / p_args.h
        T = np.eye(2) - np.outer(p_args.ground_n, p_args.ground_n)

        ret = 0.0
        for i in range(v.shape[0]):
            v_bar = T.T @ v[i]
            potential_i = p_args.mu_lambda[i] * FrictionPotential.f0(
                np.linalg.norm(v_bar * p_args.h), p_args.h
            )
            ret += potential_i
        return ret

    @staticmethod
    def grad(
        p_args: PotentialArgs,
    ) -> np.ndarray:
        grad = np.zeros_like(p_args.x)

        v = (p_args.x - p_args.x_n) / p_args.h
        T = np.eye(2) - np.outer(p_args.ground_n, p_args.ground_n)
        for i in range(len(p_args.x)):
            # If there is contact
            if p_args.mu_lambda[i] > 0:
                v_bar = T.T @ v[i]
                v_bar_norm = np.linalg.norm(v_bar)
                f1_eval = FrictionPotential.f1_div_v_bar_norm(v_bar_norm)
                grad[i] = p_args.mu_lambda[i] * f1_eval * T @ v_bar
        grad = grad.reshape(-1)

        return grad

    @staticmethod
    def hess(
        p_args: PotentialArgs,
    ) -> coo_matrix:
        i_coords, j_coords, vals = [], [], []

        v = (p_args.x - p_args.x_n) / p_args.h
        T = np.eye(2) - np.outer(p_args.ground_n, p_args.ground_n)
        for i in range(len(p_args.x)):
            # Exists contact
            if p_args.mu_lambda[i] > 0:
                v_bar = T.T @ v[i]
                hess_local = (
                    p_args.mu_lambda[i]
                    * T
                    @ FrictionPotential.f1_hess_local_inner(v_bar)
                    @ T.T
                ) / p_args.h

                for x_coord in range(2):
                    for y_coord in range(2):
                        i_coords.append(2 * i + x_coord)
                        j_coords.append(2 * i + y_coord)
                        vals.append(hess_local[x_coord, y_coord])

        hess = coo_matrix(
            (vals, (i_coords, j_coords)), shape=(2 * len(p_args.x), 2 * len(p_args.x))
        )
        return hess
