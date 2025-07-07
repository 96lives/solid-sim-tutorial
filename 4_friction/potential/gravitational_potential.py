import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from potential.base_potential import Potential
from potential.potential_args import PotentialArgs


class GravitationalPotential(Potential):
    g = 9.81  # Gravitational acceleration in m/s^2

    @staticmethod
    def val(
        p_args: PotentialArgs,
    ) -> float:
        m_array = np.array(p_args.m)
        energy = GravitationalPotential.g * sum(m_array * p_args.x[:, 1])
        return energy

    @staticmethod
    def grad(
        p_args: PotentialArgs,
    ) -> np.ndarray:
        grad = np.zeros_like(p_args.x)
        grad[:, 1] = GravitationalPotential.g * np.array(p_args.m)
        grad = grad.reshape(-1)
        return grad

    @staticmethod
    def hess(
        p_args: PotentialArgs,
    ) -> coo_matrix:
        hess_zero = coo_matrix(
            ([], ([], [])), shape=(len(p_args.x) * 2, len(p_args.x) * 2)
        )
        return hess_zero
