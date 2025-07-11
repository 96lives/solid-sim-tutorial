from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from potential.base_potential import Potential
from potential.potential_args import PotentialArgs


class InertiaPotential(Potential):
    @staticmethod
    def val(p_args: PotentialArgs) -> float:
        diff = p_args.x - p_args.x_tilde  # n x 2
        m_arr = np.array(p_args.m)[..., np.newaxis]  # n x 1
        ret = 0.5 * np.sum(diff * m_arr * diff)
        return ret

    @staticmethod
    def grad(p_args: PotentialArgs) -> np.ndarray:
        diff = p_args.x - p_args.x_tilde  # n x 2
        m_arr = np.array(p_args.m)[..., np.newaxis]  # n x 1
        ret = (diff * m_arr).reshape(-1)
        return ret

    @staticmethod
    def hess(p_args: PotentialArgs) -> coo_matrix:
        m_arr = np.array(p_args.m)
        m_arr_axis = np.stack([m_arr] * 2, axis=1).reshape(-1)
        hess = np.diag(m_arr_axis)
        hess_sparse = coo_matrix(hess)
        return hess_sparse
