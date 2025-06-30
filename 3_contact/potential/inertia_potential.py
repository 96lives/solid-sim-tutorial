from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from potential.base_potential import Potential


class InertiaPotential(Potential):
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
        diff = x - x_tilde  # n x 2
        m_arr = np.array(m)[..., np.newaxis]  # n x 1
        ret = 0.5 * np.sum(diff * m_arr * diff)
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
        diff = x - x_tilde  # n x 2
        m_arr = np.array(m)[..., np.newaxis]  # n x 1
        ret = (diff * m_arr).reshape(-1)
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
        m_arr = np.array(m)
        m_arr_axis = np.stack([m_arr] * 2, axis=1).reshape(-1)
        hess = np.diag(m_arr_axis)
        hess_sparse = coo_matrix(hess)
        return hess_sparse
