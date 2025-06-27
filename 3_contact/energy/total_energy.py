import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from energy.base_energy import Energy
from energy.gravity_energy import GravityEnergy


class TotalEnergy(Energy):

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
        gravity = GravityEnergy.val(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        return gravity

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
        gravity = GravityEnergy.grad(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        return gravity

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
        gravity = GravityEnergy.hess(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        return gravity
