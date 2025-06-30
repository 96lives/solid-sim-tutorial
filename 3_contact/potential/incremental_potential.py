import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from potential.base_potential import Potential
from potential.gravitational_potential import GravitationalPotential
from potential.inertia_potential import InertiaPotential
from potential.mass_spring_potential import MassSpringPotential


class IncrementalPotential(Potential):
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
        inertia = InertiaPotential.val(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        gravity = GravitationalPotential.val(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        mass_spring = MassSpringPotential.val(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )

        ret = inertia + h**2 * (gravity + mass_spring)
        # print(
        #     f"Inertia: {inertia:.2f}, "
        #     f"Gravity: {h ** 2 * gravity:.2f}, "
        #     f"MassSpring: {h**2 * mass_spring :.2f}, "
        #     f"Total: {ret:.2f}"
        # )
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
        inertia = InertiaPotential.grad(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        gravity = GravitationalPotential.grad(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        mass_spring = MassSpringPotential.grad(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        ret = inertia + h**2 * (gravity + mass_spring)
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
        inertia = InertiaPotential.hess(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        gravity = GravitationalPotential.hess(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        mass_spring = MassSpringPotential.hess(
            x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h
        )
        ret = inertia + h**2 * (gravity + mass_spring)
        return ret
