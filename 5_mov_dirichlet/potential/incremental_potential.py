import numpy as np
from typing import List, Tuple
from scipy.sparse import coo_matrix
from potential.base_potential import Potential
from potential.gravitational_potential import GravitationalPotential
from potential.inertia_potential import InertiaPotential
from potential.mass_spring_potential import MassSpringPotential
from potential.barrier_potential import BarrierPotential
from potential.potential_args import PotentialArgs
from potential.friction_potential import FrictionPotential


class IncrementalPotential(Potential):
    @staticmethod
    def val(
        p_args: PotentialArgs,
    ) -> float:
        inertia = InertiaPotential.val(p_args)
        gravity = GravitationalPotential.val(p_args)
        mass_spring = MassSpringPotential.val(p_args)
        barrier = BarrierPotential.val(p_args)
        friction = FrictionPotential.val(p_args)

        ret = inertia + p_args.h**2 * (gravity + mass_spring + barrier + friction)
        print(
            f"Inertia: {inertia:.2f}, "
            f"Gravity: {p_args.h ** 2 * gravity:.2f}, "
            f"MassSpring: {p_args.h**2 * mass_spring :.2f}, "
            f"Barrier: {p_args.h**2 * barrier:.2f}, "
            f"Friction: {p_args.h**2 * friction:.2f},"
            f"Total: {ret:.2f}"
        )
        return ret

    @staticmethod
    def grad(p_args: PotentialArgs) -> np.ndarray:
        inertia = InertiaPotential.grad(p_args)
        gravity = GravitationalPotential.grad(p_args)
        mass_spring = MassSpringPotential.grad(p_args)
        barrier = BarrierPotential.grad(p_args)
        friction = FrictionPotential.grad(p_args)

        ret = inertia + p_args.h**2 * (gravity + mass_spring + barrier + friction)
        return ret

    @staticmethod
    def hess(p_args: PotentialArgs) -> coo_matrix:
        inertia = InertiaPotential.hess(p_args)
        gravity = GravitationalPotential.hess(p_args)
        mass_spring = MassSpringPotential.hess(p_args)
        barrier = BarrierPotential.hess(p_args)
        friction = FrictionPotential.hess(p_args)

        ret = inertia + p_args.h**2 * (gravity + mass_spring + barrier + friction)
        return ret
