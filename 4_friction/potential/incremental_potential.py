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
import FrictionEnergy
import InertiaEnergy
import BarrierEnergy
import MassSpringEnergy
import GravityEnergy


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
        # inertia = InertiaPotential.grad(p_args)
        inertia = InertiaEnergy.grad(p_args.x, p_args.x_tilde, p_args.m).reshape(-1)
        # gravity = GravitationalPotential.grad(p_args)
        gravity = GravityEnergy.grad(p_args.x, p_args.m).reshape(-1)
        # mass_spring = MassSpringPotential.grad(p_args)
        mass_spring = MassSpringEnergy.grad(
            p_args.x, p_args.e, p_args.l2, p_args.k
        ).reshape(-1)
        # barrier = BarrierPotential.grad(p_args)
        barrier = BarrierEnergy.grad(
            p_args.x, p_args.ground_n, p_args.ground_o, p_args.contact_area
        ).reshape(-1)
        # friction = FrictionPotential.grad(p_args)
        v = p_args.x - p_args.x_n
        friction = FrictionEnergy.grad(
            v, p_args.mu_lambda, p_args.h, p_args.ground_n
        ).reshape(-1)

        ret = inertia + p_args.h**2 * (gravity + mass_spring + barrier + friction)
        return ret

    @staticmethod
    def hess(p_args: PotentialArgs) -> coo_matrix:
        import scipy.sparse as sparse

        # inertia = InertiaPotential.hess(p_args)
        ijv = InertiaEnergy.hess(p_args.x, p_args.x_tilde, p_args.m)
        inertia = sparse.coo_matrix(
            (ijv[2], (ijv[0], ijv[1])), shape=(len(p_args.x) * 2, len(p_args.x) * 2)
        ).tocsr()

        gravity = GravitationalPotential.hess(p_args)

        # mass_spring = MassSpringPotential.hess(p_args)
        ijv = MassSpringEnergy.hess(p_args.x, p_args.e, p_args.l2, p_args.k)
        mass_spring = sparse.coo_matrix(
            (ijv[2], (ijv[0], ijv[1])), shape=(len(p_args.x) * 2, len(p_args.x) * 2)
        ).tocsr()

        # barrier = BarrierPotential.hess(p_args)
        ijv = BarrierEnergy.hess(
            p_args.x, p_args.ground_n, p_args.ground_o, p_args.contact_area
        )
        barrier = sparse.coo_matrix(
            (ijv[2], (ijv[0], ijv[1])), shape=(len(p_args.x) * 2, len(p_args.x) * 2)
        ).tocsr()

        # friction = FrictionPotential.hess(p_args)
        v = p_args.x - p_args.x_n
        ijv = FrictionEnergy.hess(v, p_args.mu_lambda, p_args.h, p_args.ground_n)
        friction = sparse.coo_matrix(
            (ijv[2], (ijv[0], ijv[1])), shape=(len(p_args.x) * 2, len(p_args.x) * 2)
        ).tocsr()

        ret = inertia + p_args.h**2 * (gravity + mass_spring + barrier + friction)
        return ret
