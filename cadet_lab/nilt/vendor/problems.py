"""Benchmark transfer functions for FFT-NILT testing.

Vendored from: https://github.com/gogipav14/nilt-cfl
License: MIT

Each problem defines:
- F(s): Transfer function in Laplace domain
- f_ref(t): Analytical inverse (if available)
- alpha_c: Abscissa of convergence
- C: Tail envelope constant
- rho: Spectral radius (optional, for frequency heuristic)
"""

from __future__ import annotations
import numpy as np
from scipy.special import erfc
from dataclasses import dataclass
from typing import Callable, Optional
import cmath


@dataclass
class Problem:
    """Container for a benchmark problem."""
    name: str
    F: Callable[[complex], complex]
    f_ref: Optional[Callable[[np.ndarray], np.ndarray]]
    alpha_c: float
    C: float
    rho: Optional[float] = None
    description: str = ""


def first_order_lag(K: float = 1.0, tau: float = 1.0) -> Problem:
    """
    First-order lag system.

    F(s) = K / (tau * s + 1)
    f(t) = (K/tau) * exp(-t/tau)

    Alpha_c = -1/tau (stable)
    """
    def F(s):
        return K / (tau * s + 1)

    def f_ref(t):
        return (K / tau) * np.exp(-t / tau)

    return Problem(
        name="lag",
        F=F,
        f_ref=f_ref,
        alpha_c=-1.0 / tau,
        C=K / tau,
        rho=1.0 / tau,
        description=f"First-order lag: K={K}, tau={tau}"
    )


def fopdt(K: float = 1.0, tau: float = 1.0, theta: float = 2.0) -> Problem:
    """
    First-order plus dead time (FOPDT) system.

    F(s) = K * exp(-theta * s) / (tau * s + 1)
    f(t) = (K/tau) * exp(-(t-theta)/tau) * H(t-theta)

    Alpha_c = -1/tau (stable)
    """
    def F(s):
        return K * cmath.exp(-theta * s) / (tau * s + 1)

    def f_ref(t):
        result = np.zeros_like(t)
        mask = t >= theta
        result[mask] = (K / tau) * np.exp(-(t[mask] - theta) / tau)
        return result

    return Problem(
        name="fopdt",
        F=F,
        f_ref=f_ref,
        alpha_c=-1.0 / tau,
        C=K / tau,
        rho=1.0 / tau,
        description=f"FOPDT: K={K}, tau={tau}, theta={theta}"
    )


def second_order(omega_n: float = 1.0, zeta: float = 0.5) -> Problem:
    """
    Second-order underdamped system (impulse response).

    F(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    f(t) = (omega_n / sqrt(1-zeta^2)) * exp(-zeta*omega_n*t) * sin(omega_d*t)

    where omega_d = omega_n * sqrt(1 - zeta^2)

    Alpha_c = -zeta * omega_n (stable for zeta > 0)
    """
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    def F(s):
        return omega_n**2 / (s**2 + 2 * zeta * omega_n * s + omega_n**2)

    def f_ref(t):
        return (omega_n / np.sqrt(1 - zeta**2)) * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)

    return Problem(
        name="secondorder",
        F=F,
        f_ref=f_ref,
        alpha_c=-zeta * omega_n,
        C=omega_n / np.sqrt(1 - zeta**2),
        rho=omega_n,
        description=f"Second-order: omega_n={omega_n}, zeta={zeta}"
    )


def dampener(omega_n: float = 1.0, zeta: float = 0.2) -> Problem:
    """
    Dampener system (underdamped second-order, Hsu-Dranoff style).

    Same as second_order but with typical dampener parameters.
    Uses lower damping ratio for more pronounced oscillation.

    F(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    """
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    def F(s):
        return omega_n**2 / (s**2 + 2 * zeta * omega_n * s + omega_n**2)

    def f_ref(t):
        return (omega_n / np.sqrt(1 - zeta**2)) * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)

    return Problem(
        name="dampener",
        F=F,
        f_ref=f_ref,
        alpha_c=-zeta * omega_n,
        C=omega_n / np.sqrt(1 - zeta**2),
        rho=omega_n,
        description=f"Dampener: omega_n={omega_n}, zeta={zeta}"
    )


def semi_infinite_diffusion(D: float = 1.0, x: float = 1.0) -> Problem:
    """
    Semi-infinite diffusion problem.

    F(s) = exp(-x * sqrt(s/D)) / s
    f(t) = erfc(x / (2 * sqrt(D * t)))

    Alpha_c = 0 (singularity at s=0)
    Note: f(t) -> 1 as t -> infinity (bounded)
    """
    def F(s):
        if abs(s) < 1e-20:
            return complex(np.inf, 0)
        return cmath.exp(-x * cmath.sqrt(s / D)) / s

    def f_ref(t):
        # Avoid t=0 singularity
        result = np.zeros_like(t)
        mask = t > 0
        result[mask] = erfc(x / (2 * np.sqrt(D * t[mask])))
        return result

    return Problem(
        name="diffusion",
        F=F,
        f_ref=f_ref,
        alpha_c=0.0,
        C=1.0,  # erfc is bounded by 1
        rho=D / x**2,  # characteristic frequency
        description=f"Semi-infinite diffusion: D={D}, x={x}"
    )


def packed_bed(Pe: float = 10.0) -> Problem:
    """
    Packed-bed axial dispersion model.

    F(s) = exp(Pe/2 * (1 - sqrt(1 + 4s/Pe)))

    This represents breakthrough curves in chromatographic systems.

    Alpha_c = 0 (branch point at s = -Pe/4)
    Note: f(t) represents normalized breakthrough, bounded in [0, 1]
    """
    def F(s):
        inner = 1 + 4 * s / Pe
        return cmath.exp(Pe / 2 * (1 - cmath.sqrt(inner)))

    # No simple closed-form inverse; use numerical reference
    return Problem(
        name="packedbed",
        F=F,
        f_ref=None,  # Will use de Hoog as reference
        alpha_c=0.0,
        C=1.0,  # Bounded breakthrough curve
        rho=Pe,
        description=f"Packed-bed dispersion: Pe={Pe}"
    )


def get_problem(name: str) -> Problem:
    """
    Get a benchmark problem by name.

    Parameters
    ----------
    name : str
        Problem name: lag, fopdt, secondorder, diffusion, packedbed, dampener

    Returns
    -------
    problem : Problem
        The benchmark problem
    """
    problems = {
        "lag": first_order_lag(),
        "fopdt": fopdt(),
        "secondorder": second_order(),
        "diffusion": semi_infinite_diffusion(),
        "packedbed": packed_bed(),
        "dampener": dampener()
    }

    if name not in problems:
        raise ValueError(f"Unknown problem: {name}. Available: {list(problems.keys())}")

    return problems[name]


def get_all_problems() -> dict:
    """Get all benchmark problems."""
    return {
        "lag": first_order_lag(),
        "fopdt": fopdt(),
        "secondorder": second_order(),
        "diffusion": semi_infinite_diffusion(),
        "packedbed": packed_bed(),
        "dampener": dampener()
    }
