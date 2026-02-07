"""Mass-balance diagnostics for NL-NILT (Approach B).

Provides physics-based convergence monitoring using global mass balance
constraints that hold regardless of binding model:

  1. Zeroth moment (F(0) = 1): Total mass conservation
  2. First moment (retention time): Matches analytical prediction
  3. Steady-state check: c_out(t -> inf) = c_feed

These diagnostics are reference-free — they validate solutions using
physical laws rather than comparison to a reference solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class MassBalanceDiagnostics:
    """Mass-balance diagnostic results for a single NL-NILT iteration.

    Attributes:
        delta_F0: |F(0) - 1| — zeroth moment deviation.
        mu1_numerical: Numerical first moment (mean retention time).
        mu1_theoretical: Theoretical first moment from physical parameters.
        delta_mu1: Relative first-moment error |mu1_num - mu1_theory| / mu1_theory.
        steady_state_error: |c_out(t_end) / c_feed - 1| if step input.
        mass_in: Cumulative mass injected (step input only).
        mass_out: Cumulative mass eluted (step input only).
    """

    delta_F0: float = np.inf
    mu1_numerical: float = np.nan
    mu1_theoretical: float = np.nan
    delta_mu1: float = np.inf
    steady_state_error: float = np.inf
    mass_in: float = np.nan
    mass_out: float = np.nan


def check_zeroth_moment(
    F: Callable[[complex], complex],
    s_probe: float = 1e-10,
) -> float:
    """Check that F(0) = 1 (mass conservation).

    For any mass-conserving transfer function, the final value theorem
    gives lim_{t->inf} c_out(t) = lim_{s->0} s * F(s) * C/s = F(0) * C.
    Since c_out -> C at steady state, F(0) must equal 1.

    Args:
        F: Laplace-domain transfer function.
        s_probe: Small positive real s to approximate F(0).

    Returns:
        |F(s_probe) - 1| — deviation from mass conservation.
    """
    F0 = F(complex(s_probe, 0.0))
    return abs(F0.real - 1.0)


def compute_numerical_first_moment(
    t: np.ndarray,
    c: np.ndarray,
    c_feed: float = 1.0,
    step_input: bool = True,
) -> float:
    """Compute the first moment (mean retention time) from time-domain data.

    For step input:
        mu1 = integral_0^inf (1 - c_out(t)/c_feed) dt

    For impulse input:
        mu1 = integral_0^inf t * c_out(t) dt / integral_0^inf c_out(t) dt

    Args:
        t: Time points.
        c: Outlet concentration.
        c_feed: Feed concentration (for step input normalization).
        step_input: Whether the input is a step (True) or impulse (False).

    Returns:
        First moment (mean retention time) in seconds.
    """
    if step_input:
        # For step input: mu1 = integral (1 - c/c_feed) dt
        # This is the area above the breakthrough curve
        if c_feed > 0:
            integrand = 1.0 - c / c_feed
            # Clip negative values (numerical artifacts)
            integrand = np.maximum(integrand, 0.0)
            return float(np.trapezoid(integrand, t))
        return 0.0
    else:
        # For impulse: mu1 = integral(t * c dt) / integral(c dt)
        total_mass = np.trapezoid(c, t)
        if total_mass > 1e-30:
            return float(np.trapezoid(t * c, t) / total_mass)
        return 0.0


def compute_theoretical_first_moment(
    velocity: float,
    length: float,
    col_porosity: float,
    par_porosity: float = 0.0,
    K_eq: float = 0.0,
) -> float:
    """Compute theoretical first moment from physical parameters.

    mu1 = tau * [1 + (1-eps_c)/eps_c * (eps_p + (1-eps_p)*K_eq)]

    where tau = L/v is the column residence time.

    Args:
        velocity: Interstitial velocity [m/s].
        length: Column length [m].
        col_porosity: Column porosity.
        par_porosity: Particle porosity.
        K_eq: Equilibrium distribution coefficient q_eq/c_eq.

    Returns:
        Theoretical first moment (retention time) in seconds.
    """
    tau = length / velocity
    phase_ratio = (1.0 - col_porosity) / col_porosity
    particle_factor = par_porosity + (1.0 - par_porosity) * K_eq
    return tau * (1.0 + phase_ratio * particle_factor)


def compute_step_mass_balance(
    t: np.ndarray,
    c_out: np.ndarray,
    c_feed: float,
    velocity: float,
    cross_section: float = 1.0,
    col_porosity: float = 0.37,
) -> tuple:
    """Compute cumulative mass in and mass out for step input.

    Args:
        t: Time points.
        c_out: Outlet concentration.
        c_feed: Feed concentration.
        velocity: Interstitial velocity.
        cross_section: Column cross-sectional area.
        col_porosity: Column porosity.

    Returns:
        (mass_in, mass_out) cumulative masses at t[-1].
    """
    Q = velocity * cross_section * col_porosity  # Volumetric flow rate
    mass_in = Q * c_feed * t[-1]
    mass_out = Q * float(np.trapezoid(c_out, t))
    return mass_in, mass_out


def compute_diagnostics(
    t: np.ndarray,
    c: np.ndarray,
    F: Callable[[complex], complex],
    c_feed: float = 1.0,
    step_input: bool = True,
    velocity: float = 1e-3,
    length: float = 0.1,
    col_porosity: float = 0.37,
    par_porosity: float = 0.33,
    K_eq: float = 0.0,
) -> MassBalanceDiagnostics:
    """Compute all mass-balance diagnostics for an NL-NILT solution.

    Args:
        t: Time grid.
        c: Outlet concentration.
        F: Transfer function (for F(0) check).
        c_feed: Feed concentration.
        step_input: Whether step input was used.
        velocity: Interstitial velocity.
        length: Column length.
        col_porosity: Column porosity.
        par_porosity: Particle porosity.
        K_eq: Equilibrium distribution coefficient.

    Returns:
        MassBalanceDiagnostics with all computed metrics.
    """
    diag = MassBalanceDiagnostics()

    # 1. Zeroth moment check
    diag.delta_F0 = check_zeroth_moment(F)

    # 2. First moment
    diag.mu1_numerical = compute_numerical_first_moment(t, c, c_feed, step_input)
    diag.mu1_theoretical = compute_theoretical_first_moment(
        velocity, length, col_porosity, par_porosity, K_eq,
    )
    if diag.mu1_theoretical > 1e-30:
        diag.delta_mu1 = abs(diag.mu1_numerical - diag.mu1_theoretical) / diag.mu1_theoretical
    else:
        diag.delta_mu1 = 0.0

    # 3. Steady-state check (for step input)
    if step_input and c_feed > 0:
        diag.steady_state_error = abs(c[-1] / c_feed - 1.0)

    # 4. Cumulative mass balance
    if step_input:
        diag.mass_in, diag.mass_out = compute_step_mass_balance(
            t, c, c_feed, velocity, col_porosity=col_porosity,
        )

    return diag


def steering_decision(
    current: MassBalanceDiagnostics,
    previous: Optional[MassBalanceDiagnostics] = None,
    delta_F0_threshold: float = 1e-6,
    delta_mu1_threshold: float = 1e-3,
) -> str:
    """Make a convergence/steering decision based on mass-balance diagnostics.

    Decision rules from the NL-NILT theory document:
    - CONVERGED: delta_F0 < threshold AND delta_mu1 < threshold
    - DIVERGING: delta_F0 increased from previous iteration
    - STALLING: delta_mu1 increased for 2+ consecutive iterations
    - CONTINUING: improving but not yet converged

    Args:
        current: Current iteration diagnostics.
        previous: Previous iteration diagnostics (None for first iteration).
        delta_F0_threshold: Convergence threshold for zeroth moment.
        delta_mu1_threshold: Convergence threshold for first moment.

    Returns:
        Decision string: "CONVERGED", "DIVERGING", "STALLING", or "CONTINUING".
    """
    if current.delta_F0 < delta_F0_threshold and current.delta_mu1 < delta_mu1_threshold:
        return "CONVERGED"

    if previous is not None:
        if current.delta_F0 > previous.delta_F0 * 1.1:
            return "DIVERGING"
        if current.delta_mu1 > previous.delta_mu1 * 1.1:
            return "STALLING"

    return "CONTINUING"
