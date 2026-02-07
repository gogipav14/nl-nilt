"""CADET-specific benchmark transfer functions for NILT verification.

Provides analytical Laplace-domain transfer functions relevant to
chromatography, packed-bed reactors, and dispersion processes.
"""

from typing import Callable, List, Tuple, Optional
import numpy as np
import cmath

from .vendor import Problem


def advection_dispersion_transfer(
    velocity: float = 1e-3,
    dispersion: float = 1e-6,
    length: float = 0.1,
) -> Callable[[complex], complex]:
    """Advection-dispersion equation transfer function.

    For a column with Danckwerts boundary conditions,
    the outlet concentration transfer function is:

    F(s) = exp(Pe/2 * (1 - sqrt(1 + 4*s*tau/Pe)))

    where:
        Pe = v*L/D (Peclet number)
        tau = L/v (residence time)

    This is equivalent to the packed_bed problem from nilt-cfl
    with appropriate parameter mapping.

    Args:
        velocity: Interstitial velocity [m/s].
        dispersion: Axial dispersion coefficient [m²/s].
        length: Column length [m].

    Returns:
        Callable F(s) for the transfer function.
    """
    Pe = velocity * length / dispersion  # Peclet number
    tau = length / velocity  # Residence time

    def F(s: complex) -> complex:
        # Dimensionless time constant
        inner = 1 + 4 * s * tau / Pe
        return cmath.exp(Pe / 2 * (1 - cmath.sqrt(inner)))

    return F


def langmuir_column_transfer(
    velocity: float = 1e-3,
    dispersion: float = 1e-6,
    length: float = 0.1,
    ka: float = 1.0,
    kd: float = 0.1,
    qmax: float = 10.0,
    porosity: float = 0.4,
) -> Callable[[complex], complex]:
    """Linearized Langmuir column transfer function.

    For linear (dilute) conditions, the Langmuir model can be linearized
    to give an effective retardation factor. The transfer function
    combines advection-dispersion with sorption dynamics.

    Args:
        velocity: Interstitial velocity [m/s].
        dispersion: Axial dispersion coefficient [m²/s].
        length: Column length [m].
        ka: Adsorption rate constant [1/(mol·s)].
        kd: Desorption rate constant [1/s].
        qmax: Maximum binding capacity [mol/m³].
        porosity: Column porosity [-].

    Returns:
        Callable F(s) for the transfer function.
    """
    # Phase ratio
    phase_ratio = (1.0 - porosity) / porosity

    # Linear partition coefficient (Henry constant)
    K_eq = ka * qmax / kd

    # Retardation factor
    R = 1.0 + phase_ratio * K_eq

    # Effective Peclet number (accounting for retardation)
    Pe_eff = velocity * length / (dispersion * R)

    # Effective residence time
    tau_eff = R * length / velocity

    # Sorption time scale
    tau_sorp = 1.0 / kd

    def F(s: complex) -> complex:
        # Transport part (advection-dispersion with retardation)
        inner = 1 + 4 * s * tau_eff / Pe_eff
        transport = cmath.exp(Pe_eff / 2 * (1 - cmath.sqrt(inner)))

        # Kinetic correction (low-pass filter for finite sorption rate)
        kinetic = 1.0 / (1.0 + s * tau_sorp / R)

        return transport * kinetic

    return F


def grm_langmuir_transfer(
    velocity: float = 1e-3,
    dispersion: float = 1e-6,
    length: float = 0.1,
    col_porosity: float = 0.37,
    par_radius: float = 1e-5,
    par_porosity: float = 0.33,
    film_diffusion: float = 1e-5,
    pore_diffusion: float = 1e-10,
    ka: float = 1.0,
    kd: float = 0.1,
    qmax: float = 10.0,
) -> Callable[[complex], complex]:
    """Full GRM transfer function with kinetic Langmuir binding.

    Includes particle dynamics (film + pore diffusion) and kinetic binding.
    Derived from first principles via Laplace transform of GRM equations.

    This transfer function accounts for:
    - Column advection-dispersion
    - Film mass transfer resistance (particle surface)
    - Pore diffusion within particles
    - Kinetic Langmuir binding (with rate constants ka, kd)

    Valid for:
    - Single component
    - Dilute regime (c << qmax, linear binding)
    - Isothermal conditions

    Mathematical derivation in docs/GRM_TRANSFER_FUNCTIONS_DERIVATION.md

    Args:
        velocity: Interstitial velocity [m/s].
        dispersion: Axial dispersion coefficient [m²/s].
        length: Column length [m].
        col_porosity: Column (interstitial) porosity [-].
        par_radius: Particle radius [m].
        par_porosity: Particle porosity [-].
        film_diffusion: Film mass transfer coefficient [m/s].
        pore_diffusion: Pore diffusion coefficient [m²/s].
        ka: Adsorption rate constant [1/s] (for dilute: dq/dt = ka*qmax*c - kd*q).
        kd: Desorption rate constant [1/s].
        qmax: Maximum binding capacity [mol/m³].

    Returns:
        Callable F(s) for the transfer function.

    Example:
        >>> F = grm_langmuir_transfer(
        ...     velocity=1e-3, dispersion=1e-6, length=0.1,
        ...     col_porosity=0.37, par_radius=1e-5, par_porosity=0.33,
        ...     film_diffusion=1e-5, pore_diffusion=1e-10,
        ...     ka=1.0, kd=0.1, qmax=100.0
        ... )
        >>> outlet_conc = F(s=0.1)  # Evaluate at s=0.1
    """
    # Dimensionless parameters
    Pe = velocity * length / dispersion  # Peclet number
    tau = length / velocity  # Residence time
    Bi = film_diffusion * par_radius / pore_diffusion  # Biot number
    phase_ratio = (1.0 - col_porosity) / col_porosity  # Solid/liquid volume ratio

    def F(s: complex) -> complex:
        # Effective binding capacity (frequency-dependent)
        # β(s) = ε_p + (1-ε_p) · ka·qmax/(s+kd)
        beta = par_porosity + (1.0 - par_porosity) * ka * qmax / (s + kd)

        # Particle diffusion parameter: ξ = r_p · √(s·β/D_p)
        xi_sq = par_radius**2 * s * beta / pore_diffusion
        xi = cmath.sqrt(xi_sq)

        # Particle response function η(s):
        # η = 3·(ξ·coth(ξ) - 1) / (ξ² · (1 + (ξ·coth(ξ) - 1)/Bi))
        # = 3·A / (ξ²·(sinh(ξ) + A/Bi))  where A = ξ·cosh(ξ) - sinh(ξ)

        # Handle small ξ (Taylor expansion to avoid numerical issues)
        if abs(xi) < 1e-6:
            # As ξ→0: A = ξ³/3, sinh ≈ ξ
            # η → 3·(ξ³/3) / (ξ²·(ξ + ξ³/(3·Bi))) = 1/(1 + ξ²/(3·Bi)) → 1
            eta = 1.0
        else:
            sinh_xi = cmath.sinh(xi)
            cosh_xi = cmath.cosh(xi)

            A = xi * cosh_xi - sinh_xi
            denom = sinh_xi + A / Bi

            # Avoid division by very small denominator
            if abs(denom) < 1e-15:
                eta = 0.0
            else:
                eta = 3.0 * A / (xi_sq * denom)

        # Effective retardation factor: R_eff(s) = 1 + F·β(s)·η(s)
        R_eff = 1.0 + phase_ratio * beta * eta

        # Column transfer function with effective retardation
        # F(s) = exp(Pe/2 · (1 - √[1 + 4·s·τ·R_eff/Pe]))
        inner = 1.0 + 4.0 * s * tau * R_eff / Pe
        sqrt_inner = cmath.sqrt(inner)

        return cmath.exp(Pe / 2.0 * (1.0 - sqrt_inner))

    return F


def grm_sma_transfer(
    velocity: float = 1e-3,
    dispersion: float = 1e-6,
    length: float = 0.1,
    col_porosity: float = 0.37,
    par_radius: float = 1e-5,
    par_porosity: float = 0.33,
    film_diffusion: float = 1e-5,
    pore_diffusion: float = 1e-10,
    ka: float = 1.0,
    kd: float = 0.1,
    Lambda: float = 10.0,
    nu: float = 4.5,
    z_protein: float = 5.0,
    z_salt: float = 1.0,
    c0: float = 1e-6,
    c_salt: float = 0.1,
) -> Callable[[complex], complex]:
    """GRM transfer function with linearized SMA (Steric Mass Action) binding.

    Linearizes SMA around base state (c₀, q₀) for small perturbations.
    Uses effective rate constants k_a_eff, k_d_eff and treats as Langmuir.

    SMA binding kinetics:
        ∂q/∂t = ka · c · (Λ - z_protein·q - z_salt·c_salt)^nu - kd · q

    Linearization around base state gives:
        ∂(δq)/∂t = k_a_eff · δc - k_d_eff · δq

    Valid for:
    - Small concentration perturbations around base state
    - Single protein component + salt
    - Dilute to moderate loading

    Mathematical derivation in docs/GRM_TRANSFER_FUNCTIONS_DERIVATION.md

    Args:
        velocity: Interstitial velocity [m/s].
        dispersion: Axial dispersion coefficient [m²/s].
        length: Column length [m].
        col_porosity: Column porosity [-].
        par_radius: Particle radius [m].
        par_porosity: Particle porosity [-].
        film_diffusion: Film mass transfer coefficient [m/s].
        pore_diffusion: Pore diffusion coefficient [m²/s].
        ka: SMA adsorption constant [m³/(mol·s)].
        kd: SMA desorption rate constant [1/s].
        Lambda: Steric capacity [mol/m³].
        nu: Characteristic charge [-].
        z_protein: Protein charge [-].
        z_salt: Salt valence [-].
        c0: Base protein concentration for linearization [mol/m³].
        c_salt: Salt concentration [mol/m³].

    Returns:
        Callable F(s) for the transfer function.

    Example:
        >>> # IgG on Protein A resin (typical values)
        >>> F = grm_sma_transfer(
        ...     velocity=1e-4, dispersion=1e-7, length=0.05,
        ...     ka=1e3, kd=0.1, Lambda=100.0, nu=4.5,
        ...     z_protein=5.0, c0=1e-6, c_salt=0.15
        ... )
    """
    # Compute base state
    Lambda_eff = Lambda - z_salt * c_salt  # Effective capacity with salt

    # Equilibrium: q0 = K_eq * c0 * (Lambda_eff - z_protein*q0)^nu
    # Low-loading approximation: q0 ≈ K_eq * c0 * Lambda_eff^nu / (1 + ...)
    K_eq = ka / kd

    # Iterative solution for base state (Newton's method, 1-2 iterations usually sufficient)
    q0 = 0.0
    for _ in range(5):
        shield = Lambda_eff - z_protein * q0
        if shield <= 0:
            # Invalid state (overloaded), use low-loading limit
            q0 = K_eq * c0 * Lambda_eff**nu / (1 + K_eq * c0 * nu * z_protein * Lambda_eff**(nu-1))
            break
        f_q = q0 - K_eq * c0 * shield**nu
        df_q = 1.0 + K_eq * c0 * nu * z_protein * shield**(nu - 1)
        q0_new = q0 - f_q / df_q
        if abs(q0_new - q0) < 1e-12:
            q0 = q0_new
            break
        q0 = q0_new

    # Linearized rate constants around base state
    shield_term = Lambda_eff - z_protein * q0

    if shield_term <= 0:
        # Invalid state, fall back to pure Langmuir
        k_a_eff = ka * Lambda_eff**nu
        k_d_eff = kd
    else:
        k_a_eff = ka * shield_term**nu
        k_d_eff = kd + ka * c0 * nu * z_protein * shield_term**(nu - 1)

    # Use Langmuir GRM with effective rate constants
    return grm_langmuir_transfer(
        velocity=velocity,
        dispersion=dispersion,
        length=length,
        col_porosity=col_porosity,
        par_radius=par_radius,
        par_porosity=par_porosity,
        film_diffusion=film_diffusion,
        pore_diffusion=pore_diffusion,
        ka=k_a_eff,
        kd=k_d_eff,
        qmax=Lambda_eff,  # Effective capacity as qmax
    )


def grm_moment_transfer(
    velocity: float = 1e-3,
    dispersion: float = 1e-6,
    length: float = 0.1,
    particle_radius: float = 5e-5,
    film_diffusion: float = 1e-5,
    pore_diffusion: float = 1e-10,
    porosity_column: float = 0.4,
    porosity_particle: float = 0.5,
) -> Callable[[complex], complex]:
    """General Rate Model (GRM) transfer function (moment-based approximation).

    This approximates the GRM outlet response using a moment-matching
    approach. The GRM accounts for:
    - Axial dispersion in interstitial volume
    - Film mass transfer at particle surface
    - Pore diffusion within particles

    Args:
        velocity: Interstitial velocity [m/s].
        dispersion: Axial dispersion coefficient [m²/s].
        length: Column length [m].
        particle_radius: Particle radius [m].
        film_diffusion: Film mass transfer coefficient [m/s].
        pore_diffusion: Pore diffusion coefficient [m²/s].
        porosity_column: Column (interstitial) porosity [-].
        porosity_particle: Particle porosity [-].

    Returns:
        Callable F(s) for the transfer function.
    """
    # Dimensionless groups
    Pe = velocity * length / dispersion

    # Particle time constant
    tau_p = particle_radius**2 / (15 * pore_diffusion)

    # Phase ratio
    phi = (1 - porosity_column) / porosity_column * porosity_particle

    # Residence time
    tau_col = length / velocity

    # Effective parameters for approximate transfer function
    n_eff = max(Pe / 2, 1)  # Effective number of stages

    def F(s: complex) -> complex:
        # Column dispersion (TIS approximation)
        col_term = 1.0 / (1.0 + s * tau_col / n_eff) ** n_eff

        # Particle dynamics (LDF approximation)
        particle_term = 1.0 / (1.0 + s * tau_p * (1 + phi))

        # Film resistance
        tau_film = particle_radius / (3 * film_diffusion)
        film_term = 1.0 / (1.0 + s * tau_film)

        return col_term * particle_term * film_term

    return F


def get_benchmark_functions() -> List[Tuple[str, Callable, Tuple[float, float], bool, float]]:
    """Get list of all benchmark transfer functions for NILT verification.

    Returns:
        List of tuples: (name, F, (t_min, t_max), has_analytical_solution, alpha_c)
    """
    return [
        (
            "advection_dispersion_Pe100",
            advection_dispersion_transfer(velocity=1e-3, dispersion=1e-6, length=0.1),
            (1.0, 500.0),
            False,  # No simple closed-form
            0.0,  # alpha_c
        ),
        (
            "advection_dispersion_Pe10",
            advection_dispersion_transfer(velocity=1e-3, dispersion=1e-5, length=0.1),
            (1.0, 500.0),
            False,
            0.0,
        ),
        (
            "langmuir_linear",
            langmuir_column_transfer(
                velocity=1e-3,
                dispersion=1e-6,
                length=0.1,
                ka=1.0,
                kd=0.1,
                qmax=10.0,
            ),
            (1.0, 1000.0),
            False,
            0.0,
        ),
        (
            "grm_moment_approx",
            grm_moment_transfer(
                velocity=1e-3,
                dispersion=1e-6,
                length=0.1,
                particle_radius=5e-5,
                film_diffusion=1e-5,
                pore_diffusion=1e-10,
            ),
            (1.0, 500.0),
            False,
            0.0,
        ),
    ]


def create_benchmark_problem(
    name: str,
    F: Callable[[complex], complex],
    alpha_c: float = 0.0,
    C: float = 1.0,
    rho: Optional[float] = None,
    description: str = "",
) -> Problem:
    """Create a Problem object for a custom benchmark.

    Args:
        name: Problem name.
        F: Transfer function.
        alpha_c: Abscissa of convergence.
        C: Tail envelope constant.
        rho: Spectral radius (optional).
        description: Problem description.

    Returns:
        Problem object compatible with nilt-cfl.
    """
    return Problem(
        name=name,
        F=F,
        f_ref=None,
        alpha_c=alpha_c,
        C=C,
        rho=rho,
        description=description,
    )
