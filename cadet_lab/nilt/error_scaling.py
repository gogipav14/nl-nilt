"""Dimensionless group analysis and systematic error scaling for NL-NILT.

Provides tools for:
  - Computing dimensionless groups (Pe, Bi, NL*, S, Da) from physical parameters
  - Defining parameter sets spanning different chromatographic regimes
  - Running systematic parameter sweeps comparing NL-NILT vs CADET
  - Building 2D empirical regime maps (NL* x Pe -> expected L2 error)

Created for the CACE revision to address systematic error characterization.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .nonlinear import LangmuirBinding, nl_nilt_solve


# ---------------------------------------------------------------------------
# CADET config generator for nonlinear Langmuir
# ---------------------------------------------------------------------------

def _write_str(group: "h5py.Group", name: str, value: str) -> None:
    """Write a string dataset to an HDF5 group."""
    dt = h5py.string_dtype(encoding="ascii")
    group.create_dataset(name, data=value, dtype=dt)


def _create_nonlinear_langmuir_config(
    output_path: Path,
    ka: float, kd: float, qmax: float,
    c_feed: float = 1.0,
    transport: Optional[dict] = None,
    n_times: int = 501,
    end_time: float = 500.0,
    n_col: int = 64,
) -> Path:
    """Create CADET config with MULTI_COMPONENT_LANGMUIR binding.

    Based on the config generator in run_nl_nilt_benchmarks.py.
    """
    DEFAULT_TRANSPORT = dict(
        velocity=1e-3, dispersion=1e-6, length=0.1, col_porosity=0.37,
        par_radius=1e-5, par_porosity=0.33, film_diffusion=1e-5,
        pore_diffusion=1e-10,
    )
    if transport is None:
        transport = DEFAULT_TRANSPORT

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solution_times = np.linspace(0, end_time, n_times)
    flow_rate = transport["velocity"] * transport["col_porosity"] * 1e-4

    with h5py.File(output_path, "w") as f:
        inp = f.create_group("input")
        model = inp.create_group("model")
        solver = inp.create_group("solver")
        ret = inp.create_group("return")

        model.create_dataset("NUNITS", data=3)

        # Unit 000: INLET
        u0 = model.create_group("unit_000")
        _write_str(u0, "UNIT_TYPE", "INLET")
        _write_str(u0, "INLET_TYPE", "PIECEWISE_CUBIC_POLY")
        u0.create_dataset("NCOMP", data=1)
        sec = u0.create_group("sec_000")
        sec.create_dataset("CONST_COEFF", data=[c_feed])
        sec.create_dataset("LIN_COEFF", data=[0.0])
        sec.create_dataset("QUAD_COEFF", data=[0.0])
        sec.create_dataset("CUBE_COEFF", data=[0.0])

        # Unit 001: GRM
        u1 = model.create_group("unit_001")
        _write_str(u1, "UNIT_TYPE", "GENERAL_RATE_MODEL")
        u1.create_dataset("NCOMP", data=1)
        u1.create_dataset("NPARTYPE", data=1)
        u1.create_dataset("COL_LENGTH", data=transport["length"])
        u1.create_dataset("COL_POROSITY", data=transport["col_porosity"])
        u1.create_dataset("CROSS_SECTION_AREA", data=1e-4)
        u1.create_dataset("COL_DISPERSION", data=transport["dispersion"])
        u1.create_dataset("VELOCITY", data=transport["velocity"])
        u1.create_dataset("INIT_C", data=[0.0])
        u1.create_dataset("INIT_CS", data=[0.0])

        disc = u1.create_group("discretization")
        _write_str(disc, "SPATIAL_METHOD", "FV")
        disc.create_dataset("USE_ANALYTIC_JACOBIAN", data=1)
        disc.create_dataset("NCOL", data=n_col)
        _write_str(disc, "RECONSTRUCTION", "WENO")
        disc.create_dataset("GS_TYPE", data=1)
        disc.create_dataset("MAX_KRYLOV", data=0)
        disc.create_dataset("MAX_RESTARTS", data=10)
        disc.create_dataset("SCHUR_SAFETY", data=1e-8)
        weno = disc.create_group("weno")
        weno.create_dataset("BOUNDARY_MODEL", data=0)
        weno.create_dataset("WENO_EPS", data=1e-10)
        weno.create_dataset("WENO_ORDER", data=3)

        # Particle type with MULTI_COMPONENT_LANGMUIR
        pt = u1.create_group("particle_type_000")
        _write_str(pt, "PAR_GEOM", "SPHERE")
        pt.create_dataset("PAR_POROSITY", data=transport["par_porosity"])
        pt.create_dataset("PAR_RADIUS", data=transport["par_radius"])
        pt.create_dataset("PAR_CORERADIUS", data=0.0)
        pt.create_dataset("NBOUND", data=[1])
        pt.create_dataset("FILM_DIFFUSION", data=[transport["film_diffusion"]])
        pt.create_dataset("FILM_DIFFUSION_MULTIPLEX", data=0)
        pt.create_dataset("PORE_DIFFUSION", data=[transport["pore_diffusion"]])
        pt.create_dataset("SURFACE_DIFFUSION", data=[0.0])
        pt.create_dataset("HAS_FILM_DIFFUSION", data=1)
        pt.create_dataset("HAS_PORE_DIFFUSION", data=1)
        pt.create_dataset("HAS_SURFACE_DIFFUSION", data=0)

        _write_str(pt, "ADSORPTION_MODEL", "MULTI_COMPONENT_LANGMUIR")
        ads = pt.create_group("adsorption")
        ads.create_dataset("IS_KINETIC", data=1)
        ads.create_dataset("MCL_KA", data=[ka])
        ads.create_dataset("MCL_KD", data=[kd])
        ads.create_dataset("MCL_QMAX", data=[qmax])

        par_disc = pt.create_group("discretization")
        _write_str(par_disc, "PAR_DISC_TYPE", "EQUIDISTANT_PAR")
        par_disc.create_dataset("NCELLS", data=4)
        par_disc.create_dataset("SPATIAL_METHOD", data=0)
        par_disc.create_dataset("FV_BOUNDARY_ORDER", data=2)

        # Unit 002: OUTLET
        u2 = model.create_group("unit_002")
        _write_str(u2, "UNIT_TYPE", "OUTLET")
        u2.create_dataset("NCOMP", data=1)

        # Connections
        conn = model.create_group("connections")
        conn.create_dataset("NSWITCHES", data=1)
        sw = conn.create_group("switch_000")
        sw.create_dataset("SECTION", data=0)
        sw.create_dataset("CONNECTIONS", data=[
            0, 1, -1, -1, flow_rate,
            1, 2, -1, -1, flow_rate,
        ])

        # Model solver
        msolver = model.create_group("solver")
        msolver.create_dataset("GS_TYPE", data=1)
        msolver.create_dataset("MAX_KRYLOV", data=0)
        msolver.create_dataset("MAX_RESTARTS", data=10)
        msolver.create_dataset("SCHUR_SAFETY", data=1e-8)

        # Time solver
        solver.create_dataset("NTHREADS", data=1)
        solver.create_dataset("USER_SOLUTION_TIMES", data=solution_times)
        sections = solver.create_group("sections")
        sections.create_dataset("NSEC", data=1)
        sections.create_dataset("SECTION_TIMES", data=[0.0, end_time])
        sections.create_dataset("SECTION_CONTINUITY", data=[])
        ti = solver.create_group("time_integrator")
        ti.create_dataset("ABSTOL", data=1e-8)
        ti.create_dataset("ALGTOL", data=1e-10)
        ti.create_dataset("RELTOL", data=1e-8)
        ti.create_dataset("INIT_STEP_SIZE", data=1e-6)
        ti.create_dataset("MAX_STEPS", data=500000)

        # Return config
        ret.create_dataset("SPLIT_COMPONENTS_DATA", data=0)
        ret.create_dataset("SPLIT_PORTS_DATA", data=0)
        ret.create_dataset("WRITE_SOLVER_STATISTICS", data=1)
        for uname in ["unit_000", "unit_001", "unit_002"]:
            ur = ret.create_group(uname)
            ur.create_dataset("WRITE_SOLUTION_INLET", data=1)
            ur.create_dataset("WRITE_SOLUTION_OUTLET", data=1)
            ur.create_dataset("WRITE_SOLUTION_BULK", data=0)
            ur.create_dataset("WRITE_SOLUTION_PARTICLE", data=0)
            ur.create_dataset("WRITE_SOLUTION_SOLID", data=0)
            ur.create_dataset("WRITE_SOLUTION_FLUX", data=0)
            ur.create_dataset("WRITE_SOLUTION_VOLUME", data=0)
            ur.create_dataset("WRITE_COORDINATES", data=0)
            ur.create_dataset("WRITE_SENS_OUTLET", data=0)

    return output_path


def _create_nonlinear_sma_config(
    output_path: Path,
    ka: float, kd: float, Lambda: float, nu: float,
    z_protein: float, z_salt: float, c_salt: float,
    c_feed: float = 1.0,
    transport: Optional[dict] = None,
    n_times: int = 501,
    end_time: float = 500.0,
    n_col: int = 64,
) -> Path:
    """Create CADET config with STERIC_MASS_ACTION binding (2 components: salt + protein)."""
    DEFAULT_TRANSPORT = dict(
        velocity=1e-3, dispersion=1e-6, length=0.1, col_porosity=0.37,
        par_radius=1e-5, par_porosity=0.33, film_diffusion=1e-5,
        pore_diffusion=1e-10,
    )
    if transport is None:
        transport = DEFAULT_TRANSPORT

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solution_times = np.linspace(0, end_time, n_times)
    flow_rate = transport["velocity"] * transport["col_porosity"] * 1e-4
    n_comp = 2  # salt (comp 0) + protein (comp 1)

    with h5py.File(output_path, "w") as f:
        inp = f.create_group("input")
        model = inp.create_group("model")
        solver = inp.create_group("solver")
        ret = inp.create_group("return")

        model.create_dataset("NUNITS", data=3)

        # Unit 000: INLET (salt constant, protein step)
        u0 = model.create_group("unit_000")
        _write_str(u0, "UNIT_TYPE", "INLET")
        _write_str(u0, "INLET_TYPE", "PIECEWISE_CUBIC_POLY")
        u0.create_dataset("NCOMP", data=n_comp)
        sec = u0.create_group("sec_000")
        sec.create_dataset("CONST_COEFF", data=[c_salt, c_feed])
        sec.create_dataset("LIN_COEFF", data=[0.0, 0.0])
        sec.create_dataset("QUAD_COEFF", data=[0.0, 0.0])
        sec.create_dataset("CUBE_COEFF", data=[0.0, 0.0])

        # Unit 001: GRM
        u1 = model.create_group("unit_001")
        _write_str(u1, "UNIT_TYPE", "GENERAL_RATE_MODEL")
        u1.create_dataset("NCOMP", data=n_comp)
        u1.create_dataset("NPARTYPE", data=1)
        u1.create_dataset("COL_LENGTH", data=transport["length"])
        u1.create_dataset("COL_POROSITY", data=transport["col_porosity"])
        u1.create_dataset("CROSS_SECTION_AREA", data=1e-4)
        u1.create_dataset("COL_DISPERSION", data=transport["dispersion"])
        u1.create_dataset("VELOCITY", data=transport["velocity"])
        u1.create_dataset("INIT_C", data=[c_salt, 0.0])
        u1.create_dataset("INIT_CS", data=[c_salt, 0.0])

        disc = u1.create_group("discretization")
        _write_str(disc, "SPATIAL_METHOD", "FV")
        disc.create_dataset("USE_ANALYTIC_JACOBIAN", data=1)
        disc.create_dataset("NCOL", data=n_col)
        _write_str(disc, "RECONSTRUCTION", "WENO")
        disc.create_dataset("GS_TYPE", data=1)
        disc.create_dataset("MAX_KRYLOV", data=0)
        disc.create_dataset("MAX_RESTARTS", data=10)
        disc.create_dataset("SCHUR_SAFETY", data=1e-8)
        weno = disc.create_group("weno")
        weno.create_dataset("BOUNDARY_MODEL", data=0)
        weno.create_dataset("WENO_EPS", data=1e-10)
        weno.create_dataset("WENO_ORDER", data=3)

        # Particle type with STERIC_MASS_ACTION
        pt = u1.create_group("particle_type_000")
        _write_str(pt, "PAR_GEOM", "SPHERE")
        pt.create_dataset("PAR_POROSITY", data=transport["par_porosity"])
        pt.create_dataset("PAR_RADIUS", data=transport["par_radius"])
        pt.create_dataset("PAR_CORERADIUS", data=0.0)
        # SMA requires NBOUND=1 for salt (pseudo-bound state) and protein
        pt.create_dataset("NBOUND", data=[1, 1])
        pt.create_dataset("FILM_DIFFUSION", data=[transport["film_diffusion"]] * n_comp)
        pt.create_dataset("FILM_DIFFUSION_MULTIPLEX", data=0)
        pt.create_dataset("PORE_DIFFUSION", data=[transport["pore_diffusion"]] * n_comp)
        pt.create_dataset("SURFACE_DIFFUSION", data=[0.0] * n_comp)
        pt.create_dataset("HAS_FILM_DIFFUSION", data=1)
        pt.create_dataset("HAS_PORE_DIFFUSION", data=1)
        pt.create_dataset("HAS_SURFACE_DIFFUSION", data=0)

        _write_str(pt, "ADSORPTION_MODEL", "STERIC_MASS_ACTION")
        ads = pt.create_group("adsorption")
        ads.create_dataset("IS_KINETIC", data=1)
        ads.create_dataset("SMA_KA", data=[0.0, ka])  # salt=0, protein=ka
        ads.create_dataset("SMA_KD", data=[0.0, kd])
        ads.create_dataset("SMA_NU", data=[0.0, nu])
        ads.create_dataset("SMA_SIGMA", data=[0.0, z_protein])
        ads.create_dataset("SMA_LAMBDA", data=Lambda)
        ads.create_dataset("SMA_REFC0", data=1.0)
        ads.create_dataset("SMA_REFQ", data=1.0)

        par_disc = pt.create_group("discretization")
        _write_str(par_disc, "PAR_DISC_TYPE", "EQUIDISTANT_PAR")
        par_disc.create_dataset("NCELLS", data=4)
        par_disc.create_dataset("SPATIAL_METHOD", data=0)
        par_disc.create_dataset("FV_BOUNDARY_ORDER", data=2)

        # Unit 002: OUTLET
        u2 = model.create_group("unit_002")
        _write_str(u2, "UNIT_TYPE", "OUTLET")
        u2.create_dataset("NCOMP", data=n_comp)

        # Connections
        conn = model.create_group("connections")
        conn.create_dataset("NSWITCHES", data=1)
        sw = conn.create_group("switch_000")
        sw.create_dataset("SECTION", data=0)
        sw.create_dataset("CONNECTIONS", data=[
            0, 1, -1, -1, flow_rate,
            1, 2, -1, -1, flow_rate,
        ])

        # Model solver
        msolver = model.create_group("solver")
        msolver.create_dataset("GS_TYPE", data=1)
        msolver.create_dataset("MAX_KRYLOV", data=0)
        msolver.create_dataset("MAX_RESTARTS", data=10)
        msolver.create_dataset("SCHUR_SAFETY", data=1e-8)

        # Time solver
        solver.create_dataset("NTHREADS", data=1)
        solver.create_dataset("USER_SOLUTION_TIMES", data=solution_times)
        sections = solver.create_group("sections")
        sections.create_dataset("NSEC", data=1)
        sections.create_dataset("SECTION_TIMES", data=[0.0, end_time])
        sections.create_dataset("SECTION_CONTINUITY", data=[])
        ti = solver.create_group("time_integrator")
        ti.create_dataset("ABSTOL", data=1e-8)
        ti.create_dataset("ALGTOL", data=1e-10)
        ti.create_dataset("RELTOL", data=1e-8)
        ti.create_dataset("INIT_STEP_SIZE", data=1e-6)
        ti.create_dataset("MAX_STEPS", data=500000)

        # Return config
        ret.create_dataset("SPLIT_COMPONENTS_DATA", data=0)
        ret.create_dataset("SPLIT_PORTS_DATA", data=0)
        ret.create_dataset("WRITE_SOLVER_STATISTICS", data=1)
        for uname in ["unit_000", "unit_001", "unit_002"]:
            ur = ret.create_group(uname)
            ur.create_dataset("WRITE_SOLUTION_INLET", data=1)
            ur.create_dataset("WRITE_SOLUTION_OUTLET", data=1)
            ur.create_dataset("WRITE_SOLUTION_BULK", data=0)
            ur.create_dataset("WRITE_SOLUTION_PARTICLE", data=0)
            ur.create_dataset("WRITE_SOLUTION_SOLID", data=0)
            ur.create_dataset("WRITE_SOLUTION_FLUX", data=0)
            ur.create_dataset("WRITE_SOLUTION_VOLUME", data=0)
            ur.create_dataset("WRITE_COORDINATES", data=0)
            ur.create_dataset("WRITE_SENS_OUTLET", data=0)

    return output_path


# ---------------------------------------------------------------------------
# SMA benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class SMABenchmarkResult:
    """Result from a single SMA NL-NILT vs CADET comparison."""
    c_feed: float
    c_salt: float
    nu: float
    # NL-NILT
    nilt_converged: bool
    nilt_iterations: int
    nilt_wall_ms: float
    nilt_t: np.ndarray
    nilt_c: np.ndarray
    # CADET
    cadet_t: np.ndarray
    cadet_c: np.ndarray
    cadet_wall_ms: float
    # Metrics
    rel_l2_error: float = np.nan


def run_sma_benchmark(
    cadet_cli: str,
    output_dir: Path,
    sma_params: Optional[dict] = None,
    c_feeds: Optional[list[float]] = None,
    c_salts: Optional[list[float]] = None,
    nus: Optional[list[float]] = None,
) -> list[SMABenchmarkResult]:
    """Run SMA NL-NILT vs CADET sweep over feed concentration, salt, and/or charge."""
    from .nonlinear import SMABinding

    if sma_params is None:
        sma_params = dict(
            ka=0.1, kd=10.0, Lambda=10.0, nu=2.0,
            z_protein=2.0, z_salt=1.0, c_salt=0.05,
        )
    transport = dict(
        velocity=1e-3, dispersion=1e-6, length=0.1, col_porosity=0.37,
        par_radius=1e-5, par_porosity=0.33, film_diffusion=1e-5,
        pore_diffusion=1e-10,
    )

    # Build sweep combinations
    cases = []
    base_c_feed = 0.1
    base_c_salt = sma_params["c_salt"]
    base_nu = sma_params["nu"]

    if c_feeds is not None:
        for cf in c_feeds:
            cases.append((cf, base_c_salt, base_nu, "c_feed sweep"))
    if c_salts is not None:
        for cs in c_salts:
            cases.append((base_c_feed, cs, base_nu, "c_salt sweep"))
    if nus is not None:
        for n in nus:
            cases.append((base_c_feed, base_c_salt, n, "nu sweep"))
    if not cases:
        # Default: just c_feed sweep
        for cf in [0.01, 0.1, 1.0]:
            cases.append((cf, base_c_salt, base_nu, "default"))

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (c_feed, c_salt, nu, label) in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] SMA {label}: c_feed={c_feed:.3f}, c_salt={c_salt:.3f}, nu={nu:.1f}")

        # Compute t_end from effective retention
        Lambda_eff = sma_params["Lambda"] - sma_params["z_salt"] * c_salt
        K_eq_eff = (sma_params["ka"] / sma_params["kd"]) * (Lambda_eff / c_salt) ** nu
        phi = (1 - transport["col_porosity"]) / transport["col_porosity"]
        mu1 = (transport["length"] / transport["velocity"]) * (
            1 + phi * (transport["par_porosity"] + (1 - transport["par_porosity"]) * K_eq_eff)
        )
        t_end = max(5.0 * mu1, 200.0)
        t_end = min(t_end, 50000.0)  # Cap at ~14 hours

        try:
            # NL-NILT
            params_copy = dict(sma_params)
            params_copy["c_salt"] = c_salt
            params_copy["nu"] = nu
            binding = SMABinding(
                c0=c_feed, **params_copy, **transport,
            )
            t0 = time.perf_counter()
            nilt_result = nl_nilt_solve(
                binding, t_end=t_end, c_feed=c_feed,
                max_iterations=10, eps_conv=1e-4,
            )
            nilt_wall = (time.perf_counter() - t0) * 1000

            # CADET
            tag = f"sma_cf{c_feed:.3f}_cs{c_salt:.3f}_nu{nu:.1f}"
            cfg_path = output_dir / f"cadet_{tag}.h5"
            _create_nonlinear_sma_config(
                cfg_path,
                ka=sma_params["ka"], kd=sma_params["kd"],
                Lambda=sma_params["Lambda"], nu=nu,
                z_protein=sma_params["z_protein"],
                z_salt=sma_params["z_salt"], c_salt=c_salt,
                c_feed=c_feed, transport=transport,
                end_time=t_end, n_col=64, n_times=501,
            )
            t0 = time.perf_counter()
            proc = subprocess.run(
                [cadet_cli, str(cfg_path)],
                capture_output=True, text=True, timeout=300,
            )
            cadet_wall = (time.perf_counter() - t0) * 1000

            if proc.returncode != 0:
                print(f"  -> CADET FAILED: {proc.stderr[:200]}")
                continue

            with h5py.File(cfg_path, "r") as fh:
                cadet_t = fh["output/solution/SOLUTION_TIMES"][:]
                cadet_outlet = fh["output/solution/unit_002/SOLUTION_OUTLET"][:]
                cadet_c = cadet_outlet[:, 1]  # protein is component 1

            # Compare
            nilt_interp = np.interp(cadet_t, nilt_result.t, nilt_result.c)
            diff = nilt_interp - cadet_c
            cadet_norm = np.sqrt(np.mean(cadet_c**2))
            rel_l2 = np.sqrt(np.mean(diff**2)) / cadet_norm if cadet_norm > 1e-15 else np.nan

            results.append(SMABenchmarkResult(
                c_feed=c_feed, c_salt=c_salt, nu=nu,
                nilt_converged=nilt_result.converged,
                nilt_iterations=nilt_result.n_iterations,
                nilt_wall_ms=nilt_wall,
                nilt_t=nilt_result.t, nilt_c=nilt_result.c,
                cadet_t=cadet_t, cadet_c=cadet_c,
                cadet_wall_ms=cadet_wall,
                rel_l2_error=rel_l2,
            ))
            print(f"  -> OK: L2={rel_l2:.4f}, iters={nilt_result.n_iterations}, "
                  f"NILT={nilt_wall:.0f}ms, CADET={cadet_wall:.0f}ms")

        except Exception as e:
            print(f"  -> FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Dimensionless groups
# ---------------------------------------------------------------------------

@dataclass
class DimensionlessGroups:
    """Dimensionless groups characterizing a GRM problem."""
    Pe: float        # Peclet number = u*L/D_ax
    Bi: float        # Biot number = k_f*R_p/D_p
    phi: float       # Phase ratio = (1-eps_c)/eps_c
    K_eq: float      # Equilibrium constant = k_a*q_max/k_d
    K_a: float       # Affinity = k_a/k_d
    A: float         # Affinity-loading = K_a * c_feed
    S: float         # Saturation ratio = c_feed / q_max
    NL_star: float   # Bounded nonlinearity = A / (1 + A) in [0, 1]
    Da: float        # Binding Damkohler = k_a * q_max * L / u
    tau_transport: float  # Residence time = L/u
    tau_binding: float    # Binding timescale = 1/k_d
    tau_diffusion: float  # Diffusion timescale = R_p^2 / D_p


def compute_dimensionless_groups(
    velocity: float, dispersion: float, length: float,
    col_porosity: float, par_radius: float, par_porosity: float,
    film_diffusion: float, pore_diffusion: float,
    ka: float, kd: float, qmax: float, c_feed: float,
) -> DimensionlessGroups:
    """Compute all relevant dimensionless groups from physical parameters."""
    Pe = velocity * length / dispersion
    Bi = film_diffusion * par_radius / pore_diffusion
    phi = (1 - col_porosity) / col_porosity
    K_a = ka / kd
    K_eq = ka * qmax / kd
    A = K_a * c_feed
    S = c_feed / qmax
    NL_star = A / (1 + A)
    Da = ka * qmax * length / velocity
    tau_transport = length / velocity
    tau_binding = 1.0 / kd
    tau_diffusion = par_radius**2 / pore_diffusion

    return DimensionlessGroups(
        Pe=Pe, Bi=Bi, phi=phi, K_eq=K_eq, K_a=K_a,
        A=A, S=S, NL_star=NL_star, Da=Da,
        tau_transport=tau_transport, tau_binding=tau_binding,
        tau_diffusion=tau_diffusion,
    )


def theoretical_error_bound(K_a: float, c_feed: float) -> float:
    """Theoretical relative truncation error from Taylor expansion (Eq. 17).

    Returns |E|/q_max ~ K_a^2 * c_feed^2 / (2 * (1 + K_a*c_feed/2)^3).
    """
    return K_a**2 * c_feed**2 / (2.0 * (1.0 + K_a * c_feed / 2.0)**3)


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

@dataclass
class ParameterSet:
    """A named set of GRM transport + binding parameters."""
    name: str
    description: str
    velocity: float
    dispersion: float
    length: float
    col_porosity: float
    par_radius: float
    par_porosity: float
    film_diffusion: float
    pore_diffusion: float
    ka: float
    kd: float
    qmax: float

    @property
    def Pe(self) -> float:
        return self.velocity * self.length / self.dispersion

    @property
    def Bi(self) -> float:
        return self.film_diffusion * self.par_radius / self.pore_diffusion

    @property
    def K_a(self) -> float:
        return self.ka / self.kd

    @property
    def K_eq(self) -> float:
        return self.ka * self.qmax / self.kd


# Five parameter sets spanning different chromatographic regimes
PARAMETER_SETS = {
    "set1_analytical": ParameterSet(
        name="Set 1", description="Standard analytical column",
        velocity=1e-3, dispersion=1e-6, length=0.1,
        col_porosity=0.37, par_radius=1e-5, par_porosity=0.33,
        film_diffusion=1e-5, pore_diffusion=1e-10,
        ka=1.0, kd=5.0, qmax=10.0,
    ),
    "set2_preparative": ParameterSet(
        name="Set 2", description="Preparative column, large particles",
        velocity=5e-4, dispersion=5e-7, length=0.25,
        col_porosity=0.37, par_radius=2.5e-5, par_porosity=0.33,
        film_diffusion=1e-5, pore_diffusion=1e-10,
        ka=1.0, kd=5.0, qmax=10.0,
    ),
    "set3_diffusion_dominated": ParameterSet(
        name="Set 3", description="Diffusion-dominated (low Pe)",
        velocity=5e-3, dispersion=1e-4, length=0.05,
        col_porosity=0.37, par_radius=5e-6, par_porosity=0.33,
        film_diffusion=1e-5, pore_diffusion=1e-10,
        ka=1.0, kd=5.0, qmax=10.0,
    ),
    "set4_stiff_kinetics": ParameterSet(
        name="Set 4", description="Strong binding (K_eq=20)",
        velocity=1e-3, dispersion=1e-6, length=0.1,
        col_porosity=0.37, par_radius=1e-5, par_porosity=0.33,
        film_diffusion=1e-5, pore_diffusion=1e-10,
        ka=2.0, kd=1.0, qmax=10.0,  # K_eq=20 (10x baseline)
    ),
    "set5_sharp_shock": ParameterSet(
        name="Set 5", description="Sharp shock (high Pe)",
        velocity=1e-3, dispersion=1e-7, length=0.1,
        col_porosity=0.37, par_radius=1e-5, par_porosity=0.33,
        film_diffusion=1e-5, pore_diffusion=1e-10,
        ka=1.0, kd=5.0, qmax=10.0,
    ),
}

LOADING_LEVELS = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0]  # c_feed values


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result from a single NL-NILT vs CADET comparison."""
    param_set: str
    c_feed: float
    groups: DimensionlessGroups
    # NL-NILT results
    nilt_converged: bool
    nilt_iterations: int
    nilt_wall_ms: float
    nilt_t: np.ndarray
    nilt_c: np.ndarray
    # CADET results
    cadet_t: np.ndarray
    cadet_c: np.ndarray
    cadet_wall_ms: float
    # Comparison metrics
    l2_error: float = np.nan
    linf_error: float = np.nan
    rel_l2_error: float = np.nan


def run_single_benchmark(
    pset: ParameterSet,
    c_feed: float,
    cadet_cli: str,
    output_dir: Path,
    t_end: Optional[float] = None,
    ncol: int = 64,
    n_times: int = 501,
) -> BenchmarkResult:
    """Run NL-NILT and CADET for a single parameter set + loading level."""

    # Compute t_end if not specified: 5x the linear retention time
    if t_end is None:
        K_eq = pset.ka * pset.qmax / pset.kd
        phi = (1 - pset.col_porosity) / pset.col_porosity
        mu1 = (pset.length / pset.velocity) * (
            1 + phi * (pset.par_porosity + (1 - pset.par_porosity) * K_eq)
        )
        t_end = max(5.0 * mu1, 100.0)

    groups = compute_dimensionless_groups(
        pset.velocity, pset.dispersion, pset.length, pset.col_porosity,
        pset.par_radius, pset.par_porosity, pset.film_diffusion,
        pset.pore_diffusion, pset.ka, pset.kd, pset.qmax, c_feed,
    )

    # --- Run NL-NILT ---
    binding = LangmuirBinding(
        ka=pset.ka, kd=pset.kd, qmax=pset.qmax,
        velocity=pset.velocity, dispersion=pset.dispersion, length=pset.length,
        col_porosity=pset.col_porosity, par_radius=pset.par_radius,
        par_porosity=pset.par_porosity, film_diffusion=pset.film_diffusion,
        pore_diffusion=pset.pore_diffusion,
    )
    t0 = time.perf_counter()
    nilt_result = nl_nilt_solve(
        binding, t_end=t_end, c_feed=c_feed,
        max_iterations=10, eps_conv=1e-8,
    )
    nilt_wall = (time.perf_counter() - t0) * 1000

    # --- Run CADET ---
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{pset.name.replace(' ', '_')}_cfeed{c_feed:.1f}"
    cfg_path = output_dir / f"cadet_{tag}.h5"

    _create_nonlinear_langmuir_config(
        cfg_path,
        ka=pset.ka, kd=pset.kd, qmax=pset.qmax, c_feed=c_feed,
        transport=dict(
            velocity=pset.velocity, dispersion=pset.dispersion,
            length=pset.length, col_porosity=pset.col_porosity,
            par_radius=pset.par_radius, par_porosity=pset.par_porosity,
            film_diffusion=pset.film_diffusion, pore_diffusion=pset.pore_diffusion,
        ),
        end_time=t_end, n_col=ncol, n_times=n_times,
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [cadet_cli, str(cfg_path)],
        capture_output=True, text=True, timeout=300,
    )
    cadet_wall = (time.perf_counter() - t0) * 1000

    if proc.returncode != 0:
        raise RuntimeError(f"CADET failed for {tag}: {proc.stderr[:300]}")

    with h5py.File(cfg_path, "r") as f:
        cadet_t = f["output/solution/SOLUTION_TIMES"][:]
        cadet_c = f["output/solution/unit_002/SOLUTION_OUTLET"][:, 0]

    # --- Compute comparison metrics ---
    # Interpolate NL-NILT onto CADET time grid
    nilt_interp = np.interp(cadet_t, nilt_result.t, nilt_result.c)

    diff = nilt_interp - cadet_c
    l2_error = np.sqrt(np.mean(diff**2))
    linf_error = np.max(np.abs(diff))
    cadet_norm = np.sqrt(np.mean(cadet_c**2))
    rel_l2_error = l2_error / cadet_norm if cadet_norm > 1e-15 else np.nan

    return BenchmarkResult(
        param_set=pset.name, c_feed=c_feed, groups=groups,
        nilt_converged=nilt_result.converged,
        nilt_iterations=nilt_result.n_iterations,
        nilt_wall_ms=nilt_wall,
        nilt_t=nilt_result.t, nilt_c=nilt_result.c,
        cadet_t=cadet_t, cadet_c=cadet_c,
        cadet_wall_ms=cadet_wall,
        l2_error=l2_error, linf_error=linf_error,
        rel_l2_error=rel_l2_error,
    )


# ---------------------------------------------------------------------------
# Full sweep runner
# ---------------------------------------------------------------------------

def run_langmuir_sweep(
    cadet_cli: str,
    output_dir: Path,
    parameter_sets: Optional[dict[str, ParameterSet]] = None,
    loading_levels: Optional[list[float]] = None,
    ncol: int = 64,
) -> list[BenchmarkResult]:
    """Run the full Langmuir parameter sweep across all sets and loadings.

    Returns list of BenchmarkResult for all successful cases.
    """
    if parameter_sets is None:
        parameter_sets = PARAMETER_SETS
    if loading_levels is None:
        loading_levels = LOADING_LEVELS

    results = []
    total = len(parameter_sets) * len(loading_levels)

    for i, (key, pset) in enumerate(parameter_sets.items()):
        for j, c_feed in enumerate(loading_levels):
            idx = i * len(loading_levels) + j + 1
            print(f"[{idx}/{total}] {pset.name}: c_feed={c_feed:.1f} "
                  f"(Pe={pset.Pe:.1f}, Bi={pset.Bi:.1f})")
            try:
                result = run_single_benchmark(
                    pset, c_feed, cadet_cli,
                    output_dir / key, ncol=ncol,
                )
                results.append(result)
                status = "OK" if result.nilt_converged else "NOT CONVERGED"
                print(f"  -> {status}: L2={result.rel_l2_error:.4f}, "
                      f"iters={result.nilt_iterations}, "
                      f"NILT={result.nilt_wall_ms:.0f}ms, "
                      f"CADET={result.cadet_wall_ms:.0f}ms")
            except Exception as e:
                print(f"  -> FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def summarize_results(results: list[BenchmarkResult]) -> str:
    """Format results as a table string."""
    header = (
        f"{'Set':<8} {'Pe':>6} {'Bi':>6} {'c_feed':>6} {'NL*':>5} "
        f"{'S':>5} {'L2_rel':>8} {'Iters':>5} {'NILT_ms':>8} {'CADET_ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.param_set:<8} {r.groups.Pe:>6.1f} {r.groups.Bi:>6.1f} "
            f"{r.c_feed:>6.1f} {r.groups.NL_star:>5.3f} "
            f"{r.groups.S:>5.3f} {r.rel_l2_error:>8.4f} "
            f"{r.nilt_iterations:>5} {r.nilt_wall_ms:>8.1f} "
            f"{r.cadet_wall_ms:>9.1f}"
        )
    return "\n".join(lines)
