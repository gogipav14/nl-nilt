#!/usr/bin/env python3
"""Run NL-NILT benchmarks for the CACE paper.

Produces all numerical results needed for the paper:
  1. Loading sweep: NL-NILT across c_feed range, measuring L2 error and iteration count
  2. CADET comparison: NL-NILT vs CADET (v6) for Langmuir binding
  3. SMA benchmark: NL-NILT with SMA binding against CADET reference
  4. Larger column benchmark: L=0.25m with more CADET cells
  5. Convergence diagnostics: mass-balance traces per iteration
  6. Parameter estimation mini-case: fit ka, kd from synthetic CADET data

Usage:
  python scripts/run_nl_nilt_benchmarks.py --cadet-cli /path/to/cadet-cli
  python scripts/run_nl_nilt_benchmarks.py --cadet-cli /path/to/cadet-cli --skip-cadet  # NL-NILT only
  python scripts/run_nl_nilt_benchmarks.py --cadet-cli /path/to/cadet-cli --benchmark loading_sweep
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadet_lab.nilt.nonlinear import (
    LangmuirBinding,
    SMABinding,
    NLNiltResult,
    nl_nilt_solve,
)
from cadet_lab.nilt.mass_balance import (
    compute_diagnostics,
    compute_numerical_first_moment,
    compute_theoretical_first_moment,
    check_zeroth_moment,
    steering_decision,
    MassBalanceDiagnostics,
)
from cadet_lab.nilt.solver import NLNiltSolver
from cadet_lab.nilt.compare_to_cadet import (
    interpolate_to_common_grid,
    compute_comparison_metrics,
    ComparisonResult,
)


# ---------------------------------------------------------------------------
# Standard GRM transport parameters (shared across all benchmarks)
# ---------------------------------------------------------------------------

GRM_TRANSPORT = dict(
    velocity=1e-3,
    dispersion=1e-6,
    length=0.1,
    col_porosity=0.37,
    par_radius=1e-5,
    par_porosity=0.33,
    film_diffusion=1e-5,
    pore_diffusion=1e-10,
)

GRM_LARGE_COLUMN = dict(
    velocity=5e-4,
    dispersion=5e-7,
    length=0.25,
    col_porosity=0.37,
    par_radius=2.5e-5,
    par_porosity=0.33,
    film_diffusion=5e-6,
    pore_diffusion=5e-11,
)

# Primary benchmark binding parameters.
# K_eq = ka*qmax/kd = 2.0 -> theoretical retention time mu1 ~ 384 s.
# This ensures breakthrough occurs within t_end = 1500 s.
# The ratio Ka*c_feed = (ka/kd)*c_feed determines the nonlinear strength:
#   c_feed=0.5 -> Ka*c=0.1 (mild), c_feed=5 -> Ka*c=1 (moderate), c_feed=10 -> Ka*c=2 (strong)
LANGMUIR_BINDING = dict(ka=1.0, kd=5.0, qmax=10.0)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LoadingSweepResult:
    c_feed: float
    loading_ratio: float  # c_feed / qmax
    loading_class: str  # "A", "B", "C"
    n_iterations: int
    converged: bool
    wall_time_us: float
    strategy: str
    final_residual: float
    final_contraction: float
    # Mass-balance diagnostics
    delta_F0: float = np.inf
    mu1_numerical: float = np.nan
    mu1_theoretical: float = np.nan
    delta_mu1: float = np.inf
    # CADET comparison (if available)
    l2_error_vs_cadet: float = np.nan
    rmse_vs_cadet: float = np.nan
    linf_vs_cadet: float = np.nan
    # Linear comparison
    l2_error_linear: float = np.nan

    def to_dict(self):
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                d[k] = str(v)
            else:
                d[k] = v
        return d


@dataclass
class BenchmarkSuite:
    name: str
    timestamp: str = ""
    results: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "timestamp": self.timestamp,
            "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.results],
            "summary": self.summary,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CADET config creation (with nonlinear Langmuir binding)
# ---------------------------------------------------------------------------

def create_nonlinear_langmuir_config(
    output_path: Path,
    ka: float, kd: float, qmax: float,
    c_feed: float = 1.0,
    transport: Optional[dict] = None,
    n_times: int = 501,
    end_time: float = 500.0,
    n_col: int = 64,
) -> Path:
    """Create CADET config with MULTI_COMPONENT_LANGMUIR binding."""
    import h5py

    if transport is None:
        transport = GRM_TRANSPORT

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
        par_disc.create_dataset("NCELLS", data=1)
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

        # Return
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


def _write_str(group, name: str, value: str):
    import h5py
    dt = h5py.string_dtype(encoding='ascii')
    group.create_dataset(name, data=value, dtype=dt)


# ---------------------------------------------------------------------------
# CADET runner
# ---------------------------------------------------------------------------

def run_cadet(cadet_cli: Path, config_path: Path) -> tuple:
    """Run CADET and return (t, c_outlet, wall_time_ms)."""
    import subprocess
    import h5py

    t0 = time.perf_counter()
    result = subprocess.run(
        [str(cadet_cli), str(config_path)],
        capture_output=True, text=True, timeout=300,
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    if result.returncode != 0:
        raise RuntimeError(f"CADET failed: {result.stderr[:500]}")

    with h5py.File(config_path, "r") as f:
        t = f["output/solution/SOLUTION_TIMES"][:]
        outlet = f["output/solution/unit_001/SOLUTION_OUTLET"][:]
        if outlet.ndim == 2:
            c_out = outlet[:, 0]
        else:
            c_out = outlet

    return t, c_out, wall_ms


# ---------------------------------------------------------------------------
# Benchmark 1: Loading sweep
# ---------------------------------------------------------------------------

def run_loading_sweep(
    cadet_cli: Optional[Path],
    output_dir: Path,
    skip_cadet: bool = False,
) -> BenchmarkSuite:
    """Run NL-NILT across loading levels, optionally comparing to CADET."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Langmuir Loading Sweep")
    print("=" * 70)

    ka = LANGMUIR_BINDING["ka"]
    kd = LANGMUIR_BINDING["kd"]
    qmax = LANGMUIR_BINDING["qmax"]
    K_eq = ka * qmax / kd
    Ka = ka / kd  # association constant
    c_feeds = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
    t_end = 1500.0

    print(f"  Binding: ka={ka}, kd={kd}, qmax={qmax}, K_eq={K_eq:.1f}")
    print(f"  Theoretical mu1 = {compute_theoretical_first_moment(velocity=GRM_TRANSPORT['velocity'], length=GRM_TRANSPORT['length'], col_porosity=GRM_TRANSPORT['col_porosity'], par_porosity=GRM_TRANSPORT['par_porosity'], K_eq=K_eq):.0f} s")

    suite = BenchmarkSuite(
        name="langmuir_loading_sweep",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    for c_feed in c_feeds:
        loading = c_feed / qmax
        Ka_c = Ka * c_feed  # nonlinear strength: Ka*c << 1 is linear, >> 1 is saturated
        if Ka_c < 0.2:
            cls = "A"
        elif Ka_c < 1.0:
            cls = "B"
        else:
            cls = "C"

        print(f"\n  c_feed={c_feed:5.1f}  c/qmax={loading:.2f}  Ka*c={Ka_c:.2f}  class={cls}")

        # NL-NILT solve
        binding = LangmuirBinding(ka=ka, kd=kd, qmax=qmax, **GRM_TRANSPORT)
        result = nl_nilt_solve(
            binding=binding,
            t_end=t_end,
            c_feed=c_feed,
            step_input=True,
            max_iterations=10,
            eps_conv=1e-4,
        )

        print(f"    NL-NILT: {result.n_iterations} iters, "
              f"converged={result.converged}, "
              f"strategy={result.metadata.get('strategy', '?')}, "
              f"wall={result.wall_time_us/1e3:.1f} ms")

        # Mass-balance diagnostics
        F_lin = binding.linear_transfer_function()
        diag = compute_diagnostics(
            result.t, result.c, F_lin,
            c_feed=c_feed, step_input=True,
            velocity=GRM_TRANSPORT["velocity"],
            length=GRM_TRANSPORT["length"],
            col_porosity=GRM_TRANSPORT["col_porosity"],
            par_porosity=GRM_TRANSPORT["par_porosity"],
            K_eq=binding.effective_keq(),
        )
        print(f"    Mass balance: delta_F0={diag.delta_F0:.2e}, delta_mu1={diag.delta_mu1:.2e}")

        # L2 error: NL-NILT vs linear NILT
        diff_lin = result.c - result.c_lin
        c_lin_norm = np.linalg.norm(result.c_lin)
        l2_vs_linear = np.linalg.norm(diff_lin) / c_lin_norm if c_lin_norm > 0 else 0.0

        entry = LoadingSweepResult(
            c_feed=c_feed,
            loading_ratio=loading,
            loading_class=cls,
            n_iterations=result.n_iterations,
            converged=result.converged,
            wall_time_us=result.wall_time_us,
            strategy=result.metadata.get("strategy", ""),
            final_residual=result.metadata.get("final_residual_norm", 0.0),
            final_contraction=result.metadata.get("final_contraction", 0.0),
            delta_F0=diag.delta_F0,
            mu1_numerical=diag.mu1_numerical,
            mu1_theoretical=diag.mu1_theoretical,
            delta_mu1=diag.delta_mu1,
            l2_error_linear=l2_vs_linear,
        )

        # CADET comparison
        if cadet_cli and not skip_cadet:
            try:
                config_path = output_dir / f"cadet_langmuir_cfeed{c_feed:.1f}.h5"
                create_nonlinear_langmuir_config(
                    config_path,
                    ka=ka, kd=kd, qmax=qmax,
                    c_feed=c_feed,
                    end_time=t_end,
                    n_times=1001,
                    n_col=64,
                )
                t_cadet, c_cadet, cadet_ms = run_cadet(cadet_cli, config_path)

                # Compare on common grid
                t_common, c_nilt_interp, c_cadet_interp = interpolate_to_common_grid(
                    result.t, result.c, t_cadet, c_cadet, n_points=500,
                )
                metrics = compute_comparison_metrics(c_cadet_interp, c_nilt_interp, t_common)

                entry.l2_error_vs_cadet = metrics.relative_l2_error
                entry.rmse_vs_cadet = metrics.rmse
                entry.linf_vs_cadet = metrics.linf_norm

                print(f"    vs CADET: L2={metrics.relative_l2_error:.4f}, "
                      f"RMSE={metrics.rmse:.4e}, "
                      f"CADET wall={cadet_ms:.1f} ms, "
                      f"speedup={cadet_ms / (result.wall_time_us / 1e3):.1f}x")

                # Save CADET breakthrough for figure generation
                np.savez(
                    output_dir / f"cadet_breakthrough_cfeed{c_feed:.1f}.npz",
                    t=t_cadet, c=c_cadet,
                )
            except Exception as e:
                print(f"    CADET comparison failed: {e}")

        # Save NL-NILT breakthrough for figure generation
        np.savez(
            output_dir / f"nlnilt_breakthrough_cfeed{c_feed:.1f}.npz",
            t=result.t, c=result.c, c_lin=result.c_lin,
        )

        suite.results.append(entry)

    # Summary
    converged_count = sum(1 for r in suite.results if r.converged)
    avg_iters = np.mean([r.n_iterations for r in suite.results])
    avg_wall = np.mean([r.wall_time_us for r in suite.results])
    suite.summary = {
        "n_cases": len(suite.results),
        "converged": converged_count,
        "avg_iterations": float(avg_iters),
        "avg_wall_time_us": float(avg_wall),
        "ka": ka, "kd": kd, "qmax": qmax,
    }

    suite.save(output_dir / "loading_sweep_results.json")
    return suite


# ---------------------------------------------------------------------------
# Benchmark 2: Convergence traces (iteration-level diagnostics)
# ---------------------------------------------------------------------------

def run_convergence_traces(output_dir: Path) -> BenchmarkSuite:
    """Run NL-NILT and record per-iteration diagnostics for Class B."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Convergence Traces (Class B Langmuir)")
    print("=" * 70)

    ka = LANGMUIR_BINDING["ka"]
    kd = LANGMUIR_BINDING["kd"]
    qmax = LANGMUIR_BINDING["qmax"]
    c_feed = 5.0  # Class B: Ka*c = 0.2*5 = 1.0 (moderate nonlinearity)
    t_end = 1500.0

    binding = LangmuirBinding(ka=ka, kd=kd, qmax=qmax, **GRM_TRANSPORT)

    # Force many iterations to capture full convergence trace
    result = nl_nilt_solve(
        binding=binding,
        t_end=t_end,
        c_feed=c_feed,
        step_input=True,
        max_iterations=10,
        eps_conv=1e-15,  # Force all 10 iterations
    )

    suite = BenchmarkSuite(
        name="convergence_traces",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    F_lin = binding.linear_transfer_function()
    prev_diag = None

    for ir in result.iterations:
        # Compute mass-balance diagnostics for each iteration
        diag = compute_diagnostics(
            result.t, ir.c, F_lin,
            c_feed=c_feed, step_input=True,
            velocity=GRM_TRANSPORT["velocity"],
            length=GRM_TRANSPORT["length"],
            col_porosity=GRM_TRANSPORT["col_porosity"],
            par_porosity=GRM_TRANSPORT["par_porosity"],
            K_eq=binding.effective_keq(),
        )
        decision = steering_decision(diag, prev_diag)

        entry = {
            "iteration": ir.iteration,
            "residual_norm": float(ir.residual_norm),
            "contraction": float(ir.contraction),
            "c_operating": float(ir.c_operating),
            "strategy": ir.strategy,
            "delta_F0": float(diag.delta_F0),
            "mu1_numerical": float(diag.mu1_numerical),
            "mu1_theoretical": float(diag.mu1_theoretical),
            "delta_mu1": float(diag.delta_mu1),
            "steering_decision": decision,
        }
        print(f"  iter {ir.iteration}: residual={ir.residual_norm:.4e}, "
              f"kappa={ir.contraction:.4f}, "
              f"dF0={diag.delta_F0:.2e}, "
              f"dmu1={diag.delta_mu1:.2e}, "
              f"decision={decision}")

        suite.results.append(entry)
        prev_diag = diag

    # Save iteration breakthrough curves for figures
    for i, ir in enumerate(result.iterations):
        np.savez(
            output_dir / f"convergence_iter{i}.npz",
            t=result.t, c=ir.c,
        )
    np.savez(
        output_dir / "convergence_linear.npz",
        t=result.t, c=result.c_lin,
    )

    suite.summary = {
        "n_iterations": result.n_iterations,
        "converged": result.converged,
        "strategy": result.metadata.get("strategy", ""),
        "c_feed": c_feed, "ka": ka, "kd": kd, "qmax": qmax,
    }

    suite.save(output_dir / "convergence_traces.json")
    return suite


# ---------------------------------------------------------------------------
# Benchmark 3: SMA benchmark
# ---------------------------------------------------------------------------

def run_sma_benchmark(
    cadet_cli: Optional[Path],
    output_dir: Path,
    skip_cadet: bool = False,
) -> BenchmarkSuite:
    """Run NL-NILT with SMA binding, optionally comparing to CADET."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: SMA Binding Benchmark")
    print("=" * 70)

    # Moderate SMA parameters (avoid cmath overflow while producing breakthrough)
    # Effective K_eq ~ ka/kd * Lambda_eff^nu with Lambda_eff = Lambda - z_salt*c_salt
    # Lambda_eff = 10 - 1*0.05 = 9.95, k_a_eff ~ 0.1 * 9.95^2 ~ 9.9
    # K_eq_eff ~ 9.9 * 9.95 / 1.0 ~ 98.5 -> too retentive
    # Use kd=10 for faster desorption
    sma_params = dict(
        ka=0.1, kd=10.0, Lambda=10.0, nu=2.0,
        z_protein=2.0, z_salt=1.0, c_salt=0.05,
    )
    c_feeds = [0.01, 0.1, 1.0]
    t_end = 1500.0

    suite = BenchmarkSuite(
        name="sma_benchmark",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    for c_feed in c_feeds:
        print(f"\n  c_feed={c_feed:.1e}")

        binding = SMABinding(
            c0=c_feed,
            **sma_params,
            **GRM_TRANSPORT,
        )

        result = nl_nilt_solve(
            binding=binding,
            t_end=t_end,
            c_feed=c_feed,
            step_input=True,
            max_iterations=10,
            eps_conv=1e-4,
        )

        print(f"    NL-NILT: {result.n_iterations} iters, "
              f"converged={result.converged}, "
              f"wall={result.wall_time_us/1e3:.1f} ms")

        entry = {
            "c_feed": c_feed,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "wall_time_us": result.wall_time_us,
            "strategy": result.metadata.get("strategy", ""),
            "final_residual": result.metadata.get("final_residual_norm", 0.0),
        }

        np.savez(
            output_dir / f"sma_breakthrough_cfeed{c_feed:.1e}.npz",
            t=result.t, c=result.c, c_lin=result.c_lin,
        )

        suite.results.append(entry)

    suite.summary = {
        "n_cases": len(suite.results),
        "converged": sum(1 for r in suite.results if r["converged"]),
        **sma_params,
    }

    suite.save(output_dir / "sma_benchmark_results.json")
    return suite


# ---------------------------------------------------------------------------
# Benchmark 4: Larger column (L=0.25m)
# ---------------------------------------------------------------------------

def run_large_column_benchmark(
    cadet_cli: Optional[Path],
    output_dir: Path,
    skip_cadet: bool = False,
) -> BenchmarkSuite:
    """Run NL-NILT on a larger column where CADET is slower."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Large Column (L=0.25m)")
    print("=" * 70)

    ka = LANGMUIR_BINDING["ka"]
    kd = LANGMUIR_BINDING["kd"]
    qmax = LANGMUIR_BINDING["qmax"]
    c_feeds = [1.0, 5.0]
    # mu1 for large column: tau=0.25/5e-4=500s, K_eq=2 -> mu1~1922s
    t_end = 5000.0

    suite = BenchmarkSuite(
        name="large_column_benchmark",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    for c_feed in c_feeds:
        loading = c_feed / qmax
        print(f"\n  c_feed={c_feed:.1f}  loading={loading:.2f}")

        binding = LangmuirBinding(ka=ka, kd=kd, qmax=qmax, **GRM_LARGE_COLUMN)

        result = nl_nilt_solve(
            binding=binding,
            t_end=t_end,
            c_feed=c_feed,
            step_input=True,
            max_iterations=10,
            eps_conv=1e-4,
        )

        print(f"    NL-NILT: {result.n_iterations} iters, "
              f"converged={result.converged}, "
              f"wall={result.wall_time_us/1e3:.1f} ms")

        entry = {
            "c_feed": c_feed,
            "loading_ratio": loading,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "wall_time_us": result.wall_time_us,
            "strategy": result.metadata.get("strategy", ""),
        }

        # CADET comparison with more cells
        if cadet_cli and not skip_cadet:
            try:
                config_path = output_dir / f"cadet_large_cfeed{c_feed:.1f}.h5"
                create_nonlinear_langmuir_config(
                    config_path,
                    ka=ka, kd=kd, qmax=qmax,
                    c_feed=c_feed,
                    transport=GRM_LARGE_COLUMN,
                    end_time=t_end,
                    n_times=1001,
                    n_col=128,
                )
                t_cadet, c_cadet, cadet_ms = run_cadet(cadet_cli, config_path)

                t_common, c_nilt_interp, c_cadet_interp = interpolate_to_common_grid(
                    result.t, result.c, t_cadet, c_cadet, n_points=500,
                )
                metrics = compute_comparison_metrics(c_cadet_interp, c_nilt_interp, t_common)

                entry["l2_error_vs_cadet"] = float(metrics.relative_l2_error)
                entry["cadet_wall_ms"] = float(cadet_ms)
                entry["speedup"] = float(cadet_ms / (result.wall_time_us / 1e3))

                print(f"    vs CADET: L2={metrics.relative_l2_error:.4f}, "
                      f"CADET={cadet_ms:.0f} ms, "
                      f"speedup={entry['speedup']:.1f}x")

                np.savez(
                    output_dir / f"cadet_large_breakthrough_cfeed{c_feed:.1f}.npz",
                    t=t_cadet, c=c_cadet,
                )
            except Exception as e:
                print(f"    CADET failed: {e}")

        np.savez(
            output_dir / f"nlnilt_large_breakthrough_cfeed{c_feed:.1f}.npz",
            t=result.t, c=result.c, c_lin=result.c_lin,
        )

        suite.results.append(entry)

    suite.summary = {
        "transport": GRM_LARGE_COLUMN,
        "ka": ka, "kd": kd, "qmax": qmax,
    }

    suite.save(output_dir / "large_column_results.json")
    return suite


# ---------------------------------------------------------------------------
# Benchmark 5: Parameter estimation mini-case
# ---------------------------------------------------------------------------

def run_param_estimation(
    cadet_cli: Path,
    output_dir: Path,
) -> BenchmarkSuite:
    """Fit ka, kd from synthetic CADET breakthrough using NL-NILT as forward model."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Parameter Estimation Mini-Case")
    print("=" * 70)

    from scipy.optimize import minimize, differential_evolution

    # Ground truth (K_eq = 1.5*10/7.5 = 2.0, same as primary benchmark)
    ka_true, kd_true, qmax = 1.5, 7.5, 10.0
    c_feed = 5.0
    t_end = 1500.0
    bounds = [(0.1, 10.0), (0.5, 50.0)]  # (ka, kd) bounds

    # Generate synthetic CADET data
    config_path = output_dir / "cadet_param_est_truth.h5"
    create_nonlinear_langmuir_config(
        config_path,
        ka=ka_true, kd=kd_true, qmax=qmax,
        c_feed=c_feed,
        end_time=t_end,
        n_times=201,
        n_col=64,
    )
    t_data, c_data, _ = run_cadet(cadet_cli, config_path)

    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01 * c_feed, size=c_data.shape)
    c_noisy = np.maximum(c_data + noise, 0.0)

    print(f"  Ground truth: ka={ka_true}, kd={kd_true}")
    print(f"  Data points: {len(c_data)}")

    # Objective function using NL-NILT
    n_nilt_evals = [0]
    nilt_wall_total = [0.0]

    def objective_nilt(params):
        ka_test, kd_test = params
        if ka_test <= 0 or kd_test <= 0:
            return 1e10

        binding = LangmuirBinding(
            ka=ka_test, kd=kd_test, qmax=qmax, **GRM_TRANSPORT,
        )
        t0 = time.perf_counter()
        result = nl_nilt_solve(
            binding=binding, t_end=t_end, c_feed=c_feed,
            step_input=True, max_iterations=5, eps_conv=1e-3,
        )
        nilt_wall_total[0] += (time.perf_counter() - t0) * 1000
        n_nilt_evals[0] += 1

        c_pred = np.interp(t_data, result.t, result.c)
        return float(np.sum((c_pred - c_noisy) ** 2))

    # Fit using NL-NILT (bounded optimization)
    print("\n  Fitting with NL-NILT forward model...")
    t0_fit = time.perf_counter()
    res_nilt = differential_evolution(objective_nilt, bounds,
                                      seed=42, maxiter=100, tol=1e-4,
                                      popsize=10, polish=True)
    fit_wall_nilt = (time.perf_counter() - t0_fit) * 1000

    print(f"    Result: ka={res_nilt.x[0]:.4f}, kd={res_nilt.x[1]:.4f}")
    print(f"    Evals: {n_nilt_evals[0]}, total wall: {fit_wall_nilt:.0f} ms")
    print(f"    Error: ka={abs(res_nilt.x[0]-ka_true)/ka_true:.2%}, "
          f"kd={abs(res_nilt.x[1]-kd_true)/kd_true:.2%}")

    # Objective using CADET
    n_cadet_evals = [0]
    cadet_wall_total = [0.0]

    def objective_cadet(params):
        ka_test, kd_test = params
        if ka_test <= 0 or kd_test <= 0:
            return 1e10

        config = output_dir / f"cadet_param_est_eval{n_cadet_evals[0]}.h5"
        create_nonlinear_langmuir_config(
            config, ka=ka_test, kd=kd_test, qmax=qmax,
            c_feed=c_feed, end_time=t_end, n_times=201, n_col=64,
        )
        try:
            t0 = time.perf_counter()
            t_pred, c_pred_raw, _ = run_cadet(cadet_cli, config)
            cadet_wall_total[0] += (time.perf_counter() - t0) * 1000
            n_cadet_evals[0] += 1
            c_pred = np.interp(t_data, t_pred, c_pred_raw)
            return float(np.sum((c_pred - c_noisy) ** 2))
        except Exception:
            n_cadet_evals[0] += 1
            return 1e10

    # Fit using CADET (same bounded optimization)
    print("\n  Fitting with CADET forward model...")
    n_cadet_evals[0] = 0
    cadet_wall_total[0] = 0.0
    t0_fit = time.perf_counter()
    res_cadet = differential_evolution(objective_cadet, bounds,
                                       seed=42, maxiter=100, tol=1e-4,
                                       popsize=10, polish=True)
    fit_wall_cadet = (time.perf_counter() - t0_fit) * 1000

    print(f"    Result: ka={res_cadet.x[0]:.4f}, kd={res_cadet.x[1]:.4f}")
    print(f"    Evals: {n_cadet_evals[0]}, total wall: {fit_wall_cadet:.0f} ms")
    print(f"    Error: ka={abs(res_cadet.x[0]-ka_true)/ka_true:.2%}, "
          f"kd={abs(res_cadet.x[1]-kd_true)/kd_true:.2%}")

    speedup = fit_wall_cadet / fit_wall_nilt if fit_wall_nilt > 0 else 0
    print(f"\n  Wall-time speedup (NILT/CADET): {speedup:.1f}x")

    suite = BenchmarkSuite(
        name="param_estimation",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    suite.results = [{
        "nilt_ka": float(res_nilt.x[0]),
        "nilt_kd": float(res_nilt.x[1]),
        "nilt_evals": n_nilt_evals[0],
        "nilt_wall_ms": float(fit_wall_nilt),
        "cadet_ka": float(res_cadet.x[0]),
        "cadet_kd": float(res_cadet.x[1]),
        "cadet_evals": n_cadet_evals[0],
        "cadet_wall_ms": float(fit_wall_cadet),
        "speedup": float(speedup),
        "true_ka": ka_true,
        "true_kd": kd_true,
    }]
    suite.summary = {
        "speedup": float(speedup),
        "nilt_converged": bool(res_nilt.success),
        "cadet_converged": bool(res_cadet.success),
    }

    suite.save(output_dir / "param_estimation_results.json")
    return suite


# ---------------------------------------------------------------------------
# Benchmark 6: Pole shift analysis (analytical, no simulation needed)
# ---------------------------------------------------------------------------

def run_pole_shift_analysis(output_dir: Path) -> BenchmarkSuite:
    """Compute pole shift data for Figure 1 (analytical calculation)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: Pole Shift Analysis")
    print("=" * 70)

    ka = LANGMUIR_BINDING["ka"]
    kd = LANGMUIR_BINDING["kd"]
    qmax = LANGMUIR_BINDING["qmax"]
    c_operating = np.linspace(0, 20.0, 200)

    suite = BenchmarkSuite(
        name="pole_shift_analysis",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # Linear pole: s = -kd
    pole_linear = -kd

    # Re-linearized pole: s = -(kd + ka * c_0)
    poles_relin = -(kd + ka * c_operating)

    # Effective K_eq at each operating point
    K_eq_vals = []
    for c0 in c_operating:
        Ka = ka / kd
        q0 = qmax * Ka * c0 / (1.0 + Ka * c0)
        qmax_eff = max(qmax - q0, 1e-30)
        kd_eff = kd + ka * c0
        K_eq_eff = ka * qmax_eff / kd_eff
        K_eq_vals.append(K_eq_eff)

    print(f"  Linear pole: s = {pole_linear}")
    print(f"  Re-linearized pole range: [{poles_relin[-1]:.2f}, {poles_relin[0]:.2f}]")
    print(f"  K_eq effective range: [{K_eq_vals[-1]:.4f}, {K_eq_vals[0]:.2f}]")

    np.savez(
        output_dir / "pole_shift_data.npz",
        c_operating=c_operating,
        pole_linear=np.full_like(c_operating, pole_linear),
        poles_relin=poles_relin,
        K_eq_eff=np.array(K_eq_vals),
        ka=ka, kd=kd, qmax=qmax,
    )

    suite.summary = {
        "pole_linear": pole_linear,
        "pole_relin_at_cfeed_1": float(-(kd + ka * 1.0)),
        "pole_relin_at_cfeed_5": float(-(kd + ka * 5.0)),
        "pole_relin_at_cfeed_10": float(-(kd + ka * 10.0)),
    }

    suite.save(output_dir / "pole_shift_analysis.json")
    return suite


# ---------------------------------------------------------------------------
# CADET config for LUMPED_RATE_MODEL_WITHOUT_PORES (plug-flow kinetic model)
# ---------------------------------------------------------------------------

def create_lrmwp_config(
    output_path: Path,
    ka: float, kd: float, qmax: float,
    c_feed: float,
    col_porosity: float = 0.5,
    velocity: float = 1e-3,
    length: float = 0.1,
    dispersion: float = 1e-10,
    end_time: float = 500.0,
    n_times: int = 501,
    n_col: int = 256,
) -> Path:
    """Create CADET config with LUMPED_RATE_MODEL_WITHOUT_PORES.

    This model has no pore diffusion — binding occurs directly from the
    mobile phase. With D_ax ≈ 0, it reduces to plug-flow with kinetics,
    matching the Chen & Hsu (1989) formulation.

    Mass balance: ∂c/∂t + u·∂c/∂z + ((1-ε_t)/ε_t)·∂q/∂t = D_ax·∂²c/∂z²
    Langmuir:     ∂q/∂t = ka·c·(qmax-q) - kd·q
    """
    import h5py

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solution_times = np.linspace(0, end_time, n_times)
    flow_rate = velocity * col_porosity * 1e-4  # Q = u * A * eps_t

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

        # Unit 001: LRMWP
        u1 = model.create_group("unit_001")
        _write_str(u1, "UNIT_TYPE", "LUMPED_RATE_MODEL_WITHOUT_PORES")
        u1.create_dataset("NCOMP", data=1)
        u1.create_dataset("COL_LENGTH", data=length)
        u1.create_dataset("TOTAL_POROSITY", data=col_porosity)
        u1.create_dataset("COL_DISPERSION", data=dispersion)
        u1.create_dataset("VELOCITY", data=velocity)
        u1.create_dataset("CROSS_SECTION_AREA", data=1e-4)
        u1.create_dataset("INIT_C", data=[0.0])

        # Binding in particle_type_000 (CADET v6 format for LRMWP)
        pt = u1.create_group("particle_type_000")
        pt.create_dataset("HAS_FILM_DIFFUSION", data=0)
        pt.create_dataset("HAS_PORE_DIFFUSION", data=0)
        pt.create_dataset("HAS_SURFACE_DIFFUSION", data=0)
        pt.create_dataset("NBOUND", data=[1])
        pt.create_dataset("INIT_CS", data=[0.0])
        _write_str(pt, "ADSORPTION_MODEL", "MULTI_COMPONENT_LANGMUIR")
        ads = pt.create_group("adsorption")
        ads.create_dataset("IS_KINETIC", data=1)
        ads.create_dataset("MCL_KA", data=[ka])
        ads.create_dataset("MCL_KD", data=[kd])
        ads.create_dataset("MCL_QMAX", data=[qmax])

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
        ti.create_dataset("ABSTOL", data=1e-10)
        ti.create_dataset("ALGTOL", data=1e-12)
        ti.create_dataset("RELTOL", data=1e-10)
        ti.create_dataset("INIT_STEP_SIZE", data=1e-6)
        ti.create_dataset("MAX_STEPS", data=500000)

        # Return
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
# Benchmark 7: Analytical 4-way comparison (Chen & Hsu plug-flow problem)
# ---------------------------------------------------------------------------

def run_analytical_comparison(
    cadet_cli: Optional[Path],
    output_dir: Path,
    skip_cadet: bool = False,
) -> BenchmarkSuite:
    """4-way comparison: Thomas analytical vs Linear FFT vs Iterative FFT vs CADET.

    Uses the Chen & Hsu (1989) plug-flow kinetic problem where the Thomas
    analytical solution provides exact reference. Validates the iterative
    FFT concept (same as NL-NILT) on a problem with known solution.

    Chen & Hsu CGS → CADET SI mapping (ε_t = 0.5):
      C0    = 1e-6 mol/cm³  →  c_feed = 1.0 mol/m³
      k1    = 1e4 cm³/(mol·s) →  ka = 0.01 m³/(mol·s)
      k2    = 0.1 s⁻¹         →  kd = 0.1 s⁻¹
      V     = 0.1 cm/s        →  velocity = 1e-3 m/s (interstitial)
      L     = 10 cm           →  length = 0.1 m
      alpha = 1e-5 mol/cm³    →  qmax = 10 mol/m³ (with ε_t=0.5)
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 7: Analytical 4-Way Comparison (Chen & Hsu 1989)")
    print("=" * 70)

    # Import Chen & Hsu functions
    sys.path.insert(0, str(Path(__file__).parent))
    from chen_hsu_reproduction import (
        analytical_solution, chen_hsu_iterative_fft,
        TABLE1_TIMES, TABLE1_ANALYTICAL, TABLE1_LINEAR, TABLE1_ITER2,
        T_OFFSET, C0, K1, K2, V, L, ALPHA,
    )

    # CADET SI parameters (mapped from Chen & Hsu CGS)
    eps_t = 0.5
    ka_si = K1 * 1e-6       # 0.01 m³/(mol·s)
    kd_si = K2               # 0.1 s⁻¹
    qmax_si = ALPHA * 1e6 * eps_t / (1 - eps_t)  # 10 mol/m³
    c_feed_si = C0 * 1e6     # 1.0 mol/m³
    velocity_si = V * 0.01   # 1e-3 m/s
    length_si = L * 0.01     # 0.1 m
    K_eq = ka_si * qmax_si / kd_si

    print(f"  Chen & Hsu: C0={C0:.1e}, k1={K1:.1e}, k2={K2}, V={V}, L={L}, alpha={ALPHA:.1e}")
    print(f"  CADET SI:   c_feed={c_feed_si}, ka={ka_si}, kd={kd_si}, qmax={qmax_si}, K_eq={K_eq:.1f}")
    print(f"              velocity={velocity_si}, length={length_si}, eps_t={eps_t}")
    print(f"  Retention:  tau={length_si/velocity_si:.0f}s, "
          f"tau_r={length_si/velocity_si * (1 + (1-eps_t)/eps_t * K_eq):.0f}s")

    suite = BenchmarkSuite(
        name="analytical_comparison",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # 1. Thomas analytical solution on fine grid
    print("\n  1. Computing Thomas analytical solution...")
    t_fine = np.linspace(110, 400, 1000)
    t_n_fine = t_fine - T_OFFSET
    C_analytical_fine = analytical_solution(t_n_fine)

    # Also at Table 1 times
    t_n_table = TABLE1_TIMES - T_OFFSET
    C_analytical_table = analytical_solution(t_n_table)
    print(f"     Max deviation from paper: {np.max(np.abs(C_analytical_table - TABLE1_ANALYTICAL)):.4f}")

    # 2. Chen & Hsu iterative FFT (linear + 2 iterations)
    print("\n  2. Running Chen & Hsu iterative FFT (2 iterations)...")
    fft_results = chen_hsu_iterative_fft(n_iterations=3)
    t_fft = fft_results["t"]
    C_linear_fft = fft_results["linear"]
    C_iter1_fft = fft_results["iterations"][0]
    C_iter2_fft = fft_results["iterations"][1]
    C_iter3_fft = fft_results["iterations"][2]

    # Interpolate to Table 1 times for error computation
    C_linear_table = np.interp(TABLE1_TIMES, t_fft, C_linear_fft)
    C_iter1_table = np.interp(TABLE1_TIMES, t_fft, C_iter1_fft)
    C_iter2_table = np.interp(TABLE1_TIMES, t_fft, C_iter2_fft)
    C_iter3_table = np.interp(TABLE1_TIMES, t_fft, C_iter3_fft)

    # Compute errors vs analytical at Table 1 times
    err_linear = np.sqrt(np.mean((C_linear_table - TABLE1_ANALYTICAL) ** 2))
    err_iter1 = np.sqrt(np.mean((C_iter1_table - TABLE1_ANALYTICAL) ** 2))
    err_iter2 = np.sqrt(np.mean((C_iter2_table - TABLE1_ANALYTICAL) ** 2))
    err_iter3 = np.sqrt(np.mean((C_iter3_table - TABLE1_ANALYTICAL) ** 2))

    print(f"     RMS vs analytical: linear={err_linear:.4f}, "
          f"iter1={err_iter1:.4f}, iter2={err_iter2:.4f}, iter3={err_iter3:.4f}")

    # 3. CADET LRMWP (optional)
    t_cadet = None
    C_cadet = None
    cadet_wall_ms = None

    if cadet_cli and not skip_cadet:
        print("\n  3. Running CADET LRMWP...")
        try:
            config_path = output_dir / "cadet_lrmwp_chen_hsu.h5"
            create_lrmwp_config(
                config_path,
                ka=ka_si, kd=kd_si, qmax=qmax_si,
                c_feed=c_feed_si,
                col_porosity=eps_t,
                velocity=velocity_si,
                length=length_si,
                dispersion=1e-10,  # Near-zero dispersion
                end_time=500.0,
                n_times=2001,
                n_col=512,  # Fine grid for accuracy
            )
            t_cadet, c_cadet_raw, cadet_wall_ms = run_cadet(cadet_cli, config_path)
            # Normalize to C/C0
            C_cadet = c_cadet_raw / c_feed_si

            # CADET uses sharp step (no 5s delay), so offset = L/V = 100s
            # The analytical at CADET times:
            t_n_cadet = t_cadet - (length_si / velocity_si)  # offset = 100s
            C_analytical_cadet = analytical_solution(t_n_cadet)

            # Error vs analytical (at times where breakthrough happens)
            mask = (t_cadet >= 110) & (t_cadet <= 400)
            err_cadet = np.sqrt(np.mean((C_cadet[mask] - C_analytical_cadet[mask]) ** 2))
            print(f"     CADET wall time: {cadet_wall_ms:.1f} ms")
            print(f"     CADET RMS vs analytical (t=110-400): {err_cadet:.4f}")

            np.savez(
                output_dir / "analytical_comparison_cadet.npz",
                t=t_cadet, C_norm=C_cadet, C_analytical=C_analytical_cadet,
            )
        except Exception as e:
            print(f"     CADET LRMWP failed: {e}")
            t_cadet = None
    else:
        print("\n  3. CADET comparison skipped")

    # 4. Print comparison table
    print(f"\n  {'':=<70}")
    print(f"  {'4-Way Comparison at Table 1 Times':^70}")
    print(f"  {'':=<70}")
    print(f"  {'Time':>6} | {'Analytical':>10} {'Linear':>10} {'Iter1':>10} "
          f"{'Iter2':>10} {'Iter3':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}")

    for i, t_val in enumerate(TABLE1_TIMES):
        print(f"  {t_val:6.0f} | {C_analytical_table[i]:10.4f} "
              f"{C_linear_table[i]:10.4f} {C_iter1_table[i]:10.4f} "
              f"{C_iter2_table[i]:10.4f} {C_iter3_table[i]:10.4f}")

    print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}")
    print(f"  {'RMS':>6} | {'ref':>10} {err_linear:10.4f} {err_iter1:10.4f} "
          f"{err_iter2:10.4f} {err_iter3:10.4f}")

    # Improvement ratios
    print(f"\n  Error reduction vs linear:")
    print(f"    Iter 1: {err_linear/err_iter1:.1f}x")
    print(f"    Iter 2: {err_linear/err_iter2:.1f}x")
    print(f"    Iter 3: {err_linear/err_iter3:.1f}x")

    # 5. Save all results
    np.savez(
        output_dir / "analytical_comparison_data.npz",
        # Fine grid
        t_fine=t_fine,
        C_analytical_fine=C_analytical_fine,
        T_OFFSET=T_OFFSET,
        # FFT data (full time grid)
        t_fft=t_fft,
        C_linear_fft=C_linear_fft,
        C_iter1_fft=C_iter1_fft,
        C_iter2_fft=C_iter2_fft,
        C_iter3_fft=C_iter3_fft,
        # Table 1 comparison
        table_times=TABLE1_TIMES,
        C_analytical_table=C_analytical_table,
        C_linear_table=C_linear_table,
        C_iter1_table=C_iter1_table,
        C_iter2_table=C_iter2_table,
        C_iter3_table=C_iter3_table,
        paper_analytical=TABLE1_ANALYTICAL,
    )

    suite.results = [{
        "rms_linear_vs_analytical": float(err_linear),
        "rms_iter1_vs_analytical": float(err_iter1),
        "rms_iter2_vs_analytical": float(err_iter2),
        "rms_iter3_vs_analytical": float(err_iter3),
        "improvement_iter1": float(err_linear / err_iter1) if err_iter1 > 0 else float("inf"),
        "improvement_iter2": float(err_linear / err_iter2) if err_iter2 > 0 else float("inf"),
        "has_cadet": t_cadet is not None,
    }]
    suite.summary = {
        "chen_hsu_params": {"C0": C0, "k1": K1, "k2": K2, "V": V, "L": L, "alpha": ALPHA},
        "cadet_si_params": {
            "ka": ka_si, "kd": kd_si, "qmax": qmax_si,
            "c_feed": c_feed_si, "velocity": velocity_si,
            "length": length_si, "eps_t": eps_t,
        },
        "K_eq": K_eq,
        "Ka_C0": ka_si / kd_si * c_feed_si,
    }

    suite.save(output_dir / "analytical_comparison_results.json")
    return suite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "loading_sweep": "Loading sweep across c_feed values",
    "convergence_traces": "Per-iteration convergence diagnostics",
    "sma_benchmark": "SMA binding benchmark",
    "large_column": "Large column (L=0.25m) benchmark",
    "param_estimation": "Parameter estimation mini-case (requires CADET)",
    "pole_shift": "Pole shift analysis (analytical)",
    "analytical_comparison": "4-way comparison vs Thomas analytical (Chen & Hsu 1989)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run NL-NILT benchmarks for the CACE paper",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--cadet-cli",
        type=Path,
        default=None,
        help="Path to cadet-cli executable (required for CADET comparisons)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts/nl_nilt_benchmarks"),
        help="Base output directory",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default=["all"],
        help="Which benchmarks to run:\n" + "\n".join(
            f"  {k}: {v}" for k, v in BENCHMARKS.items()
        ),
    )
    parser.add_argument(
        "--skip-cadet",
        action="store_true",
        help="Skip CADET comparisons (run NL-NILT only)",
    )
    args = parser.parse_args()

    if "all" in args.benchmark:
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = args.benchmark

    # Check CADET availability
    cadet_cli = args.cadet_cli
    if cadet_cli and not cadet_cli.exists():
        print(f"Warning: CADET CLI not found at {cadet_cli}")
        cadet_cli = None

    needs_cadet = {"param_estimation"}
    for bm in benchmarks:
        if bm in needs_cadet and not cadet_cli and not args.skip_cadet:
            print(f"Warning: {bm} requires --cadet-cli, will be skipped")

    print("=" * 70)
    print("NL-NILT Paper Benchmarks")
    print("=" * 70)
    print(f"Output: {args.output_dir}")
    print(f"CADET:  {cadet_cli or 'not available'}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if "pole_shift" in benchmarks:
        results["pole_shift"] = run_pole_shift_analysis(args.output_dir)

    if "loading_sweep" in benchmarks:
        results["loading_sweep"] = run_loading_sweep(
            cadet_cli, args.output_dir, args.skip_cadet,
        )

    if "convergence_traces" in benchmarks:
        results["convergence_traces"] = run_convergence_traces(args.output_dir)

    if "sma_benchmark" in benchmarks:
        results["sma_benchmark"] = run_sma_benchmark(
            cadet_cli, args.output_dir, args.skip_cadet,
        )

    if "large_column" in benchmarks:
        results["large_column"] = run_large_column_benchmark(
            cadet_cli, args.output_dir, args.skip_cadet,
        )

    if "param_estimation" in benchmarks and cadet_cli:
        results["param_estimation"] = run_param_estimation(
            cadet_cli, args.output_dir,
        )

    if "analytical_comparison" in benchmarks:
        results["analytical_comparison"] = run_analytical_comparison(
            cadet_cli, args.output_dir, args.skip_cadet,
        )

    # Final summary
    print("\n\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    for name, suite in results.items():
        status = "OK" if suite.summary else "?"
        print(f"  {name}: {status}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
