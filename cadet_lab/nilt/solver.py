"""Programmatic NILT solver API.

Provides NiltSolver for parameter estimation loops (tight inner loop)
and NLNiltSolver for nonlinear binding problems.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from .vendor.tuner import TunedParams, tune_params, refine_until_accept
from .convergence import NiltConvergenceResult
from .classify import ProblemClassification, classify_problem
from .extract_params import extract_nilt_params
from . import benchmarks
from .nonlinear import (
    nl_nilt_solve,
    LangmuirBinding,
    SMABinding,
    NLNiltResult,
)


@dataclass
class NiltSolution:
    """Result of an NILT solve.

    Attributes:
        t: Time grid (evaluation interval).
        y: Solution values (outlet concentration).
        params: CFL-tuned parameters used.
        convergence: Convergence diagnostics.
        classification: Problem classification (None if solved with direct F).
        wall_time_us: Solve time in microseconds.
        metadata: Extra diagnostics (N, eps_im, E_N, etc.).
    """

    t: np.ndarray
    y: np.ndarray
    params: TunedParams
    convergence: NiltConvergenceResult
    classification: Optional[ProblemClassification] = None
    wall_time_us: float = 0.0
    metadata: dict = field(default_factory=dict)


class NiltSolver:
    """NILT solver for chromatography transfer functions.

    The primary use case is parameter estimation loops where ``solve()``
    is called repeatedly with different transfer functions constructed
    from varying physical parameters.

    Args:
        t_end: End time for evaluation.
        t_eval_min: Minimum time for evaluation (avoids t=0 singularity).
        eps_conv: N-doubling convergence threshold.
        N_max: Maximum FFT size before giving up.
    """

    def __init__(
        self,
        t_end: float,
        t_eval_min: float = 0.1,
        eps_conv: float = 1e-6,
        N_max: int = 32768,
    ):
        self.t_end = t_end
        self.t_eval_min = t_eval_min
        self.eps_conv = eps_conv
        self.N_max = N_max

    def solve(
        self,
        F: Callable[[complex], complex],
        alpha_c: float = 0.0,
        C: float = 1.0,
        eps_im_threshold: float = 1e-2,
        n_timing_runs: int = 5,
        step_input: bool = False,
        **tune_kwargs,
    ) -> NiltSolution:
        """Solve given a transfer function directly.

        This is the tight inner loop method for parameter estimation.
        No file I/O, no classification — just solve.

        Args:
            F: Laplace-domain transfer function F(s) -> complex.
            alpha_c: Abscissa of convergence.
            C: Tail envelope constant.
            eps_im_threshold: Epsilon-Im threshold for acceptance.
            n_timing_runs: Number of timing runs for wall time estimate.
            step_input: If True, compute step response F(s)/s instead of
                impulse response F(s). Use for comparison with CADET
                step-input simulations.
            **tune_kwargs: Additional kwargs for tune_params().

        Returns:
            NiltSolution with time-domain solution and diagnostics.
        """
        t_start = time.perf_counter()

        # Phase 1: Tune parameters
        params = tune_params(
            t_end=self.t_end,
            alpha_c=alpha_c,
            C=C,
            **tune_kwargs,
        )

        if not params.feasible:
            # Return empty solution with failure info
            return NiltSolution(
                t=np.array([]),
                y=np.array([]),
                params=params,
                convergence=NiltConvergenceResult(
                    passed=False,
                    message=f"CFL infeasible: margin={params.margin:.2e}",
                ),
                wall_time_us=(time.perf_counter() - t_start) * 1e6,
                metadata={"feasible": False},
            )

        # Wrap transfer function for step input if requested
        F_solve = F
        if step_input:
            F_solve = lambda s, _F=F: _F(s) / s

        # Phase 2: Refine until acceptance
        result = refine_until_accept(
            F_solve,
            params,
            t_end=self.t_end,
            eps_im_threshold=eps_im_threshold,
            eps_conv=self.eps_conv,
            N_max=self.N_max,
            t_eval_min=self.t_eval_min,
            n_timing_runs=n_timing_runs,
        )

        wall_time_us = (time.perf_counter() - t_start) * 1e6

        # Build convergence result
        convergence = NiltConvergenceResult(
            passed=bool(result["accepted"]),
            epsilon_im=float(result["eps_im"]),
            delta_sequence=[float(d) for d in result["delta_history"]],
            final_delta=float(result["E_N"]),
            n_sequence=None,
            threshold=self.eps_conv,
            message=(
                f"N={result['N']}, eps_im={result['eps_im']:.2e}, "
                f"E_N={result['E_N']:.2e}"
            ),
        )

        # Update params with final N
        final_params = TunedParams(
            a=result["a"],
            T=result["T"],
            N=result["N"],
            delta_t=2 * result["T"] / result["N"],
            t_max=2 * result["T"],
            a_min_star=params.a_min_star,
            a_min=params.a_min,
            a_max=params.a_max,
            margin=params.margin,
            feasible=params.feasible,
        )

        return NiltSolution(
            t=result["t_eval"],
            y=result["f_eval"],
            params=final_params,
            convergence=convergence,
            wall_time_us=wall_time_us,
            metadata={
                "N": result["N"],
                "eps_im": float(result["eps_im"]),
                "eps_im_full": float(result["eps_im_full"]),
                "E_N": float(result["E_N"]),
                "iterations": result["iterations"],
                "timing_median_us": float(result["timing_median_us"]),
                "timing_mad_us": float(result["timing_mad_us"]),
                "accepted": bool(result["accepted"]),
            },
        )

    def solve_from_params(
        self, params: dict, step_input: bool = True,
    ) -> NiltSolution:
        """Solve given physical parameters (from extract_nilt_params).

        Classifies the problem, selects the appropriate transfer function,
        constructs it from the physical parameters, and solves.

        Args:
            params: Physical parameter dictionary from extract_nilt_params().
            step_input: If True (default), compute step response to match
                CADET step-input simulations. Set False for impulse response.

        Returns:
            NiltSolution with classification info attached.

        Raises:
            ValueError: If the problem type is unsupported.
        """
        classification = classify_problem(params)

        if classification.problem_type == "unsupported":
            raise ValueError(
                f"Unsupported problem: {classification.warnings}"
            )

        # Use end_time from params if solver wasn't configured with one
        if params.get("end_time"):
            self.t_end = params["end_time"]

        # Construct transfer function from physical parameters
        F = _build_transfer_function(params, classification)

        solution = self.solve(F, step_input=step_input)
        solution.classification = classification
        return solution

    def solve_from_h5(
        self, h5_path: Union[str, Path], step_input: bool = True,
    ) -> NiltSolution:
        """Solve given a CADET HDF5 config file.

        End-to-end convenience: extract params, classify, build transfer
        function, solve.

        Args:
            h5_path: Path to CADET HDF5 configuration file.
            step_input: If True (default), compute step response to match
                CADET step-input simulations.

        Returns:
            NiltSolution with full diagnostics.
        """
        params = extract_nilt_params(h5_path)
        return self.solve_from_params(params, step_input=step_input)


def _build_transfer_function(
    params: dict,
    classification: ProblemClassification,
) -> Callable[[complex], complex]:
    """Construct the appropriate transfer function from physical parameters."""
    name = classification.transfer_function_name

    if name == "advection_dispersion_transfer":
        return benchmarks.advection_dispersion_transfer(
            velocity=params["velocity"],
            dispersion=params["dispersion"],
            length=params["length"],
        )

    if name == "langmuir_column_transfer":
        bp = params.get("binding_params", {})
        return benchmarks.langmuir_column_transfer(
            velocity=params["velocity"],
            dispersion=params["dispersion"],
            length=params["length"],
            ka=bp.get("ka", 1.0),
            kd=bp.get("kd", 0.1),
            qmax=bp.get("qmax", 10.0),
            porosity=params.get("col_porosity", 0.4),
        )

    if name == "grm_langmuir_transfer":
        bp = params.get("binding_params", {})
        ptype = classification.problem_type

        if ptype == "grm_no_binding":
            # GRM with particle dynamics but no binding:
            # ka=0, kd=1 (nonzero to avoid /0), qmax=0
            ka, kd, qmax = 0.0, 1.0, 0.0
        else:
            # For LINEAR binding, derive qmax from K_eq
            ka = bp.get("ka", 1.0)
            kd = bp.get("kd", 0.1)
            qmax = bp.get("qmax", ka / kd if kd > 0 else 1.0)
        return benchmarks.grm_langmuir_transfer(
            velocity=params["velocity"],
            dispersion=params["dispersion"],
            length=params["length"],
            col_porosity=params.get("col_porosity", 0.37),
            par_radius=params.get("par_radius", 1e-5),
            par_porosity=params.get("par_porosity", 0.33),
            film_diffusion=params.get("film_diffusion", 1e-5),
            pore_diffusion=params.get("pore_diffusion", 1e-10),
            ka=ka,
            kd=kd,
            qmax=qmax,
        )

    if name == "grm_sma_transfer":
        bp = params.get("binding_params", {})
        return benchmarks.grm_sma_transfer(
            velocity=params["velocity"],
            dispersion=params["dispersion"],
            length=params["length"],
            col_porosity=params.get("col_porosity", 0.37),
            par_radius=params.get("par_radius", 1e-5),
            par_porosity=params.get("par_porosity", 0.33),
            film_diffusion=params.get("film_diffusion", 1e-5),
            pore_diffusion=params.get("pore_diffusion", 1e-10),
            ka=bp.get("ka", 1.0),
            kd=bp.get("kd", 0.1),
            Lambda=bp.get("Lambda", 10.0),
            nu=bp.get("nu", 4.5),
        )

    raise ValueError(f"Unknown transfer function: {name}")


class NLNiltSolver:
    """Nonlinear NILT solver for chromatography with fully nonlinear binding.

    Extends the linear NILT approach using iterative correction:
    c_nl(t) = c_lin(t) + delta_c(t), where delta_c is computed via
    the CFL-constrained, mass-validated correction iteration.

    Args:
        t_end: End time for evaluation.
        t_eval_min: Minimum time for evaluation.
        eps_conv: Convergence threshold for contraction factor.
        eps_mass: Mass balance convergence threshold.
        max_iterations: Maximum nonlinear iterations.
        N_max: Maximum FFT size.
    """

    def __init__(
        self,
        t_end: float,
        t_eval_min: float = 0.1,
        eps_conv: float = 1e-4,
        eps_mass: float = 1e-6,
        max_iterations: int = 10,
        N_max: int = 32768,
    ):
        self.t_end = t_end
        self.t_eval_min = t_eval_min
        self.eps_conv = eps_conv
        self.eps_mass = eps_mass
        self.max_iterations = max_iterations
        self.N_max = N_max

    def solve_langmuir(
        self,
        ka: float,
        kd: float,
        qmax: float,
        c_feed: float = 1.0,
        velocity: float = 1e-3,
        dispersion: float = 1e-6,
        length: float = 0.1,
        col_porosity: float = 0.37,
        par_radius: float = 1e-5,
        par_porosity: float = 0.33,
        film_diffusion: float = 1e-5,
        pore_diffusion: float = 1e-10,
        step_input: bool = True,
    ) -> NLNiltResult:
        """Solve with full nonlinear Langmuir binding.

        Args:
            ka: Adsorption rate constant.
            kd: Desorption rate constant.
            qmax: Maximum binding capacity.
            c_feed: Feed concentration.
            velocity: Interstitial velocity [m/s].
            dispersion: Axial dispersion [m^2/s].
            length: Column length [m].
            col_porosity: Column porosity.
            par_radius: Particle radius [m].
            par_porosity: Particle porosity.
            film_diffusion: Film transfer coefficient [m/s].
            pore_diffusion: Pore diffusion coefficient [m^2/s].
            step_input: If True, compute step response.

        Returns:
            NLNiltResult with nonlinear solution and diagnostics.
        """
        binding = LangmuirBinding(
            ka=ka, kd=kd, qmax=qmax,
            velocity=velocity, dispersion=dispersion, length=length,
            col_porosity=col_porosity, par_radius=par_radius,
            par_porosity=par_porosity, film_diffusion=film_diffusion,
            pore_diffusion=pore_diffusion,
        )
        return nl_nilt_solve(
            binding=binding,
            t_end=self.t_end,
            c_feed=c_feed,
            t_eval_min=self.t_eval_min,
            eps_conv=self.eps_conv,
            eps_mass=self.eps_mass,
            max_iterations=self.max_iterations,
            N_max=self.N_max,
            step_input=step_input,
        )

    def solve_sma(
        self,
        ka: float,
        kd: float,
        Lambda: float,
        nu: float,
        c_feed: float = 1e-6,
        z_protein: float = 5.0,
        z_salt: float = 1.0,
        c_salt: float = 0.1,
        velocity: float = 1e-3,
        dispersion: float = 1e-6,
        length: float = 0.1,
        col_porosity: float = 0.37,
        par_radius: float = 1e-5,
        par_porosity: float = 0.33,
        film_diffusion: float = 1e-5,
        pore_diffusion: float = 1e-10,
        step_input: bool = True,
    ) -> NLNiltResult:
        """Solve with full nonlinear SMA binding.

        Args:
            ka: SMA adsorption constant.
            kd: SMA desorption rate.
            Lambda: Steric capacity.
            nu: Characteristic charge.
            c_feed: Feed concentration.
            z_protein: Protein charge.
            z_salt: Salt valence.
            c_salt: Salt concentration.
            velocity: Interstitial velocity [m/s].
            dispersion: Axial dispersion [m^2/s].
            length: Column length [m].
            col_porosity: Column porosity.
            par_radius: Particle radius [m].
            par_porosity: Particle porosity.
            film_diffusion: Film transfer coefficient [m/s].
            pore_diffusion: Pore diffusion coefficient [m^2/s].
            step_input: If True, compute step response.

        Returns:
            NLNiltResult with nonlinear solution and diagnostics.
        """
        binding = SMABinding(
            ka=ka, kd=kd, Lambda=Lambda, nu=nu,
            z_protein=z_protein, z_salt=z_salt, c_salt=c_salt,
            c0=c_feed,
            velocity=velocity, dispersion=dispersion, length=length,
            col_porosity=col_porosity, par_radius=par_radius,
            par_porosity=par_porosity, film_diffusion=film_diffusion,
            pore_diffusion=pore_diffusion,
        )
        return nl_nilt_solve(
            binding=binding,
            t_end=self.t_end,
            c_feed=c_feed,
            t_eval_min=self.t_eval_min,
            eps_conv=self.eps_conv,
            eps_mass=self.eps_mass,
            max_iterations=self.max_iterations,
            N_max=self.N_max,
            step_input=step_input,
        )
