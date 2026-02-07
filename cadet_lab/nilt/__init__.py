"""NILT verification pack using CFL-informed FFT-NILT.

This module provides numerical inverse Laplace transform capabilities
for verifying CADET simulations against analytical reference solutions.

The implementation uses the vendored nilt-cfl library from:
https://github.com/gogipav14/nilt-cfl

Key Features:
- CFL-informed parameter selection (Algorithm 1)
- FFT-based Bromwich integral with DFT-consistent frequency mapping
- Paper-compliant ε_Im diagnostic: max|Im(z)|/max|Re(z)| should be ~1e-10
- N-doubling convergence test
- CADET-specific benchmark transfer functions

Note: The frequency grid uses fftfreq() to properly map DFT bins k > N/2
to negative frequencies. This ensures z_ifft is nearly real for real-valued
f(t), making ε_Im a meaningful diagnostic.
"""

# Re-export vendored library functions
from .vendor import (
    fft_nilt,
    fft_nilt_one_sided,  # Deprecated
    # Diagnostics (ε_Im)
    eps_im_max,          # Paper-compliant: max|Im|/max|Re| (~1e-10 for real f(t))
    eps_im_rms,          # Alternative: RMS(Im)/RMS(Re)
    # Deprecated aliases
    one_sided_imag_ratio,
    epsilon_im_paper,
    eps_im,
    n_doubling_error,
    # Parameter tuning
    tune_params,
    refine_until_accept,
    check_cfl_feasibility,
    TunedParams,
    # Benchmark problems
    get_problem,
    get_all_problems,
    Problem,
)

# High-level convergence wrappers
from .convergence import (
    epsilon_im_test,
    n_doubling_test,
    NiltConvergenceResult,
)

# CADET-specific benchmarks
from .benchmarks import (
    get_benchmark_functions,
    advection_dispersion_transfer,
    langmuir_column_transfer,
    grm_langmuir_transfer,
    grm_sma_transfer,
    grm_moment_transfer,
)

# Solver API
from .solver import NiltSolver, NiltSolution, NLNiltSolver
from .classify import classify_problem, ProblemClassification
from .extract_params import extract_nilt_params
from .output import write_cadet_h5, write_json

# Nonlinear NILT
from .nonlinear import (
    nl_nilt_solve,
    LangmuirBinding,
    SMABinding,
    NLNiltResult,
    NLNiltIterationResult,
    BindingModel,
)
from .mass_balance import (
    MassBalanceDiagnostics,
    compute_diagnostics,
    check_zeroth_moment,
    compute_numerical_first_moment,
    compute_theoretical_first_moment,
    steering_decision,
)

__all__ = [
    # Vendored core NILT
    "fft_nilt",
    "fft_nilt_one_sided",  # Deprecated
    "n_doubling_error",
    # Diagnostics (ε_Im) - use eps_im_max for paper-compliant behavior
    "eps_im_max",          # Paper-compliant: max|Im|/max|Re| (~1e-10 for real f(t))
    "eps_im_rms",          # Alternative: RMS(Im)/RMS(Re)
    # Deprecated aliases
    "one_sided_imag_ratio",
    "epsilon_im_paper",
    "eps_im",
    # Parameter tuning
    "tune_params",
    "refine_until_accept",
    "check_cfl_feasibility",
    "TunedParams",
    # Benchmark problems
    "get_problem",
    "get_all_problems",
    "Problem",
    # High-level wrappers
    "epsilon_im_test",
    "n_doubling_test",
    "NiltConvergenceResult",
    # CADET benchmarks
    "get_benchmark_functions",
    "advection_dispersion_transfer",
    "langmuir_column_transfer",
    "grm_langmuir_transfer",
    "grm_sma_transfer",
    "grm_moment_transfer",
    # Solver API
    "NiltSolver",
    "NiltSolution",
    "NLNiltSolver",
    "classify_problem",
    "ProblemClassification",
    "extract_nilt_params",
    "write_cadet_h5",
    "write_json",
    # Nonlinear NILT
    "nl_nilt_solve",
    "LangmuirBinding",
    "SMABinding",
    "NLNiltResult",
    "NLNiltIterationResult",
    "BindingModel",
    # Mass-balance diagnostics
    "MassBalanceDiagnostics",
    "compute_diagnostics",
    "check_zeroth_moment",
    "compute_numerical_first_moment",
    "compute_theoretical_first_moment",
    "steering_decision",
]
