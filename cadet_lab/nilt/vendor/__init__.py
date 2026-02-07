"""Vendored nilt-cfl library.

Source: https://github.com/gogipav14/nilt-cfl
License: MIT

This module provides CFL-informed FFT-based numerical inverse Laplace transform
using DFT-consistent frequency mapping.

Key change from original: frequency grid now uses fftfreq() to properly map
bins k > N/2 to negative frequencies. This ensures z_ifft is nearly real for
real-valued f(t), making ε_Im a meaningful diagnostic (~1e-10 expected).
"""

from .nilt_fft import (
    fft_nilt,
    fft_nilt_one_sided,  # Deprecated
    # ε_Im diagnostics
    eps_im_max,           # Paper-compliant: max|Im|/max|Re| (use this!)
    eps_im_rms,           # RMS-based alternative
    # Deprecated aliases (emit warnings)
    eps_im,
    one_sided_imag_ratio,
    epsilon_im_paper,
    n_doubling_error,
)
from .tuner import tune_params, refine_until_accept, check_cfl_feasibility, TunedParams
from .problems import get_problem, get_all_problems, Problem

__all__ = [
    # Core NILT functions
    "fft_nilt",
    "fft_nilt_one_sided",  # Deprecated
    "n_doubling_error",
    # Diagnostics (ε_Im) - use eps_im_max for paper-compliant behavior
    "eps_im_max",          # Paper-compliant: max|Im|/max|Re| (~1e-10 for real f(t))
    "eps_im_rms",          # Alternative: RMS(Im)/RMS(Re)
    # Deprecated aliases (for backward compatibility)
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
]
