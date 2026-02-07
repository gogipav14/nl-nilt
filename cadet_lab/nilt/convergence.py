"""High-level convergence tests for NILT verification.

Provides simplified wrappers around the vendored nilt-cfl library
convergence metrics with structured result objects.

Diagnostics:
- eps_im_max: Paper-compliant ε_Im = max|Im(z)|/max|Re(z)|.
  Should be ~1e-10 for real-valued functions with DFT-consistent frequency mapping.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from .vendor import fft_nilt, eps_im_max, n_doubling_error, tune_params


@dataclass
class NiltConvergenceResult:
    """Result of NILT convergence test.

    Attributes:
        passed: Whether the convergence test passed.
        epsilon_im: Imaginary part relative magnitude (for ε_Im test).
        delta_sequence: Sequence of N-doubling deltas (for N-doubling test).
        final_delta: Final delta value after N-doubling.
        n_sequence: Sequence of N values tested.
        threshold: Threshold used for pass/fail determination.
        message: Optional message describing result.
    """

    passed: bool
    epsilon_im: Optional[float] = None
    delta_sequence: Optional[List[float]] = None
    final_delta: Optional[float] = None
    n_sequence: Optional[List[int]] = None
    threshold: Optional[float] = None
    message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "epsilon_im": self.epsilon_im,
            "delta_sequence": self.delta_sequence,
            "final_delta": self.final_delta,
            "n_sequence": self.n_sequence,
            "threshold": self.threshold,
            "message": self.message,
        }


def epsilon_im_test(
    F: Callable[[complex], complex],
    t_end: float,
    alpha_c: float = 0.0,
    threshold: float = 1e-2,
    t_eval_min: float = 0.1,
    use_refinement: bool = True,
    C: float = 1.0,
) -> NiltConvergenceResult:
    """Test convergence via paper-compliant ε_Im diagnostic.

    Uses eps_im_max = max|Im(z)|/max|Re(z)| which should be ~1e-10 for
    real-valued f(t) when using DFT-consistent frequency mapping.

    Args:
        F: Laplace-domain transfer function F(s).
        t_end: End time for evaluation.
        alpha_c: Abscissa of convergence (default 0.0).
        threshold: Threshold for pass/fail (default 1e-2).
            With correct frequency mapping, ε_Im should be ~1e-10.
        t_eval_min: Minimum time for evaluation (default 0.1).
        use_refinement: If True, use adaptive N-doubling refinement.
        C: Tail envelope constant for parameter tuning.

    Returns:
        NiltConvergenceResult with ε_Im value and pass/fail status.
    """
    from .vendor import refine_until_accept

    # Tune parameters using CFL framework
    params = tune_params(t_end=t_end, alpha_c=alpha_c, C=C)

    if not bool(params.feasible):
        return NiltConvergenceResult(
            passed=False,
            epsilon_im=np.inf,
            threshold=threshold,
            message=f"CFL feasibility check failed: margin={params.margin:.2e}",
        )

    if use_refinement:
        # Use adaptive refinement to achieve threshold
        result = refine_until_accept(
            F, params, t_end,
            eps_im_threshold=threshold,
            eps_conv=threshold,
            t_eval_min=t_eval_min,
            n_timing_runs=5,
        )
        epsilon_im_value = float(result["eps_im"])
        passed = bool(result["accepted"])
    else:
        # Single evaluation without refinement
        f_full, t_full, z_ifft, eps_im_full = fft_nilt(F, params.a, params.T, params.N)
        mask = (t_full >= t_eval_min) & (t_full <= t_end)
        epsilon_im_value = float(eps_im_max(z_ifft[mask]))
        passed = epsilon_im_value <= threshold

    return NiltConvergenceResult(
        passed=passed,
        epsilon_im=epsilon_im_value,
        threshold=threshold,
        message=f"eps_im = {epsilon_im_value:.2e}, threshold = {threshold:.2e}",
    )


def n_doubling_test(
    F: Callable[[complex], complex],
    t_end: float,
    alpha_c: float = 0.0,
    n_initial: int = 64,
    max_doublings: int = 6,
    threshold: float = 1e-6,
    t_eval_min: float = 0.1,
) -> NiltConvergenceResult:
    """Test convergence via N-doubling strategy.

    Computes NILT with successively doubled N values and measures the
    difference between consecutive results. Convergence is indicated
    when the difference (delta) falls below a threshold.

    E_N = RMS(f_N - f_{2N}) / RMS(f_{2N})

    Args:
        F: Laplace-domain transfer function F(s).
        t_end: End time for evaluation.
        alpha_c: Abscissa of convergence (default 0.0).
        n_initial: Initial number of FFT points (default 64).
        max_doublings: Maximum number of N-doublings (default 6).
        threshold: Threshold for convergence (default 1e-6).
        t_eval_min: Minimum time for evaluation (default 0.1).

    Returns:
        NiltConvergenceResult with delta sequence and pass/fail status.
    """
    # Tune parameters
    params = tune_params(t_end=t_end, alpha_c=alpha_c, N_init=n_initial)

    if not bool(params.feasible):
        return NiltConvergenceResult(
            passed=False,
            delta_sequence=[],
            final_delta=np.inf,
            n_sequence=[],
            threshold=threshold,
            message=f"CFL feasibility check failed: margin={params.margin:.2e}",
        )

    a = params.a
    T = params.T
    N = n_initial

    n_sequence = []
    delta_sequence = []

    for _ in range(max_doublings):
        n_sequence.append(N)

        # Compute N-doubling error
        E_N, _, _ = n_doubling_error(F, a, T, N, t_eval_min, t_end)
        delta_sequence.append(float(E_N))

        # Check convergence
        if E_N < threshold:
            return NiltConvergenceResult(
                passed=True,
                delta_sequence=[float(d) for d in delta_sequence],
                final_delta=float(E_N),
                n_sequence=n_sequence,
                threshold=threshold,
                message=f"Converged at N={N} with delta={E_N:.2e}",
            )

        N *= 2

    # Did not converge within max_doublings
    final_delta = float(delta_sequence[-1]) if delta_sequence else np.inf

    return NiltConvergenceResult(
        passed=bool(final_delta < threshold),
        delta_sequence=[float(d) for d in delta_sequence],
        final_delta=final_delta,
        n_sequence=n_sequence,
        threshold=threshold,
        message=f"Final delta={final_delta:.2e} after {max_doublings} doublings",
    )
