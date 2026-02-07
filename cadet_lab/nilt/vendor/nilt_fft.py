"""FFT-based Numerical Inverse Laplace Transform.

Vendored from: https://github.com/gogipav14/nilt-cfl
License: MIT

Implements FFT acceleration of the trapezoidal Bromwich integral using
DFT-consistent frequency bins (positive and negative ω via FFT ordering).

For real-valued time-domain functions f(t), the Laplace transform obeys
conjugate symmetry: F(conj(s)) = conj(F(s)), i.e., F(a-iω) = conj(F(a+iω)).
This means the IFFT output z_ifft should be nearly real, and the imaginary
part measures numerical leakage (aliasing, truncation, roundoff).

The ε_Im diagnostic (max|Im|/max|Re|) should be ~1e-10 for well-conditioned
real-valued benchmark functions when using DFT-consistent frequency mapping.
"""

from __future__ import annotations
import numpy as np
import warnings
from typing import Callable, Tuple, Optional


def fft_nilt(
    F: Callable[[complex], complex],
    a: float,
    T: float,
    N: int,
    return_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute inverse Laplace transform using FFT-accelerated Bromwich integral.

    Uses DFT-consistent frequency bins where indices k > N/2 represent
    negative frequencies (wrap-around). This ensures z_ifft is nearly real
    for real-valued f(t), making ε_Im a meaningful diagnostic.

    Parameters
    ----------
    F : callable
        Laplace-domain transfer function F(s) -> complex
    a : float
        Bromwich shift parameter (contour Re(s) = a)
    T : float
        Half-period (aliasing period = 2T)
    N : int
        Number of FFT points (preferably power of 2)
    return_complex : bool
        Unused, kept for API compatibility

    Returns
    -------
    f : ndarray
        Time-domain function values at t_j = j * Δt
    t : ndarray
        Time points t_j for j = 0, ..., N-1
    z_ifft : ndarray
        Complex IFFT output (should be nearly real for real f(t))
    eps_im : float
        Paper-compliant ε_Im = max|Im(z)|/max|Re(z)| (should be ~1e-10)

    Notes
    -----
    For CFL-tuned parameters (a, T) satisfying the feasibility conditions:
    - a > α_c + δ_min (spectral placement)
    - a < (L - δ_s)/(2T) (dynamic range)
    - a ≥ α_c + ln(C/ε_tail)/(2T-t_end) (aliasing suppression)

    The result accuracy is controlled by N (truncation error) and the
    CFL parameters (aliasing error).
    """
    # Time step: Δt = 2T/N
    delta_t = 2 * T / N

    # Time grid: t_j = j * Δt for j = 0, ..., N-1
    t = np.arange(N) * delta_t

    # DFT-consistent frequency grid using fftfreq
    # This maps bins k > N/2 to negative frequencies (wrap-around)
    # fftfreq returns frequencies in cycles/sample, multiply by 2π to get angular freq
    omega = 2 * np.pi * np.fft.fftfreq(N, d=delta_t)
    s = a + 1j * omega

    # Evaluate F(s) at Bromwich contour points (includes negative ω for k > N/2)
    G = np.array([F(sk) for sk in s], dtype=np.complex128)

    # Compute sum via IFFT
    # IFFT: (1/N) * Σ G[k] exp(i 2π k j / N)
    # Our sum: Σ G[k] exp(i ω_k t_j) matches this with DFT-consistent ω
    # So multiply IFFT by N
    z_ifft = N * np.fft.ifft(G)

    # Compute paper-compliant ε_Im = max|Im|/max|Re| (should be ~1e-10 for real f(t))
    eps_im_value = eps_im_max(z_ifft)

    # Apply exponential factor, scaling, and extract real part
    # f(t) = exp(a*t) / (2*T) * Re[sum]
    f = np.exp(a * t) / (2 * T) * np.real(z_ifft)

    return f, t, z_ifft, eps_im_value


def fft_nilt_one_sided(
    F: Callable[[complex], complex],
    a: float,
    T: float,
    N: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy one-sided implementation (DEPRECATED).

    This function is kept for backward compatibility but uses incorrect
    frequency mapping. Use fft_nilt() instead which uses DFT-consistent
    frequency bins.

    The old approach evaluated F(s) at positive frequencies only (ω ≥ 0)
    but then used ifft which expects DFT bin ordering (k > N/2 = negative ω).
    This caused z_ifft to be complex by construction, not due to numerical error.
    """
    warnings.warn(
        "fft_nilt_one_sided() uses incorrect frequency mapping and is deprecated. "
        "Use fft_nilt() instead which uses DFT-consistent frequency bins.",
        DeprecationWarning,
        stacklevel=2
    )
    # Use correct implementation
    f, t, z_ifft, _ = fft_nilt(F, a, T, N)
    return f, t, z_ifft


def eps_im_max(z_ifft: np.ndarray) -> float:
    """
    Compute paper-compliant ε_Im = max|Im(z)| / max|Re(z)|.

    This is the ε_Im diagnostic from paper2_nilt_cfl Eq. (28).
    For real-valued f(t), z_ifft should be nearly real when using
    DFT-consistent frequency mapping. Values should be ~1e-10.

    Parameters
    ----------
    z_ifft : ndarray
        Complex IFFT output from fft_nilt

    Returns
    -------
    eps_im : float
        max|Im(z)| / max|Re(z)| - should be ~1e-10 for real functions
    """
    max_real = np.max(np.abs(np.real(z_ifft)))
    max_imag = np.max(np.abs(np.imag(z_ifft)))

    if max_real < 1e-300:
        return np.inf

    return max_imag / max_real


def eps_im_rms(z_ifft: np.ndarray) -> float:
    """
    Compute RMS-based ε_Im ratio = RMS(Im(z)) / RMS(Re(z)).

    Alternative to eps_im_max using RMS instead of max norm.
    Less sensitive to outliers but may mask localized issues.

    Parameters
    ----------
    z_ifft : ndarray
        Complex IFFT output from fft_nilt

    Returns
    -------
    ratio : float
        RMS(Im(z)) / RMS(Re(z))
    """
    real_part = np.real(z_ifft)
    imag_part = np.imag(z_ifft)

    rms_real = np.sqrt(np.mean(real_part**2))
    rms_imag = np.sqrt(np.mean(imag_part**2))

    if rms_real < 1e-300:
        return np.inf

    return rms_imag / rms_real


# Backward compatibility aliases
def one_sided_imag_ratio(z_ifft: np.ndarray) -> float:
    """Deprecated alias for eps_im_max. Use eps_im_max instead."""
    warnings.warn(
        "one_sided_imag_ratio() is deprecated. Use eps_im_max() instead. "
        "With DFT-consistent frequency mapping, ε_Im should be ~1e-10 for real f(t).",
        DeprecationWarning,
        stacklevel=2
    )
    return eps_im_max(z_ifft)


def eps_im(z_ifft: np.ndarray) -> float:
    """Deprecated alias for eps_im_max. Use eps_im_max instead."""
    warnings.warn(
        "eps_im() is deprecated. Use eps_im_max() for paper-compliant ε_Im "
        "(max|Im|/max|Re|) or eps_im_rms() for RMS-based ratio.",
        DeprecationWarning,
        stacklevel=2
    )
    return eps_im_max(z_ifft)


def epsilon_im_paper(
    G: np.ndarray,
    N: int,
    a: float,
    T: float,
    t_eval_min: float = 0.0,
    t_eval_max: Optional[float] = None
) -> float:
    """
    Deprecated - use eps_im_max(z_ifft) directly instead.

    With correct DFT frequency mapping, eps_im_max on z_ifft from fft_nilt()
    gives the paper-compliant ε_Im directly. This function is no longer needed.
    """
    warnings.warn(
        "epsilon_im_paper() is deprecated. With DFT-consistent frequency mapping "
        "in fft_nilt(), use eps_im_max(z_ifft) directly for paper-compliant ε_Im.",
        DeprecationWarning,
        stacklevel=2
    )
    # Recompute z_ifft from G (already has trapezoidal weight on G[0])
    z_ifft = N * np.fft.ifft(G)

    if t_eval_max is None:
        t_eval_max = 2 * N  # Use all points

    delta_t = 2 * (t_eval_max / 2) / N if t_eval_max else 1.0
    t = np.arange(N) * delta_t
    mask = (t >= t_eval_min) & (t <= t_eval_max)

    return eps_im_max(z_ifft[mask]) if np.any(mask) else eps_im_max(z_ifft)


def n_doubling_error(
    F: Callable[[complex], complex],
    a: float,
    T: float,
    N: int,
    t_eval_min: float = 0.1,
    t_eval_max: float = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute N-doubling convergence error.

    E_N = RMS(f_N - f_{2N}) / RMS(f_{2N})

    evaluated over [t_eval_min, t_eval_max].

    Parameters
    ----------
    F : callable
        Transfer function
    a : float
        Bromwich shift
    T : float
        Half-period
    N : int
        Current sample count
    t_eval_min : float
        Minimum time for evaluation (default 0.1 to avoid t=0 issues)
    t_eval_max : float
        Maximum time for evaluation (default T)

    Returns
    -------
    E_N : float
        Convergence error metric
    f_N : ndarray
        Solution at N points (on evaluation grid)
    f_2N : ndarray
        Solution at 2N points (on evaluation grid)
    """
    if t_eval_max is None:
        t_eval_max = T

    # Compute at N points
    f_N_full, t_N, _, _ = fft_nilt(F, a, T, N)

    # Compute at 2N points
    f_2N_full, t_2N, _, _ = fft_nilt(F, a, T, 2 * N)

    # Interpolate to common evaluation grid
    # Use the 2N time points within evaluation range
    mask_2N = (t_2N >= t_eval_min) & (t_2N <= t_eval_max)
    t_eval = t_2N[mask_2N]
    f_2N = f_2N_full[mask_2N]

    # Interpolate f_N to the same time points
    f_N = np.interp(t_eval, t_N, f_N_full)

    # Compute error
    rms_diff = np.sqrt(np.mean((f_N - f_2N)**2))
    rms_2N = np.sqrt(np.mean(f_2N**2))

    if rms_2N < 1e-300:
        return np.inf, f_N, f_2N

    E_N = rms_diff / rms_2N

    return E_N, f_N, f_2N
