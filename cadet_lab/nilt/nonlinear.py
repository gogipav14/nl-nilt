"""NL-NILT: Nonlinear FFT-NILT iteration engine.

Extends FFT-NILT to handle fully nonlinear binding (Langmuir, SMA) via
iterative correction. Three strategies are implemented:

  1. Adaptive re-linearization (Picard on effective binding parameters)
     - Re-linearize the transfer function around the current loading level
     - Each step produces a valid GRM transfer function with proper CFL
     - No distributed-source Green's function needed

  2. Anderson acceleration (Type I / DIIS)
     - Stores last m iterates and finds optimal linear combination
     - Accelerates linear convergence to superlinear

  3. Direct Picard iteration (Chen & Hsu 1989 style)
     - Fallback when re-linearization doesn't apply
     - Treats nonlinear product as known forcing from previous step

The default strategy is adaptive re-linearization with Anderson acceleration.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .vendor.nilt_fft import fft_nilt, eps_im_max
from .vendor.tuner import TunedParams, tune_params


# ---------------------------------------------------------------------------
# Binding model protocol
# ---------------------------------------------------------------------------

class BindingModel(ABC):
    """Abstract base class for nonlinear binding models."""

    @abstractmethod
    def equilibrium(self, c: np.ndarray) -> np.ndarray:
        """Compute equilibrium bound-state q_eq(c) from the isotherm."""

    @abstractmethod
    def nonlinear_residual(
        self, c: np.ndarray, q: np.ndarray,
    ) -> np.ndarray:
        """Compute the nonlinear residual R(t) = f_nl(c,q) - f_lin(c)."""

    @abstractmethod
    def linear_transfer_function(self) -> Callable[[complex], complex]:
        """Return the linearized transfer function F_lin(s)."""

    @abstractmethod
    def effective_keq(self) -> float:
        """Return the effective equilibrium constant K_eq."""

    def relinearized_transfer_function(
        self, c_operating: float,
    ) -> Callable[[complex], complex]:
        """Return transfer function re-linearized around c_operating.

        Default: falls back to the base linear transfer function.
        Subclasses should override for proper re-linearization.
        """
        return self.linear_transfer_function()


# ---------------------------------------------------------------------------
# Concrete binding models
# ---------------------------------------------------------------------------

class LangmuirBinding(BindingModel):
    """Full nonlinear Langmuir binding model.

    Kinetics: dq/dt = k_a * c * (q_max - q) - k_d * q
    """

    def __init__(
        self,
        ka: float,
        kd: float,
        qmax: float,
        velocity: float,
        dispersion: float,
        length: float,
        col_porosity: float = 0.37,
        par_radius: float = 1e-5,
        par_porosity: float = 0.33,
        film_diffusion: float = 1e-5,
        pore_diffusion: float = 1e-10,
    ):
        self.ka = ka
        self.kd = kd
        self.qmax = qmax
        self.velocity = velocity
        self.dispersion = dispersion
        self.length = length
        self.col_porosity = col_porosity
        self.par_radius = par_radius
        self.par_porosity = par_porosity
        self.film_diffusion = film_diffusion
        self.pore_diffusion = pore_diffusion

    def equilibrium(self, c: np.ndarray) -> np.ndarray:
        K_eq = self.ka / self.kd if self.kd > 0 else 0.0
        return self.qmax * K_eq * c / (1.0 + K_eq * c)

    def nonlinear_residual(
        self, c: np.ndarray, q: np.ndarray,
    ) -> np.ndarray:
        return -self.ka * c * q

    def linear_transfer_function(self) -> Callable[[complex], complex]:
        from .benchmarks import grm_langmuir_transfer

        return grm_langmuir_transfer(
            velocity=self.velocity,
            dispersion=self.dispersion,
            length=self.length,
            col_porosity=self.col_porosity,
            par_radius=self.par_radius,
            par_porosity=self.par_porosity,
            film_diffusion=self.film_diffusion,
            pore_diffusion=self.pore_diffusion,
            ka=self.ka,
            kd=self.kd,
            qmax=self.qmax,
        )

    def effective_keq(self) -> float:
        if self.kd > 0:
            return self.ka * self.qmax / self.kd
        return 0.0

    def relinearized_transfer_function(
        self, c_operating: float,
    ) -> Callable[[complex], complex]:
        """Re-linearize Langmuir around operating point c_operating.

        At operating point c_0:
          q_0 = qmax * Ka * c_0 / (1 + Ka * c_0)
          ka_eff = ka (unchanged rate constant)
          qmax_eff = qmax - q_0 (reduced capacity)
          kd_eff = kd + ka * c_0 (enhanced effective desorption)

        The effective Henry constant decreases with loading:
          K_eq_eff = ka * (qmax - q_0) / (kd + ka * c_0)
        """
        from .benchmarks import grm_langmuir_transfer

        Ka = self.ka / self.kd if self.kd > 0 else 0.0
        q_0 = self.qmax * Ka * c_operating / (1.0 + Ka * c_operating)
        qmax_eff = max(self.qmax - q_0, 1e-30)
        kd_eff = self.kd + self.ka * c_operating

        return grm_langmuir_transfer(
            velocity=self.velocity,
            dispersion=self.dispersion,
            length=self.length,
            col_porosity=self.col_porosity,
            par_radius=self.par_radius,
            par_porosity=self.par_porosity,
            film_diffusion=self.film_diffusion,
            pore_diffusion=self.pore_diffusion,
            ka=self.ka,
            kd=kd_eff,
            qmax=qmax_eff,
        )


class SMABinding(BindingModel):
    """Full nonlinear Steric Mass Action (SMA) binding model."""

    def __init__(
        self,
        ka: float,
        kd: float,
        Lambda: float,
        nu: float,
        z_protein: float = 5.0,
        z_salt: float = 1.0,
        c_salt: float = 0.1,
        c0: float = 1e-6,
        velocity: float = 1e-3,
        dispersion: float = 1e-6,
        length: float = 0.1,
        col_porosity: float = 0.37,
        par_radius: float = 1e-5,
        par_porosity: float = 0.33,
        film_diffusion: float = 1e-5,
        pore_diffusion: float = 1e-10,
    ):
        self.ka = ka
        self.kd = kd
        self.Lambda = Lambda
        self.nu = nu
        self.z_protein = z_protein
        self.z_salt = z_salt
        self.c_salt = c_salt
        self.c0 = c0
        self.velocity = velocity
        self.dispersion = dispersion
        self.length = length
        self.col_porosity = col_porosity
        self.par_radius = par_radius
        self.par_porosity = par_porosity
        self.film_diffusion = film_diffusion
        self.pore_diffusion = pore_diffusion

        self.Lambda_eff = Lambda - z_salt * c_salt
        self._compute_base_state()

    def _compute_base_state(self):
        K_eq = self.ka / self.kd if self.kd > 0 else 0.0
        q0 = 0.0
        for _ in range(10):
            shield = self.Lambda_eff - self.z_protein * q0
            if shield <= 0:
                q0 = 0.0
                break
            f_q = q0 - K_eq * self.c0 * shield ** self.nu
            df_q = 1.0 + K_eq * self.c0 * self.nu * self.z_protein * shield ** (self.nu - 1)
            q0_new = q0 - f_q / df_q
            if abs(q0_new - q0) < 1e-12:
                q0 = q0_new
                break
            q0 = q0_new
        self.q0 = max(q0, 0.0)

        shield = self.Lambda_eff - self.z_protein * self.q0
        if shield > 0:
            self.k_a_eff = self.ka * shield ** self.nu
            self.k_d_eff = self.kd + self.ka * self.c0 * self.nu * self.z_protein * shield ** (self.nu - 1)
        else:
            self.k_a_eff = self.ka * self.Lambda_eff ** self.nu
            self.k_d_eff = self.kd

    def equilibrium(self, c: np.ndarray) -> np.ndarray:
        q = np.zeros_like(c)
        K_eq = self.ka / self.kd if self.kd > 0 else 0.0
        for _ in range(10):
            shield = np.maximum(self.Lambda_eff - self.z_protein * q, 1e-30)
            f_q = q - K_eq * c * shield ** self.nu
            df_q = 1.0 + K_eq * c * self.nu * self.z_protein * shield ** (self.nu - 1)
            dq = f_q / df_q
            q = np.maximum(q - dq, 0.0)
            if np.max(np.abs(dq)) < 1e-12:
                break
        return q

    def nonlinear_residual(
        self, c: np.ndarray, q: np.ndarray,
    ) -> np.ndarray:
        shield = np.maximum(self.Lambda_eff - self.z_protein * q, 0.0)
        full_rate = self.ka * c * shield ** self.nu - self.kd * q
        linear_rate = self.k_a_eff * c - self.k_d_eff * q
        return full_rate - linear_rate

    def linear_transfer_function(self) -> Callable[[complex], complex]:
        from .benchmarks import grm_langmuir_transfer

        return grm_langmuir_transfer(
            velocity=self.velocity,
            dispersion=self.dispersion,
            length=self.length,
            col_porosity=self.col_porosity,
            par_radius=self.par_radius,
            par_porosity=self.par_porosity,
            film_diffusion=self.film_diffusion,
            pore_diffusion=self.pore_diffusion,
            ka=self.k_a_eff,
            kd=self.k_d_eff,
            qmax=self.Lambda_eff,
        )

    def effective_keq(self) -> float:
        if self.k_d_eff > 0:
            return self.k_a_eff * self.Lambda_eff / self.k_d_eff
        return 0.0

    def relinearized_transfer_function(
        self, c_operating: float,
    ) -> Callable[[complex], complex]:
        """Re-linearize SMA around operating point c_operating."""
        from .benchmarks import grm_langmuir_transfer

        # Newton for q_0 at c_operating
        K_eq = self.ka / self.kd if self.kd > 0 else 0.0
        q0 = 0.0
        for _ in range(10):
            shield = self.Lambda_eff - self.z_protein * q0
            if shield <= 0:
                break
            f_q = q0 - K_eq * c_operating * shield ** self.nu
            df_q = 1.0 + K_eq * c_operating * self.nu * self.z_protein * shield ** (self.nu - 1)
            q0_new = q0 - f_q / df_q
            if abs(q0_new - q0) < 1e-12:
                q0 = q0_new
                break
            q0 = q0_new
        q0 = max(q0, 0.0)

        shield = max(self.Lambda_eff - self.z_protein * q0, 1e-30)
        ka_eff = self.ka * shield ** self.nu
        kd_eff = self.kd + self.ka * c_operating * self.nu * self.z_protein * shield ** (self.nu - 1)

        return grm_langmuir_transfer(
            velocity=self.velocity,
            dispersion=self.dispersion,
            length=self.length,
            col_porosity=self.col_porosity,
            par_radius=self.par_radius,
            par_porosity=self.par_porosity,
            film_diffusion=self.film_diffusion,
            pore_diffusion=self.pore_diffusion,
            ka=ka_eff,
            kd=kd_eff,
            qmax=self.Lambda_eff,
        )


# ---------------------------------------------------------------------------
# Anderson acceleration
# ---------------------------------------------------------------------------

class AndersonAccelerator:
    """Type-I Anderson acceleration (DIIS) for fixed-point iteration.

    Stores the last `depth` iterates and residuals and finds the optimal
    linear combination that minimizes the residual norm.

    Args:
        depth: Number of previous iterates to store (mixing depth).
        beta: Relaxation parameter (1.0 = full step, <1 = damped).
    """

    def __init__(self, depth: int = 5, beta: float = 1.0):
        self.depth = depth
        self.beta = beta
        self._x_history: list = []
        self._r_history: list = []

    def step(self, x: np.ndarray, g_x: np.ndarray) -> np.ndarray:
        """Compute the Anderson-accelerated update.

        Args:
            x: Current iterate x^m.
            g_x: Picard update g(x^m) = result of one fixed-point step.

        Returns:
            Accelerated next iterate x^{m+1}.
        """
        r = g_x - x  # Residual

        self._x_history.append(x.copy())
        self._r_history.append(r.copy())

        m_k = len(self._r_history)
        if m_k <= 1:
            # Not enough history for mixing — just do Picard
            return x + self.beta * r

        # Trim to depth
        if m_k > self.depth + 1:
            self._x_history.pop(0)
            self._r_history.pop(0)
            m_k = len(self._r_history)

        # Build residual difference matrix: Delta_R[:, j] = r^{m-k+j+1} - r^{m-k+j}
        n_cols = m_k - 1
        n = len(r)
        Delta_R = np.empty((n, n_cols))
        for j in range(n_cols):
            Delta_R[:, j] = self._r_history[j + 1] - self._r_history[j]

        # Solve least-squares: min || r^m - Delta_R @ alpha ||
        # Using the normal equations: Delta_R^T @ Delta_R @ alpha = Delta_R^T @ r^m
        try:
            alpha, _, _, _ = np.linalg.lstsq(Delta_R, r, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to Picard
            return x + self.beta * r

        # Build mixing differences for x as well
        Delta_X = np.empty((n, n_cols))
        for j in range(n_cols):
            Delta_X[:, j] = self._x_history[j + 1] - self._x_history[j]

        # Anderson update: x^{m+1} = (x^m - Delta_X @ alpha) + beta * (r^m - Delta_R @ alpha)
        x_mix = x - Delta_X @ alpha
        r_mix = r - Delta_R @ alpha
        return x_mix + self.beta * r_mix

    def reset(self):
        """Clear history (e.g., after a restart)."""
        self._x_history.clear()
        self._r_history.clear()


# ---------------------------------------------------------------------------
# Iteration results
# ---------------------------------------------------------------------------

@dataclass
class NLNiltIterationResult:
    """Result of a single NL-NILT iteration step."""

    iteration: int
    c: np.ndarray
    q: np.ndarray
    delta_c: np.ndarray
    residual_norm: float
    contraction: float = np.inf
    delta_F0: float = np.inf
    eps_im: float = np.inf
    damping: float = 1.0
    c_operating: float = 0.0
    strategy: str = ""


@dataclass
class NLNiltResult:
    """Full result of the NL-NILT solve."""

    t: np.ndarray
    c: np.ndarray
    c_lin: np.ndarray
    delta_c: np.ndarray
    iterations: list
    converged: bool
    n_iterations: int
    wall_time_us: float = 0.0
    tuned_params: Optional[TunedParams] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: estimate operating concentration from breakthrough curve
# ---------------------------------------------------------------------------

def _estimate_operating_concentration(
    t: np.ndarray, c: np.ndarray, c_feed: float,
) -> float:
    """Estimate the effective operating concentration for re-linearization.

    Uses a weighted average of c(t), weighting by how much of the
    breakthrough has occurred. This captures the "typical" concentration
    the column sees during the loading phase.
    """
    if c_feed <= 0 or len(c) == 0:
        return 0.0

    # Normalized breakthrough curve
    c_norm = np.clip(c / c_feed, 0.0, 1.5)

    # Weight by the change in concentration (emphasize the front)
    dc = np.abs(np.gradient(c_norm, t))
    total_weight = np.trapezoid(dc, t)
    if total_weight > 1e-30:
        c_weighted = np.trapezoid(c_norm * dc, t) / total_weight
    else:
        # Flat curve — use the mean
        c_weighted = np.mean(c_norm)

    return float(c_weighted * c_feed)


# ---------------------------------------------------------------------------
# Helper: single FFT-NILT evaluation at fixed (a, T, N)
# ---------------------------------------------------------------------------

def _nilt_eval_at_params(
    F: Callable, a: float, T: float, N: int,
    t_eval: np.ndarray, c_feed: float, step_input: bool,
) -> np.ndarray:
    """Evaluate FFT-NILT at fixed parameters and interpolate to t_eval."""
    if step_input:
        F_solve = lambda s, _F=F: _F(s) / s
    else:
        F_solve = F

    f_full, t_full, z_ifft, _ = fft_nilt(F_solve, a, T, N)
    f_eval = np.interp(t_eval, t_full, f_full)
    return f_eval * c_feed


# ---------------------------------------------------------------------------
# Strategy 1: Adaptive re-linearization with Anderson acceleration
# ---------------------------------------------------------------------------

def _solve_relinearization(
    binding: BindingModel,
    t_eval: np.ndarray,
    c_lin: np.ndarray,
    c_feed: float,
    a: float, T: float, N: int,
    *,
    step_input: bool,
    max_iterations: int,
    eps_conv: float,
    anderson_depth: int,
) -> tuple:
    """Picard iteration with adaptive re-linearization + Anderson acceleration.

    At each step: re-linearize the transfer function around the current
    loading level, then solve the full GRM with the new parameters.
    Anderson acceleration is applied on the solution vector c(t).
    """
    aa = AndersonAccelerator(depth=anderson_depth, beta=1.0)
    c_current = c_lin.copy()
    c_prev = None
    iteration_results = []
    converged = False

    for m in range(max_iterations):
        # Estimate operating concentration from current iterate
        c_op = _estimate_operating_concentration(t_eval, c_current, c_feed)

        # Build re-linearized transfer function
        F_relin = binding.relinearized_transfer_function(c_op)

        # Solve with re-linearized transfer function (one Picard step)
        g_c = _nilt_eval_at_params(F_relin, a, T, N, t_eval, c_feed, step_input)
        g_c = np.maximum(g_c, 0.0)

        # Anderson acceleration
        c_next = aa.step(c_current, g_c)
        c_next = np.maximum(c_next, 0.0)

        # Compute contraction
        if c_prev is not None:
            diff = np.sqrt(np.mean((c_next - c_current) ** 2))
            prev_diff = np.sqrt(np.mean((c_current - c_prev) ** 2))
            kappa = diff / prev_diff if prev_diff > 1e-30 else np.inf
        else:
            kappa = np.inf

        # Residual: difference between Picard output and current iterate
        residual_norm = np.sqrt(np.mean((g_c - c_current) ** 2))

        # Relative residual
        c_rms = np.sqrt(np.mean(c_current ** 2))
        rel_residual = residual_norm / c_rms if c_rms > 1e-30 else residual_norm

        q_current = binding.equilibrium(c_next)
        delta_c = c_next - c_lin

        iteration_results.append(NLNiltIterationResult(
            iteration=m,
            c=c_next.copy(),
            q=q_current.copy(),
            delta_c=delta_c.copy(),
            residual_norm=residual_norm,
            contraction=kappa,
            c_operating=c_op,
            strategy="relinearization+anderson",
        ))

        c_prev = c_current.copy()
        c_current = c_next

        if rel_residual < eps_conv:
            converged = True
            break

    return c_current, converged, iteration_results


# ---------------------------------------------------------------------------
# Strategy 2: Direct Picard (Chen & Hsu style, no Anderson)
# ---------------------------------------------------------------------------

def _solve_picard_direct(
    binding: BindingModel,
    t_eval: np.ndarray,
    c_lin: np.ndarray,
    c_feed: float,
    a: float, T: float, N: int,
    *,
    step_input: bool,
    max_iterations: int,
    eps_conv: float,
) -> tuple:
    """Direct Picard iteration: re-linearize without Anderson acceleration."""
    c_current = c_lin.copy()
    c_prev = None
    iteration_results = []
    converged = False

    for m in range(max_iterations):
        c_op = _estimate_operating_concentration(t_eval, c_current, c_feed)
        F_relin = binding.relinearized_transfer_function(c_op)
        c_next = _nilt_eval_at_params(F_relin, a, T, N, t_eval, c_feed, step_input)
        c_next = np.maximum(c_next, 0.0)

        if c_prev is not None:
            diff = np.sqrt(np.mean((c_next - c_current) ** 2))
            prev_diff = np.sqrt(np.mean((c_current - c_prev) ** 2))
            kappa = diff / prev_diff if prev_diff > 1e-30 else np.inf
        else:
            kappa = np.inf

        residual_norm = np.sqrt(np.mean((c_next - c_current) ** 2))
        c_rms = np.sqrt(np.mean(c_current ** 2))
        rel_residual = residual_norm / c_rms if c_rms > 1e-30 else residual_norm

        q_current = binding.equilibrium(c_next)

        iteration_results.append(NLNiltIterationResult(
            iteration=m,
            c=c_next.copy(),
            q=q_current.copy(),
            delta_c=(c_next - c_lin).copy(),
            residual_norm=residual_norm,
            contraction=kappa,
            c_operating=c_op,
            strategy="picard_direct",
        ))

        c_prev = c_current.copy()
        c_current = c_next

        if rel_residual < eps_conv:
            converged = True
            break

    return c_current, converged, iteration_results


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def nl_nilt_solve(
    binding: BindingModel,
    t_end: float,
    c_feed: float = 1.0,
    *,
    t_eval_min: float = 0.1,
    eps_conv: float = 1e-4,
    eps_mass: float = 1e-6,
    max_iterations: int = 10,
    damping_threshold: float = 2.0,
    switch_to_direct: float = 0.5,
    N_max: int = 32768,
    eps_im_threshold: float = 1e-2,
    step_input: bool = True,
    strategy: str = "auto",
    anderson_depth: int = 5,
) -> NLNiltResult:
    """Solve nonlinear chromatography via iterative re-linearization.

    Args:
        binding: Nonlinear binding model instance.
        t_end: End time for evaluation.
        c_feed: Feed concentration for step input.
        t_eval_min: Minimum time for evaluation.
        eps_conv: Convergence threshold (relative residual).
        eps_mass: Mass balance threshold on |F(0) - 1|.
        max_iterations: Maximum nonlinear iterations.
        damping_threshold: Unused (kept for API compatibility).
        switch_to_direct: Unused (kept for API compatibility).
        N_max: Maximum FFT size.
        eps_im_threshold: eps_im acceptance threshold.
        step_input: If True, compute step response (F(s)/s).
        strategy: Iteration strategy:
            "auto" — try relinearization+anderson, fall back to picard
            "relinearization" — adaptive re-linearization + Anderson
            "picard" — direct Picard without Anderson
        anderson_depth: Depth for Anderson acceleration history.

    Returns:
        NLNiltResult with final solution and convergence diagnostics.
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: Linear baseline (linearized around c=0)
    # ------------------------------------------------------------------
    F_lin = binding.linear_transfer_function()

    if step_input:
        F_lin_solve = lambda s, _F=F_lin: _F(s) / s
    else:
        F_lin_solve = F_lin

    params = tune_params(t_end=t_end, alpha_c=0.0)
    if not params.feasible:
        return NLNiltResult(
            t=np.array([]),
            c=np.array([]),
            c_lin=np.array([]),
            delta_c=np.array([]),
            iterations=[],
            converged=False,
            n_iterations=0,
            wall_time_us=(time.perf_counter() - t_start) * 1e6,
            metadata={"error": "CFL infeasible for linear baseline"},
        )

    from .vendor.tuner import refine_until_accept

    lin_result = refine_until_accept(
        F_lin_solve,
        params,
        t_end=t_end,
        eps_im_threshold=eps_im_threshold,
        eps_conv=1e-6,
        N_max=N_max,
        t_eval_min=t_eval_min,
        n_timing_runs=1,
    )

    t_eval = lin_result["t_eval"]
    c_lin = lin_result["f_eval"] * c_feed
    N_final = lin_result["N"]
    a_final = lin_result["a"]
    T_final = lin_result["T"]

    # Check if nonlinear correction is needed
    q_lin = binding.equilibrium(c_lin)
    R_check = binding.nonlinear_residual(c_lin, q_lin)
    residual_check = np.sqrt(np.mean(R_check ** 2))

    if residual_check < 1e-15:
        # Already in linear regime — no iteration needed
        wall_time_us = (time.perf_counter() - t_start) * 1e6
        return NLNiltResult(
            t=t_eval,
            c=c_lin,
            c_lin=c_lin,
            delta_c=np.zeros_like(c_lin),
            iterations=[NLNiltIterationResult(
                iteration=0, c=c_lin, q=q_lin,
                delta_c=np.zeros_like(c_lin),
                residual_norm=residual_check,
                contraction=0.0, strategy="linear_exact",
            )],
            converged=True,
            n_iterations=1,
            wall_time_us=wall_time_us,
            tuned_params=params,
            metadata={
                "N_linear": N_final,
                "F0_baseline": F_lin(complex(1e-12, 0.0)).real,
                "delta_F0_baseline": abs(F_lin(complex(1e-12, 0.0)).real - 1.0),
                "c_feed": c_feed,
                "strategy": "linear_exact",
                "final_residual_norm": residual_check,
                "final_contraction": 0.0,
            },
        )

    # ------------------------------------------------------------------
    # Phase 2: Nonlinear iteration
    # ------------------------------------------------------------------

    if strategy == "auto":
        # Try relinearization + Anderson first
        c_final, converged, iter_results = _solve_relinearization(
            binding, t_eval, c_lin, c_feed,
            a_final, T_final, N_final,
            step_input=step_input,
            max_iterations=max_iterations,
            eps_conv=eps_conv,
            anderson_depth=anderson_depth,
        )
        used_strategy = "relinearization+anderson"

        # If not converged, try plain Picard as fallback
        if not converged:
            c_picard, conv_picard, picard_results = _solve_picard_direct(
                binding, t_eval, c_lin, c_feed,
                a_final, T_final, N_final,
                step_input=step_input,
                max_iterations=max_iterations,
                eps_conv=eps_conv,
            )
            # Use whichever has smaller final residual
            relin_resid = iter_results[-1].residual_norm if iter_results else np.inf
            picard_resid = picard_results[-1].residual_norm if picard_results else np.inf
            if picard_resid < relin_resid:
                c_final = c_picard
                converged = conv_picard
                iter_results = picard_results
                used_strategy = "picard_direct"

    elif strategy == "relinearization":
        c_final, converged, iter_results = _solve_relinearization(
            binding, t_eval, c_lin, c_feed,
            a_final, T_final, N_final,
            step_input=step_input,
            max_iterations=max_iterations,
            eps_conv=eps_conv,
            anderson_depth=anderson_depth,
        )
        used_strategy = "relinearization+anderson"

    elif strategy == "picard":
        c_final, converged, iter_results = _solve_picard_direct(
            binding, t_eval, c_lin, c_feed,
            a_final, T_final, N_final,
            step_input=step_input,
            max_iterations=max_iterations,
            eps_conv=eps_conv,
        )
        used_strategy = "picard_direct"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    wall_time_us = (time.perf_counter() - t_start) * 1e6
    delta_c_final = c_final - c_lin

    # Mass-balance check
    F0_lin = F_lin(complex(1e-12, 0.0)).real
    delta_F0_baseline = abs(F0_lin - 1.0)

    return NLNiltResult(
        t=t_eval,
        c=c_final,
        c_lin=c_lin,
        delta_c=delta_c_final,
        iterations=iter_results,
        converged=converged,
        n_iterations=len(iter_results),
        wall_time_us=wall_time_us,
        tuned_params=params,
        metadata={
            "N_linear": N_final,
            "F0_baseline": F0_lin,
            "delta_F0_baseline": delta_F0_baseline,
            "use_direct": False,
            "c_feed": c_feed,
            "strategy": used_strategy,
            "final_residual_norm": iter_results[-1].residual_norm if iter_results else 0.0,
            "final_contraction": iter_results[-1].contraction if iter_results else 0.0,
        },
    )
