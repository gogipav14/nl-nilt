#!/usr/bin/env python3
"""Reproduce Table 1 from Chen & Hsu (1989), AIChE J. 35(2), 332-334.

"Application of Fast Fourier Transform to Nonlinear Fixed-Bed Adsorption Problems"

Problem: Plug-flow column with second-order kinetics (Thomas model),
non-porous adsorbent, no axial dispersion.

  dC/dt + V dC/dz + alpha d(theta)/dt = 0          (Eq. 1)
  d(theta)/dt = k1 * C * (1 - theta) - k2 * theta  (Eq. 3)

Analytical solution (Thomas/Hiester-Vermeulen) provides exact reference.
The iterative FFT technique uses:
  1. Linear approximation (C*theta = 0) as initial guess
  2. Picard iteration: compute C*theta in time domain, FFT to frequency
     domain, re-solve the ODE in Laplace/frequency domain, IFFT back

Parameters from the paper:
  C0     = 1.0e-6 mol/cm^3    (inlet concentration)
  k1     = 1.0e4  cm^3/(mol*s) (forward rate constant)
  k2     = 0.1    s^-1         (backward rate constant)
  V      = 0.1    cm/s         (interstitial velocity)
  L      = 10     cm           (column length)
  alpha  = 1.0e-5 mol/cm^3    (max adsorption capacity per unit void volume)
  tau    = 5      s            (lag time for modified step input)
  sigma  = 0.2    s            (std dev of smoothed step edges)
  te     = 300    s            (time when input goes back to zero)
  T      = 400    s            (half-period for FFT)
  N      = 1024                (FFT sample points)

Table 1 comparison times: t_n = t - 105 where 105 = L/V + tau
  (i.e., time after a non-adsorbed component would have emerged)

Usage:
  python scripts/chen_hsu_reproduction.py
  python scripts/chen_hsu_reproduction.py --output-dir artifacts/chen_hsu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import i0, erfc

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Parameters from Chen & Hsu 1989
# ---------------------------------------------------------------------------

C0 = 1.0e-6       # mol/cm^3
K1 = 1.0e4         # cm^3/(mol*s)
K2 = 0.1           # s^-1
V = 0.1            # cm/s
L = 10.0           # cm
ALPHA = 1.0e-5     # mol/cm^3
TAU = 5.0          # s (lag time / mean of normal distribution)
SIGMA = 0.2        # s (std dev of normal distribution)
TE = 300.0         # s (time when input goes back to zero)
T_HALF = 400.0     # s (half-period for FFT)
N_FFT = 1024       # FFT sample points
T_OFFSET = L / V + TAU  # = 105 s

# Derived dimensionless parameters for analytical solution (Eqs. 15-17)
R_PARAM = 1.0 - C0 * K1 / K2       # Eq. 15: r = 1 - C0*k1/k2
N_PARAM = ALPHA * K1 * L / V       # Eq. 16: n = alpha*k1*L/V
M_PARAM_DENOM = ALPHA * L           # For Eq. 17 denominator: alpha*L


# ---------------------------------------------------------------------------
# Analytical solution: Thomas model (Hiester & Vermeulen 1952; Chase 1984)
# ---------------------------------------------------------------------------

def thomas_J(a, b):
    """Compute J(a,b) from Eq. 18 via numerical integration.

    J(a, b) = 1 - e^{-b} * integral_0^a e^{-x} * I_0(2*sqrt(b*x)) dx

    Uses high-order numerical quadrature for accuracy matching the paper.
    The asymptotic form (Eq. 19) is available for very large a*b but
    is not used here to avoid approximation errors at moderate arguments.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.zeros_like(a)

    for idx in np.ndindex(a.shape):
        ai, bi = a[idx], b[idx]
        if ai <= 0 or bi <= 0:
            result[idx] = 0.0 if bi <= 0 else 1.0
            continue

        # Numerical integration of Eq. 18 using scipy for accuracy
        from scipy.integrate import quad
        def integrand(x):
            if x <= 0:
                return 1.0  # I_0(0) = 1, e^0 = 1
            return np.exp(-x) * float(i0(2.0 * np.sqrt(bi * x)))

        integral, _ = quad(integrand, 0, ai, limit=200)
        result[idx] = 1.0 - np.exp(-bi) * integral

    return result


def analytical_solution(t_n):
    """Compute normalized C(t_n, L)/C0 using the Thomas analytical solution.

    t_n is the time measured after a non-adsorbed component would have
    emerged from the column (i.e., t_n = t - L/V - tau).

    From Eq. 14:
      C/C0 = J(n/r, nM) / [J(n/r, nM) + (1 - J(n, nM/r)) * exp((1-r^{-1})(n - nM))]

    where:
      r = 1 - C0*k1/k2            (Eq. 15)
      n = alpha*k1*L/V             (Eq. 16)
      M = t_n * (k2/k1 + C0)*V / (alpha*L)  (Eq. 17)
    """
    t_n = np.asarray(t_n, dtype=float)
    result = np.zeros_like(t_n)

    r = R_PARAM
    n = N_PARAM

    for i, tn in enumerate(t_n.flat):
        if tn <= 0:
            result.flat[i] = 0.0
            continue

        M = tn * (K2 / K1 + C0) * V / (ALPHA * L)

        a1 = n / r
        b1 = n * M
        a2 = n
        b2 = n * M / r

        J1 = thomas_J(np.array([a1]), np.array([b1]))[0]
        J2 = thomas_J(np.array([a2]), np.array([b2]))[0]

        exp_arg = (1.0 - 1.0 / r) * (n - n * M)
        # Clamp to avoid overflow
        exp_arg = np.clip(exp_arg, -500, 500)
        exp_term = np.exp(exp_arg)

        denom = J1 + (1.0 - J2) * exp_term
        if abs(denom) < 1e-30:
            result.flat[i] = 0.0
        else:
            result.flat[i] = J1 / denom

    return result


# ---------------------------------------------------------------------------
# Chen & Hsu iterative FFT method
# ---------------------------------------------------------------------------

def modified_step_input(t, C0, tau, sigma, te):
    """Construct the modified step input g(t) from Eq. 20.

    g(t) = C0 * exp(-0.5*((t-tau)/sigma)^2)     for 0 <= t < tau
         = C0                                     for tau <= t < te - tau
         = C0 * exp(-0.5*((t-te+tau)/sigma)^2)   for te-tau <= t < te
         = 0                                       for te <= t < 2T
    """
    g = np.zeros_like(t)
    for i, ti in enumerate(t):
        if 0 <= ti < tau:
            g[i] = C0 * np.exp(-0.5 * ((ti - tau) / sigma) ** 2)
        elif tau <= ti < te - tau:
            g[i] = C0
        elif te - tau <= ti < te:
            g[i] = C0 * np.exp(-0.5 * ((ti - te + tau) / sigma) ** 2)
        # else: g[i] = 0 (already initialized)
    return g


def chen_hsu_iterative_fft(n_iterations=2, n_z_points=6):
    """Implement the Chen & Hsu iterative FFT technique.

    The method solves Eq. 9 by discretizing the z-integral using n_z_points
    quadrature nodes (the paper uses 6). At each iteration:
      1. Compute C(t,z) and theta(t,z) at each z node in time domain
      2. Form C*theta at each z node, FFT to frequency domain
      3. Numerically integrate Eq. 9 in the frequency domain
      4. IFFT back to time domain

    Returns dict with time grid and concentration at each iteration stage.
    """
    dt = 2.0 * T_HALF / N_FFT
    t = np.arange(N_FFT) * dt  # Time grid: 0 to 2T
    omega = np.fft.fftfreq(N_FFT, d=dt) * 2 * np.pi  # Angular frequencies

    # Modified step input
    g = modified_step_input(t, C0, TAU, SIGMA, TE)

    # FFT of input -> frequency domain G(s=iw)
    G_freq = np.fft.fft(g) * dt  # Approximate Laplace transform at s=iw

    s_freq = 1j * omega  # s = i*omega on imaginary axis
    s_safe = s_freq.copy()
    s_safe[0] = 1e-30

    lambda1 = (1.0 / V) * (s_safe + K1 * ALPHA * s_safe / (K2 + s_safe))
    lambda2 = K1 * ALPHA * s_safe / (V * (K2 + s_safe))

    # --- Linear approximation (C*theta = 0) ---
    # Eqs. 12-13: solution at each z
    z_nodes = np.linspace(0, L, n_z_points + 1)  # Include z=0 and z=L

    # Compute linear C and theta at all z nodes
    C_z_linear = np.zeros((n_z_points + 1, N_FFT))  # C(z_i, t)
    theta_z_linear = np.zeros((n_z_points + 1, N_FFT))

    for iz, z in enumerate(z_nodes):
        C_freq_z = G_freq * np.exp(-lambda1 * z)
        C_z_linear[iz, :] = np.maximum(np.real(np.fft.ifft(C_freq_z / dt)), 0.0)

        theta_freq_z = C_freq_z * K1 / (K2 + s_safe)
        theta_z_linear[iz, :] = np.clip(np.real(np.fft.ifft(theta_freq_z / dt)), 0.0, 1.0)

    # Linear solution at outlet (z = L)
    C_linear = C_z_linear[-1, :]

    results = {
        "t": t,
        "t_offset": T_OFFSET,
        "linear": C_linear / C0,
        "iterations": [],
    }

    # --- Iterative correction ---
    C_z_current = C_z_linear.copy()
    theta_z_current = theta_z_linear.copy()

    for iteration in range(n_iterations):
        # Compute C*theta at each z node and FFT to frequency domain
        C_theta_freq_z = np.zeros((n_z_points + 1, N_FFT), dtype=complex)
        for iz in range(n_z_points + 1):
            C_theta = C_z_current[iz, :] * theta_z_current[iz, :]
            C_theta_freq_z[iz, :] = np.fft.fft(C_theta) * dt

        # Evaluate Eq. 9 at z = L using numerical quadrature over z.
        # C_bar(s, L) = G(s)*exp(-lambda1*L)
        #   + exp(-lambda1*L) * integral_0^L exp(lambda1*z') * lambda2 * C_theta_bar(s,z') dz'
        #
        # The integrand at z' is: exp(lambda1*z') * lambda2 * C_theta_bar(s,z')
        # Use the trapezoidal rule with the z_nodes.

        integral_freq = np.zeros(N_FFT, dtype=complex)
        for iz in range(n_z_points):
            z1, z2 = z_nodes[iz], z_nodes[iz + 1]
            dz = z2 - z1

            f1 = np.exp(lambda1 * z1) * lambda2 * C_theta_freq_z[iz, :]
            f2 = np.exp(lambda1 * z2) * lambda2 * C_theta_freq_z[iz + 1, :]
            integral_freq += 0.5 * (f1 + f2) * dz

        # Full solution at z = L
        exp_minus_L = np.exp(-lambda1 * L)
        C_freq_L = G_freq * exp_minus_L + exp_minus_L * integral_freq

        # IFFT to time domain at z=L
        C_L_new = np.maximum(np.real(np.fft.ifft(C_freq_L / dt)), 0.0)

        # Update C and theta at ALL z nodes for next iteration
        C_z_new = np.zeros((n_z_points + 1, N_FFT))
        theta_z_new = np.zeros((n_z_points + 1, N_FFT))

        for iz, z in enumerate(z_nodes):
            # Compute integral up to z (not just L)
            int_z = np.zeros(N_FFT, dtype=complex)
            for jz in range(iz):
                z1, z2 = z_nodes[jz], z_nodes[jz + 1]
                dz = z2 - z1
                f1 = np.exp(lambda1 * z1) * lambda2 * C_theta_freq_z[jz, :]
                f2 = np.exp(lambda1 * z2) * lambda2 * C_theta_freq_z[jz + 1, :]
                int_z += 0.5 * (f1 + f2) * dz

            exp_minus_z = np.exp(-lambda1 * z)
            C_freq_z = G_freq * exp_minus_z + exp_minus_z * int_z

            C_z_new[iz, :] = np.maximum(np.real(np.fft.ifft(C_freq_z / dt)), 0.0)

            theta_freq_z = C_freq_z * K1 / (K2 + s_safe)
            theta_z_new[iz, :] = np.clip(np.real(np.fft.ifft(theta_freq_z / dt)), 0.0, 1.0)

        results["iterations"].append(C_L_new / C0)

        C_z_current = C_z_new.copy()
        theta_z_current = theta_z_new.copy()

    return results


# ---------------------------------------------------------------------------
# Table 1 reproduction
# ---------------------------------------------------------------------------

# Table 1 reference data from the paper
TABLE1_TIMES = np.array([125, 150, 175, 200, 225, 250, 275, 300, 325, 350], dtype=float)
TABLE1_LINEAR = np.array([0.011, 0.092, 0.273, 0.502, 0.705, 0.846, 0.928, 0.969, 0.988, 0.996])
TABLE1_ITER1 = np.array([0.011, 0.102, 0.313, 0.578, 0.794, 0.923, 0.980, 0.999, 1.002, 1.002])
TABLE1_ITER2 = np.array([0.011, 0.103, 0.316, 0.582, 0.797, 0.922, 0.977, 0.996, 1.000, 1.000])
TABLE1_ANALYTICAL = np.array([0.011, 0.102, 0.316, 0.580, 0.790, 0.912, 0.967, 0.989, 0.997, 1.000])


def reproduce_table1(output_dir: Path):
    """Reproduce Table 1 from Chen & Hsu 1989."""
    print("=" * 78)
    print("REPRODUCTION: Chen & Hsu (1989), AIChE J. 35(2), Table 1")
    print("=" * 78)

    # 1. Compute analytical solution at Table 1 times
    # t_n = t - 105 (time after non-adsorbed component emerges)
    # The Table 1 times are absolute times, and t_n = t - 105
    t_n_table = TABLE1_TIMES - T_OFFSET
    C_analytical = analytical_solution(t_n_table)

    print(f"\nParameters:")
    print(f"  C0 = {C0:.1e} mol/cm^3")
    print(f"  k1 = {K1:.1e} cm^3/(mol*s)")
    print(f"  k2 = {K2} s^-1")
    print(f"  V  = {V} cm/s")
    print(f"  L  = {L} cm")
    print(f"  alpha = {ALPHA:.1e} mol/cm^3")
    print(f"  t_offset = {T_OFFSET} s")
    print(f"  r = {R_PARAM:.4f}")
    print(f"  n = {N_PARAM:.1f}")

    # 2. Run iterative FFT
    fft_results = chen_hsu_iterative_fft(n_iterations=2)

    # Interpolate FFT results to Table 1 times
    t_fft = fft_results["t"]
    C_linear_interp = np.interp(TABLE1_TIMES, t_fft, fft_results["linear"])
    C_iter1_interp = np.interp(TABLE1_TIMES, t_fft, fft_results["iterations"][0])
    C_iter2_interp = np.interp(TABLE1_TIMES, t_fft, fft_results["iterations"][1])

    # 3. Print comparison table
    print(f"\n{'':=<78}")
    print(f"{'Table 1: Normalized concentration C/C0':^78}")
    print(f"{'':=<78}")
    print(f"{'':>6} | {'--- FFT Technique (This Work) ---':^36} | {'--- Reference ---':^22}")
    print(f"{'Time':>6} | {'Linear':>8} {'Iter 1':>8} {'Iter 2':>8} | "
          f"{'Analyt.':>8} | {'Paper':>6} {'Paper':>6}")
    print(f"{'(s)':>6} | {'Approx.':>8} {'':>8} {'':>8} | "
          f"{'Soln.':>8} | {'Iter2':>6} {'Analyt':>6}")
    print(f"{'-'*6}-+-{'-'*36}-+-{'-'*8}-+-{'-'*14}")

    max_err_analytical = 0.0
    max_err_paper_iter2 = 0.0

    for i, t_val in enumerate(TABLE1_TIMES):
        our_lin = C_linear_interp[i]
        our_it1 = C_iter1_interp[i]
        our_it2 = C_iter2_interp[i]
        our_anl = C_analytical[i]
        ref_it2 = TABLE1_ITER2[i]
        ref_anl = TABLE1_ANALYTICAL[i]

        err_anl = abs(our_anl - ref_anl)
        err_it2 = abs(our_it2 - ref_it2)
        max_err_analytical = max(max_err_analytical, err_anl)
        max_err_paper_iter2 = max(max_err_paper_iter2, err_it2)

        print(f"{t_val:6.0f} | {our_lin:8.3f} {our_it1:8.3f} {our_it2:8.3f} | "
              f"{our_anl:8.3f} | {ref_it2:6.3f} {ref_anl:6.3f}")

    print(f"{'-'*6}-+-{'-'*36}-+-{'-'*8}-+-{'-'*14}")
    print(f"\nMax |our analytical - paper analytical|: {max_err_analytical:.4f}")
    print(f"Max |our FFT iter2  - paper FFT iter2 |: {max_err_paper_iter2:.4f}")

    # 4. Detailed comparison
    print(f"\n{'FFT Iteration Validation':=^78}")
    rms_err_fft = np.sqrt(np.mean((C_iter2_interp - TABLE1_ITER2) ** 2))
    print(f"  FFT Iter 2 vs paper Iter 2:")
    print(f"    RMS error: {rms_err_fft:.4f}")
    print(f"    Max error: {max_err_paper_iter2:.4f}")
    if max_err_paper_iter2 < 0.01:
        print(f"    STATUS: PASS (max error < 1%)")
    else:
        print(f"    STATUS: CLOSE (max error = {max_err_paper_iter2:.1%})")

    rms_err_anl = np.sqrt(np.mean((C_analytical - TABLE1_ANALYTICAL) ** 2))
    print(f"\n  Analytical solution vs paper:")
    print(f"    RMS error: {rms_err_anl:.4f}")
    print(f"    Max error: {max_err_analytical:.4f}")
    print(f"    Note: Paper used Eq. 19 asymptotic for J(a,b) when a*b>36;")
    print(f"    we use full numerical quadrature. Differences are expected.")

    # 5. Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full breakthrough curves
    np.savez(
        output_dir / "chen_hsu_fft_results.npz",
        t=t_fft,
        linear=fft_results["linear"],
        iter1=fft_results["iterations"][0],
        iter2=fft_results["iterations"][1],
    )

    # Save analytical solution on fine grid
    t_n_fine = np.linspace(1, 300, 500)
    C_analytical_fine = analytical_solution(t_n_fine)
    np.savez(
        output_dir / "chen_hsu_analytical.npz",
        t_n=t_n_fine,
        t_absolute=t_n_fine + T_OFFSET,
        C_norm=C_analytical_fine,
    )

    # Save table comparison as JSON
    table_data = {
        "times": TABLE1_TIMES.tolist(),
        "our_linear": C_linear_interp.tolist(),
        "our_iter1": C_iter1_interp.tolist(),
        "our_iter2": C_iter2_interp.tolist(),
        "our_analytical": C_analytical.tolist(),
        "paper_linear": TABLE1_LINEAR.tolist(),
        "paper_iter1": TABLE1_ITER1.tolist(),
        "paper_iter2": TABLE1_ITER2.tolist(),
        "paper_analytical": TABLE1_ANALYTICAL.tolist(),
        "rms_error_analytical": float(rms_err_anl),
        "max_error_analytical": float(max_err_analytical),
        "max_error_fft_iter2": float(max_err_paper_iter2),
    }
    with open(output_dir / "chen_hsu_table1_comparison.json", "w") as f:
        json.dump(table_data, f, indent=2)
    print(f"\n  Saved results to {output_dir}")

    return table_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1 from Chen & Hsu (1989)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts/chen_hsu"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    reproduce_table1(args.output_dir)


if __name__ == "__main__":
    main()
