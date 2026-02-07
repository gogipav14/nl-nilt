#!/usr/bin/env python3
"""Generate all figures for the NL-NILT CACE paper.

Reads benchmark data from artifacts/nl_nilt_benchmarks/ and produces
publication-quality figures using matplotlib.

Figures:
  1. Pole shift diagram (analytical)
  2. Iteration convergence (Class B Langmuir)
  3. L2 error comparison across loading levels
  4. Mass-balance diagnostic traces per iteration
  5. Breakthrough curves: NL-NILT vs CADET for multiple loading levels
  6. Error localization: pointwise |c_NL - c_CADET|

Usage:
  # First generate benchmark data:
  python scripts/run_nl_nilt_benchmarks.py --cadet-cli /path/to/cadet-cli
  # Then generate figures:
  python notebooks/nl_nilt_paper_figures.py
  python notebooks/nl_nilt_paper_figures.py --data-dir artifacts/nl_nilt_benchmarks --output-dir artifacts/figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Style configuration for publication
# ---------------------------------------------------------------------------

def setup_style():
    """Configure matplotlib for CACE publication quality."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })


# Color scheme (colorblind-friendly)
COLORS = {
    "nilt_nl": "#0072B2",       # Blue
    "nilt_lin": "#56B4E9",      # Light blue
    "cadet": "#D55E00",         # Red-orange
    "iter0": "#009E73",         # Green
    "iter1": "#0072B2",         # Blue
    "iter2": "#CC79A7",         # Pink
    "error": "#E69F00",         # Yellow-orange
    "pole_linear": "#D55E00",   # Red
    "pole_relin": "#0072B2",    # Blue
}


# ---------------------------------------------------------------------------
# Figure 1: Pole shift diagram
# ---------------------------------------------------------------------------

def figure1_pole_shift(data_dir: Path, output_dir: Path):
    """Pole shift diagram showing CFL improvement with re-linearization."""
    data = np.load(data_dir / "pole_shift_data.npz")
    c_op = data["c_operating"]
    pole_lin = data["pole_linear"]
    pole_relin = data["poles_relin"]
    K_eq_eff = data["K_eq_eff"]
    ka = float(data["ka"])
    kd = float(data["kd"])
    qmax = float(data["qmax"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: Pole positions
    ax1.axhline(y=pole_lin[0], color=COLORS["pole_linear"], ls="--", lw=1.5,
                label=f"Linear pole $s = -k_d = {-kd:.1f}$")
    ax1.plot(c_op, pole_relin, color=COLORS["pole_relin"], lw=2,
             label=r"Re-linearized $s = -(k_d + k_a c_0)$")
    ax1.set_xlabel(r"Operating concentration $c_0$")
    ax1.set_ylabel(r"Pole location $s^*$")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_title("(a) Pole shift with loading")

    # Right: Effective K_eq
    ax2.plot(c_op, K_eq_eff, color=COLORS["nilt_nl"], lw=2)
    ax2.axhline(y=ka * qmax / kd, color=COLORS["pole_linear"], ls="--", lw=1,
                label=f"$K_{{eq}}^{{lin}} = {ka*qmax/kd:.0f}$")
    ax2.set_xlabel(r"Operating concentration $c_0$")
    ax2.set_ylabel(r"Effective $K_{eq}^{eff}$")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_title("(b) Effective equilibrium constant")

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_pole_shift.pdf")
    fig.savefig(output_dir / "fig1_pole_shift.png")
    plt.close(fig)
    print("  Fig 1: Pole shift diagram")


# ---------------------------------------------------------------------------
# Figure 2: Iteration convergence
# ---------------------------------------------------------------------------

def figure2_iteration_convergence(data_dir: Path, output_dir: Path):
    """Iteration convergence: c^0 -> c^1 -> c^2 vs CADET for Class B."""
    # Load iteration data
    lin_data = np.load(data_dir / "convergence_linear.npz")
    t = lin_data["t"]
    c_lin = lin_data["c"]

    # Load available iteration curves
    iter_curves = []
    for i in range(10):
        path = data_dir / f"convergence_iter{i}.npz"
        if path.exists():
            d = np.load(path)
            iter_curves.append(d["c"])

    # Load CADET if available
    cadet_path = data_dir / "cadet_breakthrough_cfeed5.0.npz"
    has_cadet = cadet_path.exists()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Plot linear baseline
    ax.plot(t, c_lin, color=COLORS["nilt_lin"], ls="--", lw=1.2,
            label=r"Linear ($c^0$)", alpha=0.8)

    # Plot first few iterations
    iter_labels = [r"Iter 1 ($c^1$)", r"Iter 2 ($c^2$)", r"Iter 3 ($c^3$)"]
    iter_colors = [COLORS["iter0"], COLORS["iter1"], COLORS["iter2"]]
    for i, c_iter in enumerate(iter_curves[:3]):
        ax.plot(t, c_iter, color=iter_colors[i % 3], lw=1.5,
                label=iter_labels[i] if i < 3 else f"Iter {i+1}")

    if has_cadet:
        cadet = np.load(cadet_path)
        ax.plot(cadet["t"], cadet["c"], "k:", lw=1.5, label="CADET (reference)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Outlet concentration (mol/m$^3$)")
    ax.legend(fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title("Iteration convergence (Class B Langmuir)")

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_iteration_convergence.pdf")
    fig.savefig(output_dir / "fig2_iteration_convergence.png")
    plt.close(fig)
    print("  Fig 2: Iteration convergence")


# ---------------------------------------------------------------------------
# Figure 3: L2 error comparison across loading levels
# ---------------------------------------------------------------------------

def figure3_l2_error_comparison(data_dir: Path, output_dir: Path):
    """L2 error: NL-NILT vs linear NILT vs CADET across loading levels."""
    with open(data_dir / "loading_sweep_results.json") as f:
        sweep = json.load(f)

    results = sweep["results"]
    c_feeds = [r["c_feed"] for r in results]
    loadings = [r["loading_ratio"] for r in results]
    n_iters = [r["n_iterations"] for r in results]
    l2_linear = [r.get("l2_error_linear", 0) for r in results]

    # Check if CADET data available
    l2_cadet = [r.get("l2_error_vs_cadet", "nan") for r in results]
    has_cadet = any(v != "nan" and v != float("nan") for v in l2_cadet
                    if isinstance(v, (int, float)))
    l2_cadet = [float(v) if v != "nan" else np.nan for v in l2_cadet]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))

    # Left: L2 errors
    x = np.arange(len(c_feeds))
    width = 0.35

    ax1.bar(x - width / 2, [v * 100 for v in l2_linear], width,
            color=COLORS["nilt_lin"], label="NL-NILT vs Linear NILT", alpha=0.9)
    if has_cadet:
        ax1.bar(x + width / 2, [v * 100 if not np.isnan(v) else 0 for v in l2_cadet],
                width, color=COLORS["cadet"], label="NL-NILT vs CADET", alpha=0.9)

    ax1.set_xlabel(r"Feed concentration $c_{feed}$")
    ax1.set_ylabel("Relative L2 error (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{cf:.1f}" for cf in c_feeds], fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_title("(a) Accuracy")

    # Right: Iteration count
    ax2.bar(x, n_iters, color=COLORS["nilt_nl"], alpha=0.9)
    ax2.set_xlabel(r"Feed concentration $c_{feed}$")
    ax2.set_ylabel("Number of iterations")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{cf:.1f}" for cf in c_feeds], fontsize=8)
    ax2.set_ylim(0, max(n_iters) + 1)
    ax2.set_title("(b) Convergence")

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_l2_error_comparison.pdf")
    fig.savefig(output_dir / "fig3_l2_error_comparison.png")
    plt.close(fig)
    print("  Fig 3: L2 error comparison")


# ---------------------------------------------------------------------------
# Figure 4: Mass-balance diagnostic traces
# ---------------------------------------------------------------------------

def figure4_mass_balance_traces(data_dir: Path, output_dir: Path):
    """Mass-balance diagnostic traces (delta_F0 and delta_mu1) per iteration."""
    with open(data_dir / "convergence_traces.json") as f:
        traces = json.load(f)

    results = traces["results"]
    iters = [r["iteration"] for r in results]
    residuals = [r["residual_norm"] for r in results]
    contractions = [r["contraction"] for r in results]
    delta_F0 = [r["delta_F0"] for r in results]
    delta_mu1 = [r["delta_mu1"] for r in results]
    c_ops = [r["c_operating"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))

    # (a) Residual norm
    ax = axes[0, 0]
    ax.semilogy(iters, residuals, "o-", color=COLORS["nilt_nl"], markersize=5)
    ax.set_ylabel("Residual norm")
    ax.set_title("(a) Iteration residual")
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (b) Contraction factor
    ax = axes[0, 1]
    valid_kappa = [(i, k) for i, k in zip(iters, contractions) if k < 100 and k > 0]
    if valid_kappa:
        ax.plot([v[0] for v in valid_kappa], [v[1] for v in valid_kappa],
                "s-", color=COLORS["iter1"], markersize=5)
    ax.axhline(y=1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_ylabel(r"Contraction $\kappa$")
    ax.set_title(r"(b) Contraction factor")
    ax.set_ylim(0, max(2, max(v[1] for v in valid_kappa) * 1.2) if valid_kappa else 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # (c) delta_F0
    ax = axes[1, 0]
    ax.semilogy(iters, delta_F0, "^-", color=COLORS["error"], markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|\delta F(0)|$")
    ax.set_title(r"(c) Zeroth moment deviation")

    # (d) Operating concentration
    ax = axes[1, 1]
    ax.plot(iters, c_ops, "D-", color=COLORS["iter2"], markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$c_{op}$ (mol/m$^3$)")
    ax.set_title("(d) Operating concentration")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_mass_balance_traces.pdf")
    fig.savefig(output_dir / "fig4_mass_balance_traces.png")
    plt.close(fig)
    print("  Fig 4: Mass-balance diagnostic traces")


# ---------------------------------------------------------------------------
# Figure 5: Breakthrough curves
# ---------------------------------------------------------------------------

def figure5_breakthrough_curves(data_dir: Path, output_dir: Path):
    """Breakthrough curves: NL-NILT vs CADET for 4 loading levels."""
    c_feed_targets = [0.5, 2.5, 5.0, 10.0]
    labels = ["A: $c_f$=0.5", "B: $c_f$=2.5", "B: $c_f$=5.0", "C: $c_f$=10.0"]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))

    for idx, (cf, label) in enumerate(zip(c_feed_targets, labels)):
        ax = axes[idx // 2, idx % 2]

        # NL-NILT data
        nilt_path = data_dir / f"nlnilt_breakthrough_cfeed{cf:.1f}.npz"
        if nilt_path.exists():
            d = np.load(nilt_path)
            ax.plot(d["t"], d["c"] / cf, color=COLORS["nilt_nl"], lw=1.5,
                    label="NL-NILT")
            ax.plot(d["t"], d["c_lin"] / cf, color=COLORS["nilt_lin"], ls="--",
                    lw=1, label="Linear NILT", alpha=0.7)

        # CADET data
        cadet_path = data_dir / f"cadet_breakthrough_cfeed{cf:.1f}.npz"
        if cadet_path.exists():
            d = np.load(cadet_path)
            ax.plot(d["t"], d["c"] / cf, ":", color=COLORS["cadet"], lw=1.5,
                    label="CADET")

        ax.set_title(label, fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        if idx >= 2:
            ax.set_xlabel("Time (s)")
        if idx % 2 == 0:
            ax.set_ylabel(r"$c/c_{feed}$")
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_breakthrough_curves.pdf")
    fig.savefig(output_dir / "fig5_breakthrough_curves.png")
    plt.close(fig)
    print("  Fig 5: Breakthrough curves")


# ---------------------------------------------------------------------------
# Figure 6: Error localization
# ---------------------------------------------------------------------------

def figure6_error_localization(data_dir: Path, output_dir: Path):
    """Pointwise |c_NL - c_CADET| showing error concentrated at shock front."""
    c_feed_targets = [1.0, 5.0, 10.0]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    colors_list = [COLORS["iter0"], COLORS["nilt_nl"], COLORS["iter2"]]

    for cf, color in zip(c_feed_targets, colors_list):
        nilt_path = data_dir / f"nlnilt_breakthrough_cfeed{cf:.1f}.npz"
        cadet_path = data_dir / f"cadet_breakthrough_cfeed{cf:.1f}.npz"

        if not (nilt_path.exists() and cadet_path.exists()):
            continue

        nilt_d = np.load(nilt_path)
        cadet_d = np.load(cadet_path)

        # Interpolate to common grid
        t_min = max(nilt_d["t"][0], cadet_d["t"][0])
        t_max = min(nilt_d["t"][-1], cadet_d["t"][-1])
        t_common = np.linspace(t_min, t_max, 500)

        c_nilt = np.interp(t_common, nilt_d["t"], nilt_d["c"])
        c_cadet = np.interp(t_common, cadet_d["t"], cadet_d["c"])

        error = np.abs(c_nilt - c_cadet)
        ax.plot(t_common, error, color=color, lw=1.2,
                label=f"$c_f = {cf:.0f}$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|c_{NL} - c_{CADET}|$ (mol/m$^3$)")
    ax.legend(fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title("Pointwise error localization")

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_error_localization.pdf")
    fig.savefig(output_dir / "fig6_error_localization.png")
    plt.close(fig)
    print("  Fig 6: Error localization")


# ---------------------------------------------------------------------------
# Figure 7: 4-way analytical comparison (Chen & Hsu plug-flow problem)
# ---------------------------------------------------------------------------

def figure7_analytical_comparison(data_dir: Path, output_dir: Path):
    """4-way comparison: Analytical vs Linear FFT vs Iterative FFT vs CADET."""
    data = np.load(data_dir / "analytical_comparison_data.npz")
    t_fine = data["t_fine"]
    C_analytical = data["C_analytical_fine"]
    T_OFFSET = float(data["T_OFFSET"])
    t_fft = data["t_fft"]
    C_linear = data["C_linear_fft"]
    C_iter1 = data["C_iter1_fft"]
    C_iter2 = data["C_iter2_fft"]
    C_iter3 = data["C_iter3_fft"]
    table_times = data["table_times"]

    # Check for CADET data
    cadet_path = data_dir / "analytical_comparison_cadet.npz"
    has_cadet = cadet_path.exists()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # --- (a) Breakthrough curves ---
    # Analytical (ground truth) - uses T_OFFSET=105 (FFT time convention)
    ax1.plot(t_fine, C_analytical, "k-", lw=2, label="Thomas analytical", zorder=5)

    # Linear FFT (C*theta=0 approximation)
    mask_fft = (t_fft >= 100) & (t_fft <= 420)
    ax1.plot(t_fft[mask_fft], C_linear[mask_fft], color=COLORS["nilt_lin"],
             ls="--", lw=1.3, label="Linear FFT", alpha=0.8)

    # Iterative FFT (NL-FFT, 2 iterations)
    ax1.plot(t_fft[mask_fft], C_iter2[mask_fft], color=COLORS["nilt_nl"],
             lw=1.5, label="Iterative FFT (2 iter)")

    # CADET LRMWP (shift +5s to align with FFT time convention)
    if has_cadet:
        cadet_data = np.load(cadet_path)
        t_cadet = cadet_data["t"]
        C_cadet = cadet_data["C_norm"]
        # CADET uses sharp step (no 5s lag); shift to align with FFT convention
        t_cadet_shifted = t_cadet + 5.0
        mask_c = (t_cadet_shifted >= 100) & (t_cadet_shifted <= 420)
        ax1.plot(t_cadet_shifted[mask_c], C_cadet[mask_c], ":", color=COLORS["cadet"],
                 lw=1.8, label="CADET (LRMWP)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"$C/C_0$")
    ax1.set_xlim(100, 420)
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_title("(a) Breakthrough curves")

    # --- (b) Error vs analytical at Table 1 times ---
    C_analytical_table = data["C_analytical_table"]
    C_linear_table = data["C_linear_table"]
    C_iter1_table = data["C_iter1_table"]
    C_iter2_table = data["C_iter2_table"]
    C_iter3_table = data["C_iter3_table"]

    ax2.plot(table_times, np.abs(C_linear_table - C_analytical_table) * 100,
             "s-", color=COLORS["nilt_lin"], markersize=4, lw=1.2,
             label="Linear FFT")
    ax2.plot(table_times, np.abs(C_iter1_table - C_analytical_table) * 100,
             "^-", color=COLORS["iter0"], markersize=4, lw=1.2,
             label="Iter 1")
    ax2.plot(table_times, np.abs(C_iter2_table - C_analytical_table) * 100,
             "o-", color=COLORS["nilt_nl"], markersize=4, lw=1.2,
             label="Iter 2")
    ax2.plot(table_times, np.abs(C_iter3_table - C_analytical_table) * 100,
             "D-", color=COLORS["iter2"], markersize=4, lw=1.2,
             label="Iter 3")

    if has_cadet:
        cadet_data = np.load(cadet_path)
        t_cadet = cadet_data["t"]
        C_cadet = cadet_data["C_norm"]
        # Shift CADET to FFT time convention (+5s for lag offset), then
        # compute error at the FFT analytical reference
        C_cadet_table = np.interp(table_times, t_cadet + 5.0, C_cadet)
        ax2.plot(table_times, np.abs(C_cadet_table - C_analytical_table) * 100,
                 "x-", color=COLORS["cadet"], markersize=5, lw=1.2,
                 label="CADET")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$|C/C_0 - C_{analytical}/C_0|$ (%)")
    ax2.legend(fontsize=7)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_title("(b) Pointwise error vs analytical")

    fig.tight_layout()
    fig.savefig(output_dir / "fig7_analytical_comparison.pdf")
    fig.savefig(output_dir / "fig7_analytical_comparison.png")
    plt.close(fig)
    print("  Fig 7: 4-way analytical comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for the NL-NILT CACE paper"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./artifacts/nl_nilt_benchmarks"),
        help="Directory containing benchmark data (from run_nl_nilt_benchmarks.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts/figures"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["1", "2", "3", "4", "5", "6", "7", "all"],
        default=["all"],
        help="Which figures to generate",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    if "all" in args.figures:
        figures = ["1", "2", "3", "4", "5", "6", "7"]
    else:
        figures = args.figures

    print("=" * 60)
    print("NL-NILT Paper Figures")
    print("=" * 60)
    print(f"Data:   {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print()

    if "1" in figures:
        try:
            figure1_pole_shift(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 1 FAILED: {e}")

    if "2" in figures:
        try:
            figure2_iteration_convergence(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 2 FAILED: {e}")

    if "3" in figures:
        try:
            figure3_l2_error_comparison(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 3 FAILED: {e}")

    if "4" in figures:
        try:
            figure4_mass_balance_traces(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 4 FAILED: {e}")

    if "5" in figures:
        try:
            figure5_breakthrough_curves(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 5 FAILED: {e}")

    if "6" in figures:
        try:
            figure6_error_localization(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 6 FAILED: {e}")

    if "7" in figures:
        try:
            figure7_analytical_comparison(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  Fig 7 FAILED: {e}")

    print(f"\nFigures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
