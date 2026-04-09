# NL-NILT: Nonlinear Numerical Inverse Laplace Transform for Chromatography

Extends the FFT-based Numerical Inverse Laplace Transform (FFT-NILT) to nonlinear binding isotherms via iterative re-linearization of the General Rate Model (GRM) transfer function.

Compatible with [CADET](https://github.com/cadet/CADET-Core) v6 for reference solutions and validation.

## Method

FFT-NILT provides significant speedup for linear chromatographic transport but fails for nonlinear binding because the bilinear kinetic term destroys the multiplicative transfer-function structure. The iterative re-linearization overcomes this by:

1. Estimating the operating concentration from the current breakthrough curve
2. Re-linearizing the Langmuir/SMA binding with effective parameters
3. Evaluating the re-linearized GRM transfer function via CFL-informed FFT-NILT

Re-linearization shifts the kinetic pole further into the left half-plane, improving CFL feasibility under loaded conditions.

## Key Results

- **7x error reduction** vs linear approximation, validated against Thomas analytical solution
- **0.7–8.5% relative L² error** vs CADET across 30 benchmark configurations (Pe 2.5–1000)
- Error scales primarily with bounded nonlinearity NL\* = Kₐcfeed / (1 + Kₐcfeed)
- Empirical regime: NL\* < 0.3 → L² < 3% (reliable); NL\* > 0.7 → L² > 8% (use FV/DG)
- **~58 ms** per evaluation, independent of column discretization
- Generalizes to single-component SMA binding (ν ≤ 2, cₛ ≥ 0.15 mol/m³)

## Installation

```bash
pip install numpy scipy h5py
```

For CADET reference comparisons, install [CADET v6](https://cadet.github.io).

## Quick Start

```python
from cadet_lab.nilt.nonlinear import LangmuirBinding, nl_nilt_solve

binding = LangmuirBinding(
    ka=1.0, kd=5.0, qmax=10.0,
    velocity=1e-3, dispersion=1e-6, length=0.1,
    col_porosity=0.37, par_radius=1e-5, par_porosity=0.33,
    film_diffusion=1e-5, pore_diffusion=1e-10,
)

result = nl_nilt_solve(binding, t_end=1500.0, c_feed=5.0)
print(f"Converged: {result.converged}, iterations: {result.n_iterations}")
# result.t, result.c contain the breakthrough curve
```

## Reproducing Benchmark Results

All benchmark results are in `artifacts/` as JSON files. To regenerate from scratch (requires [CADET](https://github.com/cadet/CADET-Core)):

```bash
# Langmuir parameter sweep (30 cases across 5 transport configurations)
python -c "
from cadet_lab.nilt.error_scaling import run_langmuir_sweep
from pathlib import Path
results = run_langmuir_sweep(cadet_cli='/path/to/cadet-cli', output_dir=Path('output'))
"

# SMA benchmarks (salt, charge, and feed sweeps)
python -c "
from cadet_lab.nilt.error_scaling import run_sma_benchmark
from pathlib import Path
results = run_sma_benchmark(cadet_cli='/path/to/cadet-cli', output_dir=Path('output'))
"

# Chen & Hsu (1989) analytical validation (no CADET needed)
python scripts/chen_hsu_reproduction.py
```

## Repository Structure

```
cadet_lab/nilt/           Core NL-NILT implementation
  nonlinear.py            Iterative re-linearization engine (Langmuir, SMA)
  benchmarks.py           GRM transfer functions
  solver.py               High-level solver API
  error_scaling.py        Dimensionless groups, parameter sweeps, regime analysis
  mass_balance.py         Reference-free convergence diagnostics
  vendor/                 Vendored FFT-NILT core (CFL-tuned parameter selection)
scripts/                  Benchmark and reproduction scripts
artifacts/                Pre-computed results and figures
tests/                    Test suite
```

## Software Versions

- Python 3.12, NumPy 1.26, SciPy 1.14, h5py 3.11
- [CADET](https://github.com/cadet/CADET-Core) v6.0.0 (for reference solutions)

## References

- G. Pavlov, "Systematic parameter selection for FFT-based NILT: CFL-informed tuning rules and quality diagnostics," *Chem. Eng. Sci.* 328 (2026) 123776. [DOI: 10.1016/j.ces.2026.123776](https://doi.org/10.1016/j.ces.2026.123776)
- E. von Lieres, J. Andersson, "A fast and accurate solver for the general rate model of column liquid chromatography," *Comput. Chem. Eng.* 34 (2010) 1180–1191.
- T.-L. Chen, J.-T. Hsu, "Application of the fast Fourier transform to nonlinear fixed-bed adsorption problems," *AIChE J.* 35 (1989) 332–334.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
