# NL-NILT: Nonlinear Numerical Inverse Laplace Transform for Chromatography

Reproduction repository for:

> **Extending FFT-based numerical inverse Laplace transform to nonlinear chromatography via adaptive re-linearization**
>
> *Computers and Chemical Engineering* (submitted 2026)

## Overview

FFT-NILT provides 100-1600x speedup for linear chromatographic transport but fails for kinetic binding due to poles near the imaginary axis. This work overcomes that limitation via adaptive re-linearization: the binding isotherm is linearized around the current operating concentration at each iteration, producing a valid CFL-feasible transfer function. A single re-linearization step reduces error by 7x compared to the linear approximation.

**Key results:**
- 7.7x error reduction vs linear NILT, validated against Thomas analytical solution
- 0.7-7.6% L2 error vs CADET across dilute-to-overloaded Langmuir conditions
- Single-step convergence for all tested loading regimes
- Generalizes to SMA binding (2-3 iterations)

## Quick Start

```bash
# Clone and install
git clone https://github.com/gogipav14/nl-nilt.git
cd nl-nilt
pip install -e ".[dev,figures]"

# Run tests (no CADET needed)
pytest tests/ -q

# Reproduce all figures from precomputed data
make figures

# Compile the paper
make paper
```

## Full Reproduction (requires CADET)

To regenerate all benchmark data from scratch, you need [CADET](https://github.com/modsim/CADET) v6.0.0:

```bash
# Install CADET (conda recommended)
conda install -c conda-forge cadet

# Or build from source (see https://cadet.github.io/master/getting_started/installation.html)

# Set CADET path
export CADET_CLI_PATH=$(which cadet-cli)

# Reproduce everything: benchmarks -> figures -> paper
make all
```

### Step-by-step reproduction

```bash
# 1. Chen & Hsu (1989) Table 1 reproduction (no CADET needed)
python scripts/chen_hsu_reproduction.py

# 2. Run all NL-NILT benchmarks (CADET needed for reference solutions)
python scripts/run_nl_nilt_benchmarks.py --cadet-cli $(which cadet-cli)

# 3. Run without CADET (NL-NILT results only, no reference comparison)
python scripts/run_nl_nilt_benchmarks.py --cadet-cli $(which cadet-cli) --skip-cadet

# 4. Generate paper figures
python notebooks/nl_nilt_paper_figures.py

# 5. Compile paper
cd paper && pdflatex nl_nilt_paper.tex && pdflatex nl_nilt_paper.tex
```

## Repository Structure

```
nl-nilt/
├── cadet_lab/nilt/           # Core NL-NILT implementation
│   ├── nonlinear.py          #   Iterative re-linearization engine
│   ├── mass_balance.py       #   Reference-free convergence diagnostics
│   ├── benchmarks.py         #   GRM/Langmuir/SMA transfer functions
│   ├── solver.py             #   High-level solver API
│   └── vendor/               #   Vendored FFT-NILT core (nilt-cfl)
│       ├── nilt_fft.py       #     de Hoog FFT-NILT algorithm
│       └── tuner.py          #     CFL parameter selection
├── scripts/
│   ├── run_nl_nilt_benchmarks.py   # All paper benchmarks (Tables 2-4)
│   └── chen_hsu_reproduction.py    # Chen & Hsu 1989 validation (Table 1)
├── notebooks/
│   └── nl_nilt_paper_figures.py    # All paper figures (Figs 1-7)
├── tests/                    # 77 tests (pytest)
├── paper/
│   └── nl_nilt_paper.tex     # Manuscript (elsarticle/CACE format)
├── artifacts/
│   ├── nl_nilt_benchmarks/   # Precomputed benchmark data (.json, .npz)
│   └── figures/              # Generated figures (.pdf, .png)
├── pyproject.toml
├── Makefile
└── README.md
```

## Benchmarks

| Benchmark | Script | CADET needed | Paper reference |
|-----------|--------|:------------:|-----------------|
| Chen & Hsu Table 1 | `chen_hsu_reproduction.py` | Optional | Table 2 |
| Langmuir loading sweep | `run_nl_nilt_benchmarks.py --benchmark loading_sweep` | Yes | Table 3 |
| SMA binding | `run_nl_nilt_benchmarks.py --benchmark sma` | No | Table 4 |
| Convergence diagnostics | `run_nl_nilt_benchmarks.py --benchmark convergence` | Yes | Fig 4 |
| Analytical 4-way comparison | `run_nl_nilt_benchmarks.py --benchmark analytical_comparison` | Yes | Fig 7 |

## Dependencies

**Required** (pure Python, no CADET):
- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7

**For figures:**
- Matplotlib >= 3.5

**For CADET comparison benchmarks:**
- [CADET](https://github.com/modsim/CADET) >= 6.0.0
- h5py >= 3.0

## Citation

```bibtex
@article{nlnilt2026,
  title={Extending FFT-based numerical inverse Laplace transform to nonlinear
         chromatography via adaptive re-linearization},
  author={},
  journal={Computers and Chemical Engineering},
  year={2026},
  note={Submitted}
}
```

## Related Work

- **Paper 1 (CFL theory)**: FFT-NILT with CFL-informed parameter selection for linear transport. *Chemical Engineering Science* (2026, under review).
- **CADET**: [github.com/modsim/CADET](https://github.com/modsim/CADET) - Reference solver for chromatographic process simulation.
- **nilt-cfl**: [github.com/gogipav14/nilt-cfl](https://github.com/gogipav14/nilt-cfl) - Standalone CFL-tuned FFT-NILT library (vendored here).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
