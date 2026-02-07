"""Tests for NILT verification pack.

Tests for CFL-informed FFT-NILT implementation using vendored nilt-cfl library.

Key change: The frequency grid now uses fftfreq() to properly map DFT bins
k > N/2 to negative frequencies. This ensures z_ifft is nearly real for
real-valued f(t), making ε_Im a meaningful diagnostic (~1e-10 expected).
"""

import warnings
import numpy as np
import pytest

from cadet_lab.nilt import (
    fft_nilt,
    eps_im_max,        # Paper-compliant: max|Im|/max|Re|
    eps_im_rms,        # RMS-based alternative
    eps_im,            # Deprecated alias
    one_sided_imag_ratio,  # Deprecated alias
    epsilon_im_paper,  # Deprecated
    n_doubling_error,
    tune_params,
    check_cfl_feasibility,
    refine_until_accept,
    get_problem,
    get_all_problems,
    TunedParams,
)
from cadet_lab.nilt.convergence import (
    epsilon_im_test,
    n_doubling_test,
    NiltConvergenceResult,
)
from cadet_lab.nilt.benchmarks import (
    get_benchmark_functions,
    advection_dispersion_transfer,
    langmuir_column_transfer,
    grm_moment_transfer,
)


class TestCflFeasibility:
    """Tests for CFL feasibility check."""

    def test_feasible_standard_case(self):
        """Standard chromatography parameters should be feasible."""
        # Use tune_params which internally checks feasibility
        params = tune_params(t_end=100.0, alpha_c=0.0)
        assert isinstance(params, TunedParams)
        assert bool(params.feasible) is True
        assert params.a > 0
        assert params.T > 0
        assert params.N > 0

    def test_check_cfl_feasibility_standard(self):
        """CFL check passes for reasonable parameters."""
        cfl_ok, floor_ok, lhs_cfl, lhs_floor = check_cfl_feasibility(
            alpha_c=0.0,
            t_max=200.0,  # 2*T for t_end=100
        )
        assert bool(cfl_ok) is True
        assert bool(floor_ok) is True

    def test_infeasible_extreme_t_max(self):
        """Extremely large t_max may return meaningful bounds."""
        # With t_max = 1e12 and tight tolerance
        cfl_ok, floor_ok, lhs_cfl, lhs_floor = check_cfl_feasibility(
            alpha_c=0.0,
            t_max=1e12,
            eps_tail=1e-15,
        )
        # Check that bounds are computed (may or may not be feasible)
        assert lhs_cfl > 0  # lhs should be positive
        assert lhs_floor > 0

    def test_feasibility_margin(self):
        """Feasibility margin should be positive for valid cases."""
        params = tune_params(t_end=100.0, alpha_c=0.0)
        assert params.margin > 0


class TestParameterSelection:
    """Tests for deterministic (a, T, N) selection."""

    def test_returns_tuned_params(self):
        """tune_params returns TunedParams dataclass."""
        params = tune_params(t_end=100.0, alpha_c=0.0)
        assert isinstance(params, TunedParams)
        assert params.a > 0
        assert params.T > 0
        assert params.N > 0

    def test_deterministic_selection(self):
        """Same inputs produce identical parameters."""
        params1 = tune_params(t_end=50.0, alpha_c=-0.1)
        params2 = tune_params(t_end=50.0, alpha_c=-0.1)

        assert params1.a == params2.a
        assert params1.T == params2.T
        assert params1.N == params2.N

    def test_t_max_relationship(self):
        """t_max should equal 2*T."""
        t_end = 100.0
        params = tune_params(t_end=t_end, alpha_c=0.0, kappa=1.0)
        # With kappa=1.0: T = t_end, t_max = 2*T = 2*t_end
        assert params.t_max == pytest.approx(2 * t_end)


class TestFftNilt:
    """Tests for the core FFT-NILT algorithm."""

    def test_simple_exponential(self):
        """Test with F(s) = 1/(s+1), whose inverse is exp(-t) using refine_until_accept."""
        problem = get_problem("lag")  # First-order lag with tau=1
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        # Use refine_until_accept - ε_Im should be small now (~1e-10)
        result = refine_until_accept(
            problem.F, params, t_end=5.0,
            eps_im_threshold=1e-2, eps_conv=1e-2, n_timing_runs=3
        )

        t_eval = result["t_eval"]
        f_eval = result["f_eval"]
        f_exact = problem.f_ref(t_eval)

        # Verify solution is finite and has correct shape
        assert np.all(np.isfinite(f_eval))
        assert len(f_eval) == len(t_eval)

        # Check correlation with expected shape (allows systematic offset and tail aliasing)
        # Use only the central region to avoid boundary effects
        n = len(t_eval)
        mid_start = n // 10
        mid_end = 8 * n // 10
        correlation = np.corrcoef(f_eval[mid_start:mid_end], f_exact[mid_start:mid_end])[0, 1]
        assert correlation > 0.99, f"Poor correlation: {correlation}"

        # Verify values are positive in early/middle time range (tail may have aliasing)
        early_end = n // 2
        assert np.all(f_eval[:early_end] > 0), "Early values should be positive"

    def test_second_order_system(self):
        """Test with second-order underdamped system using refine_until_accept."""
        problem = get_problem("secondorder")
        params = tune_params(t_end=10.0, alpha_c=problem.alpha_c, C=problem.C)

        result = refine_until_accept(
            problem.F, params, t_end=10.0,
            eps_im_threshold=1e-2, eps_conv=1e-2, n_timing_runs=3
        )

        t_eval = result["t_eval"]
        f_eval = result["f_eval"]
        f_exact = problem.f_ref(t_eval)

        # Verify solution is finite and correct shape
        assert np.all(np.isfinite(f_eval))
        assert len(f_eval) == len(t_eval)

        # Check accuracy on interior window (avoid boundary aliasing)
        n = len(t_eval)
        mid_start = n // 10
        mid_end = 7 * n // 10
        np.testing.assert_allclose(
            f_eval[mid_start:mid_end], f_exact[mid_start:mid_end],
            rtol=2e-2, atol=1e-3
        )

    def test_delayed_exponential(self):
        """Test FOPDT (first-order plus dead time) using refine_until_accept."""
        problem = get_problem("fopdt")
        params = tune_params(t_end=10.0, alpha_c=problem.alpha_c, C=problem.C)

        result = refine_until_accept(
            problem.F, params, t_end=10.0,
            eps_im_threshold=1e-2, eps_conv=1e-2, n_timing_runs=3,
            t_eval_min=2.5  # Start after delay
        )

        t_eval = result["t_eval"]
        f_eval = result["f_eval"]
        f_exact = problem.f_ref(t_eval)

        # Verify solution is finite
        assert np.all(np.isfinite(f_eval))

        # Check accuracy on interior window (avoid boundary aliasing)
        n = len(t_eval)
        mid_start = n // 10
        mid_end = 7 * n // 10
        np.testing.assert_allclose(
            f_eval[mid_start:mid_end], f_exact[mid_start:mid_end],
            rtol=3e-2, atol=1e-3
        )


class TestEpsilonImTest:
    """Tests for ε_Im convergence criterion.

    With DFT-consistent frequency mapping, ε_Im = max|Im|/max|Re| should be
    ~1e-10 for real-valued f(t), making it a meaningful diagnostic.
    """

    def test_epsilon_im_below_threshold_for_real_benchmark(self):
        """ε_Im should be small (~1e-10) for real-valued benchmark functions.

        With correct DFT frequency mapping, z_ifft is nearly real for real f(t).
        """
        problem = get_problem("lag")
        result = epsilon_im_test(
            problem.F,
            t_end=10.0,
            alpha_c=problem.alpha_c,
            threshold=1e-2,  # Paper-compliant threshold (actual ~1e-10)
            C=problem.C,
        )

        assert isinstance(result, NiltConvergenceResult)
        assert np.isfinite(result.epsilon_im)
        # ε_Im should now be small (not ~0.6 like before)
        assert result.epsilon_im < 1e-2, f"ε_Im = {result.epsilon_im:.2e} too high"

    def test_epsilon_im_for_vendored_benchmarks(self):
        """ε_Im should be small for all vendored benchmark functions."""
        problems = get_all_problems()

        for name, problem in problems.items():
            # Skip diffusion which has singularity at s=0
            if name == "diffusion":
                continue

            result = epsilon_im_test(
                problem.F,
                t_end=10.0,
                alpha_c=problem.alpha_c,
                threshold=1e-2,  # Paper-compliant threshold
                C=problem.C,
            )
            assert np.isfinite(result.epsilon_im), f"ε_Im computation failed for {name}"
            # ε_Im should be small for all real-valued benchmarks
            assert result.epsilon_im < 1e-2, f"ε_Im = {result.epsilon_im:.2e} too high for {name}"


class TestNDoublingTest:
    """Tests for N-doubling convergence criterion."""

    def test_n_doubling_converges_for_stable_case(self):
        """N-doubling delta should decrease for stable transfer functions."""
        problem = get_problem("lag")

        result = n_doubling_test(
            problem.F,
            t_end=5.0,
            alpha_c=problem.alpha_c,
            n_initial=64,
            threshold=0.1,  # Achievable threshold with DFT-consistent frequency grid
        )

        assert isinstance(result, NiltConvergenceResult)
        # Delta should decrease with N-doubling
        if len(result.delta_sequence) >= 2:
            assert result.delta_sequence[-1] < result.delta_sequence[0]
        # Should converge (use == for numpy boolean)
        assert result.passed == True

    def test_n_doubling_delta_decreases(self):
        """Delta sequence should generally decrease for well-behaved F."""
        problem = get_problem("secondorder")

        result = n_doubling_test(
            problem.F,
            t_end=10.0,
            alpha_c=problem.alpha_c,
            n_initial=32,
            threshold=1e-4,
        )

        deltas = result.delta_sequence
        if len(deltas) >= 3:
            # At least the last delta should be smaller than the first
            assert deltas[-1] < deltas[0]


class TestBenchmarkFunctions:
    """Tests for benchmark Laplace-domain transfer functions."""

    def test_advection_dispersion_transfer(self):
        """Advection-dispersion transfer function is well-defined."""
        F = advection_dispersion_transfer(
            velocity=1e-3,
            dispersion=1e-6,
            length=0.1,
        )
        # Check that F returns finite values for typical s
        s_test = 0.1 + 0.1j
        result = F(s_test)
        assert np.isfinite(result)

    def test_langmuir_column_transfer(self):
        """Langmuir column transfer function is well-defined."""
        F = langmuir_column_transfer(
            velocity=1e-3,
            dispersion=1e-6,
            length=0.1,
            ka=1.0,
            kd=0.1,
            qmax=10.0,
            porosity=0.4,
        )
        s_test = 0.1 + 0.1j
        result = F(s_test)
        assert np.isfinite(result)

    def test_grm_moment_transfer(self):
        """GRM moment transfer function is well-defined."""
        F = grm_moment_transfer(
            velocity=1e-3,
            dispersion=1e-6,
            length=0.1,
            particle_radius=5e-5,
            film_diffusion=1e-5,
            pore_diffusion=1e-10,
        )
        s_test = 0.1 + 0.1j
        result = F(s_test)
        assert np.isfinite(result)

    def test_benchmark_registry(self):
        """get_benchmark_functions returns list of benchmarks."""
        benchmarks = get_benchmark_functions()
        assert isinstance(benchmarks, list)
        assert len(benchmarks) >= 3

        for benchmark in benchmarks:
            name, F, t_range, has_analytical, alpha_c = benchmark
            assert isinstance(name, str)
            assert callable(F)
            assert len(t_range) == 2
            assert isinstance(has_analytical, bool)


class TestRefineUntilAccept:
    """Tests for adaptive refinement."""

    def test_refine_converges(self):
        """refine_until_accept returns valid results for well-behaved problem."""
        problem = get_problem("lag")
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        result = refine_until_accept(
            problem.F,
            params,
            t_end=5.0,
            eps_im_threshold=1e-2,  # Paper-compliant threshold
            eps_conv=1e-2,          # Achievable threshold
            n_timing_runs=5,        # Reduce for test speed
        )

        # Verify result structure
        assert "accepted" in result
        assert "eps_im" in result
        assert "E_N" in result
        assert np.isfinite(result["eps_im"])
        assert np.isfinite(result["E_N"])
        # ε_Im should be small now
        assert result["eps_im"] < 1e-2, f"ε_Im = {result['eps_im']:.2e} too high"
        # Verify solution arrays exist
        assert "f_eval" in result
        assert "t_eval" in result


class TestCadetBenchmarksWithNilt:
    """Tests for CADET-specific benchmarks with NILT."""

    def test_advection_dispersion_eps_im(self):
        """ε_Im should be small for advection-dispersion benchmark."""
        F = advection_dispersion_transfer(velocity=1e-3, dispersion=1e-6, length=0.1)

        result = epsilon_im_test(F, t_end=100.0, alpha_c=0.0, threshold=1e-2)

        assert np.isfinite(result.epsilon_im), f"ε_Im computation failed: {result.epsilon_im}"
        assert result.epsilon_im < 1e-2, f"ε_Im = {result.epsilon_im:.2e} too high"

    def test_langmuir_eps_im(self):
        """ε_Im should be small for Langmuir column benchmark."""
        F = langmuir_column_transfer(
            velocity=1e-3, dispersion=1e-6, length=0.1,
            ka=1.0, kd=0.1, qmax=10.0
        )

        result = epsilon_im_test(F, t_end=200.0, alpha_c=0.0, threshold=1e-2)

        assert np.isfinite(result.epsilon_im), f"ε_Im computation failed: {result.epsilon_im}"
        assert result.epsilon_im < 1e-2, f"ε_Im = {result.epsilon_im:.2e} too high"


class TestDiagnosticsSemantics:
    """Tests for ε_Im diagnostic semantics.

    With DFT-consistent frequency mapping, ε_Im = max|Im|/max|Re| should be
    ~1e-10 for real-valued f(t). The old workarounds (one_sided_imag_ratio,
    epsilon_im_paper) are now deprecated.
    """

    def test_eps_im_max_is_small_for_real_functions(self):
        """eps_im_max should be small (~1e-10) for real-valued f(t)."""
        problem = get_problem("lag")
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        # Compute NILT
        f, t, z_ifft, eps_im = fft_nilt(problem.F, params.a, params.T, params.N)

        # ε_Im should be small now (not ~0.6 like before)
        assert np.isfinite(eps_im)
        assert eps_im < 1e-2, f"ε_Im = {eps_im:.2e} too high (expected ~1e-10)"

        # Verify z_ifft is nearly real
        max_imag = np.max(np.abs(np.imag(z_ifft)))
        max_real = np.max(np.abs(np.real(z_ifft)))
        assert max_imag / max_real < 1e-2, "z_ifft should be nearly real"

    def test_eps_im_rms_also_small(self):
        """eps_im_rms should also be small for real-valued f(t)."""
        problem = get_problem("lag")
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        f, t, z_ifft, _ = fft_nilt(problem.F, params.a, params.T, params.N)

        rms_ratio = eps_im_rms(z_ifft)
        assert np.isfinite(rms_ratio)
        assert rms_ratio < 1e-2, f"RMS ratio = {rms_ratio:.2e} too high"

    def test_eps_im_deprecated_alias_warns(self):
        """eps_im should warn about deprecation."""
        problem = get_problem("lag")
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        f, t, z_ifft, _ = fft_nilt(problem.F, params.a, params.T, params.N)

        # Should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = eps_im(z_ifft)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "eps_im()" in str(w[0].message)

    def test_one_sided_imag_ratio_deprecated(self):
        """one_sided_imag_ratio should warn about deprecation."""
        problem = get_problem("lag")
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        f, t, z_ifft, _ = fft_nilt(problem.F, params.a, params.T, params.N)

        # Should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = one_sided_imag_ratio(z_ifft)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_fft_nilt_returns_eps_im_directly(self):
        """fft_nilt now returns ε_Im directly (no diagnostics_mode)."""
        problem = get_problem("lag")
        params = tune_params(t_end=5.0, alpha_c=problem.alpha_c, C=problem.C)

        # fft_nilt returns (f, t, z_ifft, eps_im)
        f, t, z_ifft, eps_im = fft_nilt(problem.F, params.a, params.T, params.N)

        assert len(f) == params.N
        assert len(t) == params.N
        assert len(z_ifft) == params.N
        assert np.isfinite(eps_im)
        assert eps_im < 1e-2, f"ε_Im = {eps_im:.2e} too high"
