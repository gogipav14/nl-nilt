"""CADET comparison module for NILT verification.

Provides utilities to compare NILT reference solutions with CADET simulations
on a common time grid, computing RMSE, L2, and L∞ error metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union
import json
import numpy as np

from ..config_gen.minimal_grm import create_minimal_grm_config


@dataclass
class ComparisonResult:
    """Result of comparing NILT reference with CADET simulation.

    Attributes:
        rmse: Root mean squared error.
        l2_norm: L2 norm of difference.
        linf_norm: L∞ (maximum) norm of difference.
        relative_l2_error: Relative L2 error (||diff||_2 / ||ref||_2).
        t_grid: Common time grid used for comparison.
        y_ref: Reference (NILT) values on common grid.
        y_cadet: CADET values on common grid.
    """

    rmse: float
    l2_norm: float
    linf_norm: float
    relative_l2_error: float
    t_grid: Optional[np.ndarray] = None
    y_ref: Optional[np.ndarray] = None
    y_cadet: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rmse": self.rmse,
            "l2_norm": self.l2_norm,
            "linf_norm": self.linf_norm,
            "relative_l2_error": self.relative_l2_error,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def interpolate_to_common_grid(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two signals to a common time grid.

    Args:
        t1: Time points for first signal.
        y1: Values for first signal.
        t2: Time points for second signal.
        y2: Values for second signal.
        n_points: Number of points in common grid.

    Returns:
        Tuple of (t_common, y1_interp, y2_interp).
    """
    # Find common time range
    t_min = max(t1[0], t2[0])
    t_max = min(t1[-1], t2[-1])

    # Create common grid
    t_common = np.linspace(t_min, t_max, n_points)

    # Interpolate both signals
    y1_interp = np.interp(t_common, t1, y1)
    y2_interp = np.interp(t_common, t2, y2)

    return t_common, y1_interp, y2_interp


def compute_comparison_metrics(
    y_ref: np.ndarray,
    y_test: np.ndarray,
    t_grid: Optional[np.ndarray] = None,
) -> ComparisonResult:
    """Compute comparison metrics between reference and test signals.

    Args:
        y_ref: Reference signal values.
        y_test: Test signal values (must be same length as y_ref).
        t_grid: Optional time grid (for storing in result).

    Returns:
        ComparisonResult with all metrics.
    """
    diff = y_ref - y_test

    # RMSE
    rmse = np.sqrt(np.mean(diff**2))

    # L2 norm
    l2_norm = np.linalg.norm(diff)

    # L∞ norm
    linf_norm = np.max(np.abs(diff))

    # Relative L2 error
    ref_norm = np.linalg.norm(y_ref)
    if ref_norm < 1e-300:
        relative_l2_error = np.inf if l2_norm > 1e-300 else 0.0
    else:
        relative_l2_error = l2_norm / ref_norm

    return ComparisonResult(
        rmse=rmse,
        l2_norm=l2_norm,
        linf_norm=linf_norm,
        relative_l2_error=relative_l2_error,
        t_grid=t_grid,
        y_ref=y_ref,
        y_cadet=y_test,
    )


def create_matching_cadet_config(
    output_path: Union[str, Path],
    velocity: float,
    dispersion: float,
    length: float,
    t_max: float,
    n_times: int,
    binding_model: Optional[str] = None,
    ka: float = 1.0,
    kd: float = 0.1,
    qmax: float = 10.0,
    porosity: float = 0.4,
    **kwargs,
) -> Path:
    """Create a CADET configuration matching NILT benchmark parameters.

    Args:
        output_path: Path for output HDF5 file.
        velocity: Interstitial velocity [m/s].
        dispersion: Axial dispersion coefficient [m²/s].
        length: Column length [m].
        t_max: End time for simulation [s].
        n_times: Number of output time points.
        binding_model: Optional binding model ("LANGMUIR" or None).
        ka: Adsorption rate constant (if binding_model set).
        kd: Desorption rate constant (if binding_model set).
        qmax: Maximum binding capacity (if binding_model set).
        porosity: Column porosity.
        **kwargs: Additional arguments for create_minimal_grm_config.

    Returns:
        Path to created HDF5 configuration file.
    """
    output_path = Path(output_path)

    # Map to minimal_grm_config parameters
    config_kwargs = {
        "velocity": velocity,
        "col_dispersion": dispersion,
        "col_length": length,
        "end_time": t_max,
        "n_times": n_times,
        "col_porosity": porosity,
        **kwargs,
    }

    if binding_model == "LANGMUIR":
        config_kwargs["binding_ka"] = ka
        config_kwargs["binding_kd"] = kd
        config_kwargs["binding_qmax"] = qmax

    return create_minimal_grm_config(output_path=output_path, **config_kwargs)


def compare_nilt_to_cadet_output(
    nilt_t: np.ndarray,
    nilt_y: np.ndarray,
    cadet_output_path: Union[str, Path],
    unit_id: int = 1,
    component: int = 0,
    port: str = "outlet",
) -> ComparisonResult:
    """Compare NILT solution to CADET output file.

    Args:
        nilt_t: NILT time points.
        nilt_y: NILT solution values.
        cadet_output_path: Path to CADET output HDF5 file.
        unit_id: CADET unit operation ID (default 1 for column).
        component: Component index (default 0).
        port: Output port ("outlet" or "inlet").

    Returns:
        ComparisonResult with metrics.
    """
    import h5py

    cadet_output_path = Path(cadet_output_path)

    with h5py.File(cadet_output_path, "r") as f:
        # Read solution times
        solution_times = f["output/solution/SOLUTION_TIMES"][:]

        # Read outlet concentration
        outlet_key = f"output/solution/unit_{unit_id:03d}/SOLUTION_{port.upper()}"
        if outlet_key not in f:
            # Try alternative key format
            outlet_key = f"output/solution/unit_{unit_id:03d}/SOLUTION_OUTLET"

        outlet_data = f[outlet_key][:]

        # Extract component (handle different array shapes)
        if outlet_data.ndim == 1:
            cadet_y = outlet_data
        elif outlet_data.ndim == 2:
            cadet_y = outlet_data[:, component]
        else:
            cadet_y = outlet_data[:, 0, component]  # (time, port, component)

    # Interpolate to common grid
    t_common, nilt_interp, cadet_interp = interpolate_to_common_grid(
        nilt_t, nilt_y, solution_times, cadet_y
    )

    return compute_comparison_metrics(nilt_interp, cadet_interp, t_common)
