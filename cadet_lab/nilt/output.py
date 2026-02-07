"""CADET-compatible output for NILT solutions.

Writes NILT solutions in formats compatible with existing CADET tooling
(read_solution from telemetry) and JSON for diagnostics.
"""

import json
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from .solver import NiltSolution


def write_cadet_h5(
    solution: NiltSolution,
    output_path: Union[str, Path],
    unit_id: int = 1,
    n_comp: int = 1,
) -> Path:
    """Write NILT solution as CADET-style HDF5.

    Creates the same HDF5 structure that CADET produces, so that
    read_solution() from telemetry can read it directly.

    Args:
        solution: NILT solution to write.
        output_path: Path for output HDF5 file.
        unit_id: Unit operation ID (default 1 for column).
        n_comp: Number of components (default 1).

    Returns:
        Path to created HDF5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Create output/solution structure matching CADET format
        solution_grp = f.create_group("output/solution")

        # Solution times
        solution_grp.create_dataset("SOLUTION_TIMES", data=solution.t)

        # Outlet data: shape (n_times, n_comp)
        unit_key = f"unit_{unit_id:03d}"
        unit_grp = solution_grp.create_group(unit_key)

        if n_comp == 1:
            # Shape (n_times, 1) for single component
            outlet_data = solution.y.reshape(-1, 1)
        else:
            # For multi-component, y should already be (n_times, n_comp)
            outlet_data = solution.y
            if outlet_data.ndim == 1:
                outlet_data = outlet_data.reshape(-1, 1)

        unit_grp.create_dataset("SOLUTION_OUTLET", data=outlet_data)

        # NILT diagnostics section
        diag_grp = f.create_group("output/nilt_diagnostics")
        diag_grp.create_dataset("N", data=solution.params.N)
        diag_grp.create_dataset(
            "EPS_IM", data=solution.metadata.get("eps_im", 0.0)
        )
        diag_grp.create_dataset("WALL_TIME_US", data=solution.wall_time_us)

        if solution.classification is not None:
            dt = h5py.string_dtype(encoding="ascii")
            diag_grp.create_dataset(
                "CLASSIFICATION",
                data=solution.classification.problem_type,
                dtype=dt,
            )
            diag_grp.create_dataset(
                "PRODUCTION_READY",
                data=int(solution.classification.production_ready),
            )

        diag_grp.create_dataset(
            "CONVERGED", data=int(solution.convergence.passed)
        )

        if solution.convergence.epsilon_im is not None:
            diag_grp.create_dataset(
                "CONVERGENCE_EPS_IM", data=solution.convergence.epsilon_im
            )
        if solution.convergence.final_delta is not None:
            diag_grp.create_dataset(
                "CONVERGENCE_E_N", data=solution.convergence.final_delta
            )

    return output_path


def write_json(
    solution: NiltSolution,
    output_path: Union[str, Path],
) -> Path:
    """Write NILT solution as JSON with full diagnostics.

    Args:
        solution: NILT solution to write.
        output_path: Path for output JSON file.

    Returns:
        Path to created JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "t": solution.t.tolist(),
        "y": solution.y.tolist(),
        "wall_time_us": solution.wall_time_us,
        "convergence": solution.convergence.to_dict(),
        "params": {
            "a": solution.params.a,
            "T": solution.params.T,
            "N": solution.params.N,
            "delta_t": solution.params.delta_t,
            "feasible": bool(solution.params.feasible),
        },
        "metadata": {
            k: v for k, v in solution.metadata.items()
            if not isinstance(v, np.ndarray)
        },
    }

    if solution.classification is not None:
        data["classification"] = {
            "problem_type": solution.classification.problem_type,
            "transfer_function_name": solution.classification.transfer_function_name,
            "production_ready": solution.classification.production_ready,
            "warnings": solution.classification.warnings,
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path
