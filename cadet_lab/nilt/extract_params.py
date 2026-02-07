"""Extract physical parameters from CADET HDF5 config for NILT.

Reads a CADET HDF5 configuration file and extracts the physical parameters
needed to construct the matching analytical transfer function.

Supports both CADET v6 format (particle_type_000 subgroup) and legacy format
(parameters directly under unit group).
"""

from pathlib import Path
from typing import Union

import h5py
import numpy as np


def _read_scalar(group: h5py.Group, name: str, default=None):
    """Read a scalar value from an HDF5 group, returning default if missing."""
    if name not in group:
        return default
    val = group[name][()]
    if hasattr(val, "item"):
        return val.item()
    return val


def _read_array_first(group: h5py.Group, name: str, default=None):
    """Read first element of an array dataset, or scalar."""
    if name not in group:
        return default
    val = group[name][()]
    if hasattr(val, "__len__") and len(val) > 0:
        v = val[0]
        return v.item() if hasattr(v, "item") else v
    if hasattr(val, "item"):
        return val.item()
    return val


def _decode_string(val) -> str:
    """Decode bytes to string if needed."""
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def extract_nilt_params(h5_path: Union[str, Path]) -> dict:
    """Extract physical parameters from a CADET HDF5 config for NILT.

    Reads column geometry, transport, and binding parameters from a CADET
    configuration file. Handles both CADET v6 format (particle_type_000
    subgroup) and legacy format.

    Args:
        h5_path: Path to CADET HDF5 configuration file.

    Returns:
        Dictionary with keys:
            velocity, dispersion, length, col_porosity,
            par_porosity, par_radius, film_diffusion, pore_diffusion,
            binding_model, binding_params (dict with ka, kd, qmax/Lambda/etc.),
            n_comp, end_time

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        ValueError: If required datasets are missing.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # Find the column unit (GENERAL_RATE_MODEL)
        unit = _find_column_unit(f)
        unit_path = unit.name

        # Column-level parameters
        velocity = _read_scalar(unit, "VELOCITY")
        if velocity is None:
            raise ValueError(f"VELOCITY not found in {unit_path}")

        dispersion = _read_scalar(unit, "COL_DISPERSION")
        if dispersion is None:
            raise ValueError(f"COL_DISPERSION not found in {unit_path}")

        length = _read_scalar(unit, "COL_LENGTH")
        if length is None:
            raise ValueError(f"COL_LENGTH not found in {unit_path}")

        col_porosity = _read_scalar(unit, "COL_POROSITY", default=0.37)
        n_comp = _read_scalar(unit, "NCOMP", default=1)

        # Particle-level parameters: try v6 format first, then legacy
        par_porosity, par_radius, film_diffusion, pore_diffusion = \
            _read_particle_params(unit)

        # Binding model and parameters
        binding_model, binding_params = _read_binding_params(unit)

        # End time from solver sections
        end_time = _read_end_time(f)

    return {
        "velocity": velocity,
        "dispersion": dispersion,
        "length": length,
        "col_porosity": col_porosity,
        "par_porosity": par_porosity,
        "par_radius": par_radius,
        "film_diffusion": film_diffusion,
        "pore_diffusion": pore_diffusion,
        "binding_model": binding_model,
        "binding_params": binding_params,
        "n_comp": n_comp,
        "end_time": end_time,
    }


def _find_column_unit(f: h5py.File) -> h5py.Group:
    """Find the GRM column unit in the HDF5 file."""
    if "input/model" not in f:
        raise ValueError("No input/model group found in HDF5 file")

    model = f["input/model"]
    for key in sorted(model.keys()):
        if not key.startswith("unit_"):
            continue
        unit = model[key]
        if "UNIT_TYPE" not in unit:
            continue
        unit_type = _decode_string(unit["UNIT_TYPE"][()])
        if unit_type in ("GENERAL_RATE_MODEL", "LUMPED_RATE_MODEL_WITH_PORES",
                         "LUMPED_RATE_MODEL_WITHOUT_PORES"):
            return unit

    raise ValueError("No column unit (GRM/LRMP/LRM) found in HDF5 file")


def _read_particle_params(unit: h5py.Group):
    """Read particle-level parameters from unit group.

    Tries CADET v6 format (particle_type_000 subgroup) first,
    then falls back to legacy format (directly under unit).
    """
    # v6 format: parameters under particle_type_000
    if "particle_type_000" in unit:
        pt = unit["particle_type_000"]
        par_porosity = _read_scalar(pt, "PAR_POROSITY", default=0.33)
        par_radius = _read_scalar(pt, "PAR_RADIUS", default=1e-5)
        film_diffusion = _read_array_first(pt, "FILM_DIFFUSION", default=1e-5)
        pore_diffusion = _read_array_first(pt, "PORE_DIFFUSION", default=1e-10)
        return par_porosity, par_radius, film_diffusion, pore_diffusion

    # Legacy format: parameters directly under unit
    par_porosity = _read_scalar(unit, "PAR_POROSITY", default=0.33)
    par_radius = _read_scalar(unit, "PAR_RADIUS", default=1e-5)
    film_diffusion = _read_array_first(unit, "FILM_DIFFUSION", default=1e-5)
    pore_diffusion = _read_array_first(unit, "PAR_DIFFUSION",
                                        default=_read_array_first(
                                            unit, "PORE_DIFFUSION", default=1e-10))
    return par_porosity, par_radius, film_diffusion, pore_diffusion


def _read_binding_params(unit: h5py.Group) -> tuple:
    """Read binding model type and parameters.

    Returns:
        Tuple of (binding_model: str, binding_params: dict).
        binding_model is one of: "NONE", "LINEAR", "MULTI_COMPONENT_LANGMUIR",
        "STERIC_MASS_ACTION", or the raw string from the file.
    """
    # v6 format: ADSORPTION_MODEL in particle_type_000
    ads_model = None
    ads_group = None

    if "particle_type_000" in unit:
        pt = unit["particle_type_000"]
        if "ADSORPTION_MODEL" in pt:
            ads_model = _decode_string(pt["ADSORPTION_MODEL"][()])
        if "adsorption" in pt:
            ads_group = pt["adsorption"]
    else:
        # Legacy format
        if "adsorption" in unit:
            legacy_ads = unit["adsorption"]
            if "ADSORPTION_MODEL" in legacy_ads:
                ads_model = _decode_string(legacy_ads["ADSORPTION_MODEL"][()])
            ads_group = legacy_ads
        elif "ADSORPTION_MODEL" in unit:
            ads_model = _decode_string(unit["ADSORPTION_MODEL"][()])

    if ads_model is None or ads_model == "NONE":
        return "NONE", {}

    params = {}

    if ads_group is not None:
        if ads_model == "LINEAR":
            params["ka"] = _read_array_first(ads_group, "LIN_KA", default=0.0)
            params["kd"] = _read_array_first(ads_group, "LIN_KD", default=0.0)
        elif ads_model == "MULTI_COMPONENT_LANGMUIR":
            params["ka"] = _read_array_first(ads_group, "MCL_KA", default=0.0)
            params["kd"] = _read_array_first(ads_group, "MCL_KD", default=0.0)
            params["qmax"] = _read_array_first(ads_group, "MCL_QMAX", default=0.0)
        elif ads_model == "STERIC_MASS_ACTION":
            params["ka"] = _read_array_first(ads_group, "SMA_KA", default=0.0)
            params["kd"] = _read_array_first(ads_group, "SMA_KD", default=0.0)
            params["Lambda"] = _read_scalar(ads_group, "SMA_LAMBDA", default=0.0)
            params["nu"] = _read_array_first(ads_group, "SMA_NU", default=0.0)

    return ads_model, params


def _read_end_time(f: h5py.File) -> float:
    """Read end time from solver sections."""
    # Try SECTION_TIMES
    section_paths = [
        "input/solver/sections/SECTION_TIMES",
        "input/solver/SECTION_TIMES",
    ]
    for path in section_paths:
        if path in f:
            times = f[path][()]
            if hasattr(times, "__len__") and len(times) > 0:
                return float(times[-1])

    # Try USER_SOLUTION_TIMES
    sol_time_paths = [
        "input/solver/USER_SOLUTION_TIMES",
    ]
    for path in sol_time_paths:
        if path in f:
            times = f[path][()]
            if hasattr(times, "__len__") and len(times) > 0:
                return float(times[-1])

    raise ValueError("Could not determine end time from HDF5 file")
