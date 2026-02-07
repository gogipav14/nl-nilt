"""Problem classification for NILT solver selection.

Given extracted physical parameters from a CADET configuration,
determines which analytical transfer function to use and whether
NILT is production-ready for the problem type.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProblemClassification:
    """Result of classifying a chromatography problem for NILT.

    Attributes:
        problem_type: One of "transport_only", "transport_binding_linear",
            "grm_langmuir", "grm_sma", "unsupported".
        transfer_function_name: Name of the transfer function to use from
            benchmarks.py (e.g., "advection_dispersion_transfer").
        production_ready: True if NILT is validated for this problem type.
        warnings: List of warning messages about limitations.
    """

    problem_type: str
    transfer_function_name: str
    production_ready: bool
    warnings: List[str] = field(default_factory=list)


def classify_problem(params: dict) -> ProblemClassification:
    """Classify a chromatography problem for NILT solver selection.

    Args:
        params: Dictionary of physical parameters (from extract_nilt_params).
            Required keys: binding_model, binding_params.
            Optional keys used for validation: velocity, dispersion, length,
            par_porosity, par_radius, film_diffusion, pore_diffusion.

    Returns:
        ProblemClassification with problem type, transfer function, and
        production readiness.
    """
    binding_model = params.get("binding_model", "NONE")
    binding_params = params.get("binding_params", {})
    warnings = []

    # Check if we have GRM-level particle parameters
    has_particle_params = (
        params.get("par_radius") is not None
        and params.get("pore_diffusion") is not None
        and params.get("film_diffusion") is not None
    )

    # Determine if binding is effectively inactive
    no_effective_binding = (
        binding_model == "NONE"
        or (binding_model == "LINEAR"
            and binding_params.get("ka", 0.0) == 0.0
            and binding_params.get("kd", 0.0) == 0.0)
    )

    # Case 1: No effective binding
    if no_effective_binding:
        if has_particle_params:
            # GRM with particle dynamics but no binding — still need
            # grm_langmuir_transfer with ka=0 to capture film + pore diffusion
            return ProblemClassification(
                problem_type="grm_no_binding",
                transfer_function_name="grm_langmuir_transfer",
                production_ready=True,
                warnings=warnings,
            )
        # Simple column: no particles, no binding
        return ProblemClassification(
            problem_type="transport_only",
            transfer_function_name="advection_dispersion_transfer",
            production_ready=True,
            warnings=warnings,
        )

    # Case 2: Linear binding with nonzero rates
    if binding_model == "LINEAR":
        if has_particle_params:
            warnings.append(
                "GRM with linear binding: NILT uses grm_langmuir_transfer "
                "with qmax derived from linear K_eq. Experimental."
            )
            return ProblemClassification(
                problem_type="transport_binding_linear",
                transfer_function_name="grm_langmuir_transfer",
                production_ready=False,
                warnings=warnings,
            )

        warnings.append(
            "Linear binding without GRM particle parameters: "
            "using langmuir_column_transfer approximation. Experimental."
        )
        return ProblemClassification(
            problem_type="transport_binding_linear",
            transfer_function_name="langmuir_column_transfer",
            production_ready=False,
            warnings=warnings,
        )

    # Case 3: Multi-component Langmuir
    if binding_model == "MULTI_COMPONENT_LANGMUIR":
        warnings.append(
            "GRM with Langmuir binding: linearized transfer function. "
            "Valid for dilute regime (c << qmax). Experimental."
        )
        return ProblemClassification(
            problem_type="grm_langmuir",
            transfer_function_name="grm_langmuir_transfer",
            production_ready=False,
            warnings=warnings,
        )

    # Case 4: Steric Mass Action
    if binding_model == "STERIC_MASS_ACTION":
        warnings.append(
            "GRM with SMA binding: linearized around base state. "
            "Valid for small perturbations, single component. Experimental."
        )
        return ProblemClassification(
            problem_type="grm_sma",
            transfer_function_name="grm_sma_transfer",
            production_ready=False,
            warnings=warnings,
        )

    # Unsupported binding model
    return ProblemClassification(
        problem_type="unsupported",
        transfer_function_name="",
        production_ready=False,
        warnings=[f"Unsupported binding model: {binding_model}"],
    )
