"""Pytest fixtures for cadet-lab-tools tests."""

import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cadet_cli_path():
    """Get path to cadet-cli executable.

    Searches in common locations and environment variable.
    """
    # Check environment variable first
    env_path = os.environ.get("CADET_CLI_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check common locations
    common_paths = [
        "/usr/local/bin/cadet-cli",
        "/usr/bin/cadet-cli",
        os.path.expanduser("~/cadet/bin/cadet-cli"),
        # Add build directory paths
        str(Path(__file__).parent.parent.parent / "build" / "bin" / "cadet-cli"),
        str(Path(__file__).parent.parent.parent / "install" / "bin" / "cadet-cli"),
    ]

    for path in common_paths:
        if os.path.isfile(path):
            return path

    # Return None if not found - tests should skip gracefully
    return None


@pytest.fixture
def cadet_test_data_dir():
    """Path to CADET-Core test data directory."""
    return Path(__file__).parent.parent.parent / "test" / "data"


@pytest.fixture
def sample_ref_h5(cadet_test_data_dir):
    """Path to a sample reference HDF5 file for testing."""
    ref_file = cadet_test_data_dir / "ref_LRM_dynLin_1comp_sensbenchmark1_FV_Z32.h5"
    if ref_file.exists():
        return ref_file
    # Fallback to any available ref file
    ref_files = list(cadet_test_data_dir.glob("ref_*.h5"))
    if ref_files:
        return ref_files[0]
    pytest.skip("No reference HDF5 files found in test data")
