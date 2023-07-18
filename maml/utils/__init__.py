"""Utilities package."""
from __future__ import annotations

from ._data_conversion import convert_docs, pool_from, to_array
from ._data_split import DataSplitter, ShuffleSplitter
from ._dummy import feature_dim_from_test_system, get_describer_dummy_obj
from ._inspect import get_full_args
from ._jit import njit
from ._lammps import (
    check_structures_forces_stresses,
    get_lammps_lattice_and_rotation,
    stress_format_change,
    stress_list_to_matrix,
    stress_matrix_to_list,
    write_data_from_structure,
)
from ._material import to_composition
from ._preprocessing import DummyScaler, Scaler, StandardScaler
from ._signal_processing import cwt, fft_magnitude, get_sp_method, spectrogram, wvd
from ._stats import STATS_KWARGS, Stats, get_full_stats_and_funcs, stats_list_conversion
from ._tempfile import MultiScratchDir
from ._value_profile import ConstantValue, LinearProfile, ValueProfile

__all__ = [
    "pool_from",
    "convert_docs",
    "to_array",
    "ValueProfile",
    "ConstantValue",
    "LinearProfile",
    "njit",
    "MultiScratchDir",
    "DataSplitter",
    "ShuffleSplitter",
    "write_data_from_structure",
    "check_structures_forces_stresses",
    "stress_format_change",
    "stress_list_to_matrix",
    "stress_matrix_to_list",
    "get_lammps_lattice_and_rotation",
    "Stats",
    "STATS_KWARGS",
    "stats_list_conversion",
    "get_full_stats_and_funcs",
    "spectrogram",
    "cwt",
    "wvd",
    "fft_magnitude",
    "get_sp_method",
    "to_composition",
    "get_full_args",
    "get_describer_dummy_obj",
    "feature_dim_from_test_system",
    "StandardScaler",
    "Scaler",
    "DummyScaler",
]
