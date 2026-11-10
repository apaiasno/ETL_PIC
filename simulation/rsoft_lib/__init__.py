"""rsoft_lib: helpers for RSoft simulations.

    import rsoft_lib as rlib

    _env        RSoft install paths, RSoftCircuit
    utils       param_str
    symbols     read_all_symbols, read_symbol, export_symbols      (.ind symbol table)
    monitors    MonitorData, read_monitor                          (BeamPROP .mon files)
    fields      FieldData, read_fld, read_mode, write_fld, add_fields, regrid, edge_to_peak_ratio
    modes       mode_field, ex_mode_index, classify_modes, MODE_LIB_DIR   (mode library)
    bend_modes  te_mode_index, read_te_mode, read_straight_mode, best_offset, straight_offset, junction_offset
                                                                (bend-to-straight offset)
                get_loss, sbend_loss, build_bend_loss_table, ...  (bend loss from Im(n_eff), loss table lookup)
    materials   n_clad, material_nk, read_nk, find_material_file  (index from RSoft material files)
    simulation  run_sim                                         (set symbols, run, clean up)
    geometry    sbend_lengths, euler_sbend, parallel_path, save_gds
"""
from ._env import RSoftCircuit
from .utils import param_str
from .symbols import read_all_symbols, read_symbol, export_symbols
from .monitors import MonitorData, read_monitor
from .fields import (FieldData, read_fld, read_mode, write_fld, add_fields, regrid,
                     edge_to_peak_ratio)
from .modes import MODE_LIB_DIR, ex_mode_index, mode_field, classify_modes
from .bend_modes import (BEND_MODE_DIR, BEND_MODE_ROOT, te_mode_index, read_te_mode,
                         read_straight_mode, best_offset, straight_offset, junction_offset, mirror_x, bend_offsets,
                         BEND_LOSS_CSV, load_loss_table, get_loss, sbend_loss,
                         loss_dB_per_cm_from_neff_imag, build_bend_loss_table)
from .materials import n_clad, material_nk, read_nk, find_material_file
from .simulation import run_sim
from .geometry import (sbend_lengths, euler_sbend, euler_ramp_length, euler_sbend_from_ramp,
                       euler_sbend_fixed_ramp, bend_curvature, parallel_path, save_gds,
                       sidestep_offset, sidestep_straight, sidestep_angle, ind2gds, delay_arm, euler_wave)

__all__ = [
    "RSoftCircuit", "param_str",
    "read_all_symbols", "read_symbol", "export_symbols",
    "MonitorData", "read_monitor",
    "FieldData", "read_fld", "read_mode", "write_fld", "add_fields", "regrid", "edge_to_peak_ratio",
    "MODE_LIB_DIR", "ex_mode_index", "mode_field", "classify_modes",
    "BEND_MODE_DIR", "BEND_MODE_ROOT", "te_mode_index", "read_te_mode", "read_straight_mode", "best_offset",
    "straight_offset", "junction_offset", "mirror_x", "bend_offsets",
    "BEND_LOSS_CSV", "load_loss_table", "get_loss", "sbend_loss", "loss_dB_per_cm_from_neff_imag",
    "build_bend_loss_table",
    "n_clad", "material_nk", "read_nk", "find_material_file",
    "run_sim",
    "sbend_lengths", "euler_sbend", "euler_ramp_length", "euler_sbend_from_ramp",
    "euler_sbend_fixed_ramp", "bend_curvature", "parallel_path", "save_gds",
    "sidestep_offset", "sidestep_straight", "sidestep_angle", "ind2gds", "delay_arm", "euler_wave",
]