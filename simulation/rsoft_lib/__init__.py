"""rsoft_lib: helpers for RSoft simulations.

    import rsoft_lib as rlib

    _env        RSoft install paths, RSoftCircuit
    utils       param_str
    symbols     read_all_symbols, read_symbol, export_symbols      (.ind symbol table)
    monitors    MonitorData, read_monitor                          (BeamPROP .mon files)
    fields      FieldData, read_fld, read_mode, write_fld, add_fields, regrid, edge_to_peak_ratio
    modes       mode_field, ex_mode_index, classify_modes, MODE_LIB_DIR   (mode library)
                straight_neff, StraightModel                         (n_eff of straight guides vs wavelength and width)
    bend_modes  te_mode_index, read_te_mode, read_straight_mode, best_offset, straight_offset, junction_offset
                                                                (bend-to-straight offset)
                get_loss, sbend_loss, build_bend_loss_table, ...  (bend loss from Im(n_eff), loss table lookup)
                bend_neff, build_bend_neff_table, BendModel          (n_eff of the bend mode vs radius and wavelength)
    materials   n_clad, material_nk, read_nk, find_material_file  (index from RSoft material files)
    simulation  run_sim                                         (set symbols, run, clean up)
                run_scan, scan_wavelengths, read_pathways, read_scan (run a circuit over a parameter grid or wavelengths, read its pathway monitors)
                estimate_pathways                                    (phase and loss of the pathways of a circuit from the n_eff tables)
    transfer    ideal transfer matrices (y_splitter, mmi_2x2, thermo_optic_phase_shifter, pathway, chain, parallel, mzi_response, power):
                use as rlib.transfer.<name>; combine with estimate_pathways to get an interference spectrum
    device_setup  settings and paths shared by the device scripts (BASE_DIR, WIDTH, WAVELENGTHS, SIM, device_paths, export_device)
    geometry    paths and widths: sbend_lengths, euler_sbend, delay_arm, euler_wave, raised_cosine, cosine_tapers, ...
    circuit     the structures as RSoft CAD: Circuit, paths_to_circuit, mzi, radiation_witness, ind2gds (RSoft's GDS export), pathway_file_symbols
    gds         independent GDS tools: save_gds, read_gds_shapes, gds_waveguides, gds_to_ind
"""
from ._env import RSoftCircuit
from .utils import param_str
from .symbols import read_all_symbols, read_symbol, export_symbols
from .monitors import MonitorData, read_monitor
from .fields import (FieldData, read_fld, read_mode, write_fld, add_fields, regrid,
                     edge_to_peak_ratio)
from .modes import MODE_LIB_DIR, ex_mode_index, mode_field, classify_modes, StraightModel, straight_neff
from .bend_modes import (BEND_MODE_DIR, BEND_MODE_ROOT, te_mode_index, read_te_mode,
                         read_straight_mode, best_offset, straight_offset, junction_offset, mirror_x, bend_offsets,
                         BEND_LOSS_CSV, load_loss_table, get_loss, sbend_loss,
                         loss_dB_per_cm_from_neff_imag, build_bend_loss_table,
                         BEND_NEFF_CSV, BendModel, bend_neff, build_bend_neff_table, load_bend_neff_table)
from .materials import n_clad, material_nk, read_nk, find_material_file
from .simulation import run_sim, run_scan, scan_wavelengths, read_pathways, read_scan, estimate_pathways, PathwayEstimate, PathwayEstimates
from . import transfer
from . import device_setup
from .geometry import (sbend_lengths, euler_sbend, euler_ramp_length, euler_sbend_from_ramp, euler_sbend_fixed_ramp,
                       bend_curvature, parallel_path, path_list, eval_width, sidestep_offset, sidestep_straight, sidestep_angle,
                       delay_arm_parameters, delay_arm, euler_wave, raised_cosine, cosine_tapers)
from .circuit import Circuit, mzi, radiation_witness, paths_to_circuit, arc_end_point, fit_path, pathway_file_symbols, ind2gds
from .gds import save_gds, read_gds_shapes, gds_waveguides, gds_to_ind

__all__ = [
    "RSoftCircuit", "param_str",
    "read_all_symbols", "read_symbol", "export_symbols",
    "MonitorData", "read_monitor",
    "FieldData", "read_fld", "read_mode", "write_fld", "add_fields", "regrid", "edge_to_peak_ratio",
    "MODE_LIB_DIR", "ex_mode_index", "mode_field", "classify_modes", "StraightModel", "straight_neff",
    "BEND_MODE_DIR", "BEND_MODE_ROOT", "te_mode_index", "read_te_mode", "read_straight_mode", "best_offset",
    "straight_offset", "junction_offset", "mirror_x", "bend_offsets",
    "BEND_LOSS_CSV", "load_loss_table", "get_loss", "sbend_loss", "loss_dB_per_cm_from_neff_imag",
    "build_bend_loss_table", "BEND_NEFF_CSV", "BendModel", "bend_neff", "build_bend_neff_table", "load_bend_neff_table",
    "n_clad", "material_nk", "read_nk", "find_material_file",
    "run_sim", "run_scan", "scan_wavelengths", "read_pathways", "read_scan", "estimate_pathways", "PathwayEstimate", "PathwayEstimates", "transfer", "device_setup",
    "sbend_lengths", "euler_sbend", "euler_ramp_length", "euler_sbend_from_ramp", "euler_sbend_fixed_ramp", "bend_curvature",
    "parallel_path", "path_list", "eval_width", "sidestep_offset", "sidestep_straight", "sidestep_angle", "delay_arm_parameters",
    "delay_arm", "euler_wave", "raised_cosine", "cosine_tapers",
    "Circuit", "mzi", "radiation_witness", "paths_to_circuit", "arc_end_point", "fit_path", "pathway_file_symbols", "ind2gds",
    "save_gds", "read_gds_shapes", "gds_waveguides", "gds_to_ind",
]