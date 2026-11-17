"""Settings and paths shared by the NullingPIC Gen2 device scripts (P-00, P-02, P-03, P-05, ...).

Every device is written, simulated and analysed with the settings below, so change a numerical setting here once rather than in
each script. The straight reference (P-00) normalizes the other devices only if it is run with the same settings.
Use from a device script as: from rsoft_lib.device_setup import SIM, WAVELENGTHS, WIDTH, device_paths, export_device
"""
import os

from .modes import mode_field

BASE_DIR = r"D:\DATA\ETL_PIC\simulation\NullingPIC_Gen2_sims"

WIDTH = 3.4                                   # um, guide width (and height) of every device
DESIGN_WAVELENGTH = 1.55                      # um, the wavelength written into the .ind files
WAVELENGTHS = [1.45, 1.50, 1.55, 1.60, 1.65]  # um, the BeamPROP scan; each must be in the straight mode library

# Numerical settings of every BeamPROP run (Circuit.write options). Step_z and pade_order as chosen in the S-bend convergence study.
SIM = dict(Grid_Size=0.1, Grid_Edge=0.5, Step_z=0.1, pade_order=2, Padding=20.0, PML_width=5.0)


def device_paths(filename_root, device_name=None):
    """Folder and file names of one device: dict(name, working_dir, ind, gds). The outputs of each scan go to scan_dir(paths, ...)."""
    name = device_name or filename_root
    working_dir = os.path.join(BASE_DIR, filename_root)
    return dict(name=name, working_dir=working_dir, ind=os.path.join(working_dir, f"{name}.ind"),
                gds=os.path.join(working_dir, f"{name}.gds"))


def export_device(c, paths, layer=1):
    """Write the device's .ind (shared numerical settings, the mode of the design wavelength on every pathway) and its GDS.
    The wavelength scan (scan_wavelengths) then sets the wavelength and the mode files of each run."""
    os.makedirs(paths["working_dir"], exist_ok=True)
    mode = mode_field(DESIGN_WAVELENGTH, WIDTH)[0]
    n = len(c.pathways)
    return c.export(paths["ind"], paths["gds"], layer=layer, wavelength=DESIGN_WAVELENGTH,
                    launch_files=[mode] * n, monitor_files=[mode] * n, **SIM)

def scan_dir(paths, scan):
    """The folder for the outputs of one scan of a device: <working_dir>\\<name>_scan_<scan>, e.g. scan="Wav" -> ..._scan_Wav,
    scan="Stepz" -> ..._scan_Stepz."""
    return os.path.join(paths["working_dir"], f"{paths['name']}_scan_{scan}")