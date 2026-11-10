"""Mode library: fundamental-mode files and n_eff for straight waveguides."""
import os

import numpy as np

from .utils import param_str

# Directory of StraightWaveguide_Wav{wav}_Width{width}.nef / _ex.m00 / _ex.m01 files.
MODE_LIB_DIR = r"D:\DATA\ActivePIC\SingleWaveguideConvergence\StraightWaveguide_neff_table"


def ex_mode_index(width, height=3.4):
    """Index (0 or 1) of the Ex-dominant mode of a width x height guide.

    The fundamental mode is polarized along the longer cross-section
    dimension, so for width < height the Ex-dominant mode is the second
    (.m01); otherwise it is the fundamental (.m00). At width == height the
    two polarizations are degenerate, so the result is not meaningful there.
    """
    return 1 if round(width, 3) < height else 0


def mode_field(wav, width, mode_dir=MODE_LIB_DIR, height=3.4):
    """Filename and n_eff of the Ex-dominant mode (the one used for launching).

    Parameters
    ----------
    wav, width : float
        Wavelength and guide width in microns.
    mode_dir : str
        Directory of the mode library.
    height : float
        Guide height in microns (sets which mode is Ex-dominant).

    Returns
    -------
    mode_filename : str
        Mode field file for launch injection or monitor reference.
    neff : float
    """
    idx = ex_mode_index(width, height)
    stem = os.path.join(mode_dir, f"StraightWaveguide_Wav{param_str(wav)}_Width{param_str(width)}")
    nef = np.atleast_2d(np.genfromtxt(stem + ".nef"))     # one row per mode: [index, neff]
    return f"{stem}_ex.m0{idx}", float(nef[idx, 1])


def _lobes(profile, threshold):
    """Number of lobes (sign-alternating regions above threshold*peak) in a 1-D profile."""
    p = profile[np.abs(profile) > threshold * np.abs(profile).max()]
    sign = np.sign(p)
    return 1 + int((sign[1:] != sign[:-1]).sum())


def classify_modes(stem, threshold=0.05):
    """Polarization, lobe counts and symmetry of every mode computed by FemSIM.

    Reads `<stem>.nef` and `<stem>_ex.mNN` / `<stem>_ey.mNN` for each mode NN.

    Returns a DataFrame with one row per mode:
      number    mode number (the NN in the file name; modes are ordered by n_eff)
      neff      real part of the effective index
      pol       'Ex' or 'Ey' if one component dominates (peak ratio > 3), else 'mixed'
      x_lobes   lobes of the dominant component along x, through the field maximum
      y_lobes   lobes along y, through the field maximum
      parity_x  +1 even / -1 odd under x -> -x (NaN if the grid is not symmetric)
      parity_y  same for y -> -y

    A mode with x_lobes = 3 and y_lobes = 1 is the second laterally even mode
    (even modes have an odd number of lobes). Lobes smaller than
    `threshold` x the peak are ignored.
    """
    import pandas as pd
    from .fields import read_mode

    nef = np.atleast_2d(np.genfromtxt(stem + ".nef"))
    rows = []
    for k in range(len(nef)):
        ex = read_mode(f"{stem}_ex.m{k:02d}")
        ey = read_mode(f"{stem}_ey.m{k:02d}")
        ratio = ex.peak_amp() / ey.peak_amp()
        pol = "Ex" if ratio > 3 else "Ey" if ratio < 1 / 3 else "mixed"
        f = (ey if pol == "Ey" else ex).signed_field()          # shape (ny, nx)
        iy, ix = np.unravel_index(np.abs(f).argmax(), f.shape)
        norm = (f ** 2).sum()
        sym_x = np.allclose(ex.x, -ex.x[::-1], atol=1e-6)
        sym_y = np.allclose(ex.y, -ex.y[::-1], atol=1e-6)
        rows.append(dict(
            number=k, neff=float(nef[k, 1]), pol=pol,
            x_lobes=_lobes(f[iy, :], threshold), y_lobes=_lobes(f[:, ix], threshold),
            parity_x=float((f * f[:, ::-1]).sum() / norm) if sym_x else np.nan,
            parity_y=float((f * f[::-1, :]).sum() / norm) if sym_y else np.nan,
        ))
    return pd.DataFrame(rows)