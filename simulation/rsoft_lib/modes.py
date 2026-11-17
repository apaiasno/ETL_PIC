"""Mode library: fundamental-mode files and n_eff for straight waveguides."""
import os
import re
from functools import lru_cache

import numpy as np

from .utils import interp_pchip, param_str, parse_param

# Directory of StraightWaveguide_Wav{wav}_Width{width}.nef / _ex.m00 / _ex.m01 files.
MODE_LIB_DIR = r"D:\DATA\ETL_PIC\simulation\rsoft_lib\StraightWaveguide_neff_table"


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


# --------------------------------------------------------------------------- #
# n_eff of straight guides as a function of wavelength and width
# --------------------------------------------------------------------------- #

_STRAIGHT_NEF = re.compile(r"^StraightWaveguide_Wav(\d+p\d+)_Width(\d+p\d+)\.nef$", re.IGNORECASE)


class StraightModel:
    """n_eff of the Ex-dominant mode of the straight guides in a mode library, interpolated in wavelength and width.

    Reads every StraightWaveguide_Wav<wav>_Width<width>.nef in mode_dir (one row per mode: index, Re n_eff and, if there is a
    third column, Im n_eff; the row used is the one of the Ex-dominant mode, as in mode_field). The wavelength is interpolated first
    (PCHIP, per width), then the width (PCHIP), and nothing is extrapolated: a wavelength or width outside the table raises
    ValueError, or with clamp=True uses the nearest width in the table (the result says which values were clamped)."""

    def __init__(self, mode_dir=MODE_LIB_DIR, height=3.4):
        self.mode_dir, self.height = mode_dir, height
        found = {}
        for name in os.listdir(mode_dir):
            hit = _STRAIGHT_NEF.match(name)
            if not hit:
                continue
            wav, width = parse_param(hit.group(1)), parse_param(hit.group(2))
            nef = np.atleast_2d(np.genfromtxt(os.path.join(mode_dir, name)))
            row = ex_mode_index(width, height)
            if row >= len(nef):
                continue
            im = float(nef[row, 2]) if nef.shape[1] > 2 else 0.0
            found.setdefault(width, []).append((wav, float(nef[row, 1]), im))
        if not found:
            raise FileNotFoundError(f"No StraightWaveguide_Wav*_Width*.nef files in {mode_dir}")
        self.data = {}
        for width, rows in sorted(found.items()):
            a = np.array(sorted(rows))
            self.data[width] = (a[:, 0], a[:, 1], a[:, 2])
        self.widths = np.array(list(self.data))
        self.wavelengths = np.array(sorted({w for wv, _, _ in self.data.values() for w in wv}))
        self._cache = {}
        self._grid_key, self._grid_val = None, None

    def _along_width(self, wav):
        """Widths whose table covers wav, with Re and Im n_eff at wav."""
        key = round(float(wav), 9)
        if key not in self._cache:
            ws, re_, im_ = [], [], []
            for width, (wv, r, i) in self.data.items():
                a = interp_pchip(wv, r, [wav])[0]
                if np.isfinite(a):
                    ws.append(width)
                    re_.append(a)
                    im_.append(interp_pchip(wv, i, [wav])[0])
            self._cache[key] = (np.array(ws), np.array(re_), np.array(im_))
        return self._cache[key]

    def _grid(self, lams):
        """Re and Im n_eff of every table width at the wavelengths lams: two arrays (n_widths, len(lams)), NaN where the width's own
        table does not cover the wavelength. Kept for the latest lams only."""
        key = lams.tobytes()
        if self._grid_key != key:
            re_, im_ = np.empty((len(self.widths), len(lams))), np.empty((len(self.widths), len(lams)))
            for k, width in enumerate(self.widths):
                wv, r, i = self.data[width]
                re_[k], im_[k] = interp_pchip(wv, r, lams), interp_pchip(wv, i, lams)
            self._grid_key, self._grid_val = key, (re_, im_)
        return self._grid_val

    def neff(self, wav, width, clamp=False):
        """Complex n_eff (Re from the table, Im the table's Im n_eff, 0 if it has none) at the wavelengths and widths given (they
        broadcast). Returns (neff, clamped): clamped is True where the width was outside the table and clamp=True used the nearest.
        Vectorized over the wavelengths: the wavelength is interpolated for all of them at once, then the width."""
        from scipy.interpolate import PchipInterpolator
        wav, width = np.broadcast_arrays(np.asarray(wav, float), np.asarray(width, float))
        flat_w, flat_x = wav.ravel(), width.ravel()
        out = np.empty(flat_w.shape, complex)
        clamped = np.zeros(flat_w.shape, bool)
        lams, inverse = np.unique(flat_w, return_inverse=True)
        re_, im_ = self._grid(lams)
        full = ~np.isnan(re_).any(axis=0)                                           # wavelengths covered by every width's table
        sel = full[inverse]
        if sel.any():
            ws = self.widths
            xs = flat_x[sel]
            bad = (xs < ws[0] - 1e-9) | (xs > ws[-1] + 1e-9)
            if bad.any() and not clamp:
                raise ValueError(f"Width {xs[bad][0]:g} um is outside the straight-guide table ({ws[0]:g}-{ws[-1]:g} um) in {self.mode_dir}; "
                                 f"clamp=True uses the nearest width")
            xc = np.clip(xs, ws[0], ws[-1])
            cols = (np.cumsum(full) - 1)[inverse[sel]]
            if len(ws) == 1:
                out[sel] = re_[0, full][cols] + 1j * im_[0, full][cols]
            else:
                uw, iw = np.unique(xc, return_inverse=True)
                out[sel] = (PchipInterpolator(ws, re_[:, full], axis=0)(uw)[iw, cols]
                            + 1j * PchipInterpolator(ws, im_[:, full], axis=0)(uw)[iw, cols])
            clamped[sel] = bad
        for lam in lams[~full]:                                                     # wavelengths some widths do not cover: per wavelength
            ws, r, i = self._along_width(lam)
            if len(ws) == 0:
                raise ValueError(f"Wavelength {lam:g} um is outside the straight-guide table "
                                 f"({self.wavelengths[0]:g}-{self.wavelengths[-1]:g} um) in {self.mode_dir}")
            sel = flat_w == lam
            wq = flat_x[sel]
            bad = (wq < ws[0] - 1e-9) | (wq > ws[-1] + 1e-9)
            if bad.any() and not clamp:
                raise ValueError(f"Width {wq[bad][0]:g} um is outside the straight-guide table ({ws[0]:g}-{ws[-1]:g} um) in {self.mode_dir}; "
                                 f"clamp=True uses the nearest width")
            wc = np.clip(wq, ws[0], ws[-1])
            out[sel] = interp_pchip(ws, r, wc) + 1j * interp_pchip(ws, i, wc)
            clamped[sel] = bad
        return out.reshape(wav.shape), clamped.reshape(wav.shape)


@lru_cache(maxsize=4)
def straight_model(mode_dir=MODE_LIB_DIR, height=3.4):
    """The StraightModel of a mode library, read once per directory (call straight_model.cache_clear() after adding files)."""
    return StraightModel(mode_dir, height)


def straight_neff(wav, width, mode_dir=MODE_LIB_DIR, height=3.4, clamp=False):
    """Complex n_eff of the Ex-dominant mode of a straight guide of this width (um) at this wavelength (um), interpolated in the
    mode library's tables (see StraightModel). Scalars or arrays."""
    return straight_model(mode_dir, height).neff(wav, width, clamp)[0][()]