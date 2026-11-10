"""Bend-to-straight offset and bend loss from FemSIM conformal-transform bend modes.

A bend pushes the mode toward the outside of the curve, so a straight guide
butted against a bend couples with a mismatch. Shifting the bend guide toward
the inside by the right amount (`Straight_offset` in the .ind) recovers it.
That amount is the lateral shift of the bend mode which maximizes its
overlap with the straight-guide mode.

Expected files (from the FemSIM bend scan), for each radius and wavelength:

    <mode_dir>\\<root>_Wav1p450_Radius500p000.mon        mode list: z column = mode number
    <mode_dir>\\<root>_Wav1p450_Radius500p000_ex.m00     Ex field of mode 0 (.m01, .m02, ...)

The straight-guide reference is the fundamental mode of the straight
waveguide library (see modes.mode_field), by default the 3.4 um wide guide.
It is resampled onto the bend-mode grid if the grids differ. In a 3.4 x 3.4 um
guide the TE and TM modes are degenerate, so the stored mode may be a TE/TM
mixture; the overlap uses only the Ex field, normalized, and both polarizations
share the same Ex/Ey profile in a square guide, so the mixture does not matter
to first order. Both fields must have the guide centered at x = 0.

Offsets are signed: positive means "move the bend guide toward +x".

Main functions: straight_offset(radius, wav) and junction_offset(radius, wav). The (radius, wavelength) pair must
be one of the points computed in the FemSIM bend scan.

Bend loss (end of this module): the imaginary part of the TE bend mode's n_eff gives a loss in dB/cm
(build_bend_loss_table); get_loss looks it up in the resulting table and sbend_loss gives the loss
of a whole S-bend.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.optimize import minimize_scalar

from .fields import FieldData, read_mode, regrid
from .geometry import sbend_lengths
from .modes import MODE_LIB_DIR, mode_field
from .monitors import read_monitor
from .utils import param_str

BEND_MODE_DIR = r"D:\DATA\ActivePIC\SingleWaveguideConvergence\StraightWaveguide_simulatedBend_neff_table"
BEND_MODE_ROOT = "StraightWaveguide_simulatedBend"


# --------------------------------------------------------------------------- #
# Locating and reading the TE mode
# --------------------------------------------------------------------------- #

def bend_stem(radius, wav, mode_dir=BEND_MODE_DIR, root=BEND_MODE_ROOT):
    """Common path prefix of the files for one (radius, wavelength)."""
    return os.path.join(mode_dir, f"{root}_Wav{param_str(wav)}_Radius{param_str(radius)}")


def te_mode_index(radius, wav, mode_dir=BEND_MODE_DIR, root=BEND_MODE_ROOT, n_candidates=2):
    """Mode number of the fundamental TE (Ex-dominant) mode.

    Takes the `n_candidates` modes with the largest first-monitor value in
    the .mon file and returns the one whose Ex field has the largest peak.
    """
    stem = bend_stem(radius, wav, mode_dir, root)
    mon = read_monitor(stem + ".mon")
    cand = mon.z[np.argsort(mon.values[:, 0])[::-1]][:n_candidates].astype(int)
    peaks = [read_mode(f"{stem}_ex.m{k:02d}").peak_amp() for k in cand]
    return int(cand[int(np.argmax(peaks))])


def read_te_mode(radius, wav, mode_dir=BEND_MODE_DIR, root=BEND_MODE_ROOT, n_candidates=2):
    """Ex field (FieldData) of the fundamental TE mode at this radius and wavelength."""
    k = te_mode_index(radius, wav, mode_dir, root, n_candidates)
    return read_mode(f"{bend_stem(radius, wav, mode_dir, root)}_ex.m{k:02d}")


@lru_cache(maxsize=64)
def read_straight_mode(wav, width=3.4, mode_dir=MODE_LIB_DIR):
    """Ex field (FieldData) of the straight-guide fundamental mode from the mode library."""
    return read_mode(mode_field(wav, width, mode_dir=mode_dir)[0])


# --------------------------------------------------------------------------- #
# Offset from two modes
# --------------------------------------------------------------------------- #

def mode_centroid_x(f: FieldData) -> float:
    """Intensity-weighted centroid of |E|^2 along x (um)."""
    p = (f.amp ** 2).sum(axis=0)            # sum over y -> (nx,)
    return float((f.x * p).sum() / p.sum())


def best_offset(f_straight: FieldData, f_bend: FieldData, smax=0.5):
    """Shift of the bend guide that maximizes overlap with the straight mode.

    Shifting the bend mode by s along x is equivalent to moving the bend
    guide by s. Uses the scalar overlap of the signed Ex fields.

    Parameters
    ----------
    f_straight, f_bend : FieldData
        If their grids differ, the straight mode is resampled onto the bend grid.
    smax : search range +-smax (um)

    Returns
    -------
    s_star : float        optimal shift (um, + = toward +x)
    eta_0 : float         power overlap with no shift
    eta_best : float      power overlap at s_star (1 - eta = mismatch loss)
    """
    f_straight = regrid(f_straight, f_bend)
    dx = f_straight.x[1] - f_straight.x[0]
    a, b = f_straight.signed_field(), f_bend.signed_field()
    na, nb = np.sqrt((a ** 2).sum()), np.sqrt((b ** 2).sum())

    def eta(s):
        bs = nd_shift(b, (0, s / dx), order=3, mode="constant", cval=0.0)
        return ((a * bs).sum() / (na * nb)) ** 2

    res = minimize_scalar(lambda s: -eta(s), bounds=(-smax, smax),
                          method="bounded", options={"xatol": 1e-4})
    return float(res.x), float(eta(0.0)), float(-res.fun)


def straight_offset(radius, wav, ref_width=3.4, mode_dir=BEND_MODE_DIR, root=BEND_MODE_ROOT,
                    ref_dir=MODE_LIB_DIR, smax=0.5, details=False):
    """Signed bend-to-straight offset (um) for one (radius, wavelength).

    Positive = move the bend guide toward +x; the magnitude is the value for
    `Straight_offset` in the .ind. The reference is the straight-guide mode of
    width `ref_width` from `ref_dir`; the bend mode comes from the FemSIM scan
    in `mode_dir`. Raises FileNotFoundError if that (radius, wav) was not scanned.

    With details=True, returns a dict instead: offset, centroid_shift (an
    independent estimate from the mode centroids), eta_no_offset and
    eta_offset (power overlap without and with the offset; 1 - eta is the
    mismatch loss).
    """
    ref = read_straight_mode(wav, ref_width, ref_dir)
    bend = read_te_mode(radius, wav, mode_dir, root)
    s, eta0, eta1 = best_offset(ref, bend, smax)
    if not details:
        return s
    return dict(offset=s,
                centroid_shift=-(mode_centroid_x(bend) - mode_centroid_x(regrid(ref, bend))),
                eta_no_offset=eta0, eta_offset=eta1)


def mirror_x(f: FieldData) -> FieldData:
    """The field reflected about x = 0, i.e. the mode of the same bend curving the other way.

    (The sign of the Ex component also flips under this reflection; the overlap used here does not
    depend on that sign.)
    """
    g = FieldData(x=-f.x[::-1].copy(), y=f.y.copy(), amp=f.amp[:, ::-1].copy(), phase=f.phase[:, ::-1].copy(),
                  wavelength=f.wavelength, y_extra_tokens=list(f.y_extra_tokens),
                  preamble=list(f.preamble), keyword=f.keyword)
    return regrid(g, f)


def junction_offset(radius, wav, radius2=None, same_direction=False, mode_dir=BEND_MODE_DIR,
                    root=BEND_MODE_ROOT, ref_dir=MODE_LIB_DIR, smax=2.0, details=False):
    """Optimal bend-to-bend offset (um) at the junction of two circular arcs.

    The field leaving the first arc (radius `radius`) is its bend mode, displaced toward the outside of that
    arc. This finds the lateral shift of the second arc (radius `radius2`, default: the same radius) that maximizes
    the overlap between the two arcs' bend modes at the junction.

    same_direction=False (default): the arcs curve in opposite directions, as in an S-bend; the second arc's mode is
    the mirror image of the mode of a bend of radius2. To first order the offset is the sum of the two bend-to-straight
    offsets, which is twice the straight offset when the radii are equal.
    same_direction=True: both arcs curve the same way (a compound curve); to first order the offset is the difference
    of the two bend-to-straight offsets.

    The result is the shift to apply to the second arc, signed in the frame of the FemSIM bend (use abs() as for
    straight_offset). Both (radius, wavelength) points must be in the FemSIM bend scan.

    With details=True, returns a dict: offset, centroid_shift (the difference of the mode centroids, an independent
    estimate), eta_no_offset and eta_offset (power overlap of the two arcs' modes without and with the offset;
    1 - eta is the mismatch loss at the junction), straight_offset_1 and straight_offset_2 (the bend-to-straight offsets
    of the two radii, computed against the straight modes in ref_dir, or None), first_order (their sum or difference)
    and ratio (|offset| / first_order).
    """
    radius2 = radius if radius2 is None else radius2
    bend1 = read_te_mode(radius, wav, mode_dir, root)
    bend2 = bend1 if radius2 == radius else read_te_mode(radius2, wav, mode_dir, root)
    second = bend2 if same_direction else mirror_x(bend2)         # the second arc's mode in the frame of the first
    s, eta0, eta1 = best_offset(bend1, second, smax)
    if not details:
        return s
    try:
        s1 = abs(straight_offset(radius, wav, mode_dir=mode_dir, root=root, ref_dir=ref_dir))
        s2 = s1 if radius2 == radius else abs(straight_offset(radius2, wav, mode_dir=mode_dir, root=root, ref_dir=ref_dir))
        first_order = abs(s1 - s2) if same_direction else s1 + s2
    except (FileNotFoundError, OSError, ValueError):
        s1 = s2 = first_order = None
    return dict(offset=s,
                centroid_shift=mode_centroid_x(bend1) - mode_centroid_x(second),
                eta_no_offset=eta0, eta_offset=eta1,
                straight_offset_1=s1, straight_offset_2=s2, first_order=first_order,
                ratio=(abs(s) / first_order) if first_order else None)


def bend_offsets(radii, directions, wav, mode_dir=BEND_MODE_DIR, root=BEND_MODE_ROOT, ref_dir=MODE_LIB_DIR,
                 exact=False):
    """Signed lateral shifts (um, in CAD x) for a chain of circular arcs, from the bend-mode displacement.

    A bend mode is displaced toward the OUTSIDE of its arc, so each arc is shifted toward its INSIDE (the side of
    its center of curvature) by the displacement of its radius. Parameters:

      radii       radius of each arc in order, e.g. [500, 500] for an S-bend
      directions  +1 if the arc turns toward +x as z increases (center of curvature at +x), -1 if toward -x;
                  an S-bend is [+1, -1] (or [-1, +1])
      wav         wavelength (um); each radius must be in the FemSIM bend scan

    Returns a dict with
      entry       shift of the first arc relative to the straight before it
      junctions   list: shift of each arc relative to the end of the previous one
      exit        shift of the straight after the last arc relative to that arc's end
    The shifts add up to zero, so the straights before and after the chain stay where the geometry puts them.

    With exact=False the displacement of a radius is abs(straight_offset); with exact=True each junction magnitude
    comes from the full mode overlap of the two arcs (junction_offset) and only its sign from the rule above.
    """
    delta = [abs(straight_offset(R, wav, mode_dir=mode_dir, root=root, ref_dir=ref_dir)) for R in radii]
    shifts = [d * k for d, k in zip(directions, delta)]                  # each arc, toward its inside
    junctions = [shifts[k + 1] - shifts[k] for k in range(len(radii) - 1)]
    if exact:
        for k in range(len(junctions)):
            same = directions[k] == directions[k + 1]
            magnitude = abs(junction_offset(radii[k], wav, radii[k + 1], same_direction=same, mode_dir=mode_dir,
                                            root=root, ref_dir=ref_dir))
            junctions[k] = np.sign(junctions[k]) * magnitude
    return dict(entry=shifts[0], junctions=junctions, exit=-shifts[-1])


# --------------------------------------------------------------------------- #
# Bend loss from the imaginary part of n_eff, and lookup of the loss table
#
# 1. build_bend_loss_table turns Im(n_eff) of the TE bend mode at each (radius, wavelength) into a
#    loss in dB/cm and writes the CSV (radius, wavelength, loss_dB_per_cm).
# 2. get_loss(radius, wavelength) reads that CSV and interpolates.
# 3. sbend_loss combines it with the S-bend length (geometry.sbend_lengths) into the expected loss of
#    a whole S-bend, as a dB value and as a fraction of the power.
# --------------------------------------------------------------------------- #

BEND_LOSS_CSV = (r"D:\DATA\ActivePIC\SingleWaveguideConvergence\StraightWaveguide_simulatedBend_neff_table"
                 r"\StraightWaveguide_simulatedBend_bendloss_dB_per_cm.csv")


def loss_dB_per_cm_from_neff_imag(neff_imag, wav):
    """Power loss (dB/cm) from the imaginary part of n_eff at wavelength wav (um).

    The field decays as exp(-k0*Im(n_eff)*z), so the power decays with alpha = 4*pi*|Im(n_eff)|/wav (1/um).
    """
    alpha_per_um = 4 * np.pi * np.abs(neff_imag) / wav
    return alpha_per_um * 1e4 * 10 / np.log(10)


@lru_cache(maxsize=4)
def load_loss_table(path=BEND_LOSS_CSV):
    """The bend-loss table as {radius: (wavelengths, loss_dB_per_cm)}, sorted by radius. Cached per path;
    call load_loss_table.cache_clear() after regenerating the CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    table = {}
    for r, sub in df.groupby("radius"):
        sub = sub.sort_values("wavelength")
        table[float(r)] = (sub.wavelength.to_numpy(float), sub.loss_dB_per_cm.to_numpy(float))
    return dict(sorted(table.items()))


def get_loss(radius, wavelength, loss_file=None, log_radius=False, strict=False):
    """Bend loss (dB/cm) at a given bend radius (um) and wavelength (um) from the loss table.

    Linear interpolation in wavelength at each tabulated radius, then in radius. Loss falls roughly
    exponentially with radius, so log_radius=True interpolates log(loss) in radius instead, which is
    more accurate between widely spaced radii. Outside the table, np.interp clamps to the end values;
    with strict=True a ValueError is raised instead. radius is a single value; wavelength may be an array.
    """
    table = load_loss_table(loss_file or BEND_LOSS_CSV)
    radii = np.array(list(table))
    if strict:
        wmin = max(w[0] for w, _ in table.values())
        wmax = min(w[-1] for w, _ in table.values())
        if not (radii[0] <= radius <= radii[-1]) or np.any(np.asarray(wavelength) < wmin) \
                or np.any(np.asarray(wavelength) > wmax):
            raise ValueError(f"(radius={radius}, wavelength={wavelength}) is outside the table "
                             f"(radii {radii[0]:g}-{radii[-1]:g} um, wavelengths {wmin:g}-{wmax:g} um)")
    at_radii = np.array([np.interp(wavelength, w, loss) for w, loss in table.values()])   # (n_radii[, n_wav])

    def over_radius(values):
        if log_radius:
            return np.exp(np.interp(radius, radii, np.log(values)))
        return np.interp(radius, radii, values)

    if at_radii.ndim == 1:
        return over_radius(at_radii)
    return np.array([over_radius(at_radii[:, j]) for j in range(at_radii.shape[1])])     # wavelength array


def sbend_loss(R, d, wav, junction_offset=0.0, straight_offset=0.0, loss_file=None, log_radius=False):
    """Expected loss of a circular S-bend of radius R (um) and offset d (um) at wavelength wav (um).

    Uses the loss table along the arc length of the bend (geometry.sbend_lengths). Returns
    (loss_dB, loss_fraction): the total loss in dB and the fraction of the power lost, 1 - 10**(-dB/10).
    wav may be an array.
    """
    arc_length_um = sbend_lengths(R, d, junction_offset, straight_offset)[0]
    loss_dB = get_loss(R, wav, loss_file, log_radius) * arc_length_um * 1e-4      # dB/cm * cm
    return loss_dB, 1 - 10 ** (-loss_dB / 10)


def build_bend_loss_table(radii, wavs, mode_dir=BEND_MODE_DIR, root=BEND_MODE_ROOT, csv_path=None):
    """Bend-loss table from the FemSIM bend-mode scan, as a DataFrame (saved to csv_path if given).

    For each (radius, wavelength) it finds the fundamental TE mode (te_mode_index), reads the imaginary
    part of its n_eff from the .nef file (assumed to be column 2 of the row for that mode) and converts
    it to dB/cm. Columns: radius, wavelength, neff_imag, loss_dB_per_cm. Missing runs are skipped.
    """
    import pandas as pd

    rows, skipped = [], 0
    for R in radii:
        for wav in wavs:
            try:
                k = te_mode_index(R, wav, mode_dir, root)
                nef = np.atleast_2d(np.genfromtxt(bend_stem(R, wav, mode_dir, root) + ".nef"))
                im = nef[k, 2]
            except (FileNotFoundError, OSError, IndexError, ValueError):
                skipped += 1
                continue
            rows.append(dict(radius=R, wavelength=wav, neff_imag=im,
                             loss_dB_per_cm=float(loss_dB_per_cm_from_neff_imag(im, wav))))
    if skipped:
        print(f"Warning: {skipped} (radius, wavelength) combinations were skipped (missing or unreadable files).")
    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df