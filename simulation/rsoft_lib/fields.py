"""Transverse field data: OUTPUT_AMP_PHASE_3D files (.fld, .m00/.m01 mode files).

Conventions
-----------
FieldData.x is the first axis in the file header (per-row coordinate),
FieldData.y the second. `amp` and `phase` have shape (ny, nx), so
`plt.pcolormesh(field.x, field.y, field.amp)` works with no transpose.
**Phase is in degrees** throughout.

.fld layout (reverse-engineered from exported files, not an official spec):

    /rn/ls1
    /r
    /rq
    <n1> <min1> <max1> <extra...> OUTPUT_AMP_PHASE_3D Wavelength=<wl>
    <n2> <min2> <max2>
    0 <v2_0> <v2_1> ... <v2_{n2-1}>
    <v1_0> <amp_00> <phase_00> <amp_01> <phase_01> ...
    ...

.m00/.m01 mode-file layout (no coordinate rows; coordinates are evenly spaced):

    /rn,a,b/nx0/ls1
    /r,qa,qb
    <n1> <min1> <max1> <extra...> OUTPUT_AMP_PHASE_3D <neff> 0 Wavelength=<wl>
    <n2> <min2> <max2>
    <amp_00> <phase_00> <amp_01> <phase_01> ...   (one row per value of axis 1)

Main API: read_fld, read_mode, write_fld, add_fields (or `a + b`), regrid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import numpy as np


@dataclass
class FieldData:
    x: np.ndarray                   # (nx,)
    y: np.ndarray                   # (ny,)
    amp: np.ndarray                 # (ny, nx)
    phase: np.ndarray               # (ny, nx), degrees
    wavelength: Optional[float] = None
    y_extra_tokens: List[str] = dc_field(default_factory=list)  # tokens between max1 and the keyword
    preamble: List[str] = dc_field(default_factory=lambda: ["/rn/ls1", "/r", "/rq"])
    keyword: str = "OUTPUT_AMP_PHASE_3D"

    @property
    def nx(self) -> int:
        return self.x.size

    @property
    def ny(self) -> int:
        return self.y.size

    def complex_field(self) -> np.ndarray:
        """E = amp * exp(i*phase), shape (ny, nx)."""
        return self.amp * np.exp(1j * np.deg2rad(self.phase))

    def signed_field(self) -> np.ndarray:
        """Real part of the complex field: amp*cos(phase). Useful for plotting a mode with its sign."""
        return self.amp * np.cos(np.deg2rad(self.phase))

    def peak_amp(self) -> float:
        return float(np.max(self.amp))

    def edge_amp_max(self) -> float:
        """Max amplitude anywhere on the outer boundary of the grid."""
        edge = np.concatenate([self.amp[0, :], self.amp[-1, :], self.amp[:, 0], self.amp[:, -1]])
        return float(np.max(edge))

    def __add__(self, other):
        if not isinstance(other, FieldData):
            return NotImplemented
        return add_fields(self, other)

    def __radd__(self, other):
        # lets sum([f1, f2, f3]) work (sum starts from 0 + f1)
        if other == 0:
            return self
        if not isinstance(other, FieldData):
            return NotImplemented
        return add_fields(other, self)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

_WAVELENGTH_RE = re.compile(r"Wavelength\s*=\s*([-+0-9.eE]+)")


def _parse_header_line4(line: str):
    """Parse '933 -20 70 19 OUTPUT_AMP_PHASE_3D Wavelength=1.55'.

    Returns (n1, min1, max1, extra_tokens, keyword, wavelength). Numeric tokens
    after max1 are kept as `extra_tokens`; the first non-numeric token is the keyword.
    """
    tokens = line.split()
    n1, min1, max1 = int(float(tokens[0])), float(tokens[1]), float(tokens[2])

    extra_tokens, keyword = [], None
    idx = 3
    while idx < len(tokens):
        tok = tokens[idx]
        idx += 1
        try:
            float(tok)
            extra_tokens.append(tok)
        except ValueError:
            keyword = tok
            break

    m = _WAVELENGTH_RE.search(" ".join(tokens[idx:]))
    wavelength = float(m.group(1)) if m else None
    return n1, min1, max1, extra_tokens, keyword, wavelength


def read_fld(path: str) -> FieldData:
    """Read an OUTPUT_AMP_PHASE_3D .fld file (with coordinate rows)."""
    with open(path, "r") as f:
        content = [ln.rstrip("\n") for ln in f if ln.strip()]

    preamble = content[0:3]
    n1, min1, max1, extra, keyword, wavelength = _parse_header_line4(content[3])
    n2 = int(float(content[4].split()[0]))

    # first token on the coordinate row is a placeholder 0
    col_coords = np.array(content[5].split(), dtype=float)[1:1 + n2]
    if col_coords.size != n2:
        raise ValueError(f"{path}: expected {n2} column coordinates, found {col_coords.size}.")

    data_lines = content[6:6 + n1]
    if len(data_lines) != n1:
        raise ValueError(f"{path}: expected {n1} data rows, found {len(data_lines)}.")
    data = np.loadtxt(data_lines, ndmin=2)
    if data.shape != (n1, 1 + 2 * n2):
        raise ValueError(f"{path}: data shape {data.shape}, expected {(n1, 1 + 2 * n2)}.")

    return FieldData(
        x=data[:, 0],
        y=col_coords,
        amp=data[:, 1::2].T,
        phase=data[:, 2::2].T,
        wavelength=wavelength,
        y_extra_tokens=extra,
        preamble=preamble,
        keyword=keyword or "OUTPUT_AMP_PHASE_3D",
    )


def read_mode(path: str) -> FieldData:
    """Read an RSoft mode file (.m00, .m01, ...), which has no coordinate rows.

    Coordinates are rebuilt as evenly spaced points from the header ranges.
    The first header axis is taken as x and the second as y, as in read_fld;
    if a plot of an asymmetric mode looks rotated 90 degrees, that assumption
    is the thing to check.
    """
    with open(path, "r") as f:
        f.readline()
        f.readline()
        n1, min1, max1, extra, keyword, wavelength = _parse_header_line4(f.readline())
        axis2 = f.readline().split()
    n2, min2, max2 = int(float(axis2[0])), float(axis2[1]), float(axis2[2])

    data = np.loadtxt(path, skiprows=4, ndmin=2)
    if data.shape != (n1, 2 * n2):
        raise ValueError(f"{path}: data shape {data.shape}, expected {(n1, 2 * n2)}.")

    return FieldData(
        x=np.linspace(min1, max1, n1),
        y=np.linspace(min2, max2, n2),
        amp=data[:, 0::2].T,
        phase=data[:, 1::2].T,
        wavelength=wavelength,
        y_extra_tokens=extra,
        keyword=keyword or "OUTPUT_AMP_PHASE_3D",
    )


# --------------------------------------------------------------------------- #
# Combining
# --------------------------------------------------------------------------- #

def add_fields(*fields: FieldData, grid: Optional[FieldData] = None) -> FieldData:
    """Coherently add two or more fields (complex sum).

    Fields on the same grid are summed directly; otherwise each is linearly
    interpolated (real and imaginary parts separately, zero outside its
    extent) onto the target grid. The target grid is `grid` if given, else
    the first field's. Metadata is copied from the target.
    """
    if len(fields) < 2:
        raise ValueError("Provide at least two fields to add.")

    target = grid if grid is not None else fields[0]
    tx, ty = target.x, target.y
    total = np.zeros((ty.size, tx.size), dtype=complex)

    for fd in fields:
        same_grid = (fd.x.shape == tx.shape and np.allclose(fd.x, tx)
                     and fd.y.shape == ty.shape and np.allclose(fd.y, ty))
        total += fd.complex_field() if same_grid else _interp_complex(fd, tx, ty)

    return FieldData(
        x=tx.copy(), y=ty.copy(),
        amp=np.abs(total), phase=np.rad2deg(np.angle(total)),
        wavelength=target.wavelength,
        y_extra_tokens=list(target.y_extra_tokens),
        preamble=list(target.preamble),
        keyword=target.keyword,
    )


def _interp_complex(fd: FieldData, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """Linearly interpolate a field's complex values onto the (tx, ty) grid."""
    from scipy.interpolate import RegularGridInterpolator

    cf = fd.complex_field()
    interp_re = RegularGridInterpolator((fd.y, fd.x), cf.real, bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator((fd.y, fd.x), cf.imag, bounds_error=False, fill_value=0.0)

    TY, TX = np.meshgrid(ty, tx, indexing="ij")
    pts = np.stack([TY.ravel(), TX.ravel()], axis=-1)
    return (interp_re(pts) + 1j * interp_im(pts)).reshape(ty.size, tx.size)


def regrid(f: FieldData, like: FieldData) -> FieldData:
    """Resample `f` onto the grid of `like` (linear interpolation, zero outside f's extent)."""
    if (f.x.shape == like.x.shape and np.allclose(f.x, like.x)
            and f.y.shape == like.y.shape and np.allclose(f.y, like.y)):
        return f
    cf = _interp_complex(f, like.x, like.y)
    return FieldData(
        x=like.x.copy(), y=like.y.copy(),
        amp=np.abs(cf), phase=np.rad2deg(np.angle(cf)),
        wavelength=f.wavelength, y_extra_tokens=list(f.y_extra_tokens),
        preamble=list(f.preamble), keyword=f.keyword,
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def write_fld(path: str, field: FieldData) -> None:
    """Write a FieldData as an OUTPUT_AMP_PHASE_3D .fld file (inverse of read_fld)."""
    nx, ny = field.nx, field.ny
    amp_raw, phase_raw = field.amp.T, field.phase.T      # (nx, ny): row-major file layout

    lines = list(field.preamble)

    wl = f" Wavelength={field.wavelength}" if field.wavelength is not None else ""
    extra = f" {' '.join(field.y_extra_tokens)}" if field.y_extra_tokens else ""
    lines.append(f"{nx} {field.x.min():g} {field.x.max():g}{extra} {field.keyword}{wl}")
    lines.append(f"{ny} {field.y.min():g} {field.y.max():g}")
    lines.append("0 " + " ".join(f"{v:.10g}" for v in field.y))

    for i in range(nx):
        row = [f"{field.x[i]:.10g}"]
        for j in range(ny):
            row += [f"{amp_raw[i, j]:.5E}", f"{phase_raw[i, j]:.5E}"]
        lines.append(" ".join(row))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def edge_to_peak_ratio(field: FieldData) -> float:
    """(max amplitude on the grid boundary) / (global peak amplitude). Small = domain is big enough."""
    return field.edge_amp_max() / field.peak_amp()