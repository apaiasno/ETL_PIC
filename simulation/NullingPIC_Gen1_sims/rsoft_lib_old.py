# from __future__ import annotations

# """
# rsoft_field.py

# Utilities for reading, combining, and writing RSoft OUTPUT_AMP_PHASE_3D
# field-monitor files (.fld).

# Raw file layout this module expects (as inferred from a real exported file):

#     /rn/ls1
#     /r
#     /rq
#     <n1> <min1> <max1> <extra...> OUTPUT_AMP_PHASE_3D Wavelength=<wl>
#     <n2> <min2> <max2>
#     0 <v2_0> <v2_1> ... <v2_{n2-1}>
#     <v1_0> <amp_00> <phase_00> <amp_01> <phase_01> ... <amp_0,n2-1> <phase_0,n2-1>
#     <v1_1> <amp_10> <phase_10> ...
#     ...
#     <v1_{n1-1}> ...

# i.e. after a 3-line preamble and a 2-line grid header, there is one row of
# coordinates (prefixed with a placeholder "0"), followed by one data row per
# value of the *other* axis. Each data row starts with that row's coordinate
# and is then followed by (amplitude, phase) pairs, one pair per column
# coordinate.

# FieldData convention used by this module
# -----------------------------------------
# `FieldData.x` corresponds to the per-row coordinate (line 4 header/values),
# `FieldData.y` corresponds to the per-column coordinate (line 5/6
# header/values). `amp`/`phase` are stored with shape (ny, nx) — i.e. axis 0
# indexes y, axis 1 indexes x — which matches what plotting functions like
# `matplotlib.pyplot.pcolormesh(x, y, C)` expect (C.shape == (len(y), len(x))),
# so `plt.pcolormesh(field.x, field.y, field.amp)` works with no transpose
# needed.

# Main functions/methods:

#     read_fld(path)                  -> FieldData
#     FieldData.__add__ / add_fields  -> FieldData (coherent complex sum)
#     write_fld(path, field)          -> None

# NOTE: This parser was reverse-engineered from a sample file rather than an
# official format spec. It has been validated to round-trip the sample data
# byte-for-byte in terms of numeric content. If your files come from a
# different RSoft tool/version with a different header, check the header
# parsing (`_parse_header_line4`) and adjust as needed.

# Reads RSoft BeamPROP monitor files -- the plain-text output with a
# 5-line header (Monitor #, Path, Type, Tilt, Mode) followed by one data
# row per Z step:
 
#     Monitor #:                1             2             3             4
#     Monitor Path:             1             1             2             2
#     Monitor Type:      WG_POWER  LAUNCH_PHASE      WG_POWER  LAUNCH_PHASE
#     Monitor Tilt:             1             1             1             1
#     Monitor Mode:             0             0             0             0
#      0.000000E+00  8.016629E-01  0.000000E+00  0.000000E+00  0.000000E+00
#      1.000000E-01  8.017489E-01  4.039808E-02  0.000000E+00  5.814786E-02
#      ...
 
# This is a different file format from rsoft_field.py's OUTPUT_AMP_PHASE_3D
# .fld reader -- this one is a simple Z-vs-scalar time/propagation series
# per monitor, not a 2D transverse field snapshot.
# """

# from __future__ import annotations

# import re
# from dataclasses import dataclass, field as dc_field
# from typing import List, Optional, Dict, Tuple

# import numpy as np
# import json

# import sys
# rsoft_python_paths = [
#     r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\products\most\python",
#     r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\bin",
#     r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\products\python\python3.11\Lib\site-packages"
# ]
# for p in rsoft_python_paths:
#     if p not in sys.path:
#         sys.path.insert(0, p)

# from rstools import RSoftCircuit

# def read_all_symbols(ind_file):
#     symbols = {}
#     with open(ind_file) as f:
#         for line in f:
#             m = re.match(r'\s*([A-Za-z_]\w*)\s*=\s*(.+)', line.split('#')[0])
#             if m:
#                 symbols[m.group(1)] = m.group(2).strip()
#     return symbols

# def read_symbol(ind_file, symbol_name):
#     symbols = read_all_symbols(ind_file)
#     symbol = symbols[symbol_name]
#     return symbol

# def export_symbols(ind_file, out_file):
#     symbols = read_all_symbols(ind_file)
#     with open(out_file, 'w') as f:
#         json.dump(symbols, f, indent=2)
#     return symbols


# # --------------------------------------------------------------------------- #
# # Field Data container
# # --------------------------------------------------------------------------- #

# @dataclass
# class FieldData:
#     x: np.ndarray                  # shape (nx,)
#     y: np.ndarray                  # shape (ny,)
#     amp: np.ndarray                 # shape (ny, nx)
#     phase: np.ndarray               # shape (ny, nx), radians
#     wavelength: Optional[float] = None
#     y_extra_tokens: List[str] = dc_field(default_factory=list)  # tokens between max1 and the keyword on line 4
#     preamble: List[str] = dc_field(default_factory=lambda: ["/rn/ls1", "/r", "/rq"])
#     keyword: str = "OUTPUT_AMP_PHASE_3D"

#     @property
#     def nx(self) -> int:
#         return self.x.size

#     @property
#     def ny(self) -> int:
#         return self.y.size

#     def complex_field(self) -> np.ndarray:
#         """Return the complex field E = amp * exp(i*phase), shape (ny, nx)."""
#         return self.amp * np.exp(1j * self.phase)

#     def peak_amp(self) -> float:
#         return float(np.max(self.amp))

#     def edge_amp_max(self) -> float:
#         """Max amplitude found anywhere on the outer boundary of the grid."""
#         edge_vals = np.concatenate([
#             self.amp[0, :], self.amp[-1, :], self.amp[:, 0], self.amp[:, -1]
#         ])
#         return float(np.max(edge_vals))

#     # ----------------------------------------------------------------- #
#     # Operator support
#     # ----------------------------------------------------------------- #

#     def __add__(self, other: "FieldData") -> "FieldData":
#         if not isinstance(other, FieldData):
#             return NotImplemented
#         return add_fields(self, other)

#     def __radd__(self, other) -> "FieldData":
#         # Allows sum([field1, field2, field3]) to work, since sum() starts
#         # with 0 + field1.
#         if other == 0:
#             return self
#         if not isinstance(other, FieldData):
#             return NotImplemented
#         return add_fields(other, self)

# # --------------------------------------------------------------------------- #
# # Field Data container
# # --------------------------------------------------------------------------- #

# from __future__ import annotations


# @dataclass
# class MonitorData:
#     z: np.ndarray                  # shape (n_z,)
#     values: np.ndarray              # shape (n_z, n_monitors)
#     number: List[int]
#     path: List[int]
#     type: List[str]
#     tilt: List[int]
#     mode: List[int]

#     @property
#     def n_monitors(self) -> int:
#         return len(self.number)

#     def get(self, path: int = None, type: str = None) -> np.ndarray:
#         """Return the (n_z,) column(s) matching the given path and/or type.
#         If multiple monitors match, returns shape (n_z, n_matches)."""
#         idx = [
#             i for i in range(self.n_monitors)
#             if (path is None or self.path[i] == path)
#             and (type is None or self.type[i] == type)
#         ]
#         if not idx:
#             raise KeyError(f"No monitor found with path={path}, type={type}")
#         cols = self.values[:, idx]
#         return cols[:, 0] if cols.shape[1] == 1 else cols


# def _header_values(line: str) -> List[str]:
#     """Split a header line like 'Monitor Type:   WG_POWER  LAUNCH_PHASE'
#     into its values, after the label/colon."""
#     _, _, rest = line.partition(":")
#     return rest.split()


# def read_monitor(path: str) -> MonitorData:
#     with open(path, "r") as f:
#         lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]

#     number = [int(v) for v in _header_values(lines[0])]
#     mon_path = [int(v) for v in _header_values(lines[1])]
#     mon_type = _header_values(lines[2])
#     tilt = [int(v) for v in _header_values(lines[3])]
#     mode = [int(v) for v in _header_values(lines[4])]

#     n_monitors = len(number)
#     data_lines = lines[5:]
#     n_z = len(data_lines)

#     z = np.empty(n_z, dtype=float)
#     values = np.empty((n_z, n_monitors), dtype=float)

#     for i, line in enumerate(data_lines):
#         vals = np.fromstring(line, sep=" ")
#         if vals.size != 1 + n_monitors:
#             raise ValueError(
#                 f"Row {i} has {vals.size} values, expected {1 + n_monitors} "
#                 f"(Z + {n_monitors} monitors)."
#             )
#         z[i] = vals[0]
#         values[i, :] = vals[1:]

#     return MonitorData(
#         z=z, values=values,
#         number=number, path=mon_path, type=mon_type, tilt=tilt, mode=mode,
#     )
# # --------------------------------------------------------------------------- #
# # Reading
# # --------------------------------------------------------------------------- #

# _WAVELENGTH_RE = re.compile(r"Wavelength\s*=\s*([-+0-9.eE]+)")


# def _parse_header_line4(line: str):
#     """
#     Parse a line like:
#         933 -20 70 19 OUTPUT_AMP_PHASE_3D Wavelength=1.55
#     Returns (n1, min1, max1, extra_tokens, keyword, wavelength)
#     """
#     tokens = line.split()
#     n1 = int(float(tokens[0]))
#     min1 = float(tokens[1])
#     max1 = float(tokens[2])

#     # Everything after max1 up to (and not including) the first token that
#     # is not parseable as a float is "extra" numeric metadata (e.g. a slice
#     # index). The first non-numeric token is taken as the keyword.
#     extra_tokens = []
#     keyword = None
#     wavelength = None
#     idx = 3
#     while idx < len(tokens):
#         tok = tokens[idx]
#         try:
#             float(tok)
#             extra_tokens.append(tok)
#             idx += 1
#         except ValueError:
#             keyword = tok
#             idx += 1
#             break

#     rest = " ".join(tokens[idx:])
#     m = _WAVELENGTH_RE.search(rest)
#     if m:
#         wavelength = float(m.group(1))

#     return n1, min1, max1, extra_tokens, keyword, wavelength


# def read_fld(path: str) -> FieldData:
#     """Read an RSoft OUTPUT_AMP_PHASE_3D .fld file into a FieldData object.

#     Returns a FieldData where `.x` is the per-row (line 4) coordinate,
#     `.y` is the per-column (line 5/6) coordinate, and `.amp`/`.phase` have
#     shape (ny, nx) (transposed relative to the file's raw row/column
#     layout, so axis 0 is y and axis 1 is x — ready for
#     `plt.pcolormesh(field.x, field.y, field.amp)` with no extra transpose).
#     """
#     with open(path, "r") as f:
#         lines = [ln.rstrip("\n") for ln in f]

#     # Skip blank lines defensively but keep track of real content lines
#     content_lines = [ln for ln in lines if ln.strip() != ""]

#     preamble = content_lines[0:3]
#     n1, min1, max1, extra_tokens, keyword, wavelength = _parse_header_line4(content_lines[3])

#     n2_line_tokens = content_lines[4].split()
#     n2 = int(float(n2_line_tokens[0]))

#     col_tokens = [float(t) for t in content_lines[5].split()]
#     # First token is a placeholder (observed to be 0); the remaining n2
#     # tokens are the actual column coordinates.
#     col_coords = np.array(col_tokens[1:1 + n2], dtype=float)
#     if col_coords.size != n2:
#         raise ValueError(
#             f"Expected {n2} column coordinates, found {col_coords.size} "
#             f"in line: {content_lines[5][:80]}..."
#         )

#     row_coords = np.empty(n1, dtype=float)
#     amp_raw = np.empty((n1, n2), dtype=float)     # raw (row, col) = (line4, line5) order
#     phase_raw = np.empty((n1, n2), dtype=float)

#     data_lines = content_lines[6:6 + n1]
#     if len(data_lines) != n1:
#         raise ValueError(
#             f"Expected {n1} data rows, found {len(data_lines)} in file '{path}'."
#         )

#     for i, line in enumerate(data_lines):
#         vals = np.fromstring(line, sep=" ")
#         expected_len = 1 + 2 * n2
#         if vals.size != expected_len:
#             raise ValueError(
#                 f"Row {i} (coord={vals[0] if vals.size else '?'}) has {vals.size} "
#                 f"values, expected {expected_len}."
#             )
#         row_coords[i] = vals[0]
#         pairs = vals[1:].reshape(n2, 2)
#         amp_raw[i, :] = pairs[:, 0]
#         phase_raw[i, :] = pairs[:, 1]

#     # Relabel: x = row_coords (line 4), y = col_coords (line 5/6).
#     # amp_raw/phase_raw are (n1, n2) = (len(x), len(y)); transpose so that
#     # the stored amp/phase are (len(y), len(x)) = (ny, nx).
#     return FieldData(
#         x=row_coords,
#         y=col_coords,
#         amp=amp_raw.T,
#         phase=phase_raw.T,
#         wavelength=wavelength,
#         y_extra_tokens=extra_tokens,
#         preamble=preamble,
#         keyword=keyword or "OUTPUT_AMP_PHASE_3D",
#     )


# # --------------------------------------------------------------------------- #
# # Combining fields
# # --------------------------------------------------------------------------- #

# def add_fields(*fields: FieldData, grid: Optional[FieldData] = None) -> FieldData:
#     """
#     Coherently add two or more fields (complex sum: amp*exp(i*phase)).

#     If the fields share the same x/y grid, they are summed directly.
#     If grids differ, every field is interpolated (via linear interpolation
#     of its real and imaginary parts) onto a common grid before summing.

#     Prefer `field1 + field2` for two fields; use this function directly
#     when adding more than two fields at once or to force a specific
#     target grid via `grid=`.

#     Parameters
#     ----------
#     *fields : FieldData
#         Two or more fields to add.
#     grid : FieldData, optional
#         A FieldData whose x/y grid should be used as the target grid.
#         Defaults to the grid of the first field in `fields`.

#     Returns
#     -------
#     FieldData
#         A new FieldData with the summed complex field, expressed back as
#         amplitude/phase, on the target grid. Metadata (wavelength, header
#         info) is copied from the target grid's source field.
#     """
#     if len(fields) < 2:
#         raise ValueError("Provide at least two fields to add.")

#     target = grid if grid is not None else fields[0]
#     tx, ty = target.x, target.y

#     total = np.zeros((ty.size, tx.size), dtype=complex)

#     for fdata in fields:
#         if fdata.x.shape == tx.shape and np.allclose(fdata.x, tx) and \
#            fdata.y.shape == ty.shape and np.allclose(fdata.y, ty):
#             total += fdata.complex_field()
#         else:
#             total += _interp_complex(fdata, tx, ty)

#     return FieldData(
#         x=tx.copy(),
#         y=ty.copy(),
#         amp=np.abs(total),
#         phase=np.angle(total),
#         wavelength=target.wavelength,
#         y_extra_tokens=list(target.y_extra_tokens),
#         preamble=list(target.preamble),
#         keyword=target.keyword,
#     )


# def _interp_complex(fdata: FieldData, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
#     """Interpolate a field's complex values onto a target (tx, ty) grid.

#     Uses separable linear interpolation (RegularGridInterpolator), which
#     works for any monotonically increasing, possibly non-uniform, x/y
#     coordinate arrays. Points outside the source grid are filled with 0.
#     """
#     from scipy.interpolate import RegularGridInterpolator

#     cf = fdata.complex_field()  # shape (ny, nx)

#     interp_re = RegularGridInterpolator(
#         (fdata.y, fdata.x), cf.real, bounds_error=False, fill_value=0.0
#     )
#     interp_im = RegularGridInterpolator(
#         (fdata.y, fdata.x), cf.imag, bounds_error=False, fill_value=0.0
#     )

#     TY, TX = np.meshgrid(ty, tx, indexing="ij")
#     pts = np.stack([TY.ravel(), TX.ravel()], axis=-1)

#     re = interp_re(pts).reshape(ty.size, tx.size)
#     im = interp_im(pts).reshape(ty.size, tx.size)
#     return re + 1j * im


# # --------------------------------------------------------------------------- #
# # Writing
# # --------------------------------------------------------------------------- #

# def _fmt_num(v: float, sig: int = 10) -> str:
#     """Format a coordinate value compactly (used for the coordinate row)."""
#     return f"{v:.{sig}g}"


# def _fmt_sci(v: float) -> str:
#     """Format an amplitude/phase value in the file's scientific notation style."""
#     return f"{v:.5E}"


# def write_fld(path: str, field: FieldData) -> None:
#     """
#     Write a FieldData object out as an RSoft OUTPUT_AMP_PHASE_3D .fld file.

#     `field.x` (per-row) is written into the line-4 header/rows, and
#     `field.y` (per-column) is written into the line-5/6 header/coordinate
#     row, matching the layout produced by `read_fld`. Since `field.amp`/
#     `field.phase` are stored as (ny, nx), they are transposed back to
#     (nx, ny) for the raw row-major file layout.
#     """
#     nx, ny = field.nx, field.ny
#     amp_raw = field.amp.T      # (nx, ny) for row-major (x-row, y-col) file layout
#     phase_raw = field.phase.T

#     lines = list(field.preamble)

#     extra = " ".join(field.y_extra_tokens)
#     wl_str = f" Wavelength={field.wavelength}" if field.wavelength is not None else ""
#     header4 = f"{nx} {field.x.min():g} {field.x.max():g}"
#     if extra:
#         header4 += f" {extra}"
#     header4 += f" {field.keyword}{wl_str}"
#     lines.append(header4)

#     lines.append(f"{ny} {field.y.min():g} {field.y.max():g}")

#     col_row = "0 " + " ".join(_fmt_num(v) for v in field.y)
#     lines.append(col_row)

#     for i in range(nx):
#         row_vals = [_fmt_num(field.x[i])]
#         for j in range(ny):
#             row_vals.append(_fmt_sci(amp_raw[i, j]))
#             row_vals.append(_fmt_sci(phase_raw[i, j]))
#         lines.append(" ".join(row_vals))

#     with open(path, "w") as f:
#         f.write("\n".join(lines) + "\n")


# # --------------------------------------------------------------------------- #
# # Convenience: edge vs. peak amplitude ratio
# # --------------------------------------------------------------------------- #

# def edge_to_peak_ratio(field: FieldData) -> float:
#     """Return (max amplitude on the outer boundary) / (global peak amplitude)."""
#     return field.edge_amp_max() / field.peak_amp()


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) > 1:
#         fd = read_fld(sys.argv[1])
#         print(f"Read '{sys.argv[1]}': nx={fd.nx}, ny={fd.ny}, "
#               f"wavelength={fd.wavelength}")
#         print(f"Peak amplitude:  {fd.peak_amp():.5E}")
#         print(f"Max edge amplitude: {fd.edge_amp_max():.5E}")
#         print(f"Edge/peak ratio: {edge_to_peak_ratio(fd):.5E} "
#               f"({20*np.log10(edge_to_peak_ratio(fd)):.1f} dB)")


"""
rsoft_lib.py

Utilities for reading, combining, and writing RSoft OUTPUT_AMP_PHASE_3D
field-monitor files (.fld), RSoft BeamPROP monitor files, and RSoft .ind
symbol tables.

Raw .fld file layout this module expects (as inferred from a real exported file):

    /rn/ls1
    /r
    /rq
    <n1> <min1> <max1> <extra...> OUTPUT_AMP_PHASE_3D Wavelength=<wl>
    <n2> <min2> <max2>
    0 <v2_0> <v2_1> ... <v2_{n2-1}>
    <v1_0> <amp_00> <phase_00> <amp_01> <phase_01> ... <amp_0,n2-1> <phase_0,n2-1>
    <v1_1> <amp_10> <phase_10> ...
    ...
    <v1_{n1-1}> ...

i.e. after a 3-line preamble and a 2-line grid header, there is one row of
coordinates (prefixed with a placeholder "0"), followed by one data row per
value of the *other* axis. Each data row starts with that row's coordinate
and is then followed by (amplitude, phase) pairs, one pair per column
coordinate.

FieldData convention used by this module
-----------------------------------------
`FieldData.x` corresponds to the per-row coordinate (line 4 header/values),
`FieldData.y` corresponds to the per-column coordinate (line 5/6
header/values). `amp`/`phase` are stored with shape (ny, nx) -- i.e. axis 0
indexes y, axis 1 indexes x -- which matches what plotting functions like
`matplotlib.pyplot.pcolormesh(x, y, C)` expect (C.shape == (len(y), len(x))),
so `plt.pcolormesh(field.x, field.y, field.amp)` works with no transpose
needed.

Main functions/methods:

    read_fld(path)                  -> FieldData
    FieldData.__add__ / add_fields  -> FieldData (coherent complex sum)
    write_fld(path, field)          -> None

NOTE: This parser was reverse-engineered from a sample file rather than an
official format spec. It has been validated to round-trip the sample data
byte-for-byte in terms of numeric content. If your files come from a
different RSoft tool/version with a different header, check the header
parsing (`_parse_header_line4`) and adjust as needed.

Also reads RSoft BeamPROP monitor files -- the plain-text output with a
5-line header (Monitor #, Path, Type, Tilt, Mode) followed by one data
row per Z step:

    Monitor #:                1             2             3             4
    Monitor Path:             1             1             2             2
    Monitor Type:      WG_POWER  LAUNCH_PHASE      WG_POWER  LAUNCH_PHASE
    Monitor Tilt:             1             1             1             1
    Monitor Mode:             0             0             0             0
     0.000000E+00  8.016629E-01  0.000000E+00  0.000000E+00  0.000000E+00
     1.000000E-01  8.017489E-01  4.039808E-02  0.000000E+00  5.814786E-02
     ...

This is a different file format from the OUTPUT_AMP_PHASE_3D .fld reader
above -- this one is a simple Z-vs-scalar propagation series per monitor,
not a 2D transverse field snapshot.
"""

from __future__ import annotations

import re
import sys
import json
from dataclasses import dataclass, field as dc_field
import struct
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.optimize import brentq

rsoft_python_paths = [
    r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\products\most\python",
    r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\bin",
    r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\products\python\python3.11\Lib\site-packages"
]
for p in rsoft_python_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from rstools import RSoftCircuit


# --------------------------------------------------------------------------- #
# Symbol table (.ind) reading
# --------------------------------------------------------------------------- #

def read_all_symbols(ind_file):
    symbols = {}
    with open(ind_file) as f:
        for line in f:
            m = re.match(r'\s*([A-Za-z_]\w*)\s*=\s*(.+)', line.split('#')[0])
            if m:
                symbols[m.group(1)] = m.group(2).strip()
    return symbols


def read_symbol(ind_file, symbol_name):
    symbols = read_all_symbols(ind_file)
    symbol = symbols[symbol_name]
    return symbol


def export_symbols(ind_file, out_file):
    symbols = read_all_symbols(ind_file)
    with open(out_file, 'w') as f:
        json.dump(symbols, f, indent=2)
    return symbols

def mode_field(wav, width):
    ''' Returns the mode field filename and n_eff value for the fundamental mode of a given wavelength and width.

        Parameters
        ----------
        wav : float
            Wavelength
        width : float
            Rectangular waveguide width in microns (height is assumed to be 3.4 microns)

        Returns
        -------
        mode_filename : str
            Filename for the corresponding mode field file (used for launch injection in BeamPROP or monitor file reference)
        neff : float
            Effective index of mode
    '''

    mode_dir = 'D:\\DATA\\ActivePIC\\SingleWaveguideConvergence\\StraightWaveguide_neff_table\\'
    wav_str, width_str = f"{wav:0.3f}".replace(".", "p"), f"{width:0.3f}".replace(".", "p")
    df = np.genfromtxt(f'{mode_dir}StraightWaveguide_Wav{wav_str}_Width{width_str}.nef')
    if width < 3.4:
        mode_filename = f'{mode_dir}StraightWaveguide_Wav{wav_str}_Width{width_str}_ex.m01'
        neff = df[1,1]
    else:
        mode_filename = f'{mode_dir}StraightWaveguide_Wav{wav_str}_Width{width_str}_ex.m00'
        neff = df[1]
    return mode_filename, neff


# --------------------------------------------------------------------------- #
# Field Data container
# --------------------------------------------------------------------------- #

@dataclass
class FieldData:
    x: np.ndarray                  # shape (nx,)
    y: np.ndarray                  # shape (ny,)
    amp: np.ndarray                 # shape (ny, nx)
    phase: np.ndarray               # shape (ny, nx), radians
    wavelength: Optional[float] = None
    y_extra_tokens: List[str] = dc_field(default_factory=list)  # tokens between max1 and the keyword on line 4
    preamble: List[str] = dc_field(default_factory=lambda: ["/rn/ls1", "/r", "/rq"])
    keyword: str = "OUTPUT_AMP_PHASE_3D"

    @property
    def nx(self) -> int:
        return self.x.size

    @property
    def ny(self) -> int:
        return self.y.size

    def complex_field(self) -> np.ndarray:
        """Return the complex field E = amp * exp(i*phase), shape (ny, nx)."""
        return self.amp * np.exp(1j * np.deg2rad(self.phase))

    def peak_amp(self) -> float:
        return float(np.max(self.amp))

    def edge_amp_max(self) -> float:
        """Max amplitude found anywhere on the outer boundary of the grid."""
        edge_vals = np.concatenate([
            self.amp[0, :], self.amp[-1, :], self.amp[:, 0], self.amp[:, -1]
        ])
        return float(np.max(edge_vals))

    # ----------------------------------------------------------------- #
    # Operator support
    # ----------------------------------------------------------------- #

    def __add__(self, other: "FieldData") -> "FieldData":
        if not isinstance(other, FieldData):
            return NotImplemented
        return add_fields(self, other)

    def __radd__(self, other) -> "FieldData":
        # Allows sum([field1, field2, field3]) to work, since sum() starts
        # with 0 + field1.
        if other == 0:
            return self
        if not isinstance(other, FieldData):
            return NotImplemented
        return add_fields(other, self)


# --------------------------------------------------------------------------- #
# Monitor Data container
# --------------------------------------------------------------------------- #

@dataclass
class MonitorData:
    z: np.ndarray                  # shape (n_z,)
    values: np.ndarray              # shape (n_z, n_monitors)
    number: List[int]
    path: List[int]
    type: List[str]
    tilt: List[int]
    mode: List[int]

    @property
    def n_monitors(self) -> int:
        return len(self.number)

    def get(self, path: int = None, type: str = None) -> np.ndarray:
        """Return the (n_z,) column(s) matching the given path and/or type.
        If multiple monitors match, returns shape (n_z, n_matches)."""
        idx = [
            i for i in range(self.n_monitors)
            if (path is None or self.path[i] == path)
            and (type is None or self.type[i] == type)
        ]
        if not idx:
            raise KeyError(f"No monitor found with path={path}, type={type}")
        cols = self.values[:, idx]
        return cols[:, 0] if cols.shape[1] == 1 else cols


def _header_values(line: str) -> List[str]:
    """Split a header line like 'Monitor Type:   WG_POWER  LAUNCH_PHASE'
    into its values, after the label/colon."""
    _, _, rest = line.partition(":")
    return rest.split()


def read_monitor(path: str) -> MonitorData:
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]

    number = [int(v) for v in _header_values(lines[0])]
    mon_path = [int(v) for v in _header_values(lines[1])]
    mon_type = _header_values(lines[2])
    tilt = [int(v) for v in _header_values(lines[3])]
    mode = [int(v) for v in _header_values(lines[4])]

    n_monitors = len(number)
    data_lines = lines[5:]
    n_z = len(data_lines)

    z = np.empty(n_z, dtype=float)
    values = np.empty((n_z, n_monitors), dtype=float)

    for i, line in enumerate(data_lines):
        vals = np.fromstring(line, sep=" ")
        if vals.size != 1 + n_monitors:
            raise ValueError(
                f"Row {i} has {vals.size} values, expected {1 + n_monitors} "
                f"(Z + {n_monitors} monitors)."
            )
        z[i] = vals[0]
        values[i, :] = vals[1:]

    return MonitorData(
        z=z, values=values,
        number=number, path=mon_path, type=mon_type, tilt=tilt, mode=mode,
    )


# --------------------------------------------------------------------------- #
# Reading .fld
# --------------------------------------------------------------------------- #

_WAVELENGTH_RE = re.compile(r"Wavelength\s*=\s*([-+0-9.eE]+)")


def _parse_header_line4(line: str):
    """
    Parse a line like:
        933 -20 70 19 OUTPUT_AMP_PHASE_3D Wavelength=1.55
    Returns (n1, min1, max1, extra_tokens, keyword, wavelength)
    """
    tokens = line.split()
    n1 = int(float(tokens[0]))
    min1 = float(tokens[1])
    max1 = float(tokens[2])

    # Everything after max1 up to (and not including) the first token that
    # is not parseable as a float is "extra" numeric metadata (e.g. a slice
    # index). The first non-numeric token is taken as the keyword.
    extra_tokens = []
    keyword = None
    wavelength = None
    idx = 3
    while idx < len(tokens):
        tok = tokens[idx]
        try:
            float(tok)
            extra_tokens.append(tok)
            idx += 1
        except ValueError:
            keyword = tok
            idx += 1
            break

    rest = " ".join(tokens[idx:])
    m = _WAVELENGTH_RE.search(rest)
    if m:
        wavelength = float(m.group(1))

    return n1, min1, max1, extra_tokens, keyword, wavelength


def read_fld(path: str) -> FieldData:
    """Read an RSoft OUTPUT_AMP_PHASE_3D .fld file into a FieldData object.

    Returns a FieldData where `.x` is the per-row (line 4) coordinate,
    `.y` is the per-column (line 5/6) coordinate, and `.amp`/`.phase` have
    shape (ny, nx) (transposed relative to the file's raw row/column
    layout, so axis 0 is y and axis 1 is x -- ready for
    `plt.pcolormesh(field.x, field.y, field.amp)` with no extra transpose).
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Skip blank lines defensively but keep track of real content lines
    content_lines = [ln for ln in lines if ln.strip() != ""]

    preamble = content_lines[0:3]
    n1, min1, max1, extra_tokens, keyword, wavelength = _parse_header_line4(content_lines[3])

    n2_line_tokens = content_lines[4].split()
    n2 = int(float(n2_line_tokens[0]))

    col_tokens = [float(t) for t in content_lines[5].split()]
    # First token is a placeholder (observed to be 0); the remaining n2
    # tokens are the actual column coordinates.
    col_coords = np.array(col_tokens[1:1 + n2], dtype=float)
    if col_coords.size != n2:
        raise ValueError(
            f"Expected {n2} column coordinates, found {col_coords.size} "
            f"in line: {content_lines[5][:80]}..."
        )

    row_coords = np.empty(n1, dtype=float)
    amp_raw = np.empty((n1, n2), dtype=float)     # raw (row, col) = (line4, line5) order
    phase_raw = np.empty((n1, n2), dtype=float)

    data_lines = content_lines[6:6 + n1]
    if len(data_lines) != n1:
        raise ValueError(
            f"Expected {n1} data rows, found {len(data_lines)} in file '{path}'."
        )

    for i, line in enumerate(data_lines):
        vals = np.fromstring(line, sep=" ")
        expected_len = 1 + 2 * n2
        if vals.size != expected_len:
            raise ValueError(
                f"Row {i} (coord={vals[0] if vals.size else '?'}) has {vals.size} "
                f"values, expected {expected_len}."
            )
        row_coords[i] = vals[0]
        pairs = vals[1:].reshape(n2, 2)
        amp_raw[i, :] = pairs[:, 0]
        phase_raw[i, :] = pairs[:, 1]

    # Relabel: x = row_coords (line 4), y = col_coords (line 5/6).
    # amp_raw/phase_raw are (n1, n2) = (len(x), len(y)); transpose so that
    # the stored amp/phase are (len(y), len(x)) = (ny, nx).
    return FieldData(
        x=row_coords,
        y=col_coords,
        amp=amp_raw.T,
        phase=phase_raw.T,
        wavelength=wavelength,
        y_extra_tokens=extra_tokens,
        preamble=preamble,
        keyword=keyword or "OUTPUT_AMP_PHASE_3D",
    )


# --------------------------------------------------------------------------- #
# Combining fields
# --------------------------------------------------------------------------- #

def add_fields(*fields: FieldData, grid: Optional[FieldData] = None) -> FieldData:
    """
    Coherently add two or more fields (complex sum: amp*exp(i*phase)).

    If the fields share the same x/y grid, they are summed directly.
    If grids differ, every field is interpolated (via linear interpolation
    of its real and imaginary parts) onto a common grid before summing.

    Prefer `field1 + field2` for two fields; use this function directly
    when adding more than two fields at once or to force a specific
    target grid via `grid=`.

    Parameters
    ----------
    *fields : FieldData
        Two or more fields to add.
    grid : FieldData, optional
        A FieldData whose x/y grid should be used as the target grid.
        Defaults to the grid of the first field in `fields`.

    Returns
    -------
    FieldData
        A new FieldData with the summed complex field, expressed back as
        amplitude/phase, on the target grid. Metadata (wavelength, header
        info) is copied from the target grid's source field.
    """
    if len(fields) < 2:
        raise ValueError("Provide at least two fields to add.")

    target = grid if grid is not None else fields[0]
    tx, ty = target.x, target.y

    total = np.zeros((ty.size, tx.size), dtype=complex)

    for fdata in fields:
        if fdata.x.shape == tx.shape and np.allclose(fdata.x, tx) and \
           fdata.y.shape == ty.shape and np.allclose(fdata.y, ty):
            total += fdata.complex_field()
        else:
            total += _interp_complex(fdata, tx, ty)

    return FieldData(
        x=tx.copy(),
        y=ty.copy(),
        amp=np.abs(total),
        phase=np.rad2deg(np.angle(total)),
        wavelength=target.wavelength,
        y_extra_tokens=list(target.y_extra_tokens),
        preamble=list(target.preamble),
        keyword=target.keyword,
    )


def _interp_complex(fdata: FieldData, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """Interpolate a field's complex values onto a target (tx, ty) grid.

    Uses separable linear interpolation (RegularGridInterpolator), which
    works for any monotonically increasing, possibly non-uniform, x/y
    coordinate arrays. Points outside the source grid are filled with 0.
    """
    from scipy.interpolate import RegularGridInterpolator

    cf = fdata.complex_field()  # shape (ny, nx)

    interp_re = RegularGridInterpolator(
        (fdata.y, fdata.x), cf.real, bounds_error=False, fill_value=0.0
    )
    interp_im = RegularGridInterpolator(
        (fdata.y, fdata.x), cf.imag, bounds_error=False, fill_value=0.0
    )

    TY, TX = np.meshgrid(ty, tx, indexing="ij")
    pts = np.stack([TY.ravel(), TX.ravel()], axis=-1)

    re = interp_re(pts).reshape(ty.size, tx.size)
    im = interp_im(pts).reshape(ty.size, tx.size)
    return re + 1j * im


# --------------------------------------------------------------------------- #
# Writing .fld
# --------------------------------------------------------------------------- #

def _fmt_num(v: float, sig: int = 10) -> str:
    """Format a coordinate value compactly (used for the coordinate row)."""
    return f"{v:.{sig}g}"


def _fmt_sci(v: float) -> str:
    """Format an amplitude/phase value in the file's scientific notation style."""
    return f"{v:.5E}"


def write_fld(path: str, field: FieldData) -> None:
    """
    Write a FieldData object out as an RSoft OUTPUT_AMP_PHASE_3D .fld file.

    `field.x` (per-row) is written into the line-4 header/rows, and
    `field.y` (per-column) is written into the line-5/6 header/coordinate
    row, matching the layout produced by `read_fld`. Since `field.amp`/
    `field.phase` are stored as (ny, nx), they are transposed back to
    (nx, ny) for the raw row-major file layout.
    """
    nx, ny = field.nx, field.ny
    amp_raw = field.amp.T      # (nx, ny) for row-major (x-row, y-col) file layout
    phase_raw = field.phase.T

    lines = list(field.preamble)

    extra = " ".join(field.y_extra_tokens)
    wl_str = f" Wavelength={field.wavelength}" if field.wavelength is not None else ""
    header4 = f"{nx} {field.x.min():g} {field.x.max():g}"
    if extra:
        header4 += f" {extra}"
    header4 += f" {field.keyword}{wl_str}"
    lines.append(header4)

    lines.append(f"{ny} {field.y.min():g} {field.y.max():g}")

    col_row = "0 " + " ".join(_fmt_num(v) for v in field.y)
    lines.append(col_row)

    for i in range(nx):
        row_vals = [_fmt_num(field.x[i])]
        for j in range(ny):
            row_vals.append(_fmt_sci(amp_raw[i, j]))
            row_vals.append(_fmt_sci(phase_raw[i, j]))
        lines.append(" ".join(row_vals))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Convenience: edge vs. peak amplitude ratio
# --------------------------------------------------------------------------- #

def edge_to_peak_ratio(field: FieldData) -> float:
    """Return (max amplitude on the outer boundary) / (global peak amplitude)."""
    return field.edge_amp_max() / field.peak_amp()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        fd = read_fld(sys.argv[1])
        print(f"Read '{sys.argv[1]}': nx={fd.nx}, ny={fd.ny}, "
              f"wavelength={fd.wavelength}")
        print(f"Peak amplitude:  {fd.peak_amp():.5E}")
        print(f"Max edge amplitude: {fd.edge_amp_max():.5E}")
        print(f"Edge/peak ratio: {edge_to_peak_ratio(fd):.5E} "
              f"({20*np.log10(edge_to_peak_ratio(fd)):.1f} dB)")


### Functions for drawing GDS structures ###
def euler_sbend(d, R_min, p=1.0, ds=1.0):
    """Partial-Euler S-bend centerline with lateral offset d (um) and minimum radius R_min (um).
    p = Euler fraction (0 = circular arcs, 1 = full Euler). Returns x, z, phi (z = propagation)."""
    def path(theta, n):
        sc, sa = p * theta * R_min, (1 - p) * theta * R_min   # clothoid and arc lengths
        Lh = 2 * sc + sa
        s = np.linspace(0, 2 * Lh, n + 1)
        u = np.where(s <= Lh, s, s - Lh)
        if sc > 0:
            ph = np.where(u < sc, u**2 / (2 * sc * R_min),
                 np.where(u <= sc + sa, sc / (2 * R_min) + (u - sc) / R_min,
                          theta - (Lh - u)**2 / (2 * sc * R_min)))
        else:
            ph = u / R_min
        phi = np.where(s <= Lh, ph, theta - ph)
        x = np.concatenate([[0], np.cumsum((np.sin(phi[1:]) + np.sin(phi[:-1])) / 2 * np.diff(s))])
        z = np.concatenate([[0], np.cumsum((np.cos(phi[1:]) + np.cos(phi[:-1])) / 2 * np.diff(s))])
        return x, z, phi
 
    theta = brentq(lambda t: path(t, 20000)[0][-1] - abs(d), 1e-9, np.pi / 2 - 1e-6)
    n = max(int(np.ceil(2 * (1 + p) * theta * R_min / ds)), 2)
    x, z, phi = path(theta, n)
    return np.sign(d) * x, z, np.sign(d) * phi
 
 
def parallel_path(x, z, phi, s):
    """Curve running parallel to (x, z) at a constant perpendicular distance s (um).
    s > 0 is on the +x side where the path heads along +z."""
    return x + s * np.cos(phi), z - s * np.sin(phi), phi
 
 
def save_gds(filename, x, z, phi, width, layer=1):
    """Write the waveguide outline (centerline +- width/2) to a GDSII file. Units: um, 1 nm grid."""
    def rec(rtype, dtype, data=b""):
        data += b"\x00" * (len(data) % 2)
        return struct.pack(">HBB", 4 + len(data), rtype, dtype) + data
 
    def real8(v):                                   # GDSII 8-byte real
        e = 64
        while v >= 1: v /= 16; e += 1
        while v < 1 / 16: v *= 16; e -= 1
        return bytes([e]) + int(round(v * 2**56)).to_bytes(7, "big")
 
    # accept a single path or a list of paths: save_gds(fn, [x1, x2], [z1, z2], [p1, p2], width)
    paths = list(zip(x, z, phi)) if isinstance(x, (list, tuple)) else [(x, z, phi)]
 
    out = [rec(0x00, 0x02, struct.pack(">h", 600)), rec(0x01, 0x02, bytes(24)),
           rec(0x02, 0x06, b"LIB"), rec(0x03, 0x05, real8(1e-3) + real8(1e-9)),
           rec(0x05, 0x02, bytes(24)), rec(0x06, 0x06, b"EULER_SBEND")]
    step = 4000                                     # keep each polygon under GDSII's 8191-point limit
    for x, z, phi in paths:
        nx, nz = np.cos(phi), -np.sin(phi)          # normal to the path
        left = np.column_stack([x + width / 2 * nx, z + width / 2 * nz])
        right = np.column_stack([x - width / 2 * nx, z - width / 2 * nz])
        for i in range(0, len(x) - 1, step):
            j = min(i + step, len(x) - 1)
            poly = np.vstack([left[i:j + 1], right[i:j + 1][::-1], left[i:i + 1]])
            pts = np.round(poly * 1000).astype(">i4")   # um -> nm
            out += [rec(0x08, 0x00), rec(0x0D, 0x02, struct.pack(">h", layer)),
                    rec(0x0E, 0x02, struct.pack(">h", 0)), rec(0x10, 0x03, pts.tobytes()),
                    rec(0x11, 0x00)]
    out += [rec(0x07, 0x00), rec(0x04, 0x00)]
    open(filename, "wb").write(b"".join(out))