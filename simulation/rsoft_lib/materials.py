"""Refractive index of the materials used in an RSoft design, read from RSoft's material data files.

RSoft materials in a .ind file usually define their index through a tabulated data file, e.g.

    material 1
        name = SiO2
        optical
            nr = userreal("<rsoftmat>\\SiO2_nk.dat", free_space_wavelength)
            ni = userimag("<rsoftmat>\\SiO2_nk.dat", free_space_wavelength)

The data file has three columns: wavelength (um), real index, imaginary index. This module finds
the material block, locates that file, and interpolates it, which is what RSoft's nreal(material)
does. `<rsoftmat>` is RSoft's materials folder; its location is not known to this module, so it
is searched for (see find_material_file) or can be given with the mat_dir argument or the
RSOFT_MAT_DIR environment variable.

Main function: n_clad(wav, ind_file) = real index of the design's background material.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._env import RSOFT_ROOT
from .symbols import read_symbol

_USERFUNC = re.compile(r'(userreal|userimag)\(\s*"([^"]+)"')


def _material_block(ind_file, material):
    """Text of the `material N ... end material` block whose name is `material`."""
    text = Path(ind_file).read_text(errors="replace").replace("\r\n", "\n")
    for m in re.finditer(r"^material \d+\n(.*?)^end material", text, re.S | re.M):
        block = m.group(1)
        if re.search(rf"^\s*name\s*=\s*{re.escape(material)}\s*$", block, re.M):
            return block
    raise KeyError(f"No material named '{material}' in {ind_file}")


def _data_file_name(block, key):
    """File name used by the nr (key='userreal') or ni (key='userimag') expression, or None."""
    for kind, path in _USERFUNC.findall(block):
        if kind == key:
            return re.split(r"[\\/]", path)[-1]          # drop the <rsoftmat>\ prefix
    return None


def _constant(block, name):
    """Value of a constant nr / ni line, or None if the line is missing or not a number."""
    m = re.search(rf"^\s*{name}\s*=\s*(.+?)\s*$", block, re.M)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


@lru_cache(maxsize=32)
def find_material_file(filename, mat_dir=None, ind_dir=None):
    """Locate an RSoft material data file.

    Searched in order: mat_dir, the RSOFT_MAT_DIR environment variable, ind_dir (the folder of
    the .ind file), then the whole RSoft install folder (first match).
    """
    candidates = [mat_dir, os.environ.get("RSOFT_MAT_DIR"), ind_dir]
    for d in candidates:
        if d and (Path(d) / filename).is_file():
            return str(Path(d) / filename)
    if Path(RSOFT_ROOT).is_dir():
        for p in Path(RSOFT_ROOT).rglob(filename):
            return str(p)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched {[c for c in candidates if c]} and {RSOFT_ROOT}. "
        f"Pass mat_dir=<folder containing it> or set the RSOFT_MAT_DIR environment variable.")


def read_nk(path):
    """Read an RSoft nk data file: columns wavelength (um), n, k. Header and comment lines are skipped.

    Returns wav, n, k as arrays sorted by wavelength.
    """
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            try:
                rows.append([float(v) for v in parts[:3]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No (wavelength, n, k) rows found in {path}")
    a = np.array(sorted(rows))
    return a[:, 0], a[:, 1], a[:, 2]


def material_nk(material, wav, ind_file, mat_dir=None):
    """Complex-index parts (n, k) of the named material of an .ind file at wavelength wav (um).

    Uses the material's data file (linear interpolation) or a constant nr / ni. Raises ValueError if
    wav is outside the data range, and NotImplementedError for index expressions of another kind.
    """
    block = _material_block(ind_file, material)
    out = []
    for name, key in (("nr", "userreal"), ("ni", "userimag")):
        fname = _data_file_name(block, key)
        if fname is None:
            const = _constant(block, name)
            if const is None:
                if name == "ni":
                    out.append(0.0)                       # no imaginary part defined
                    continue
                expr = re.search(rf"^\s*{name}\s*=\s*(.+)$", block, re.M)
                raise NotImplementedError(f"Material '{material}': cannot evaluate {name} = "
                                          f"{expr.group(1) if expr else '(missing)'}")
            out.append(const)
            continue
        path = find_material_file(fname, mat_dir, str(Path(ind_file).parent))
        print(path)
        w, n, k = read_nk(path)
        if not (w[0] <= wav <= w[-1]):
            raise ValueError(f"{wav} um is outside the data range {w[0]:g} to {w[-1]:g} um of {fname}")
        out.append(float(np.interp(wav, w, n if key == "userreal" else k)))
    return out[0], out[1]


def n_clad(wav, ind_file, mat_dir=None, material=None):
    """Real refractive index of the cladding (the design's background material) at wavelength wav (um).

    The material is the `background_material` symbol of the .ind file, unless `material` is given.
    """
    if material is None:
        material = read_symbol(ind_file, "background_material")
    return material_nk(material, wav, ind_file, mat_dir)[0]