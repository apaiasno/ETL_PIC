"""RSoft circuits: the structures as RSoft CAD (.ind files), exportable to GDS with RSoft's own export.

Paths and widths come from rsoft_lib.geometry. A path becomes straight lines and circular arcs (an arc is one RSoft segment, a raised-cosine
taper is one segment with a user taper), the segments are put into pathways with launch fields and monitors, and the file is written
and checked with RSoft's own parser. rsoft_lib.gds has the independent GDS reader and writer.

Contents (in this order)
    Global parameters   defaults of the files: mode file, segment limit, materials, taper profiles
    File text           _material_block, _segment_text, _ind_text, _write_ind
    Arcs                arc_end_point, fit_path (x, z, phi -> lines and free arcs)
    Circuit             segments, pathways and the file: line, arc, free_arc, add_path, add_pathway, path_length, extent, write, export, plot;
                        builders add_mmi, add_delay_arm, add_sbend, add_sidestep, add_route, add_tapered_arm
    Any path            paths_to_circuit (x, z, phi, width -> a circuit with one pathway per path)
    mzi                 an unbalanced Mach-Zehnder: two 2x2 MMIs joined by a bump delay arm and a tapered arm
    Run helpers         pathway_file_symbols (the symbols to set per run), ind2gds (RSoft's GDS export)
"""


import re
import subprocess
import warnings
from pathlib import Path

import numpy as np

from . import _env
from .geometry import delay_arm_parameters, euler_sbend, eval_width, parallel_path, path_list, sbend_lengths


# --------------------------------------------------------------------------- #
# Global parameters
# --------------------------------------------------------------------------- #
# Defaults of the RSoft files (Circuit.write, gds_to_ind). Edit them here, or override them in the call.
# Launch field and monitor file given to every pathway (a mode file from the neff table), and the most segments RSoft
# accepts in one pathway
DEFAULT_MODE_FILE = r"D:\DATA\ETL_PIC\simulation\rsoft_lib\StraightWaveguide_neff_table\StraightWaveguide_Wav1p550_Width3p400_ex.m00"
MAX_PATHWAY_SEGMENTS = 200                       # RSoft rejects a pathway with more segments ("Maximum pathway length exceeded")
# Materials: the cladding is the library material, the core has its refractive index times CORE_INDEX_FACTOR
CLADDING_MATERIAL = "SiO2"                          # background material (cladding)
CORE_MATERIAL = "DopedSiO2"                         # default material (core)
CORE_INDEX_FACTOR = "Contrast"                      # core index = CORE_INDEX_FACTOR x library index (Contrast is a symbol)
MATERIAL_NK_FILE = r"<rsoftmat>\\SiO2_nk.dat"       # refractive index library file (n, k versus wavelength)
# Everything else in a material block (dispersion, electro-, thermo- and stress-optic data of the SiO2 library entry)
MATERIAL_TEMPLATE = """material {number}
\tname = {name}
\torigin = Path=/RSoft/Dielectrics/SiO2 Date=03/25/2022 Time=10:00
\tcomment = Lambda_Min=0.2, Lambda_Max=3.37, Src=J.Opt.Soc.Am/Vol-55/pp-1205
\toptical
\t\tni = {scale}userimag("{nk_file}",free_space_wavelength)
\t\tnr = {scale}userreal("{nk_file}",free_space_wavelength)
\tend optical
\tepsinf = 1
\tnum_dsp = 3
\tdelta_1 = 5873.621094
\ta_1 = 1
\tb_1 = 0
\tc_1 = 8437.094835
\tdelta_2 = 1191.892625
\ta_2 = 1
\tb_2 = 0
\tc_2 = 2921.716498
\tdelta_3 = 0.3622641
\ta_3 = 1
\tb_3 = 0
\tc_3 = 0.4036462
\telectro_optic
\t\teps_xx = 3.8
\t\teps_yy = 3.8
\t\teps_zz = 3.8
\tend electro_optic
\tthermo_optic
\t\tkappa_xx = 1.38*1e-6
\t\tkappa_yy = 1.38*1e-6
\t\tkappa_zz = 1.38*1e-6
\t\tr = 1e-05
\tend thermo_optic
\tstress_optic
\t\tE = 76.7
\t\talpha = 5.4e-07
\t\tnu = 0.186
\t\tp1 = 0.16
\t\tp2 = 0.27
\tend stress_optic
\tsemiconductor
\t\telem_matfile = Oxide.mat
\t\treg_regtype = oxide
\tend semiconductor
end material
"""
# User taper profiles: f(z) goes from 0 to 1 along a segment (z from 0 to 1, the cosine in degrees), and the width is
# begin.width + (end.width - begin.width) * f(z). "cosine" is the raised cosine of your Taper.ind example.
PROFILES = {"cosine": "0.5 - 0.5*cos(180*z)"}


# --------------------------------------------------------------------------- #
# File text
# --------------------------------------------------------------------------- #


def _material_block(number, name, index_factor=None):
    """The material block of the .ind file: the library index of MATERIAL_NK_FILE, times index_factor if there is one."""
    return MATERIAL_TEMPLATE.format(number=number, name=name, nk_file=MATERIAL_NK_FILE,
                                    scale=f"{index_factor}*" if index_factor else "")


def _segment_text(n, kind, A, B, wa, wb, width, extended=False, user_taper=None, free_arc=None):
    """The text of one RSoft segment: a line or an arc (arc_iangle = 0, tangent to z at A) from A to B, with widths wa and wb.
    `width` is the default width symbol; widths are written when they differ from it. user_taper is the number k of a
    user_taper block (width_taper = TAPER_USER_k); otherwise a width change is linear. free_arc = (radius, iangle, fangle)
    writes the arc as ARC_FREE (any start heading) instead of arc_iangle = 0."""
    lines = [f"segment {n}"]
    if extended:
        lines.append("\textended = 1")
    if kind == "arc" and free_arc is not None:               # (radius, initial heading, final heading): A is where it starts, B where it ends
        lines += ["\tposition_taper = TAPER_ARC", "\tarc_type = ARC_FREE", f"\tarc_radius = {free_arc[0]:.10g}",
                  f"\tarc_iangle = {free_arc[1]:.10g}", f"\tarc_fangle = {free_arc[2]:.10g}"]
    elif kind == "arc":
        lines += ["\tposition_taper = TAPER_ARC", "\tarc_type = ARC_IANGLE", "\tarc_iangle = 0"]
    lines += [f"\tbegin.x = {A[0]:.6f}", f"\tbegin.z = {A[1]:.6f}", f"\tend.x = {B[0]:.6f}", f"\tend.z = {B[1]:.6f}"]
    if user_taper is not None:
        lines.append(f"\twidth_taper = TAPER_USER_{user_taper}")
    if user_taper is not None or abs(wa - width) > 1e-3 or abs(wb - width) > 1e-3:
        if user_taper is None and abs(wa - wb) > 1e-3:
            lines.append("\twidth_taper = TAPER_LINEAR")
        lines += [f"\tbegin.width = {wa:.6f}", f"\tend.width = {wb:.6f}"]
    lines.append("end segment\n")
    return "\n".join(lines)


def _ind_text(segments, pathways, launch_x, xmin, xmax, width, height, Padding, PML_width, Contrast, Grid_Size, Grid_Edge, Step_z,
              pade_order, wavelength, core_material, cladding_material, write_materials, launch_files, monitor_files,
              launch_pathways=None, user_tapers=()):
    """The text of a BeamPROP .ind file: symbols, materials, user tapers, segments, pathways, monitors and launch fields.

    segments : the text of every segment (see _segment_text); pathways : for every pathway the list of its segment numbers;
    launch_x : the lateral position (x) of the start of every pathway (where its launch field goes);
    launch_pathways : the pathways (1, 2, ...) that get a launch field (default: all); user_tapers : the text of the
    user_taper blocks. Every pathway gets a power and a phase monitor."""
    n = len(pathways)
    esc = lambda s: str(s).replace("\\", "\\\\")
    symbols = {
        "Contrast": Contrast, "Grid_Edge": Grid_Edge, "Grid_Size": Grid_Size, "PML_width": PML_width, "Padding": Padding,
        "Step_z": Step_z, "Circuit_xmin": f"{xmin:.6f}", "Circuit_xmax": f"{xmax:.6f}",
        "alpha": 0, "background_alpha": "nimag($background_material)", "background_index": "nreal($background_material)",
        "background_material": cladding_material, "bc_type": "BC_PML",
        "boundary_max": "Circuit_xmax+Padding", "boundary_max_y": "(height/2+Padding)",
        "boundary_min": "Circuit_xmin-Padding", "boundary_min_y": "-(height/2+Padding)",
        "cad_aspectratio_y": 1, "default_material": core_material, "delta": 0.1, "dimension": 3, "eim": 0,
        "free_space_wavelength": wavelength, "grid_edge_factor_x": "Grid_Edge", "grid_edge_factor_y": "Grid_Edge",
        "grid_nonuniform": 1, "grid_size": "Grid_Size", "grid_size_y": "Grid_Size", "height": height,
        "k0": "(2*pi)/free_space_wavelength", "launch_align_file": 1, "launch_file": "$Launch_filename_1",
        "launch_file_explicit": 1, "launch_normalization": 2, "launch_position": 0, "launch_position_y": 0,
        "launch_tilt": 1, "launch_type": "LAUNCH_FILE", "monitor_file_explicit": 1, "pade_order": pade_order,
        "pade_order_warning_3d": 0, "pml_width_x": "PML_width", "pml_width_y": "PML_width", "sim_tool": "ST_BEAMPROP",
        "slice_display_mode": "DISPLAY_CONTOURMAPXZ", "slice_output_format": "OUTPUT_AMP_PHASE", "step_size": "Step_z",
        "structure": "STRUCT_CHANNEL", "vector": 1, "width": f"{width:.6f}".rstrip("0").rstrip("."),
    }
    for k in range(n):
        symbols[f"Launch_filename_{k + 1}"] = esc(launch_files[k])
        symbols[f"Monitor_filename_{k + 1}"] = esc(monitor_files[k])
    text = "\n" + "\n".join(f"{k} = {v}" for k, v in sorted(symbols.items())) + "\n\n\n"
    if write_materials:
        text += _material_block(1, core_material, CORE_INDEX_FACTOR) + "\n" + _material_block(2, cladding_material) + "\n\n"
    for block in user_tapers:
        text += block + "\n\n"
    text += "\n".join(segments) + "\n"
    for k, numbers in enumerate(pathways, start=1):
        text += f"pathway {k}\n" + "".join(f"\t{s}\n" for s in numbers) + "end pathway\n\n"
    text += "\n"
    for k in range(1, n + 1):
        for j, kind in enumerate(("POWER", "PHASE")):
            text += (f"monitor {2 * k - 1 + j}\n\tpathway = {k}\n\tmonitor_type = MONITOR_FILE_{kind}\n\tmonitor_tilt = 1\n"
                     f"\tmonitor_component = COMPONENT_BOTH\n\tmonitor_file = $Monitor_filename_{k}\nend monitor\n\n")
    text += "\n"
    launched = range(1, n + 1) if launch_pathways is None else launch_pathways
    for number, k in enumerate(launched, start=1):        # launch_position is the lateral position (x) of the field: the start of its pathway
        text += (f"launch_field {number}\n\tlaunch_pathway = {k}\n\tlaunch_type = LAUNCH_FILE\n\tlaunch_tilt = 1\n"
                 f"\tlaunch_normalization = 2\n\tlaunch_align_file = 1\n\tlaunch_position = {launch_x[k - 1]:.6f}\n"
                 f"\tlaunch_position_y = 0\n\tlaunch_file = $Launch_filename_{k}\nend launch_field\n\n")
    return text


def _write_ind(ind_path, text, validate=True):
    """Write the .ind file (CRLF line ends) and, if validate and RSoft's tools are installed, read it back with RSoft's own parser
    (rstools.RSoftCircuit.read returns 1 when it reads the file and 0 when it fails), so a syntax error or a limit such as the
    pathway length shows up in Python and not when you open the file in the CAD."""
    with open(ind_path, "w", newline="") as f:
        f.write(text.replace("\n", "\r\n"))
    if validate and _env.RSoftCircuit is not None and not _env.RSoftCircuit().read(str(ind_path)):
        raise RuntimeError(f"RSoft could not read {ind_path} (RSoftCircuit.read returned 0): the file has a syntax error or exceeds a limit, "
                           f"for example a pathway with more than {MAX_PATHWAY_SEGMENTS} segments. Open it in RSoft CAD for the message, "
                           f"or pass validate=False to skip this check.")


# --------------------------------------------------------------------------- #
# Arcs
# --------------------------------------------------------------------------- #


def arc_end_point(begin, radius, iangle, fangle):
    """End point (x, z) of an RSoft free arc (arc_type = ARC_FREE): it starts at `begin` with the heading `iangle` and ends
    with the heading `fangle` (degrees, measured from +z toward +x), on a circle of the given radius. It turns toward +x when
    fangle > iangle and toward -x otherwise. Checked against the polygons RSoft exports for three arcs."""
    p0, p1 = np.radians(iangle), np.radians(fangle)
    sign = np.sign(fangle - iangle)
    return np.asarray(begin, float) + sign * radius * np.array([np.cos(p0) - np.cos(p1), np.sin(p1) - np.sin(p0)])


def fit_path(x, z, phi=None, tol=5e-3, min_run=5):
    """Straight lines and circular arcs (RSoft free arcs) that follow a path given by x, z and the tangent angle phi (rad).

    Constant-curvature stretches of the path (the circular S-bend arcs, the plateau of a partial Euler bend) and straight
    stretches become exact primitives. A stretch whose curvature changes (the clothoid ramps of an Euler bend) is approximated
    by arcs that each follow the heading of the path: an arc starts at a point of the path with the heading there and ends with the
    heading at its last point, and is made as long as the path stays within `tol` (um) of it. The path must head toward +z.
    phi=None computes it from x and z. min_run: the fewest samples of a constant-curvature stretch.
    Returns a list of dicts: kind "line" (begin, end) or "arc" (begin, end, radius, iangle, fangle; degrees), each with its
    deviation (um) from the path."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    P = np.column_stack([x, z])
    N = len(x)
    phi = np.arctan2(np.gradient(x), np.gradient(z)) if phi is None else np.unwrap(np.asarray(phi, float))
    ds = np.hypot(np.diff(x), np.diff(z))
    s = np.r_[0.0, np.cumsum(ds)]
    kappa = np.diff(phi) / ds                                         # curvature of every interval

    def arc_from(i, j):
        """The arc that starts at point i with its heading and turns by the heading change to point j."""
        dphi = phi[j] - phi[i]
        radius = float(np.hypot(*(P[j] - P[i]))) / (2 * np.sin(abs(dphi) / 2))      # from the chord: exact for a circular arc
        end = arc_end_point(P[i], radius, np.degrees(phi[i]), np.degrees(phi[j]))
        sign = np.sign(dphi)
        n0 = np.array([np.cos(phi[i]), -np.sin(phi[i])]) * sign
        center = P[i] + radius * n0
        deviation = np.abs(np.hypot(*(P[i:j + 1] - center).T) - radius).max()
        return radius, end, max(deviation, float(np.hypot(*(end - P[j]))))

    def line_error(i, j):
        d = P[j] - P[i]
        rel = P[i:j + 1] - P[i]
        return np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max() / np.hypot(*d)

    def piece(i, j):
        if abs(phi[j] - phi[i]) < 1e-9:
            return ("line", i, j, None) if line_error(i, j) <= tol else None
        radius, end, err = arc_from(i, j)
        return ("arc", i, j, radius) if err <= tol else None

    # constant-curvature runs (exact), then arc-spline pieces for the rest
    runs, k = [], 0
    while k < N - 1:
        ref = kappa[k]
        m = k
        while m + 1 < N - 1 and abs(kappa[m + 1] - ref) <= 1e-5 * abs(ref) + 1e-9:
            m += 1
        if m - k + 1 >= min_run:
            runs.append((k, m + 1))                                    # point indices k .. m + 1
            k = m + 1
        else:
            k += 1
    pieces, i = [], 0

    def spline(a, b):                                                  # arcs along the points a .. b whose curvature changes
        out, i = [], a
        while i < b:
            lo, step = i + 1, 1
            while lo + step <= b and piece(i, lo + step) is not None:
                lo += step
                step *= 2
            hi = min(lo + step, b)
            while hi - lo > 1:                                         # the longest ok piece lies between lo and hi
                mid = (lo + hi) // 2
                if piece(i, mid) is not None:
                    lo = mid
                else:
                    hi = mid
            if hi == b and lo < b and piece(i, b) is not None:
                lo = b
            out.append((i, lo))
            i = lo
        return out

    for a, b in runs:
        if a > i:
            pieces += spline(i, a)
        pieces.append((a, b))
        i = b
    if i < N - 1:
        pieces += spline(i, N - 1)
    prims = []
    for a, b in pieces:
        if abs(phi[b] - phi[a]) < 1e-9:
            prims.append(dict(kind="line", begin=P[a], end=P[b], i=a, j=b, deviation=float(line_error(a, b))))
        else:
            radius, end, err = arc_from(a, b)
            prims.append(dict(kind="arc", begin=P[a], end=end, radius=float(radius), iangle=float(np.degrees(phi[a])),
                              fangle=float(np.degrees(phi[b])), i=a, j=b, deviation=float(err)))
    return prims


# --------------------------------------------------------------------------- #
# Circuit
# --------------------------------------------------------------------------- #


class Pathway(dict):
    """A pathway of a Circuit: a dict with segments (the segment numbers in route order), launch (True if it has a launch field),
    start and end ((x, z) where the guide begins and ends, as drawn, including bend offsets). The keys can also be read as
    attributes: p.start is p["start"]."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Circuit:
    """Segments and pathways of an RSoft circuit.

    Segments are added with line() and arc() (or the builders below) and get their numbers in the order they are added.
    A pathway is a list of segment numbers; it gets a power and a phase monitor, and a launch field if launch=True.

    orientation: the segment type RSoft draws. "z" (default) writes only the segments you flagged with extended=True (so the others are
    z-axis segments, as in your P-02 and P-03a files). "extended" writes every segment with extended = 1, RSoft CAD's Extended type
    (the CAD default for new segments), whose curves are fractured into polygons at the Import/Export Resolution of the CAD
    preferences. Measured in RSoft CAD for a 40 degree arc: the width along x is 4.5 um (= 3.4 / cos 40 degrees) for both types, so the
    width stays perpendicular to the path either way; tilted straight segments have not been compared."""

    def __init__(self, width=3.4, height=3.4, orientation="z"):
        self.width, self.height = width, height              # default waveguide width and height (um)
        if orientation not in ("extended", "z"):
            raise ValueError("orientation is 'extended' or 'z'")
        self.orientation = orientation
        self.segments = []                                  # dicts, number = index + 1
        self.pathways = []                                  # dicts: segments, launch
        self._tapers = []                                   # (expression, length) of the user tapers, number = index + 1

    # ---- segments
    def line(self, begin, end, width=None, profile=None, extended=False):
        """A straight segment from begin to end (x, z). width: a number, or (width at begin, width at end) for a taper.
        profile: how the width changes: None for linear, "cosine", or an expression in z (0 to 1) that goes from 0 to 1.
        Returns the segment number."""
        wa, wb = (self.width, self.width) if width is None else (width, width) if np.ndim(width) == 0 else tuple(width)
        taper = None
        if profile is not None and wa != wb:
            expression = PROFILES.get(profile, profile)
            length = float(np.hypot(end[0] - begin[0], end[1] - begin[1]))
            if (expression, length) not in self._tapers:
                self._tapers.append((expression, length))
            taper = self._tapers.index((expression, length)) + 1
        self.segments.append(dict(kind="line", A=tuple(map(float, begin)), B=tuple(map(float, end)), wa=float(wa), wb=float(wb),
                                  extended=extended, taper=taper))
        return len(self.segments)

    def arc(self, begin, end, width=None, extended=False):
        """A circular arc from begin to end that is tangent to the z direction at begin (RSoft's arc_iangle = 0). An arc that
        is tangent to z at its far end is given backward: begin at that end, as in your other circuits. Returns the number."""
        w = self.width if width is None else width
        self.segments.append(dict(kind="arc", A=tuple(map(float, begin)), B=tuple(map(float, end)), wa=float(w), wb=float(w),
                                  extended=extended, taper=None))
        return len(self.segments)

    def free_arc(self, begin, radius, iangle, fangle, width=None, extended=False):
        """A circular arc of the given radius that starts at begin with the heading iangle and ends with the heading fangle
        (degrees from +z toward +x; it turns toward +x if fangle > iangle, toward -x otherwise), written as ARC_FREE.
        Returns (segment number, end point): start the next arc there to keep the path smooth."""
        end = arc_end_point(begin, radius, iangle, fangle)
        wa, wb = (self.width, self.width) if width is None else (width, width) if np.ndim(width) == 0 else tuple(width)
        self.segments.append(dict(kind="arc", A=tuple(map(float, begin)), B=tuple(map(float, end)), wa=float(wa), wb=float(wb),
                                  extended=extended, taper=None, free=(float(radius), float(iangle), float(fangle))))
        return len(self.segments), end

    def add_path(self, x, z, phi=None, width=None, tol=5e-3, width_tol=5e-3, extended_ends=False):
        """Add any path given by x, z and the tangent angle phi (rad; phi=None computes it) as straight lines and circular arcs
        (see fit_path): exact for constant-curvature parts, and arcs that follow the heading of the path to within tol (um) where
        the curvature changes. The path must head toward +z.
        width: one number, a width at every point, or a function of the path length s (um) from the start of the path, such as
        lambda s: 3.4 + 1.6 * s / 500. A width that changes is followed to within width_tol (um): a straight
        piece becomes one segment with a raised-cosine user taper if the widths follow one, otherwise linear tapers (split as
        needed). extended_ends: extend the first and last segments if they are lines.
        Returns a dict with the segment numbers (in order), n_lines, n_arcs, n_tapers (user tapers), max_gap (the largest distance
        between the end of one piece and the start of the next; the pieces are smooth in heading), max_deviation (the largest distance
        between the path and the lines and arcs) and max_width_error: the accuracy of the circuit against the path you gave."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        if np.any(np.diff(z) < 0):
            raise ValueError("The path turns back toward -z; BeamPROP propagates along +z, so a path must keep increasing in z")
        if width is None:
            w = np.full(len(x), float(self.width))
        elif callable(width):                                          # a function of the path length s
            w = eval_width(width, np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
        elif np.ndim(width) == 0:
            w = np.full(len(x), float(width))
        else:
            w = np.asarray(width, float)
        phi_arr = np.arctan2(np.gradient(x), np.gradient(z)) if phi is None else np.unwrap(np.asarray(phi, float))
        if np.abs(phi_arr).max() > np.pi / 2:
            raise ValueError("The path heads more than 90 degrees away from +z")
        pieces = _fit_with_widths(x, z, phi_arr, w, tol, width_tol)
        numbers, n_tapers, arc_taper, werr = [], 0, False, 0.0
        for k, (pr, spec, err) in enumerate(pieces):
            kind, wa, wb = spec
            werr = max(werr, err)
            if pr["kind"] == "line":
                n = self.line(pr["begin"], pr["end"], width=(wa, wb), profile="cosine" if kind == "cosine" else None,
                              extended=extended_ends and k in (0, len(pieces) - 1))
                n_tapers += kind == "cosine"
            else:
                n, _ = self.free_arc(pr["begin"], pr["radius"], pr["iangle"], pr["fangle"], width=(wa, wb))
                arc_taper |= abs(wa - wb) > 1e-3
            numbers.append(n)
        if arc_taper:
            warnings.warn("A width change on an arc is written as a linear taper along the arc; how RSoft interpolates along an arc "
                          "has not been checked.")
        prims = [p for p, _, _ in pieces]
        gap = max((float(np.hypot(*(a["end"] - b["begin"]))) for a, b in zip(prims[:-1], prims[1:])), default=0.0)
        return dict(segments=numbers, n_lines=sum(p["kind"] == "line" for p in prims), n_arcs=sum(p["kind"] == "arc" for p in prims),
                    n_tapers=n_tapers, max_gap=gap, max_deviation=float(max(p["deviation"] for p in prims)), max_width_error=float(werr))

    def add_pathway(self, numbers, launch=False):
        """A pathway through the segments `numbers`, in order; launch=True puts a launch field at its start. It is stored in
        self.pathways as a Pathway, with its start and end points (p.start, p.end). Returns the pathway number."""
        if len(numbers) > MAX_PATHWAY_SEGMENTS:
            raise ValueError(f"The pathway has {len(numbers)} segments, but RSoft accepts at most {MAX_PATHWAY_SEGMENTS} in a pathway "
                             f"('Maximum pathway length exceeded'). A larger tol in add_path gives fewer arcs.")
        seg = [self.segments[n - 1] for n in numbers]
        start = min((seg[0]["A"], seg[0]["B"]), key=lambda p: p[1])           # pathways run toward +z: the lower-z end of the first
        end = max((seg[-1]["A"], seg[-1]["B"]), key=lambda p: p[1])            # segment, the higher-z end of the last (arcs can be backward)
        self.pathways.append(Pathway(segments=list(numbers), launch=bool(launch), start=tuple(map(float, start)),
                                     end=tuple(map(float, end))))
        return len(self.pathways)

    def path_length(self, numbers):
        """The length (um) along the segments `numbers`: straight lengths plus arc lengths (radius x angle swept)."""
        total = 0.0
        for n in numbers:
            s = self.segments[n - 1]
            A, B = np.array(s["A"]), np.array(s["B"])
            if s["kind"] == "line":
                total += float(np.hypot(*(B - A)))
            elif s.get("free"):
                total += s["free"][0] * np.radians(abs(s["free"][2] - s["free"][1]))
            else:                                              # an arc that is tangent to z at A
                rho = ((B - A) ** 2).sum() / (2 * (B[0] - A[0]))
                ra, rb = np.array([-rho, 0.0]), np.array([B[0] - A[0] - rho, B[1] - A[1]])
                total += abs(rho) * float(np.arctan2(abs(ra[0] * rb[1] - ra[1] * rb[0]), ra @ rb))
        return float(total)

    def extent(self):
        """The smallest and largest x (um) of the circuit, including the widths."""
        lo = min(min(self._outline(s, 60)[0][:, 0]) - max(s["wa"], s["wb"]) / 2 for s in self.segments)
        hi = max(max(self._outline(s, 60)[0][:, 0]) + max(s["wa"], s["wb"]) / 2 for s in self.segments)
        return lo, hi

    # ---- the file
    def write(self, ind_path, launch_files=None, monitor_files=None, default_file=DEFAULT_MODE_FILE, height=None, Padding=20.0,
              PML_width=5.0, Contrast=1.02, Grid_Size=0.1, Grid_Edge=0.5, Step_z=0.2, pade_order=4, wavelength=1.55,
              core_material=None, cladding_material=None, write_materials=True, validate=True):
        """Write the BeamPROP .ind file. The options are those of gds_to_ind: the same defaults, symbols
        Launch_filename_<k> and Monitor_filename_<k> for every pathway (default_file unless launch_files / monitor_files are
        given), a power and a phase monitor on every pathway, and a launch field on the pathways added with launch=True.
        validate: read the file back with RSoft's own parser (if RSoft's Python tools are installed) and raise if it fails.
        Returns a dict with the pathways (segment numbers, start x), the x extent and the number of segments."""
        if not self.pathways:
            raise ValueError("The circuit has no pathway: add one with add_pathway")
        n = len(self.pathways)
        launch_files = list(launch_files) if launch_files else [default_file] * n
        monitor_files = list(monitor_files) if monitor_files else [default_file] * n
        if len(launch_files) != n or len(monitor_files) != n:
            raise ValueError(f"{n} pathways, so launch_files and monitor_files need {n} entries each")
        texts = [_segment_text(k, s["kind"], s["A"], s["B"], s["wa"], s["wb"], self.width, s["extended"] or self.orientation == "extended", s["taper"], s.get("free"))
                 for k, s in enumerate(self.segments, start=1)]
        blocks = [f"user_taper {k}\n\ttype = UF_EXPRESSION\n\texpression = {expression}\n\txpoints = {length:.6g}/Step_z\nend user_taper"
                  for k, (expression, length) in enumerate(self._tapers, start=1)]
        starts = [p["start"][0] for p in self.pathways]                    # launch x: where each pathway begins
        xmin, xmax = self.extent()
        text = _ind_text(texts, [p["segments"] for p in self.pathways], starts, xmin, xmax, self.width,
                         self.height if height is None else height, Padding, PML_width, Contrast, Grid_Size, Grid_Edge, Step_z,
                         pade_order, wavelength, core_material or CORE_MATERIAL, cladding_material or CLADDING_MATERIAL,
                         write_materials, launch_files, monitor_files,
                         launch_pathways=[k for k, p in enumerate(self.pathways, start=1) if p["launch"]], user_tapers=blocks)
        _write_ind(ind_path, text, validate)
        self._domain = dict(padding=float(Padding), pml_width=float(PML_width))
        return dict(pathways=[dict(segments=p["segments"], start_x=x, launch=p["launch"]) for p, x in zip(self.pathways, starts)],
                    xmin=float(xmin), xmax=float(xmax), n_segments=len(self.segments), n_user_tapers=len(self._tapers))

    def export(self, ind_path, gds_path=None, layer=1, resolution=0.001, **write_options):
        """Write the .ind file (write_options go to write) and export the GDS from it with RSoft's ind2gds
        (layer 0 is mapped to `layer`). The GDS goes next to the .ind file unless gds_path is given. RSoft converts every
        object to polygons, and `resolution` (um, default 1 nm) is the largest distance between a polygon edge and the true curve
        (None uses the CAD preference). By default RSoft clips the export to the simulation domain. Returns the info of write."""
        info = self.write(ind_path, **write_options)
        info["gds"] = str(ind2gds(ind_path, gds_path, layer=layer, resolution=resolution))
        return info

    def plot(self, equal=False, xlim=None, ylim=None, labels=None, legend=True, domain=True, padding=None, pml_width=None,
             figsize=None):
        """Draw the circuit (x across, z up): each pathway in its own colour, segments shared by pathways (or in none) gray.
        equal: equal axes (a true-to-scale picture, but a long structure becomes thin). xlim, ylim: (min, max) in um, to zoom in.
        labels: a name for each pathway, in pathway order (default "pathway 1", "pathway 2", ...); legend=False leaves the legend out.
        domain: draw the BeamPROP simulation domain and set the axes to it: x from the circuit's extent minus to plus the padding,
        z over the circuit, with the PML bands (pml_width) shaded inside its x edges. padding and pml_width default to the values
        of the last write / export of this circuit, else the writer's defaults (20 and 5 um).
        figsize: (width, height) of the figure in inches (default (7, 8), or (5, 9) with equal=True)."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        member = {}
        for k, p in enumerate(self.pathways):
            for s in p["segments"]:
                member.setdefault(s, set()).add(k)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=figsize or ((5, 9) if equal else (7, 8)))
        for number, s in enumerate(self.segments, start=1):
            cc, w, nrm = self._outline(s)
            left, right = cc + nrm * w[:, None] / 2, cc - nrm * w[:, None] / 2
            who = member.get(number, set())
            color = "0.55" if len(who) != 1 else colors[next(iter(who)) % len(colors)]
            light = tuple(0.5 + 0.5 * v for v in to_rgb(color))          # the colour at 50% on white, opaque so that neighbouring fills leave no seam
            poly = np.vstack([left, right[::-1]])
            # snap=False: matplotlib would otherwise snap a vertical straight to the pixel grid but not the arcs next to it,
            # which draws a one-pixel step at the joint
            ax.fill(poly[:, 0], poly[:, 1], facecolor=light, edgecolor=light, lw=0.5, zorder=1, snap=False)
            ax.plot(left[:, 0], left[:, 1], color=color, lw=0.6, zorder=2, snap=False)   # the two edges only: no cap across the joint
            ax.plot(right[:, 0], right[:, 1], color=color, lw=0.6, zorder=2, snap=False)
        if domain and self.segments:
            from matplotlib.patches import Rectangle
            last = getattr(self, "_domain", {})
            pad = last.get("padding", 20.0) if padding is None else padding
            pml = last.get("pml_width", 5.0) if pml_width is None else pml_width
            x0, x1 = self.extent()
            zs = np.concatenate([self._outline(s, 60)[0][:, 1] for s in self.segments])
            bx0, bx1, z0, z1 = x0 - pad, x1 + pad, float(zs.min()), float(zs.max())
            ax.add_patch(Rectangle((bx0, z0), bx1 - bx0, z1 - z0, fill=False, edgecolor="0.3", lw=1.0, ls="--", zorder=3))
            for xa in (bx0, bx1 - pml):
                ax.add_patch(Rectangle((xa, z0), pml, z1 - z0, facecolor="0.85", edgecolor="none", zorder=0))
            mx, mz = 0.02 * (bx1 - bx0), 0.02 * (z1 - z0)
            ax.set_xlim(bx0 - mx, bx1 + mx)
            ax.set_ylim(z0 - mz, z1 + mz)
        if legend and self.pathways:
            from matplotlib.lines import Line2D
            names = list(labels) if labels is not None else []
            names += [f"pathway {k}" for k in range(len(names) + 1, len(self.pathways) + 1)]
            handles = [Line2D([], [], color=colors[k % len(colors)], lw=3, label=f"{k + 1}: {names[k]}" if labels is not None else names[k])
                       for k in range(len(self.pathways))]
            if any(len(member.get(n, ())) != 1 for n in range(1, len(self.segments) + 1)):
                handles.append(Line2D([], [], color="0.55", lw=3, label="shared or in no pathway"))
            if domain and self.segments:
                from matplotlib.patches import Patch
                handles.append(Line2D([], [], color="0.3", lw=1.0, ls="--", label=f"simulation domain (padding {pad:g} um)"))
                handles.append(Patch(facecolor="0.85", label=f"PML ({pml:g} um)"))
            ax.legend(handles=handles, loc="best", fontsize=9)
        if equal:
            ax.set_aspect("equal")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title(f"{len(self.segments)} segments, {len(self.pathways)} pathways")
        plt.show()

    def _outline(self, s, n=400):
        """Centerline points (n, 2), widths (n,) and unit normals (n, 2) of a segment, for plotting."""
        A, B = np.array(s["A"]), np.array(s["B"])
        t = np.linspace(0, 1, n)
        if s["kind"] == "line":
            c = A + t[:, None] * (B - A)
            d = (B - A) / np.hypot(*(B - A))
            nrm = np.tile([d[1], -d[0]], (n, 1))
        else:
            if s.get("free"):
                radius, i0, i1 = s["free"]
                p0, p1, sign = np.radians(i0), np.radians(i1), np.sign(i1 - i0)
                ph = p0 + t * (p1 - p0)
                c = A + sign * radius * np.column_stack([np.cos(p0) - np.cos(ph), np.sin(ph) - np.sin(p0)])
                C = A + sign * radius * np.array([np.cos(p0), -np.sin(p0)])
            else:
                dx = B[0] - A[0]
                rho = ((B - A) ** 2).sum() / (2 * dx)
                C = np.array([A[0] + rho, A[1]])
                a0, a1 = np.arctan2(A[0] - C[0], A[1] - C[1]), np.arctan2(B[0] - C[0], B[1] - C[1])
                d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
                a = a0 + t * d
                c = C + abs(rho) * np.column_stack([np.sin(a), np.cos(a)])
            nrm = (c - C) / np.hypot(*(c - C).T)[:, None]            # the radial direction, exact
        f = t
        if s["taper"] is not None:
            expression = self._tapers[s["taper"] - 1][0]
            f = eval(expression, {"__builtins__": {}}, {"cos": lambda d: np.cos(np.radians(d)), "sin": lambda d: np.sin(np.radians(d)),
                                                        "pi": np.pi, "z": t})
        return c, s["wa"] + (s["wb"] - s["wa"]) * f, nrm

    # ---- builders
    def add_mmi(self, z, x=0.0, slab_width=17.0, slab_length=682.0, taper_width=5.0, taper_gap=7.0, taper_length=100.0):
        """A 2x2 MMI as in your MMI circuits: two input tapers (guide width to taper_width over taper_length), the slab
        (centered on x), and two output tapers back to the guide width. Starts at z. The ports are at x -/+ (taper_gap +
        taper_width) / 2. Returns a dict with the segment numbers (in_left, in_right, slab, out_left, out_right) and the
        z of the end of the MMI (z_out) and the port x positions (x_left, x_right)."""
        pitch = (taper_gap + taper_width) / 2
        z_slab, z_end = z + taper_length, z + taper_length + slab_length
        out = {}
        for side, name in ((-1, "left"), (1, "right")):
            out[f"in_{name}"] = self.line((x + side * pitch, z), (x + side * pitch, z_slab), width=(self.width, taper_width))
        out["slab"] = self.line((x, z_slab), (x, z_end), width=slab_width)
        for side, name in ((-1, "left"), (1, "right")):
            out[f"out_{name}"] = self.line((x + side * pitch, z_end), (x + side * pitch, z_end + taper_length), width=(taper_width, self.width))
        out.update(z_out=z_end + taper_length, x_left=x - pitch, x_right=x + pitch)
        return out

    def add_delay_arm(self, x, z, z_length, deltaL, R, direction=1, end_straight=None):
        """The delay arm of delay_arm(): straight, S-bend out (two circular arcs of radius R), straight, S-bend back, straight.
        It starts at (x, z), spans z_length in z, bumps toward +x (direction=1) or -x, and is deltaL longer than a straight.
        end_straight: length of the first and last straight (default: a quarter of what the S-bends leave).
        Returns a dict with the segment numbers (in route order), theta_deg, bump_height, path_length, L_end and L_mid."""
        p = delay_arm_parameters(z_length, deltaL, R, end_straight)
        theta, L_end, L_mid = p["theta"], p["L_end"], p["L_mid"]
        d = 1.0 if direction >= 0 else -1.0
        dx_half, dz_half = d * R * (1 - np.cos(theta)), R * np.sin(theta)          # one arc
        h = 2 * dx_half
        numbers, cx, cz = [], x, z

        def straight(length, cx, cz):
            if length > 1e-9:
                numbers.append(self.line((cx, cz), (cx, cz + length)))
                return cz + length
            return cz

        cz = straight(L_end, cx, cz)
        if theta > 0:
            numbers.append(self.arc((cx, cz), (cx + dx_half, cz + dz_half)))                       # out, heading 0 -> theta
            numbers.append(self.arc((cx + h, cz + 2 * dz_half), (cx + dx_half, cz + dz_half)))     # out, theta -> 0 (written backward)
            cx, cz = cx + h, cz + 2 * dz_half
        cz = straight(L_mid, cx, cz)
        if theta > 0:
            numbers.append(self.arc((cx, cz), (cx - dx_half, cz + dz_half)))                       # back, heading 0 -> -theta
            numbers.append(self.arc((cx - h, cz + 2 * dz_half), (cx - dx_half, cz + dz_half)))     # back, -theta -> 0 (written backward)
            cx, cz = cx - h, cz + 2 * dz_half
        cz = straight(L_end, cx, cz)
        return dict(segments=numbers, theta_deg=float(np.degrees(theta)), bump_height=float(abs(h)), path_length=float(z_length + deltaL),
                    L_end=float(L_end), L_mid=float(L_mid))

    def add_sbend(self, x, z, offset, R, straight_offset=0.0, junction_offset=None, direction=1, entry_offset=None, exit_offset=None):
        """A circular S-bend of two arcs of radius R that moves the guide sideways by `offset` (um) from the straight before it,
        which is at x and ends at z, to the straight after it, at x + offset (direction=1) or x - offset (direction=-1).

        straight_offset: the bend-to-straight shift at each end (the arcs sit that far toward the direction of the S-bend's first turn
        from the straights, rlib.straight_offset gives the optimum); junction_offset: the step between the two arcs (default
        2 * straight_offset, the value of your P-02 design). entry_offset and exit_offset set the two ends separately (default
        straight_offset). The arcs then cover offset + junction_offset - entry_offset - exit_offset.
        Returns a dict with the segment numbers (in route order), z_end (where the straight after it begins), x_end, z_length, theta_deg
        (the angle each arc turns) and path_length (the arcs only)."""
        jo = 2 * straight_offset if junction_offset is None else junction_offset
        entry = straight_offset if entry_offset is None else entry_offset
        leave = straight_offset if exit_offset is None else exit_offset
        d_arc = offset + jo - entry - leave
        arc_length, z_length, theta = sbend_lengths(R, d_arc)
        s = 1.0 if direction >= 0 else -1.0
        start = (x + s * entry, z)                                                     # arc 1 begins (heading 0)
        junction = (start[0] + s * d_arc / 2, z + z_length / 2)                        # arc 1 ends
        junction2 = (junction[0] - s * jo, junction[1])                                # arc 2 ends there, a junction_offset away
        finish = (x + s * (offset - leave), z + z_length)                              # arc 2 begins (heading 0), written backward
        numbers = [self.arc(start, junction), self.arc(finish, junction2)]
        return dict(segments=numbers, z_end=float(z + z_length), x_end=float(x + s * offset), z_length=float(z_length),
                    theta_deg=float(np.degrees(theta)), path_length=float(arc_length))

    def add_route(self, x, z, heading, pieces, offset_wavelength=None, width=None):
        """A guide built piece by piece from the point (x, z) and the heading (degrees from +z toward +x).

        pieces: a list of ("line", length) and ("arc", radius, turn), turn in degrees, positive toward +x (the heading increases),
        negative toward -x. The pieces join smoothly along a continuous nominal path. Straights lie on that path; arcs are free arcs.
        Every piece is written as an extended segment, so its width is measured across the guide at any heading (a z-axis segment
        keeps its width along x, which narrows a tilted straight to width * cos(angle) and distorts arcs at steep headings).
        offset_wavelength: if given, every arc is drawn on the circle concentric with its nominal arc and closer to the center by
        abs(straight_offset(radius, offset_wavelength)) (bend_modes, interpolated between scanned radii). The guide then steps by
        exactly that offset perpendicular to itself where an arc meets a straight, and by the sum (opposite turns) or the difference
        (same turn) of the two offsets where two arcs meet, at any heading.
        Returns a dict: segments (route order), x_end, z_end and heading_end (the nominal end), path_length and arc_length (nominal)."""
        offsets = {}
        if offset_wavelength is not None:
            from .bend_modes import straight_offset
        px, pz, phi = float(x), float(z), float(heading)
        numbers, path_length, arc_length = [], 0.0, 0.0
        for piece in pieces:
            kind = piece[0]
            t = np.radians(phi)
            if kind == "line":
                L = float(piece[1])
                end = (px + L * np.sin(t), pz + L * np.cos(t))
                numbers.append(self.line((px, pz), end, width=width, extended=True))      # width across the segment, at any tilt
                px, pz = end
                path_length += L
            elif kind == "arc":
                R, turn = float(piece[1]), float(piece[2])
                sign = 1.0 if turn > 0 else -1.0
                center = (px + sign * R * np.cos(t), pz - sign * R * np.sin(t))         # on the side the arc turns to
                phi1 = phi + turn
                t1 = np.radians(phi1)
                if offset_wavelength is not None and R not in offsets:
                    offsets[R] = abs(straight_offset(R, offset_wavelength))
                r = R - offsets.get(R, 0.0)                                              # the guide's radius: closer to the center
                begin = (center[0] - sign * r * np.cos(t), center[1] + sign * r * np.sin(t))
                n, _ = self.free_arc(begin, r, phi, phi1, width=width, extended=True)
                numbers.append(n)
                px, pz, phi = center[0] - sign * R * np.cos(t1), center[1] + sign * R * np.sin(t1), phi1
                path_length += R * abs(np.radians(turn))
                arc_length += R * abs(np.radians(turn))
            else:
                raise ValueError(f"Unknown piece {piece!r}: use ('line', length) or ('arc', radius, turn_degrees)")
        return dict(segments=numbers, x_end=float(px), z_end=float(pz), heading_end=float(phi), path_length=float(path_length),
                    arc_length=float(arc_length), offsets=offsets)

    def add_sidestep(self, x, z, R, angle_deg, straight, R2=None, shifts=None, direction=1, offsets=None):
        """An arc of radius R that turns the guide by angle_deg, a straight tilted by that angle (length `straight`), and a second arc
        of radius R2 (default R) that turns it back to z: a side-step, of lateral displacement
        R (1 - cos a) + straight sin a + R2 (1 - cos a). direction=1 steps toward +x, -1 toward -x.

        offsets=(s1, s2): the bend-to-straight offsets of the two arcs (um, e.g. abs(rlib.straight_offset(R, wav))). Each arc is drawn
        on a circle concentric with the nominal one and s closer to its center, so the guide steps by exactly s perpendicular to itself
        at both ends of each arc, also at the tilted straight; the straights stay on the nominal path, which is continuous.
        shifts=(s0, s1, s2, s3): instead, x-shifts (um, positive toward the direction of the step) at the four joints (guide before ->
        arc 1, arc 1 -> straight, straight -> arc 2, arc 2 -> guide after), as in an .ind with "begin.x = ... rel end segment" offsets.
        At the two tilted joints an x-shift of s moves the guide sideways by only s cos(angle).
        Returns a dict with the segment numbers (route order), x_end and z_end (where the guide after it begins) and path_length."""
        if offsets is not None and shifts is not None:
            raise ValueError("Give offsets or shifts, not both")
        R2 = R if R2 is None else R2
        a, s = np.radians(angle_deg), (1.0 if direction >= 0 else -1.0)
        if offsets is not None:
            s1, s2 = offsets
            n1 = (x + s * R * (1 - np.cos(a)), z + R * np.sin(a))                       # nominal end of arc 1
            n2 = (n1[0] + s * straight * np.sin(a), n1[1] + straight * np.cos(a))         # nominal end of the straight
            n3 = (n2[0] + s * R2 * (1 - np.cos(a)), n2[1] + R2 * np.sin(a))              # nominal end of arc 2
            c1, r1 = (x + s * R, z), R - s1                                               # arc 1: same center, s1 closer to it
            c2, r2 = (n3[0] - s * R2, n3[1]), R2 - s2                                     # arc 2: same center, s2 closer to it
            g1 = (c1[0] - s * r1, c1[1])
            g2 = (c1[0] - s * r1 * np.cos(a), c1[1] + r1 * np.sin(a))
            g4 = (c2[0] + s * r2, c2[1])
            g3 = (c2[0] + s * r2 * np.cos(a), c2[1] - r2 * np.sin(a))
            numbers = [self.arc(g1, g2, extended=True), self.line(n1, n2, extended=True), self.arc(g4, g3, extended=True)]   # 2nd arc backward
            return dict(segments=numbers, x_end=float(n3[0]), z_end=float(n3[1]), path_length=float(r1 * a + straight + r2 * a))
        sh0, sh1, sh2, sh3 = shifts if shifts is not None else (0.0, 0.0, 0.0, 0.0)
        p1 = (x + s * sh0, z)
        p2 = (p1[0] + s * R * (1 - np.cos(a)), p1[1] + R * np.sin(a))
        p3 = (p2[0] + s * sh1, p2[1])
        p4 = (p3[0] + s * straight * np.sin(a), p3[1] + straight * np.cos(a))
        p5 = (p4[0] + s * sh2, p4[1])
        p6 = (p5[0] + s * R2 * (1 - np.cos(a)), p5[1] + R2 * np.sin(a))
        p7 = (p6[0] + s * sh3, p6[1])
        numbers = [self.arc(p1, p2), self.line(p3, p4), self.arc(p6, p5)]               # the second arc is written backward
        return dict(segments=numbers, x_end=float(p7[0]), z_end=float(p7[1]), path_length=float(R * a + straight + R2 * a))

    def add_tapered_arm(self, x, z, z_length, taper_width, taper_length, hold_length=0.0, start=None, profile="cosine"):
        """A straight arm with a width change: guide width, a taper up to taper_width over taper_length, hold_length at
        taper_width, a taper back down, guide width again. The block starts `start` after z (default: centered in z_length).
        profile: "cosine" (raised cosine, a user taper) or None for linear tapers. Returns the segment numbers (route order)."""
        block = 2 * taper_length + hold_length
        if block > z_length:
            raise ValueError(f"The tapers ({block:g} um) are longer than the arm ({z_length:g} um)")
        start = (z_length - block) / 2 if start is None else start
        if start < 0 or start + block > z_length + 1e-9:
            raise ValueError("The tapers do not fit in the arm")
        w, numbers, cz = self.width, [], z

        def piece(length, width, profile=None):
            nonlocal cz
            if length > 1e-9:
                numbers.append(self.line((x, cz), (x, cz + length), width=width, profile=profile))
                cz += length

        piece(start, w)
        piece(taper_length, (w, taper_width), profile)
        piece(hold_length, taper_width)
        piece(taper_length, (taper_width, w), profile)
        piece(z + z_length - cz, w)
        return dict(segments=numbers, path_length=float(z_length))


# --------------------------------------------------------------------------- #
# Any path
# --------------------------------------------------------------------------- #


def _width_spec(ws, s, kind, width_tol):
    """How the widths ws (at path lengths s) of one piece are written: ("linear", w0, w1) for a constant or linearly changing width,
    ("cosine", w0, w1) for a raised cosine (straight pieces only, a user taper), or None if neither follows them. Also the error."""
    if np.ptp(ws) <= width_tol:
        m = float(np.mean(ws))
        return ("linear", m, m), float(np.abs(ws - m).max())
    length = s[-1] - s[0]
    if length <= 0:
        return ("linear", float(ws[0]), float(ws[-1])), 0.0
    u = (s - s[0]) / length
    for kind_name, profile in (("linear", u), ("cosine", 0.5 - 0.5 * np.cos(np.pi * u))):
        if kind_name == "cosine" and kind != "line":
            continue
        err = float(np.abs(ws - (ws[0] + (ws[-1] - ws[0]) * profile)).max())
        if err <= width_tol:
            return (kind_name, float(ws[0]), float(ws[-1])), err
    return None, None


def _width_breaks(ws, s, rel=0.005, min_run=5):
    """Indices (inside the piece) where the width goes from flat to changing or back: the ends of the tapers. The width is flat where
    its slope is below `rel` of its steepest slope; runs shorter than min_run samples (an abrupt step) are not breaks."""
    slope = np.abs(np.gradient(ws, s))
    if slope.max() <= 0:
        return []
    changing = slope > rel * slope.max()
    runs, start = [], 0
    for i in range(1, len(ws) + 1):
        if i == len(ws) or changing[i] != changing[start]:
            runs.append([bool(changing[start]), start, i - 1])
            start = i
    while len(runs) > 1:                                          # merge the runs that are too short to be a flat or a tapered part
        lengths = [r[2] - r[1] + 1 for r in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_run:
            break
        nb = 1 if k == 0 else k - 1 if k == len(runs) - 1 or lengths[k - 1] >= lengths[k + 1] else k + 1
        lo, hi = min(k, nb), max(k, nb)
        runs[lo] = [runs[nb][0], runs[lo][1], runs[hi][2]]
        del runs[hi]
        k = 0
        while k < len(runs) - 1:
            if runs[k][0] == runs[k + 1][0]:
                runs[k][2] = runs[k + 1][2]
                del runs[k + 1]
            else:
                k += 1
    return [r[1] for r in runs[1:]]


def _fit_with_widths(x, z, phi, w, tol, width_tol):
    """The pieces (primitive, width spec, width error) that follow the path to within tol and its widths to within width_tol.
    A piece whose widths are not linear or a raised cosine is split at the ends of its tapers (where the width goes from flat to
    changing), or, if that does not help, where the width departs most from a straight line."""
    prims = fit_path(x, z, phi, tol)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
    out = []
    for pr in prims:
        i, j = pr["i"], pr["j"]
        spec, err = _width_spec(w[i:j + 1], s[i:j + 1], pr["kind"], width_tol)
        if spec is not None:
            out.append((pr, spec, err))
        elif j - i <= 1:
            out.append((pr, ("linear", float(w[i]), float(w[j])), 0.0))
        else:
            breaks = [i + b for b in _width_breaks(w[i:j + 1], s[i:j + 1])]
            if not breaks:
                u = (s[i:j + 1] - s[i]) / (s[j] - s[i])
                breaks = [i + 1 + int(np.argmax(np.abs(w[i:j + 1] - (w[i] + (w[j] - w[i]) * u))[1:-1]))]
            edges = [i] + breaks + [j]
            for a, b in zip(edges[:-1], edges[1:]):
                out += _fit_with_widths(x[a:b + 1], z[a:b + 1], phi[a:b + 1], w[a:b + 1], tol, width_tol)
    return out


def paths_to_circuit(x, z, phi=None, width=3.4, tol=5e-3, width_tol=5e-3, launch=True, extended_ends=True, height=3.4,
                    orientation="z"):
    """Turn any paths into an RSoft circuit: one pathway per path, built from straight lines, circular arcs and tapers.

    The arguments are those of save_gds: x, z (um) and the tangent angle phi (rad; None computes it) of one path (1D arrays),
    of several paths (a 2D array with one row per path, NaN-padded if they differ in length, or a list of 1D arrays); width: one
    value, one per path, one per point, or a function of the path length s (um) along the path. tol and width_tol (um) are how closely the geometry and the widths are followed
    (see Circuit.add_path). launch: True for a launch field on every pathway, False for none, or a list of the pathway
    numbers (1, 2, ...) that get one. extended_ends: extend the first and last segment of a path if they are lines.
    orientation: "z" (default) or "extended", the segment type RSoft draws (see Circuit).
    Returns (circuit, info): circuit.write(ind) or circuit.export(ind, gds) writes the file(s); info is a list with the
    add_path result of every path."""
    paths = path_list(x, z, phi, width)
    default_width = float(np.median(np.concatenate([p[3] for p in paths])))
    c = Circuit(default_width, height, orientation)
    launched = range(1, len(paths) + 1) if launch is True else [] if launch is False else list(launch)
    info = []
    for k, (px, pz, pphi, pw) in enumerate(paths, start=1):
        r = c.add_path(px, pz, pphi, pw, tol, width_tol, extended_ends)
        c.add_pathway(r["segments"], launch=k in launched)
        info.append(r)
    return c, info


# --------------------------------------------------------------------------- #
# mzi
# --------------------------------------------------------------------------- #


def mzi(z_length, deltaL, R, arm_taper_width, arm_taper_length, arm_hold_length=0.0, arm_taper_start=None, bump_end_straight=None,
        launch="bump", mmi=None, lead_length=100.0, output_length=300.0, bump_side=-1, width=3.4, height=3.4, profile="cosine",
        orientation="z"):
    """An unbalanced Mach-Zehnder: a 2x2 MMI, two arms of the same z-length, and a second 2x2 MMI.

    One arm is the bump delay arm (deltaL longer than a straight, S-bends of radius R); the other is straight with a width
    taper (a taper up to arm_taper_width over arm_taper_length, arm_hold_length at that width, a taper back; cosine).
    z_length is the z-length of the arms between the two MMIs. The bump arm is at x = bump_side * port pitch (bump_side=-1:
    on the -x side, bumping further toward -x). The MMIs are centered on x = 0; mmi overrides their parameters (slab_width,
    slab_length, taper_width, taper_gap, taper_length; the defaults are those of your 2x2MMI circuits).
    lead_length / output_length: the straight guides before the first MMI and after the second.
    Pathway 1 follows the bump arm and pathway 2 the tapered arm, each from its input guide to its output guide.
    launch: "bump" (a launch field on pathway 1), "taper" (pathway 2) or "both".
    Returns (circuit, info); circuit.write(...) writes the .ind file."""
    c = Circuit(width, height, orientation)
    m = dict(slab_width=17.0, slab_length=682.0, taper_width=5.0, taper_gap=7.0, taper_length=100.0)
    m.update(mmi or {})
    pitch = (m["taper_gap"] + m["taper_width"]) / 2
    xb, xt = bump_side * pitch, -bump_side * pitch
    lead_in = {s: c.line((s * pitch, 0.0), (s * pitch, lead_length), extended=True) for s in (-1, 1)}
    mmi1 = c.add_mmi(lead_length, **m)
    z1 = mmi1["z_out"]
    bump = c.add_delay_arm(xb, z1, z_length, deltaL, R, direction=bump_side, end_straight=bump_end_straight)
    taper = c.add_tapered_arm(xt, z1, z_length, arm_taper_width, arm_taper_length, arm_hold_length, arm_taper_start, profile)
    mmi2 = c.add_mmi(z1 + z_length, **m)
    z3 = mmi2["z_out"]
    lead_out = {s: c.line((s * pitch, z3), (s * pitch, z3 + output_length), extended=True) for s in (-1, 1)}

    def route(side, arm):
        name = "left" if side < 0 else "right"
        return ([lead_in[side], mmi1[f"in_{name}"], mmi1["slab"], mmi1[f"out_{name}"]] + arm["segments"]
                + [mmi2[f"in_{name}"], mmi2["slab"], mmi2[f"out_{name}"], lead_out[side]])

    c.add_pathway(route(bump_side, bump), launch=launch in ("bump", "both"))
    c.add_pathway(route(-bump_side, taper), launch=launch in ("taper", "both"))
    info = dict(bump=bump, taper=taper, z_arms=(z1, z1 + z_length), z_total=z3 + output_length, port_pitch=pitch,
                x_bump_arm=xb, x_tapered_arm=xt, deltaL=deltaL)
    return c, info


def radiation_witness(kind="circular", bend_radius=1400.0, bend_angle=30.0, bend_straight=100.0, bend_offset=3360.0, euler_p=0.5,
                      straight_offset=0.0, junction_offset=None, input_length=100.0, witness_offset=100.0, witness_return=400.0,
                      witness_radius=1400.0, witness_straight=1000.0, detector_offsets=(-100.0, -15.0, 300.0), detector_length=1000.0,
                      width=3.4, height=3.4, launch=True, tol=5e-3, orientation="z"):
    """Your P-03 radiation-witness structures: a main guide with a bend, a witness guide 100 um to its -x side, and three straight
    detector guides next to the main guide's output.

    kind selects the main guide's bend:
      "circular": an arc of bend_radius turning by bend_angle, a tilted straight of bend_straight, and an arc back (P-03a).
                  The witness runs concentric to it (radius bend_radius + witness_offset, then bend_radius - witness_offset).
      "euler":    a partial-Euler S-bend with lateral offset bend_offset, minimum radius bend_radius and Euler fraction euler_p
                  (P-03b; the witness runs parallel to it at witness_offset).
      "reference": no bend, the main guide is straight (P-03ref).
    The witness then returns witness_return um toward -x in a circular S-bend of radius witness_radius and ends in a straight of
    witness_straight; the main guide has the same z-length after its bend, so both end together. The detectors are straights of
    detector_length that end there, at detector_offsets (um) from the main guide's x.
    straight_offset: the bend-to-straight offsets of the arcs (junction_offset: of the witness S-bend, default 2 * straight_offset
    for "euler"); see the notes in the code for where each kind applies them.
    tol: the accuracy of the lines and arcs fitted to the Euler paths (um).
    Pathways: 1 the main guide (launch field if launch), 2 the witness, 3, 4 and 5 the detectors (in the order of detector_offsets).
    Returns (circuit, info); circuit.write(...) or circuit.export(...) writes the .ind (and GDS)."""
    if kind not in ("circular", "euler", "reference"):
        raise ValueError('kind must be "circular", "euler" or "reference"')
    c = Circuit(width, height, orientation)
    so = straight_offset
    lead = c.line((0.0, 0.0), (0.0, input_length), extended=True)
    z0 = float(input_length)
    main, wit, info = [lead], [], {}
    sb_args = dict(entry_offset=0.0, exit_offset=0.0, junction_offset=0.0)           # the witness S-bend: no offsets, as in P-03a and P-03ref
    if kind == "circular":
        a = witness_offset
        m = c.add_sidestep(0.0, z0, bend_radius, bend_angle, bend_straight, shifts=(so, -so, -so, so))
        # the witness: arcs of radius bend_radius + a and bend_radius - a, the shifts as in your P-03a file (its second
        # straight-to-arc shift has the opposite sign to the main guide's, so the witness does not return to the nominal x by so)
        w = c.add_sidestep(-a, z0, bend_radius + a, bend_angle, bend_straight, R2=bend_radius - a, shifts=(so, -so, so, 0.0))
        x_out, z_out = m["x_end"], m["z_end"]
        x_w, z_w = w["x_end"], w["z_end"]
        main += m["segments"]
        wit += w["segments"]
        info.update(bend_path_length=m["path_length"], witness_bend_path_length=w["path_length"])
    elif kind == "euler":
        x, z, phi = euler_sbend(bend_offset, bend_radius, euler_p)
        pm = c.add_path(x, z + z0, phi, tol=tol)
        xw, zw, _ = parallel_path(x, z, phi, -witness_offset)
        pw = c.add_path(xw, zw + z0, phi, tol=tol)
        x_out, z_out = float(x[-1]), float(z[-1] + z0)
        x_w, z_w = float(xw[-1]), float(zw[-1] + z0)
        main += pm["segments"]
        wit += pw["segments"]
        jo = 2 * so if junction_offset is None else junction_offset
        sb_args = dict(entry_offset=0.0, exit_offset=so, junction_offset=jo)         # P-03b: the straight after the S-bend is shifted by so
        info.update(euler_z_length=float(z[-1]), euler_max_deviation=max(pm["max_deviation"], pw["max_deviation"]),
                    euler_segments=(len(pm["segments"]), len(pw["segments"])))
    else:
        x_out, z_out = 0.0, z0
        x_w, z_w = -witness_offset, z0
    # the witness S-bend and straight; the main guide's straight after its bend has the same z-length, so both end at z_end
    offset = witness_return - sb_args["junction_offset"] + sb_args["entry_offset"] + sb_args["exit_offset"]
    sb = c.add_sbend(x_w, z_w, offset, witness_radius, direction=-1, **sb_args)
    wit += sb["segments"]
    wit.append(c.line((sb["x_end"], sb["z_end"]), (sb["x_end"], sb["z_end"] + witness_straight), extended=True))
    z_end = z_out + sb["z_length"] + witness_straight
    main.append(c.line((x_out, z_out), (x_out, z_end), extended=True))
    c.add_pathway(main, launch=launch)
    c.add_pathway(wit, launch=False)
    for dx in detector_offsets:
        c.add_pathway([c.line((x_out + dx, z_end - detector_length), (x_out + dx, z_end), extended=True)], launch=False)
    info.update(kind=kind, x_out=float(x_out), z_out=float(z_out), z_end=float(z_end), witness_sbend_z_length=sb["z_length"],
                main_path_length=c.path_length(main), witness_path_length=c.path_length(wit))
    return c, info


# --------------------------------------------------------------------------- #
# Run helpers
# --------------------------------------------------------------------------- #


def pathway_file_symbols(pathways, launch=None, monitor=None):
    """The symbols {Launch_filename_k, Monitor_filename_k} of a file written by gds_to_ind, for run_sim(..., symbols=...).

    pathways : the path of the .ind file (the number of pathways is read from its Launch_filename_<k> symbols, so the file can
               come from an earlier session), or the number of pathways, or the list info["pathways"] that gds_to_ind returned.
    launch, monitor : one file name for every pathway, or a list with one file name per pathway (in the order of the
                      pathways), or a function of the pathway number k = 1, 2, ... that returns the file name.
                      None leaves those symbols as they are in the file."""
    if isinstance(pathways, (str, Path)):
        found = [int(k) for k in re.findall(r"^Launch_filename_(\d+) =", Path(pathways).read_text(errors="replace"), re.M)]
        if not found:
            raise ValueError(f"{pathways} has no Launch_filename_<k> symbols; was it written by gds_to_ind?")
        n = max(found)
    else:
        n = pathways if isinstance(pathways, int) else len(pathways)
    out = {}
    for name, spec in (("Launch_filename", launch), ("Monitor_filename", monitor)):
        if spec is None:
            continue
        if not (isinstance(spec, str) or callable(spec)) and len(spec) != n:
            raise ValueError(f"{name}: {len(spec)} file names for {n} pathways")
        for k in range(1, n + 1):
            out[f"{name}_{k}"] = spec(k) if callable(spec) else spec if isinstance(spec, str) else spec[k - 1]
    return out


def ind2gds(ind_path, gds_path=None, layer=None, resolution=None, cwd=None):
    """Export an RSoft .ind file to GDS with RSoft's ind2gds utility (layer 0 -> `layer`; default resolution: CAD preference).

    ind2gds looks for files that the circuit refers to (such as an imported .gds) relative to its working directory, which is
    the folder of the .ind file unless `cwd` says otherwise."""
    ind_path = Path(ind_path).resolve()
    if not ind_path.is_file():
        raise FileNotFoundError(f"The .ind file does not exist: {ind_path}")
    cwd = Path(cwd).resolve() if cwd else ind_path.parent
    if not cwd.is_dir():
        raise NotADirectoryError(f"The working directory does not exist: {cwd}")
    gds_path = Path(gds_path).resolve() if gds_path else ind_path.with_suffix(".gds")
    gds_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ind2gds", f"-o={gds_path}"]
    if layer is not None:
        command.append(f"-l{layer}")
    if resolution is not None:
        command.append(f"-r{resolution}")
    result = subprocess.run(command + [str(ind_path)], cwd=cwd, capture_output=True, text=True)
    if not gds_path.exists():
        raise RuntimeError(f"ind2gds produced no file: {result.stdout} {result.stderr}")
    return gds_path