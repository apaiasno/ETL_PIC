"""GDS: the independent GDS writer and reader, and the conversion of a GDS into an RSoft .ind file.

The preferred route is rsoft_lib.circuit (build the circuit in RSoft terms, export the GDS with RSoft's own export). These tools write a
GDS directly from paths (save_gds, for example to compare with RSoft's export), read waveguides back from a GDS (read_gds_shapes,
gds_waveguides), and turn a GDS of waveguides into a BeamPROP .ind file (gds_to_ind).

Contents (in this order)
    GDS output          save_gds
    GDS input           read_gds_shapes, gds_waveguides
    GDS -> .ind         fitting a GDS centerline with lines and arcs (arcs tangent to z), gds_to_ind
"""


import struct
import warnings
from pathlib import Path

import numpy as np

from .circuit import (CLADDING_MATERIAL, CORE_MATERIAL, CORE_INDEX_FACTOR, DEFAULT_MODE_FILE, MAX_PATHWAY_SEGMENTS, _ind_text,
                      _segment_text, _write_ind)
from .geometry import path_list


# --------------------------------------------------------------------------- #
# GDS output
# --------------------------------------------------------------------------- #


def save_gds(filename, x, z, phi, width, layer=1, cell_name="EULER_SBEND", plot=False):
    """Write the waveguide outlines (centerline +- width/2) to a GDSII file.

    x, z, phi : lateral position (um), propagation position (um) and tangent angle from z (rad) of the centerlines. One path:
                1D arrays. Several paths: a 2D array with one row per path (rows padded with NaN at the end if the paths
                have different numbers of points), or a list of 1D arrays. phi=None computes it from x and z.
    width     : one value for every path; or one value per path; or one value per point (a taper), shaped like x; or a function
                of the path length s (um) along the path, such as lambda s: 3.4 + 1.6 * s / 500 (one function for every path, or a list
                with a number, array or function for each path).
    The GDS has x as the lateral axis and y as the propagation axis, in um on a 1 nm grid. Every polygon is tagged (property 1)
    with its path number, so that paths which share an end point stay separate paths when gds_to_ind reads the file back.
    plot     : True draws one outline per path (even if the GDS splits a long path into several polygons), in the orientation of
               the GDS (x across, propagation z up) with the pathways numbered as gds_to_ind numbers them; "equal" does the same with equal axes (to-scale, but a long structure becomes thin).
    """
    def rec(rtype, dtype, data=b""):
        data += b"\x00" * (len(data) % 2)
        return struct.pack(">HBB", 4 + len(data), rtype, dtype) + data

    def real8(v):                                   # GDSII 8-byte real (v > 0)
        e = 64
        while v >= 1:
            v /= 16
            e += 1
        while v < 1 / 16:
            v *= 16
            e -= 1
        return bytes([e]) + int(round(v * 2**56)).to_bytes(7, "big")

    paths = path_list(x, z, phi, width)
    out = [rec(0x00, 0x02, struct.pack(">h", 600)), rec(0x01, 0x02, bytes(24)),
           rec(0x02, 0x06, b"LIB"), rec(0x03, 0x05, real8(1e-3) + real8(1e-9)),
           rec(0x05, 0x02, bytes(24)), rec(0x06, 0x06, cell_name.encode())]
    step = 4000                                     # keep each polygon under GDSII's 8191-point limit
    for k, (px, pz, pphi, w) in enumerate(paths):
        nx_, nz_ = np.cos(pphi), -np.sin(pphi)      # normal to the path
        left = np.column_stack([px + w / 2 * nx_, pz + w / 2 * nz_])
        right = np.column_stack([px - w / 2 * nx_, pz - w / 2 * nz_])
        for i in range(0, len(px) - 1, step):
            j = min(i + step, len(px) - 1)
            poly = np.vstack([left[i:j + 1], right[i:j + 1][::-1], left[i:i + 1]])
            pts = np.round(poly * 1000).astype(">i4")   # um -> nm
            out += [rec(0x08, 0x00), rec(0x0D, 0x02, struct.pack(">h", layer)),
                    rec(0x0E, 0x02, struct.pack(">h", 0)), rec(0x10, 0x03, pts.tobytes()),
                    rec(0x2B, 0x02, struct.pack(">h", 1)), rec(0x2C, 0x06, str(k).encode()),
                    rec(0x11, 0x00)]
    out += [rec(0x07, 0x00), rec(0x04, 0x00)]
    with open(filename, "wb") as f:
        f.write(b"".join(out))
    if plot:
        import matplotlib.pyplot as plt
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=(5, 9) if plot == "equal" else (6, 8))
        for k, (px, pz, pphi, w) in enumerate(paths):                  # one outline per path, however many polygons it takes in the GDS
            nx_, nz_ = np.cos(pphi), -np.sin(pphi)
            outline = np.vstack([np.column_stack([px + w / 2 * nx_, pz + w / 2 * nz_]),
                                 np.column_stack([px - w / 2 * nx_, pz - w / 2 * nz_])[::-1]])
            outline = np.vstack([outline, outline[:1]])
            ax.fill(outline[:, 0], outline[:, 1], color=colors[k % len(colors)], alpha=0.45, lw=0)
            ax.plot(outline[:, 0], outline[:, 1], color=colors[k % len(colors)], lw=0.8)
        for k, (px, pz, pphi, w) in enumerate(paths):
            ax.plot(px[0], pz[0], "o", color=colors[k % len(colors)], ms=5)
            ax.annotate(f"pathway {k + 1}", (px[0], pz[0]), textcoords="offset points", xytext=(8, 6 + 12 * k), fontsize=8,
                        color=colors[k % len(colors)])
        if plot == "equal":
            ax.set_aspect("equal")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title(f"{Path(filename).name}: {len(paths)} path(s), layer {layer}")
        plt.show()


# --------------------------------------------------------------------------- #
# GDS input
# --------------------------------------------------------------------------- #


def read_gds_shapes(gds_path):
    """Boundaries and paths of a GDS file: a list of dicts (kind, layer, datatype, xy in um, width in um or None, path_id).

    path_id is the path number that save_gds writes as property 1 on every polygon (None if the polygon has none)."""
    def real8(b):
        return (-1 if b[0] & 0x80 else 1) * int.from_bytes(b[1:], "big") / 2 ** 56 * 16.0 ** ((b[0] & 0x7F) - 64)

    data = Path(gds_path).read_bytes()
    scale, i, shapes, cur, attr = 1e-3, 0, [], None, None
    while i < len(data):
        n, rtype = struct.unpack(">HB", data[i:i + 3])
        body = data[i + 4:i + n]
        if rtype == 0x03:
            scale = real8(body[8:16]) * 1e6                  # meters per database unit -> um
        elif rtype in (0x08, 0x09):
            cur = dict(kind="boundary" if rtype == 0x08 else "path", layer=0, datatype=0, xy=None, width=None, path_id=None)
        elif cur is not None and rtype == 0x0D:
            cur["layer"] = struct.unpack(">h", body)[0]
        elif cur is not None and rtype == 0x0E:
            cur["datatype"] = struct.unpack(">h", body)[0]
        elif cur is not None and rtype == 0x0F:
            cur["width"] = abs(struct.unpack(">i", body)[0]) * scale
        elif cur is not None and rtype == 0x10:
            cur["xy"] = np.frombuffer(body, ">i4").reshape(-1, 2) * scale
        elif cur is not None and rtype == 0x2B:
            attr = struct.unpack(">h", body)[0]
        elif cur is not None and rtype == 0x2C and attr == 1:
            cur["path_id"] = body.rstrip(b"\x00").decode()
        elif cur is not None and rtype == 0x11:
            if cur["xy"] is not None:
                shapes.append(cur)
            cur = None
        i += n
    return shapes


def _outline_to_centerline(poly):
    """Centerline (N, 2) and width (N,) of an outline polygon: left side forward then right side backward (as save_gds
    writes it), or a rectangle/trapezoid with any starting corner. None if the polygon is not of that kind."""
    P = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    M = len(P)
    if M < 4 or M % 2:
        return None
    N = M // 2
    candidates = []
    if M == 4:
        for r in range(4):
            Q = np.roll(P, -r, axis=0)
            candidates.append((Q[:2], Q[2:][::-1]))
    else:
        candidates.append((P[:N], P[N:][::-1]))
        candidates.append((P[N:][::-1], P[:N]))
    best = None
    for left, right in candidates:
        center, width = (left + right) / 2, np.hypot(*(left - right).T)
        if width.min() <= 0:
            continue
        direction = np.gradient(center, axis=0)
        length = np.hypot(*np.diff(center, axis=0).T).sum()
        across = left - right
        cos = np.abs(np.sum(direction * across, axis=1)) / (np.hypot(*direction.T) * width + 1e-30)
        if cos.max() < 0.2 and length > width.max() * 0.5:            # the width is across the path at every point
            if best is None or length > best[2]:
                best = (center, width, length)
    return None if best is None else (best[0], best[1])


def gds_waveguides(gds_path, layer=None, swap=False, default_width=3.4):
    """The waveguides of a GDS as a list of (centerline (N, 2) as (x, z) in um, width (N,), reversed), each running toward +z.

    Polygons tagged with a path number (as save_gds writes them) are grouped by it, in the order of the paths; other polygons
    are joined when one ends where the next begins. `reversed` says that the path ran toward -z in the GDS and was flipped.
    swap=False: RSoft x = GDS x and RSoft z = GDS y (as save_gds writes them); swap=True: x = GDS y and z = GDS x."""
    chains, skipped = [], 0                                   # each chain: [key, centerline, width]
    for s in read_gds_shapes(gds_path):
        if layer is not None and (s["layer"] != (layer[0] if isinstance(layer, tuple) else layer) or
                                  (isinstance(layer, tuple) and s["datatype"] != layer[1])):
            continue
        if s["kind"] == "path":
            c, w = s["xy"], np.full(len(s["xy"]), s["width"] if s["width"] else default_width)
        else:
            res = _outline_to_centerline(s["xy"])
            if res is None:
                skipped += 1
                continue
            c, w = res
        for chain in chains:
            touches = np.hypot(*(chain[1][-1] - c[0])) < 5e-3                 # this polygon starts where the chain ends
            if (s["path_id"] is not None and chain[0] == s["path_id"]) or (s["path_id"] is None and chain[0] is None and touches):
                skip = 1 if touches else 0                                     # chunks of one path share their end point
                chain[1], chain[2] = np.vstack([chain[1], c[skip:]]), np.concatenate([chain[2], w[skip:]])
                break
        else:
            chains.append([s["path_id"], c, w])
    if skipped:
        warnings.warn(f"{skipped} polygon(s) are not waveguide outlines (centerline plus width) and were skipped")
    if not chains:
        raise ValueError("No waveguides found in " + str(gds_path))
    out = []
    for _, c, w in chains:
        if swap:
            c = c[:, ::-1]
        flipped = bool(c[-1, 1] < c[0, 1])
        if flipped:
            c, w = c[::-1], w[::-1]
        out.append((c, w, flipped))
    return out


# --------------------------------------------------------------------------- #
# GDS -> .ind (a GDS centerline is fitted with lines and arcs that are tangent to z at one end)
# --------------------------------------------------------------------------- #


def _simplify(points, tol):
    """Ramer-Douglas-Peucker: indices of the points that keep the polyline within `tol` of the original."""
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = points[b] - points[a]
        norm = np.hypot(*d)
        rel = points[a + 1:b] - points[a]
        dist = np.hypot(*rel.T) if norm == 0 else np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / norm
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.nonzero(keep)[0]


def _circle_fit(Q):
    """Least-squares circle through the points Q (n, 2): center and radius."""
    mu = Q.mean(axis=0)
    Q0 = Q - mu
    sol = np.linalg.lstsq(np.column_stack([2 * Q0, np.ones(len(Q0))]), (Q0 ** 2).sum(axis=1), rcond=None)[0]
    return sol[:2] + mu, float(np.sqrt(sol[2] + sol[:2] @ sol[:2]))


def _line_fit(Q):
    """Least-squares line through the points Q (n, 2): a point on it and its unit direction (toward the last point)."""
    mu = Q.mean(axis=0)
    d = np.linalg.svd(Q - mu, full_matrices=False)[2][0]
    return mu, (d if d @ (Q[-1] - Q[0]) >= 0 else -d)


def _ztangent_circle(T, E):
    """Center and radius of the circle through E that is tangent to the z direction at T (what RSoft draws for an arc segment
    with arc_iangle = 0 that begins at T and ends at E). None if E lies straight ahead of T."""
    dx = E[0] - T[0]
    if abs(dx) < 1e-9:
        return None
    rho = ((E - T) ** 2).sum() / (2 * dx)
    return np.array([T[0] + rho, T[1]]), abs(rho)


def _arc_headings(A, B, C):
    """Headings (rad from +z toward +x) at the start A and at the end B of the arc around C that runs from A to B."""
    out = []
    for Q in (A, B):
        u = (Q - C) / np.linalg.norm(Q - C)
        t = np.array([u[1], -u[0]])
        if t @ (B - A) < 0:
            t = -t
        out.append(float(np.arctan2(t[0], t[1])))
    return out


def _fit_primitives(P, w, tol=1e-3, arcs=True, arc_tol=5e-3, base=20, kappa_straight=2.5e-5):
    """Split a centerline P (N, 2) of (x, z) into straight lines, circular arcs and, where neither fits, polyline pieces.

    An arc is only recognized if it is tangent to the z direction at one end (that is what RSoft's arc_iangle = 0 draws), which
    is the case for circular S-bends. Junctions between a line and an arc, or between two arcs of opposite curvature, are moved
    to the exact tangent point so that the path stays smooth. Returns a list of dicts (kind, begin, end, widths, ...)."""
    N = len(P)

    def nearest(point, lo=0, hi=N - 1):
        lo, hi = max(lo, 0), min(hi, N - 1)
        return lo + int(np.argmin(np.hypot(*(P[lo:hi + 1] - point).T)))

    def poly(begin, end, ib, ie):
        if ie > ib + 1:
            pts, wd = np.vstack([begin, P[ib + 1:ie], end]), np.r_[w[ib], w[ib + 1:ie], w[ie]]
        else:
            pts, wd = np.vstack([begin, end]), np.r_[w[ib], w[ie]]
        keep = _simplify(pts, tol)
        return dict(kind="poly", begin=begin, end=end, pts=pts[keep], wd=wd[keep])

    b = int(max(2, min(base, N // 20)))
    if not arcs or N < 8 * b + 10:
        return [poly(P[0], P[-1], 0, N - 1)]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    idx = np.arange(N)
    lo, hi = np.clip(idx - b, 0, N - 1), np.clip(idx + b, 0, N - 1)
    t = P[hi] - P[lo]
    phi = np.arctan2(t[:, 0], t[:, 1])
    kappa = (phi[hi] - phi[lo]) / np.maximum(s[hi] - s[lo], 1e-12)
    kappa[:2 * b] = kappa[2 * b]
    kappa[-2 * b:] = kappa[-2 * b - 1]
    cls = np.where(np.abs(kappa) < kappa_straight, 0, np.sign(kappa)).astype(int)
    runs, s0 = [], 0
    for i in range(1, N + 1):
        if i == N or cls[i] != cls[s0]:
            runs.append([int(cls[s0]), s0, i - 1])
            s0 = i
    minlen = 4 * b + 1
    while len(runs) > 1:                                          # merge the runs that are too short to be a primitive
        lengths = [r[2] - r[1] + 1 for r in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= minlen:
            break
        nb = 1 if k == 0 else k - 1 if k == len(runs) - 1 or lengths[k - 1] >= lengths[k + 1] else k + 1
        a, z_ = min(k, nb), max(k, nb)
        runs[a] = [runs[nb][0], runs[a][1], runs[z_][2]]
        del runs[z_]
        k = 0
        while k < len(runs) - 1:
            if runs[k][0] == runs[k + 1][0]:
                runs[k][2] = runs[k + 1][2]
                del runs[k + 1]
            else:
                k += 1
    fits = []
    for c, i0, i1 in runs:
        m = min(2 * b, (i1 - i0) // 4)
        Q = P[i0 + m:i1 - m + 1]
        if c == 0:
            p0, d = _line_fit(Q)
            ok = np.abs((Q - p0)[:, 0] * d[1] - (Q - p0)[:, 1] * d[0]).max() <= arc_tol
            fits.append(dict(kind="line", p=p0, d=d, Q=Q) if ok else None)
        else:
            C, r = _circle_fit(Q)
            ok = np.abs(np.hypot(*(Q - C).T) - r).max() <= arc_tol
            fits.append(dict(kind="arc", C=C, r=r, Q=Q) if ok else None)
    foot = lambda C, p, d: p + d * ((C - p) @ d)
    ds_mean = s[-1] / (N - 1)
    joints = []
    for k in range(len(runs) - 1):
        f0, f1, j = fits[k], fits[k + 1], None
        if f0 and f1:
            if f0["kind"] == "line" and f1["kind"] == "arc":
                j = foot(f1["C"], f0["p"], f0["d"])
            elif f0["kind"] == "arc" and f1["kind"] == "line":
                j = foot(f0["C"], f1["p"], f1["d"])
            elif f0["kind"] == "arc" and f1["kind"] == "arc" and runs[k][0] == -runs[k + 1][0]:
                u = (f1["C"] - f0["C"]) / np.linalg.norm(f1["C"] - f0["C"])
                j = f0["C"] + f0["r"] * u
        boundary = P[runs[k][2]]
        joints.append(j if j is not None and np.hypot(*(j - boundary)) < 6 * b * ds_mean else boundary)
    prims = []
    for k, (c, i0, i1) in enumerate(runs):
        begin = P[0] if k == 0 else joints[k - 1]
        end = P[-1] if k == len(runs) - 1 else joints[k]
        ib = nearest(begin, i0 - 4 * b, i1 + 4 * b)
        ie = nearest(end, i0 - 4 * b, i1 + 4 * b)
        f = fits[k]
        if f and f["kind"] == "line":
            prims.append(dict(kind="line", begin=begin, end=end, wb=w[ib], we=w[ie]))
            continue
        if f and f["kind"] == "arc":
            best = None
            for first in (True, False):                           # which end of the arc is tangent to z
                T, E = (begin, end) if first else (end, begin)
                circ = _ztangent_circle(T, E)
                if circ is not None:
                    dev = np.abs(np.hypot(*(f["Q"] - circ[0]).T) - circ[1]).max()
                    if best is None or dev < best[0]:
                        best = (dev, first, circ[0])
            if best is not None and best[0] <= arc_tol:
                prims.append(dict(kind="arc", begin=begin, end=end, wb=w[ib], we=w[ie], tangent_first=best[1], C=best[2]))
                continue
        prims.append(poly(begin, end, min(ib, ie), max(ib, ie)))
    return prims


def _kink_loss_dB(prims, width, wavelength, n_eff=1.45):
    """Rough estimate (dB) of the loss that comes from the corners where straight chords stand in for a smooth bend.

    A corner of angle d turns the guided mode by d; the overlap with the mode of the next chord is
    exp(-(2 pi n_eff w0 d / wavelength)^2 / 4), with w0 = 0.65 x width the mode radius."""
    heads = []                                                   # (heading at the start, heading at the end) of every piece
    corners = []
    for pr in prims:
        if pr["kind"] == "line":
            d = pr["end"] - pr["begin"]
            heads.append((np.arctan2(d[0], d[1]),) * 2)
        elif pr["kind"] == "arc":
            T, E = (pr["begin"], pr["end"]) if pr["tangent_first"] else (pr["end"], pr["begin"])
            C = _ztangent_circle(T, E)[0]
            heads.append(tuple(_arc_headings(pr["begin"], pr["end"], C)))
        else:
            d = np.diff(pr["pts"], axis=0)
            h = np.arctan2(d[:, 0], d[:, 1])
            corners += list(np.abs(np.diff(h)))
            heads.append((h[0], h[-1]))
    corners += [abs(a[1] - b[0]) for a, b in zip(heads[:-1], heads[1:])]
    q = (2 * np.pi * n_eff * 0.65 * width / wavelength) ** 2 / 4
    return float(10 * np.log10(np.e) * q * np.sum(np.square(corners)))          # -10 log10 of the product of the overlaps


def _primitive_segments(prims, width, first_number, all_extended=False):
    """The text of the RSoft segments for the primitives, and their numbers. Arcs are written with arc_iangle = 0 and begin at
    their z-tangent end (an arc that is tangent to z at its far end is written backward, as in your other circuits)."""
    pieces = []
    for pr in prims:
        if pr["kind"] == "line":
            pieces.append(("line", pr["begin"], pr["end"], pr["wb"], pr["we"]))
        elif pr["kind"] == "arc":
            if pr["tangent_first"]:
                pieces.append(("arc", pr["begin"], pr["end"], pr["wb"], pr["we"]))
            else:
                pieces.append(("arc", pr["end"], pr["begin"], pr["we"], pr["wb"]))
        else:
            for k in range(len(pr["pts"]) - 1):
                pieces.append(("line", pr["pts"][k], pr["pts"][k + 1], pr["wd"][k], pr["wd"][k + 1]))
    texts, numbers = [], []
    for k, (kind, A, B, wa, wb) in enumerate(pieces):
        numbers.append(first_number + k)
        # only straights are extended to the boundary, as in your circuits
        texts.append(_segment_text(first_number + k, kind, A, B, wa, wb, width,
                                   extended=all_extended or (k in (0, len(pieces) - 1) and kind == "line")))
    return texts, numbers


def gds_to_ind(gds_path, ind_path, layer=None, swap=False, launch_files=None, monitor_files=None, default_file=DEFAULT_MODE_FILE,
               height=3.4, width=None, Padding=20.0, PML_width=5.0, Contrast=1.02, Grid_Size=0.1, Grid_Edge=0.5, Step_z=0.2,
               pade_order=4, wavelength=1.55, tol=1e-3, core_material=None, cladding_material=None, write_materials=True,
               arcs=True, arc_tol=5e-3, max_segments=None, validate=True, orientation="z"):
    """Write an RSoft .ind file for BeamPROP from the waveguides of a GDS (for example one written by save_gds).

    Defaults: height 3.4 um, Padding 20 um (between the extreme x of the circuit and the volume boundary; the y boundary is
    height/2 + Padding), PML_width 5 um with BC_PML, pade_order 4 (and the 3D warning off), Grid_Size 0.1 um, Step_z 0.2 um,
    Grid_Edge 0.5, semi-vectorial, amplitude/phase slice output. The core and cladding materials are CORE_MATERIAL and
    CLADDING_MATERIAL (the global parameters at the top of rsoft_lib.circuit) unless core_material / cladding_material are given;
    their definitions are written into the file unless write_materials=False, which only sets the two names.
    `width` is the default waveguide width symbol (None: the median width in the GDS).

    Each waveguide becomes a pathway with a launch field centered on its start (symbol Launch_filename_<n>) and two monitors,
    power and phase, on it (symbol Monitor_filename_<n>); both default to `default_file`.
    launch_files / monitor_files : optional lists of file names, one per pathway, in the order of the pathways.
    arcs : write straight lines and circular arcs (tangent to z at one end, as in circular S-bends) as RSoft segments, which
           is exact and needs few segments; anything else becomes straight chords within `tol` (um) of the centerline.
    arc_tol : how far (um) the centerline may be from a fitted line or arc.
    max_segments : the most segments RSoft accepts in a pathway (default MAX_PATHWAY_SEGMENTS); a longer pathway raises an error.
    validate : read the file back with RSoft's own parser (if RSoft's Python tools are installed) and raise if it fails.
    orientation : "z" (default) writes z-axis segments except the end straights, "extended" writes every segment as RSoft's Extended
                  type (see rsoft_lib.circuit.Circuit).
    Returns a dict with the pathways (centerline, widths, segment numbers, start, end, reversed, number of lines/arcs/polyline
    pieces and chords, estimated corner loss), the x extent and the width.
    """
    core_material, cladding_material = core_material or CORE_MATERIAL, cladding_material or CLADDING_MATERIAL
    guides = gds_waveguides(gds_path, layer, swap, 3.4 if width is None else width)
    if width is None:
        width = float(np.median(np.concatenate([w for _, w, _ in guides])))
    xmin = min((c[:, 0] - w / 2).min() for c, w, _ in guides)
    xmax = max((c[:, 0] + w / 2).max() for c, w, _ in guides)

    max_segments = max_segments or MAX_PATHWAY_SEGMENTS
    segments, pathways, n_seg = [], [], 0
    for k, (c, w, flipped) in enumerate(guides, start=1):
        prims = _fit_primitives(c, w, tol, arcs, arc_tol)
        texts, numbers = _primitive_segments(prims, width, n_seg + 1, orientation == "extended")
        n_seg += len(numbers)
        segments += texts
        kinds = {kind: sum(p["kind"] == kind for p in prims) for kind in ("line", "arc", "poly")}
        n_chords = sum(len(p["pts"]) - 1 for p in prims if p["kind"] == "poly")
        loss = _kink_loss_dB(prims, width, wavelength)
        if len(numbers) > max_segments:
            raise ValueError(f"Pathway {k} needs {len(numbers)} segments, but RSoft accepts at most {max_segments} in a pathway "
                             f"('Maximum pathway length exceeded'). {n_chords} of them are straight chords standing in for a "
                             f"curve that is not a circular arc tangent to z (for example an Euler bend). Raising tol (now {tol:g} um) "
                             f"gives fewer chords but more loss at their corners; a smooth curve needs arcs with other start angles.")
        if loss > 0.005:
            warnings.warn(f"Pathway {k}: {n_chords} straight chords stand in for curved parts; the corners between them add roughly "
                          f"{loss:.3f} dB of loss that the real structure does not have.")
        pathways.append(dict(segments=numbers, points=c, widths=w, start=tuple(prims[0]["begin"]), end=tuple(prims[-1]["end"]),
                             reversed=flipped, kinds=kinds, chords=n_chords, corner_loss_dB=loss))

    n = len(pathways)
    launch_files = list(launch_files) if launch_files else [default_file] * n
    monitor_files = list(monitor_files) if monitor_files else [default_file] * n
    if len(launch_files) != n or len(monitor_files) != n:
        raise ValueError(f"{n} pathways were found, so launch_files and monitor_files need {n} entries each")

    text = _ind_text(segments, [p["segments"] for p in pathways], [p["start"][0] for p in pathways], xmin, xmax, width, height, Padding,
                     PML_width, Contrast, Grid_Size, Grid_Edge, Step_z, pade_order, wavelength, core_material, cladding_material,
                     write_materials, launch_files, monitor_files)
    _write_ind(ind_path, text, validate)
    return dict(pathways=pathways, xmin=float(xmin), xmax=float(xmax), n_segments=n_seg, width=width)