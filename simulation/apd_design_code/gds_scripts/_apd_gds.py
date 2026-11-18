"""Shared APD -> GDS builder. Import this from the per-design scripts.

Self-contained: needs only numpy and gdstk (pip install gdstk).

GEOMETRY, per arm
    input taper (3.4 um access guide -> section 1)
    section 1 | taper | section 2 | ... | section N
    output taper (section N -> 3.4 um)
    [TOP ARM ONLY] dL of extra 3.4 um guide   <-- the pi bias, see below
    2 mm thermal phase shifter on plain 3.4 um guide

Tapers are RAISED COSINE, w(s) = w0 + (w1 - w0) * [0.5 - 0.5 cos(pi s)],
which is exactly the profile the phase model integrates.

THE PATH OFFSET - the one thing to be careful with
    The pi phase bias is built into the LAYOUT as dL of extra path in one arm,
    rather than supplied by the trim heaters. The two arms are therefore
    DELIBERATELY UNEQUAL inside the APD block by dL (~100-142 nm). That
    asymmetry IS the phase bias: padding the shorter arm to match would cancel
    exactly the phase it provides. The 2 mm trim shifters are equal length, so
    the arms end up equal overall - the offset lives in the APD section only.

LAYERS
     1/0  waveguide core
    10/0  heater (2 mm thermal phase shifter, one per arm)
    60/0  device extent marker
    61/0  trench keep-out reminder, 50 um wide per arm (+/-25 um of a 3.4 um
          guide, per Bart) - a REMINDER, not a trench
    62/0  APD / trim-shifter boundary (floorplan budgets the trim separately)
    63/0  text
"""
import numpy as np
import gdstk

# ---------------------------------------------------------------- constants
ARM_PITCH = 296.278     # um, centre-to-centre; arms treated as uncoupled
GUIDE = 3.4             # um, access waveguide
TRIM_PS = 2000.0        # um, standard thermal phase shifter per arm
HEATER_W = 8.0          # um, drawn heater width
KEEPOUT = 50.0          # um, trench exclusion reminder
NPT_TAPER = 100         # points per taper (< 0.3 nm chord error)
UNIT, PRECISION = 1e-6, 1e-9      # 1 um user unit on a 1 nm database grid


def arm_points(sections, tapers, width_index):
    """[(x, w), ...] for one arm. width_index: 1 = top widths, 2 = bottom."""
    pts = []
    x = 0.0
    widths = [s[width_index] for s in sections]
    edges = [GUIDE] + widths + [GUIDE]

    def cosine(x0, w0, w1, length):
        for i in range(NPT_TAPER + 1):
            s = i / NPT_TAPER
            g = 0.5 - 0.5 * np.cos(np.pi * s)
            pts.append((x0 + s * length, w0 + (w1 - w0) * g))

    for i, (L, _, _) in enumerate(sections):
        if tapers[i] > 0:
            cosine(x, edges[i], edges[i + 1], tapers[i])
            x += tapers[i]
        else:
            pts.append((x, edges[i + 1]))
        pts.append((x, widths[i]))
        pts.append((x + L, widths[i]))
        x += L
    if tapers[-1] > 0:
        cosine(x, widths[-1], GUIDE, tapers[-1])
        x += tapers[-1]
    else:
        pts.append((x, GUIDE))
    return pts, x


def _band(pts, yc):
    """Closed polygon of a varying-width guide centred on yc."""
    return ([(x, yc + 0.5 * w) for (x, w) in pts]
            + [(x, yc - 0.5 * w) for (x, w) in reversed(pts)])


def build(name, sections, tapers, dL_um, with_trim_ps=True):
    """Return (library, cell, apd_end_x). dL_um > 0 = extra path in the TOP arm."""
    assert len(tapers) == len(sections) + 1, "need one taper per boundary"
    lib = gdstk.Library(name=name.upper()[:30], unit=UNIT, precision=PRECISION)
    cell = lib.new_cell(name.upper()[:30])

    pt, xt = arm_points(sections, tapers, 1)
    pb, xb = arm_points(sections, tapers, 2)
    assert abs(xt - xb) < 1e-6, "arms must match before the offset"

    # the path offset: extra access-width guide in one arm. No padding of the
    # other arm - the asymmetry is the phase bias.
    if dL_um > 1e-9:
        pt.append((xt + dL_um, GUIDE)); xt += dL_um
    elif dL_um < -1e-9:
        pb.append((xb - dL_um, GUIDE)); xb -= dL_um
    apd_end = max(xt, xb)

    y_top, y_bot = +ARM_PITCH / 2.0, -ARM_PITCH / 2.0
    cell.add(gdstk.Polygon(_band(pt, y_top), layer=1, datatype=0))
    cell.add(gdstk.Polygon(_band(pb, y_bot), layer=1, datatype=0))

    total = apd_end
    if with_trim_ps:
        total = apd_end + TRIM_PS
        for yc in (y_top, y_bot):
            cell.add(gdstk.rectangle((apd_end, yc - GUIDE / 2),
                                     (total, yc + GUIDE / 2), layer=1, datatype=0))
            cell.add(gdstk.rectangle((apd_end, yc - HEATER_W / 2),
                                     (total, yc + HEATER_W / 2), layer=10, datatype=0))
        cell.add(gdstk.rectangle((apd_end, y_bot - 30), (apd_end, y_top + 30),
                                 layer=62, datatype=0))
        cell.add(gdstk.Label("APD ends / 2 mm trim PS begins", (apd_end, y_top + 54),
                             layer=63, texttype=0, magnification=7))

    for yc in (y_top, y_bot):
        cell.add(gdstk.rectangle((0, yc - KEEPOUT / 2), (total, yc + KEEPOUT / 2),
                                 layer=61, datatype=0))
    cell.add(gdstk.rectangle((0, y_bot - 30), (total, y_top + 30), layer=60, datatype=0))
    cell.add(gdstk.Label(name, (0, y_top + 40), layer=63, texttype=0, magnification=10))
    cell.add(gdstk.Label(
        f"{len(sections)} sections, {apd_end:.1f} um APD, tapers "
        f"{min(tapers):.0f}-{max(tapers):.0f} um raised cosine, path offset "
        f"{1000 * dL_um:+.1f} nm in the {'top' if dL_um >= 0 else 'bottom'} arm",
        (0, y_bot - 46), layer=63, texttype=0, magnification=7))
    return lib, cell, apd_end


# ---------------------------------------------------------------- verification
def _slice_width(poly, x):
    """y-extent of a polygon on the vertical line at x."""
    P = poly.points
    ys = []
    for i in range(len(P)):
        x0, y0 = P[i]
        x1, y1 = P[(i + 1) % len(P)]
        if (x0 - x) * (x1 - x) <= 0 and abs(x1 - x0) > 1e-12:
            ys.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return (max(ys) - min(ys)) if ys else None


def verify(path, sections, tapers, dL_um, tol_um=2e-3):
    """Re-open the written GDS and check widths and the path offset.

    Catches the two mistakes that have actually happened: a section drawn at the
    wrong width, and the path offset silently cancelled by padding the arms.
    """
    lib = gdstk.read_gds(path)
    cell = lib.cells[0]
    wg = [p for p in cell.polygons if p.layer == 1]
    # the APD arms are the polygons that start at x = 0; trim blocks start later
    apd = [p for p in wg if min(q[0] for q in p.points) < 1.0]
    tops = [p for p in apd if np.mean([q[1] for q in p.points]) > 0]
    bots = [p for p in apd if np.mean([q[1] for q in p.points]) < 0]

    problems = []
    x = 0.0
    for i, (L, wt, wb) in enumerate(sections):
        x += tapers[i]
        xc = x + L / 2.0
        mt = max((_slice_width(p, xc) or 0) for p in tops)
        mb = max((_slice_width(p, xc) or 0) for p in bots)
        if abs(mt - wt) > tol_um or abs(mb - wb) > tol_um:
            problems.append(f"section {i+1}: GDS {mt:.3f}/{mb:.3f} vs "
                            f"CSV {wt:.3f}/{wb:.3f} um")
        x += L

    xt = max(q[0] for p in tops for q in p.points)
    xb = max(q[0] for p in bots for q in p.points)
    meas_nm = (xt - xb) * 1000.0
    # 1 nm database grid, so allow a nanometre of quantisation
    if abs(meas_nm - 1000 * dL_um) > 1.0:
        problems.append(f"path offset: measured {meas_nm:+.1f} nm, "
                        f"expected {1000*dL_um:+.1f} nm")

    print(f"  widths checked: {len(sections)}   "
          f"path offset measured: {meas_nm:+.1f} nm "
          f"(target {1000*dL_um:+.1f} nm)")
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print("   -", p)
    else:
        print("  verification OK")
    return not problems


def write(name, sections, tapers, dL_um, out_dir=".", svg=True):
    """Build, write <name>.gds (+ .svg), then re-open and verify."""
    import os
    lib, cell, apd_end = build(name, sections, tapers, dL_um)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.gds")
    lib.write_gds(path, max_points=8190)       # 8190 = GDSII per-polygon limit
    if svg:
        cell.write_svg(os.path.join(out_dir, f"{name}.svg"), scaling=0.35)
    bb = cell.bounding_box()
    print(f"{name}")
    print(f"  APD {apd_end:.1f} um + {TRIM_PS:.0f} um trim PS = "
          f"{apd_end + TRIM_PS:.1f} um total")
    print(f"  bounding box x {bb[0][0]:.1f}..{bb[1][0]:.1f}, "
          f"y {bb[0][1]:.1f}..{bb[1][1]:.1f} um")
    ok = verify(path, sections, tapers, dL_um)
    print(f"  wrote {path}")
    return ok
