"""Geometry: paths (circular and Euler bends, waves, delay arms) and widths (raised-cosine tapers).

A path is given by three arrays x, z, phi (um, um, rad): the position across (x) and along (z) the propagation, and the tangent angle
from z. A width is a number, an array along the path, or a function of the path length s (um). Nothing here knows about RSoft or GDS:
rsoft_lib.circuit turns paths and widths into an RSoft circuit (exportable to GDS with RSoft's own export), and rsoft_lib.gds has the
independent GDS reader and writer.

Contents (in this order)
    Path utilities      bend_curvature, parallel_path, path_list, eval_width
    Circular S-bends    sbend_lengths, sidestep_offset, sidestep_straight, sidestep_angle
    Euler S-bends       euler_sbend, euler_ramp_length, euler_sbend_from_ramp, euler_sbend_fixed_ramp
    Structures          delay_arm_parameters, delay_arm, euler_wave
    Widths              raised_cosine, cosine_tapers
Names starting with an underscore are helpers of the function that follows them.
"""


import warnings

import numpy as np
from scipy.optimize import brentq


# --------------------------------------------------------------------------- #
# Path utilities
# --------------------------------------------------------------------------- #


def bend_curvature(x, z, phi):
    """Path length s (um) and signed curvature (1/um) along a path from euler_sbend and similar."""
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    return s, np.gradient(phi, s)


def parallel_path(x, z, phi, s):
    """Curve running parallel to (x, z) at a constant perpendicular distance s (um).

    s > 0 is on the +x side where the path heads along +z.
    """
    return x + s * np.cos(phi), z - s * np.sin(phi), phi


def path_list(x, z, phi, width):
    """Paths as a list with one (x, z, phi, width) entry per path, every entry an array over the points, from the arguments of
    save_gds and paths_to_circuit: x, z, phi of one path (1D arrays) or of several (a 2D array with one row per path, NaN-padded at
    the end if they differ in length, or a list of 1D arrays); phi=None computes the tangent angle; width: one value, one per path,
    one per point, a function of the path length s (um), or a list with one of these for each path."""
    def rows(a):
        if isinstance(a, (list, tuple)):
            if len(a) and np.ndim(a[0]) == 0:                      # a single path given as a plain list
                return [np.asarray(a, float)]
            return [np.asarray(r, float) for r in a]
        a = np.asarray(a, float)
        return [a] if a.ndim == 1 else list(a)                     # 1D: one path; 2D: one path per row

    xs, zs = rows(x), rows(z)
    ps = rows(phi) if phi is not None else [None] * len(xs)
    n = len(xs)
    if not (len(zs) == len(ps) == n):
        raise ValueError(f"x, z and phi must have the same number of paths (got {n}, {len(zs)}, {len(ps)})")
    if isinstance(width, (list, tuple)) and n == 1 and len(width) == len(xs[0]) and all(np.ndim(v) == 0 for v in width):
        widths = [np.asarray(width, float)]                        # one path with a width at every point, given as a list
    elif isinstance(width, (list, tuple)):                         # one entry per path, each a number or an array over the points
        widths = list(width)
    elif np.ndim(width) == 0:
        widths = [width] * n
    elif np.ndim(width) == 2:
        widths = list(width)
    elif n == 1 and len(width) == len(xs[0]):                      # one path with a width at every point
        widths = [np.asarray(width, float)]
    else:
        widths = list(np.asarray(width, float))                    # one number per path
    if len(widths) != n:
        raise ValueError(f"width has {len(widths)} entries for {n} paths")
    out = []
    for k, (px, pz, pphi, w) in enumerate(zip(xs, zs, ps, widths)):
        if pphi is None:                                            # the tangent angle from z, from the path itself
            pphi = np.full_like(px, np.nan)
        if not (px.shape == pz.shape == pphi.shape) or px.ndim != 1:
            raise ValueError(f"path {k}: x, z and phi must be 1D arrays of the same length")
        ok = np.isfinite(px) & np.isfinite(pz)
        m = int(ok.sum())
        if not ok[:m].all():
            raise ValueError(f"path {k}: NaN values are only allowed at the end of a row (padding)")
        px, pz, pphi = px[:m], pz[:m], pphi[:m]
        if m < 2:
            raise ValueError(f"path {k} has fewer than 2 points")
        if not np.isfinite(pphi).all():
            if np.isfinite(pphi).any():
                raise ValueError(f"path {k}: phi has missing values")
            pphi = np.arctan2(np.gradient(px), np.gradient(pz))
        if callable(w):                                             # a function of the path length s (um) from the start of the path
            w = eval_width(w, np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(pz)))])
        w = np.full(m, float(w)) if np.ndim(w) == 0 else np.asarray(w, float)[:m]
        if w.shape != px.shape or not (w > 0).all():
            raise ValueError(f"path {k}: the width must be positive, a single value or one value per point")
        out.append((px, pz, pphi, w))
    return out


def eval_width(f, s):
    """The widths at the path lengths s of a width function f(s); f may be vectorized or take a single number."""
    try:
        w = np.asarray(f(s), float)
        if w.shape == s.shape:
            return w
        if w.ndim == 0:
            return np.full(s.shape, float(w))
    except Exception:
        pass
    return np.array([float(f(v)) for v in s])


# --------------------------------------------------------------------------- #
# Circular S-bends
# --------------------------------------------------------------------------- #


def sbend_lengths(R, d, junction_offset=0.0, straight_offset=0.0):
    """Circular S-bend made of two tangent arcs of radius R.

    d is the lateral offset between the two straights. The arcs themselves
    cover d_arc = d + junction_offset - 2*straight_offset, where
    junction_offset is the step between the two arcs and straight_offset is
    the bend-to-straight shift at each end. With junction_offset =
    2*straight_offset, d_arc = d.

    Returns (arc_length, z_length, theta): path length 2*R*theta, longitudinal
    extent 2*R*sin(theta) = sqrt(4*R*d_arc - d_arc**2), and the angle turned
    by each arc (rad). Accepts scalars or arrays; requires d_arc <= 4R.
    """
    R = np.asarray(R, dtype=float)
    d_arc = np.asarray(d, dtype=float) + junction_offset - 2 * straight_offset
    if np.any(d_arc > 4 * R):
        raise ValueError("Arc offset must satisfy d_arc <= 4R for two tangent arcs.")
    theta = np.arccos(1 - d_arc / (2 * R))
    return 2 * R * theta, 2 * R * np.sin(theta), theta


def sidestep_offset(radius, angle_deg, straight, junction_offset=0.0):
    """Lateral displacement (um) of two arcs of `radius` turning by `angle_deg`, joined by a tilted straight of
    length `straight`: 2 R (1 - cos a) + L sin a + 2 junction_offset (the offset applied at each end)."""
    a = np.radians(angle_deg)
    return 2 * radius * (1 - np.cos(a)) + straight * np.sin(a) + 2 * junction_offset


def sidestep_straight(radius, offset, angle_deg, junction_offset=0.0):
    """Length of the tilted straight (um) that gives the lateral displacement `offset` for arcs turning by `angle_deg`."""
    a = np.radians(angle_deg)
    return (offset - 2 * junction_offset - 2 * radius * (1 - np.cos(a))) / np.sin(a)


def sidestep_angle(radius, offset, straight, junction_offset=0.0):
    """Arc angle (degrees) that gives the lateral displacement `offset` for a tilted straight of length `straight`
    (the smallest positive solution of 2 R (1 - cos a) + L sin a = offset - 2 junction_offset)."""
    A, d = 2 * radius, offset - 2 * junction_offset
    r = np.hypot(straight, A)
    x = (d - A) / r
    if abs(x) > 1:
        raise ValueError("No angle gives this offset with these radius and straight values.")
    return float(np.degrees(np.arctan2(A, straight) + np.arcsin(x)))


# --------------------------------------------------------------------------- #
# Euler S-bends
# --------------------------------------------------------------------------- #


def _euler_path(R_min, L_c, sa, n):
    """Centerline of an S-bend whose halves each ramp the curvature from 0 to 1/R_min over a
    clothoid of length L_c, hold it over a circular plateau of length sa, and ramp it back down
    over another clothoid of length L_c (the second half mirrors the first).
    Returns x, z, phi at n + 1 points (z = propagation direction, phi = tangent angle from z)."""
    Lh = 2 * L_c + sa                          # path length of one half
    theta = (L_c + sa) / R_min                 # turn angle of one half
    s = np.linspace(0, 2 * Lh, n + 1)
    u = np.where(s <= Lh, s, s - Lh)
    if L_c > 0:
        ph = np.where(u < L_c, u**2 / (2 * L_c * R_min),
             np.where(u <= L_c + sa, L_c / (2 * R_min) + (u - L_c) / R_min,
                      theta - (Lh - u)**2 / (2 * L_c * R_min)))
    else:
        ph = u / R_min
    phi = np.where(s <= Lh, ph, theta - ph)
    x = np.concatenate([[0], np.cumsum((np.sin(phi[1:]) + np.sin(phi[:-1])) / 2 * np.diff(s))])
    z = np.concatenate([[0], np.cumsum((np.cos(phi[1:]) + np.cos(phi[:-1])) / 2 * np.diff(s))])
    return x, z, phi


def _euler_theta(d, R_min, p):
    """Turn angle of one half of the partial-Euler S-bend (d, R_min, p)."""
    return brentq(lambda t: _euler_path(R_min, p * t * R_min, (1 - p) * t * R_min, 20000)[0][-1] - abs(d),
                  1e-9, np.pi / 2 - 1e-6)


def euler_sbend(d, R_min, p=1.0, ds=1.0):
    """Partial-Euler S-bend centerline with lateral offset d (um) and minimum radius R_min (um).

    p = Euler fraction (0 = circular arcs, 1 = full Euler). Returns x, z, phi
    (z = propagation direction, phi = tangent angle from z).
    """
    theta = _euler_theta(d, R_min, p)
    n = max(int(np.ceil(2 * (1 + p) * theta * R_min / ds)), 2)
    x, z, phi = _euler_path(R_min, p * theta * R_min, (1 - p) * theta * R_min, n)
    return np.sign(d) * x, z, np.sign(d) * phi


def euler_ramp_length(d_ref, R_min, p=1.0):
    """Length (um) of each curvature ramp (clothoid) of the reference Euler S-bend (d_ref, R_min, p)."""
    return p * _euler_theta(d_ref, R_min, p) * R_min


def euler_sbend_from_ramp(d, R_min, L_c, ds=1.0, allow_lower_peak=True, return_params=False):
    """S-bend with offset d (um), peak curvature 1/R_min and curvature ramps of length L_c (um).

    The curvature ramps up at the rate 1/(L_c*R_min). For offsets that need it, a plateau at
    1/R_min is added between the ramps and its length is solved to give the offset d.
    L_c = 0 gives circular arcs.

    If d is below the smallest offset reachable with these ramps (no plateau), the same ramp
    rate cannot reach it at full peak curvature. With allow_lower_peak=True (default) the bend
    keeps the ramp rate but turns around earlier, so its peak curvature is lower than 1/R_min
    (its minimum radius is larger), and a warning reports the radius. With False a ValueError
    is raised instead.

    Returns x, z, phi like euler_sbend. Raises ValueError if d needs more than a 90 degree
    turn per half. With return_params=True it returns x, z, phi, params, where params is a dict
    describing the bend as a partial-Euler bend: R_min (the radius actually reached), p (ramp
    fraction), L_c (ramp length) and plateau (plateau length per half), all in um where relevant.
    """
    d_min = _euler_path(R_min, L_c, 0.0, 20000)[0][-1]
    if abs(d) < d_min:
        if not allow_lower_peak:
            raise ValueError(f"Offset {abs(d):g} um is below the smallest reachable with ramps of {L_c:.1f} um "
                             f"at R_min = {R_min:g} um ({d_min:.3f} um).")
        # same ramp rate k = 1/(L_c*R_min), no plateau, ramps shortened to L < L_c (peak radius = L_c*R_min/L)
        offset = lambda L: _euler_path(L_c * R_min / L, L, 0.0, 20000)[0][-1]
        L = brentq(lambda L: offset(L) - abs(d), 1e-6 * L_c, L_c)
        R_peak = L_c * R_min / L
        warnings.warn(f"Offset {abs(d):g} um is below {d_min:.1f} um, the smallest reachable with the reference "
                      f"curvature profile at full peak curvature. Using the same curvature ramp rate with a lower "
                      f"peak curvature: minimum radius {R_peak:.0f} um instead of {R_min:g} um.")
        n = max(int(np.ceil(4 * L / ds)), 2)
        x, z, phi = _euler_path(R_peak, L, 0.0, n)
        out = (np.sign(d) * x, z, np.sign(d) * phi)
        return out + (dict(R_min=R_peak, p=1.0, L_c=L, plateau=0.0),) if return_params else out
    sa_max = np.pi / 2 * R_min - L_c
    if sa_max <= 0 or _euler_path(R_min, L_c, sa_max, 20000)[0][-1] < abs(d):
        raise ValueError(f"Offset {abs(d):g} um needs more than a 90 degree turn per half at R_min = {R_min:g} um.")
    sa = brentq(lambda a: _euler_path(R_min, L_c, a, 20000)[0][-1] - abs(d), 0.0, sa_max)
    n = max(int(np.ceil(2 * (2 * L_c + sa) / ds)), 2)
    x, z, phi = _euler_path(R_min, L_c, sa, n)
    out = (np.sign(d) * x, z, np.sign(d) * phi)
    p_eff = L_c / (L_c + sa) if L_c + sa > 0 else 0.0
    return out + (dict(R_min=R_min, p=p_eff, L_c=L_c, plateau=sa),) if return_params else out


def euler_sbend_fixed_ramp(d, d_ref, R_min, p=1.0, ds=1.0, allow_lower_peak=True, return_params=False):
    """S-bend with offset d (um) and the same curvature profile as a reference Euler S-bend.

    The reference is the partial-Euler bend you would get from euler_sbend(d_ref, R_min, p).
    The new bend has the same peak curvature (1/R_min) and the same curvature ramps (same
    ramp length, hence same ramp rate); only the constant-curvature plateau between the ramps
    is lengthened or shortened to reach the new offset d. For d = d_ref it reproduces the
    reference exactly.

    Offsets below the smallest one reachable without a plateau (d_ref itself for p = 1)
    cannot keep both the ramp rate and the peak curvature. By default the bend then keeps the
    ramp rate and reaches a lower peak curvature (a warning gives its minimum radius); see
    euler_sbend_from_ramp. Pass allow_lower_peak=False to raise an error instead.

    Returns x, z, phi like euler_sbend. With return_params=True it also returns a dict giving the
    new bend as a partial-Euler bend (R_min, p, L_c, plateau); see euler_sbend_from_ramp.
    """
    return euler_sbend_from_ramp(d, R_min, euler_ramp_length(d_ref, R_min, p), ds, allow_lower_peak, return_params)


# --------------------------------------------------------------------------- #
# Structures made of S-bends
# --------------------------------------------------------------------------- #


def delay_arm_parameters(z_length, deltaL, R, end_straight=None):
    """The numbers that define a delay arm (see delay_arm): the S-bend angle theta (rad; deltaL = 4 R (theta - sin theta)), the
    lengths L_end of the first and last straight and L_mid of the middle one, and the bump height. Raises an error if the arm does not
    fit, and warns if the middle straight is short."""
    if deltaL < 0:
        raise ValueError("deltaL must be >= 0")
    if deltaL / (4 * R) > np.pi / 2 - 1:
        raise ValueError(f"deltaL is too large for R = {R:g}: the arcs would turn by more than 90 degrees "
                         f"(the largest deltaL is {4 * R * (np.pi / 2 - 1):.1f})")
    theta = brentq(lambda t: t - np.sin(t) - deltaL / (4 * R), 0.0, np.pi / 2) if deltaL > 0 else 0.0
    remaining = float(z_length - 4 * R * np.sin(theta))
    if remaining < 0:
        raise ValueError(f"z_length is too short: the two S-bends need {4 * R * np.sin(theta):.1f} um")
    L_end = remaining / 4 if end_straight is None else end_straight
    L_mid = remaining - 2 * L_end
    if L_mid < 0:
        raise ValueError(f"end_straight is too long: the straights add up to {remaining:.1f} um")
    if L_mid < 0.15 * z_length:
        warnings.warn(f"The middle straight is only {L_mid:.1f} um long: the two S-bends already take {4 * R * np.sin(theta):.1f} um "
                      f"of the {z_length:g} um z-length (theta = {np.degrees(theta):.1f} deg, offset {2 * R * (1 - np.cos(theta)):.1f} um). "
                      f"A longer z_length or a smaller R leaves more room.")
    return dict(theta=float(theta), L_end=float(L_end), L_mid=float(L_mid), bump_height=float(2 * R * (1 - np.cos(theta))))


def delay_arm(z_length, deltaL, R, end_straight=None, ds=1.0, x0=0.0, direction=1, plot=False, return_params=False):
    """Waveguide with a bump that makes it longer than a straight one over the same z-length by deltaL.

    The path is: parallel straight, S-bend out (two arcs of radius R), parallel straight, the same S-bend back,
    parallel straight. Each S-bend turns by an angle theta in each arc, and the extra path length is
    deltaL = 4 R (theta - sin theta), which gives theta. The bump height is 2 R (1 - cos theta).

    z_length      : z-length of the arm (um), from z = 0 to z = z_length
    deltaL        : path length of the arm minus z_length (um)
    R             : radius of the circular S-bends (um)
    end_straight  : length (um) of the first and of the last straight; the default is a quarter of the z-length that the
                    S-bends leave, so the middle straight gets half of it
    x0            : lateral position (um) of the straight parts
    direction     : +1 bumps toward +x, -1 toward -x
    Returns x, z, phi (z = propagation direction, phi = tangent angle from z, as in euler_sbend; they go to save_gds),
    and with return_params=True also a dict with theta_deg, bump_height, L_end, L_mid and path_length.
    """
    p = delay_arm_parameters(z_length, deltaL, R, end_straight)
    theta, L_end, L_mid = p["theta"], p["L_end"], p["L_mid"]
    sign = 1.0 if direction >= 0 else -1.0

    # build the path piece by piece: (curvature, length), with curvature 0 for the straights
    arc = R * theta
    pieces = [(0.0, L_end), (sign / R, arc), (-sign / R, arc), (0.0, L_mid), (-sign / R, arc), (sign / R, arc), (0.0, L_end)]
    x, z, phi = [x0], [0.0], [0.0]
    for kappa, length in pieces:
        if length <= 0:
            continue
        s = np.linspace(0, length, max(int(np.ceil(length / ds)), 1) + 1)[1:]
        xs, zs, p0 = x[-1], z[-1], phi[-1]
        if kappa == 0:
            x += list(xs + s * np.sin(p0))
            z += list(zs + s * np.cos(p0))
            phi += [p0] * len(s)
        else:
            p = p0 + kappa * s
            x += list(xs + (np.cos(p0) - np.cos(p)) / kappa)
            z += list(zs + (np.sin(p) - np.sin(p0)) / kappa)
            phi += list(p)
    x, z, phi = np.array(x), np.array(z), np.array(phi)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 3))
        plt.plot(z, x)
        plt.xlabel("z (um)")
        plt.ylabel("x (um)")
        plt.title(f"theta = {np.degrees(theta):.1f} deg, offset = {2 * R * (1 - np.cos(theta)):.1f} um, "
                  f"straights {L_end:.0f} / {L_mid:.0f} / {L_end:.0f} um")
        plt.show()
    if return_params:
        info = dict(theta_deg=float(np.degrees(theta)), bump_height=float(2 * R * (1 - np.cos(theta))),
                    L_end=float(L_end), L_mid=float(L_mid), path_length=float(np.sum(np.hypot(np.diff(x), np.diff(z)))))
        return x, z, phi, info
    return x, z, phi


def euler_wave(n_periods, step, R_min, p=0.5, z_length=None, straight=0.0, ds=1.0, x0=0.0, direction=1, plot=False,
               return_params=False):
    """Serpentine path of alternating Euler S-bends between an initial and a final straight.

    Each period is an S-bend out by `step` and one back, so the wave steps away on one side of the straights (between x0 and
    x0 + direction*step) and is not centered on them. S-bends start and end straight with zero curvature, so everything joins
    with continuous curvature.

    Path: straight | n_periods x (S-bend out, S-bend back) | straight

    n_periods : number of periods (each is two S-bends)
    step      : lateral offset (um) of each S-bend = the amplitude of the wave
    R_min     : minimum radius of curvature (um)
    p         : Euler fraction of each S-bend (0 = circular arcs, 1 = full Euler), as in euler_sbend
    z_length  : total z-length (um); the initial and final straights are made equal and fill what the S-bends leave.
                If None, each straight is `straight` (um).
    direction : +1 steps away toward +x, -1 toward -x
    Returns x, z, phi (as euler_sbend; they go to save_gds), and with return_params=True also a dict with z_length,
    path_length, deltaL (path_length - z_length), straight (each), sbend_z (z-length of one S-bend) and min_radius.
    """
    sign = 1.0 if direction >= 0 else -1.0
    away, back = euler_sbend(sign * step, R_min, p, ds), euler_sbend(-sign * step, R_min, p, ds)
    sbend_z = float(away[1][-1])
    if z_length is not None:
        straight = (z_length - 2 * n_periods * sbend_z) / 2
        if straight < -1e-9:
            raise ValueError(f"z_length is too short: the {2 * n_periods} S-bends need {2 * n_periods * sbend_z:.1f} um")
        straight = max(straight, 0.0)
    x, z, phi = [x0], [0.0], [0.0]

    def add_straight(length):
        if length > 0:
            n = max(int(np.ceil(length / ds)), 1)
            s = np.linspace(0, length, n + 1)[1:]
            x.extend([x[-1]] * n)
            z.extend(list(z[-1] + s))
            phi.extend([0.0] * n)

    def add_sbend(sb):
        sx, sz, sp = sb
        x.extend(list(x[-1] + sx[1:]))
        z.extend(list(z[-1] + sz[1:]))
        phi.extend(list(sp[1:]))

    add_straight(straight)
    for _ in range(n_periods):
        add_sbend(away)
        add_sbend(back)
    add_straight(straight)
    x, z, phi = np.array(x), np.array(z), np.array(phi)
    s, curvature = bend_curvature(x, z, phi)
    info = dict(z_length=float(z[-1]), path_length=float(s[-1]), deltaL=float(s[-1] - z[-1]), straight=float(straight),
                sbend_z=sbend_z, min_radius=float(1 / np.max(np.abs(curvature))))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 3))
        plt.plot(z, x)
        plt.xlabel("z (um)")
        plt.ylabel("x (um)")
        plt.title(f"{n_periods} periods, step {step:g} um, R_min {R_min:g} um: z-length {info['z_length']:.0f} um, "
                  f"straights {straight:.0f} um, deltaL {info['deltaL']:.2f} um")
        plt.show()
    return (x, z, phi, info) if return_params else (x, z, phi)


# --------------------------------------------------------------------------- #
# Widths
# --------------------------------------------------------------------------- #


def raised_cosine(s, s0, s1, w0, w1):
    """Width at the path length(s) s of a raised-cosine taper that goes from w0 at s0 to w1 at s1 (w0 before s0, w1 after s1)."""
    u = np.clip((np.asarray(s, float) - s0) / (s1 - s0), 0.0, 1.0)
    return w0 + (w1 - w0) * (0.5 - 0.5 * np.cos(np.pi * u))


def cosine_tapers(*tapers, width=3.4):
    """A width function of the path length s (um) made of raised-cosine tapers: each taper is (s0, s1, w1), from the width reached
    so far (`width` at the start) to w1 between s0 and s1, constant in between. For example
    cosine_tapers((1000, 1500, 5.0), (3000, 3500, 3.4)) widens from 3.4 to 5 um over s = 1000 to 1500 and narrows back
    over 3000 to 3500. Use it as the width of save_gds or paths_to_circuit; a taper that lies on one straight piece of the path
    becomes one segment with a user taper in the RSoft file."""
    tapers = sorted(tapers)
    for a, b in zip(tapers[:-1], tapers[1:]):
        if a[1] > b[0]:
            raise ValueError(f"The tapers {a[:2]} and {b[:2]} overlap")
    if any(t[1] <= t[0] for t in tapers):
        raise ValueError("Every taper needs s1 > s0")

    def f(s):
        s = np.asarray(s, float)
        w, previous = np.full(s.shape, float(width)), float(width)
        for s0, s1, w1 in tapers:
            w = np.where(s >= s0, raised_cosine(s, s0, s1, previous, w1), w)
            previous = float(w1)
        return w
    return f