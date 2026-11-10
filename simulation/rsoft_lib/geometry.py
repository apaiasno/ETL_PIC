"""S-bend geometry (lengths, Euler bends) and GDSII output."""
import struct
import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


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


def bend_curvature(x, z, phi):
    """Path length s (um) and signed curvature (1/um) along a path from euler_sbend and similar."""
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    return s, np.gradient(phi, s)


def parallel_path(x, z, phi, s):
    """Curve running parallel to (x, z) at a constant perpendicular distance s (um).

    s > 0 is on the +x side where the path heads along +z.
    """
    return x + s * np.cos(phi), z - s * np.sin(phi), phi


def save_gds(filename, x, z, phi, width, layer=1, cell_name="EULER_SBEND"):
    """Write the waveguide outline (centerline +- width/2) to a GDSII file.

    Units: um, 1 nm grid. Pass lists for x, z, phi to write several paths
    into one cell: save_gds(fn, [x1, x2], [z1, z2], [p1, p2], width).
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

    paths = list(zip(x, z, phi)) if isinstance(x, (list, tuple)) else [(x, z, phi)]

    out = [rec(0x00, 0x02, struct.pack(">h", 600)), rec(0x01, 0x02, bytes(24)),
           rec(0x02, 0x06, b"LIB"), rec(0x03, 0x05, real8(1e-3) + real8(1e-9)),
           rec(0x05, 0x02, bytes(24)), rec(0x06, 0x06, cell_name.encode())]
    step = 4000                                     # keep each polygon under GDSII's 8191-point limit
    for px, pz, pphi in paths:
        nx_, nz_ = np.cos(pphi), -np.sin(pphi)      # normal to the path
        left = np.column_stack([px + width / 2 * nx_, pz + width / 2 * nz_])
        right = np.column_stack([px - width / 2 * nx_, pz - width / 2 * nz_])
        for i in range(0, len(px) - 1, step):
            j = min(i + step, len(px) - 1)
            poly = np.vstack([left[i:j + 1], right[i:j + 1][::-1], left[i:i + 1]])
            pts = np.round(poly * 1000).astype(">i4")   # um -> nm
            out += [rec(0x08, 0x00), rec(0x0D, 0x02, struct.pack(">h", layer)),
                    rec(0x0E, 0x02, struct.pack(">h", 0)), rec(0x10, 0x03, pts.tobytes()),
                    rec(0x11, 0x00)]
    out += [rec(0x07, 0x00), rec(0x04, 0x00)]
    with open(filename, "wb") as f:
        f.write(b"".join(out))


# --------------------------------------------------------------------------- #
# Side-step made of two equal circular arcs joined by a tilted straight
# --------------------------------------------------------------------------- #
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
        plt.figure(figsize=(3, 10))
        plt.plot(x, z)
        plt.xlabel("x (um)")
        plt.ylabel("z (um)")
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
        plt.figure(figsize=(3, 10))
        plt.plot(x, z)
        plt.xlabel("x (um)")
        plt.ylabel("z (um)")
        plt.title(f"{n_periods} periods, step {step:g} um, R_min {R_min:g} um: z-length {info['z_length']:.0f} um, "
                  f"straights {straight:.0f} um, deltaL {info['deltaL']:.2f} um")
        plt.show()
    return (x, z, phi, info) if return_params else (x, z, phi)