"""
Tools for the APD optimizer: a conservative taper rule, and a section-length solver
with two metrics (your current one, and a "lattice-blind" one).

Units: widths and lengths in um, wavelengths in um, phases in radians.
Requires numpy and scipy only.

1. Taper rule
   boundary_tapers(top, bottom) gives a raised-cosine taper length for every boundary,
   including the steps to and from the 3.4 um access guide (rule in taper_length). Both arms taper at the same
   place, so each boundary takes the longer of its two arms' needs. The lengths are
   provisional; Anusha's simulations will refine taper_length().

2. Section lengths for fixed widths and tapers
   solve_lengths(A, T, lam, L_total, mode=...) is a linear program, so it can sit inside
   your width search. L_total caps the total section length (tapers excluded).
     A[b, j, i]  phase difference per um of section i at bias b and wavelength j:
                 2*pi/lam_j * (n_eff(top_i + bias_b) - n_eff(bottom_i + bias_b))   [rad/um]
     T[b, j]     phase difference from all tapers at bias b, wavelength j            [rad]
                 (your raised-cosine taper integral, with the lengths from boundary_tapers)
   mode="standard"       minimizes the worst error over all biases and wavelengths after
                         re-fitting the trim (c * lam0/lam) at each bias. This is your metric.
   mode="lattice_blind"  at each bias, also removes the shapes a 3-section lattice corrects
                         (piston, slope, curvature) and minimizes the worst of what is left
                         (cubic and higher). At EVERY bias, the error left for the lattice after a
                         trim of at most trim_max is kept within allowance_deg (default 0.3 deg).
                         (Before 4 Nov 2026 this was enforced at nominal width only, which let the
                         lattice be handed several degrees of piston at the bias extremes.)

3. Reporting
   worst_after_trim(err, lam) and blind_residual(err, lam) give the two metrics for any
   error curve err[b, j] = (total phase difference - pi), so both designs can be reported
   the same way. trim_report(err, lam) lists, per bias, the trim the heater must supply and
   the error left for the lattice when the trim is capped at 2 deg.
"""
import numpy as np
from scipy.optimize import linprog

GUIDE_UM = 3.4      # access waveguide width
CUTOFF_UM = 5.6     # second even lateral mode is guided above about 5.7 um; 5.6 for margin


def taper_length(w0, w1):
    """Conservative raised-cosine taper length (um) for one arm stepping from w0 to w1.

    Steps that cross 5.6 um (one side narrower, the other wider) or change width by more than
    1.2 um: 160 um. Steps of 0.5-1.2 um: 80 um. Steps of 0.5 um or less: 60 um. Each value is
    at least 1.3x the shortest low-loss length found for the hardest step of its kind within
    2.5-6.7 um (2D beam-propagation estimate, < 0.005 dB at 1450 nm).
    """
    d = abs(w1 - w0)
    if d < 1e-9:
        return 0.0
    crosses = min(w0, w1) < CUTOFF_UM < max(w0, w1)
    if crosses or d > 1.2 + 1e-9:
        return 160.0
    if d > 0.5 + 1e-9:
        return 80.0
    return 60.0


def boundary_tapers(top, bottom, guide=GUIDE_UM):
    """Taper length (um) at each of the len(top)+1 boundaries, access guides included."""
    t = [guide] + list(top) + [guide]
    b = [guide] + list(bottom) + [guide]
    return [max(taper_length(t[i], t[i + 1]), taper_length(b[i], b[i + 1])) for i in range(len(t) - 1)]


def _basis(lam, mode, lam0):
    x = (lam - lam.mean()) / (0.5 * (lam.max() - lam.min()))
    trim = lam0 / lam
    if mode == "standard":
        return trim[:, None]
    if mode == "lattice_blind":
        return np.column_stack([np.ones_like(x), x, x ** 2, trim])
    raise ValueError("mode must be 'standard' or 'lattice_blind'")


def solve_lengths(A, T, lam, L_total, mode="standard", allowance_deg=0.3, L_min=60.0,
                  nominal=None, target=np.pi, trim_max=0.035, lam0=1.55, exact_total=False,
                  offset_col=None, offset_max=1.0, allowance_bias_deg=None):
    """Section lengths (um) for fixed widths and tapers.

    Returns (lengths, worst_rad, status). worst_rad is the metric of the chosen mode.
    target is the design phase difference (pi or -pi, matching your branch).
    trim_max bounds the trim at each bias, in radians at lam0 (default 0.035 rad, about 2 deg), so the APD
    is within a few degrees of 180 deg as drawn. A large trim loads the heaters and makes the result depend
    on the heater phase following 1/lam exactly.
    offset_col[b, j] = 2*pi/lam_j * n_eff(access guide width + bias_b, lam_j) [rad/um] adds a signed path-length
    offset dL (|dL| <= offset_max um) in one arm, built into the layout instead of supplied by heaters.
    Positive dL means extra path in the top arm (the arm whose widths are listed first).
    Returns (lengths, worst_rad, status); with offset_col the status is ('ok', dL_um).
    L_total is the most section length allowed (tapers excluded); set exact_total=True to force it.
    Longer tapers carry part of the phase, so leaving the total free is usually better.
    allowance_bias_deg (lattice_blind only) sets the allowance at the non-nominal biases; None uses allowance_deg.
    """
    A = np.asarray(A, float); T = np.asarray(T, float); lam = np.asarray(lam, float)
    nb, nl, ns = A.shape
    nominal = nb // 2 if nominal is None else nominal
    B = _basis(lam, mode, lam0); nk = B.shape[1]
    extra = nb if mode == "lattice_blind" else 0         # one allowance trim per bias, capped like the others
    off = 0 if offset_col is None else 1
    nv = ns + nb * nk + extra + off + 1                   # lengths, basis weights, allowance trim, offset, t
    rows, rhs = [], []
    for b in range(nb):
        for j in range(nl):
            r = np.zeros(nv); r[:ns] = A[b, j]; r[ns + b * nk: ns + (b + 1) * nk] = -B[j]; r[-1] = -1
            if off: r[-2] = offset_col[b][j]
            rows.append(r); rhs.append(target - T[b, j])
            r2 = -r; r2[-1] = -1
            rows.append(r2); rhs.append(-(target - T[b, j]))
    if mode == "lattice_blind":
        for b in range(nb):
            a_deg = allowance_deg if (b == nominal or allowance_bias_deg is None) else allowance_bias_deg
            cap = np.radians(a_deg)
            for j in range(nl):
                r = np.zeros(nv); r[:ns] = A[b, j]; r[ns + nb * nk + b] = -lam0 / lam[j]
                if off: r[-2] = offset_col[b][j]
                rows.append(r); rhs.append(target - T[b, j] + cap)
                rows.append(-r); rhs.append(-(target - T[b, j] - cap))
    tot = np.r_[np.ones(ns), np.zeros(nv - ns)]
    if exact_total:
        A_eq, b_eq = [tot], [L_total]
    else:
        rows.append(tot); rhs.append(L_total); A_eq, b_eq = None, None
    bounds = [(L_min, None)] * ns
    for b in range(nb):
        for k in range(nk):
            bounds.append((-trim_max, trim_max) if k == nk - 1 else (None, None))
    bounds += [(-trim_max, trim_max)] * extra + [(-offset_max, offset_max)] * off + [(0, None)]
    res = linprog(np.r_[np.zeros(nv - 1), 1.0], A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=None if A_eq is None else np.array(A_eq), b_eq=None if b_eq is None else np.array(b_eq),
                  bounds=bounds, method="highs")
    if res.status != 0:
        return None, None, res.message
    return res.x[:ns], res.x[-1], ("ok", res.x[-2]) if off else "ok"


def _minimax_fit(e, B):
    """Smallest worst-case residual of e after subtracting a combination of the columns of B."""
    nl, nk = B.shape; nv = nk + 1
    rows = np.vstack([np.c_[-B, -np.ones(nl)], np.c_[B, -np.ones(nl)]])
    rhs = np.r_[-e, e]
    res = linprog(np.r_[np.zeros(nk), 1.0], A_ub=rows, b_ub=rhs, bounds=[(None, None)] * nk + [(0, None)], method="highs")
    return res.x[-1]


def trim_report(err, lam, lam0=1.55, trim_max_deg=2.0):
    """Per bias: the trim the heater must supply (deg, uncapped best fit) and the worst error left (deg) when the
    trim is capped at trim_max_deg. For a lattice-blind design, that leftover is what the lattice is handed."""
    lam = np.asarray(lam, float); B = (lam0 / lam)[:, None]; out = []
    for e in np.atleast_2d(err):
        rows = np.vstack([np.c_[-B, -np.ones(len(lam))], np.c_[B, -np.ones(len(lam))]]); rhs = np.r_[-e, e]
        res = [linprog([0, 1], A_ub=rows, b_ub=rhs, bounds=[bd, (0, None)], method="highs").x
               for bd in ((None, None), (-np.radians(trim_max_deg), np.radians(trim_max_deg)))]
        out.append({"trim_deg": float(np.degrees(res[0][0])), "left_with_capped_trim_deg": float(np.degrees(res[1][1]))})
    return out


def worst_after_trim(err, lam, lam0=1.55):
    """Your current metric: worst |error| over biases and wavelengths after a trim refit per bias (rad)."""
    B = _basis(np.asarray(lam, float), "standard", lam0)
    return max(_minimax_fit(e, B) for e in np.atleast_2d(err))


def blind_residual(err, lam, lam0=1.55):
    """Worst error a 3-section lattice cannot remove: after piston, slope, curvature and trim per bias (rad)."""
    B = _basis(np.asarray(lam, float), "lattice_blind", lam0)
    return max(_minimax_fit(e, B) for e in np.atleast_2d(err))


if __name__ == "__main__":
    print("Taper rule examples (um):", {f"{a}->{b}": taper_length(a, b) for a, b in
                                         [(3.0, 6.7), (2.5, 4.8), (3.4, 2.5), (4.7, 4.8), (6.4, 6.6)]})
    lam = np.linspace(1.45, 1.65, 21); x = lam - 1.55
    dn = [4e-4 * (1 + 0.8 * x), -3e-4 * (1 - 1.5 * x), 2e-4 * (1 + 3.0 * x), 1e-4 * (1 - 2.0 * x)]   # toy sections
    A = np.array([[2 * np.pi / lam * (d * (1 + 0.05 * bb)) for d in dn] for bb in (-1, 0, 1)]).transpose(0, 2, 1)
    T = np.zeros(A.shape[:2])
    OFF = np.array([2 * np.pi / lam * 1.458 for _ in range(3)])                                       # toy access guide
    for mode, kw in (("standard", {}), ("lattice_blind", {}), ("lattice_blind", {"allowance_bias_deg": 3.0})):
        L, w, st = solve_lengths(A, T, lam, 3000.0, mode=mode, L_min=20.0, offset_col=OFF, **kw)
        tag = mode + (f" {kw}" if kw else "")
        if L is None:
            print(f"toy example, {tag}: no solution ({st})"); continue
        print(f"toy example, {tag}: {st[0]}; lengths {np.round(L, 1)}; path offset {1000 * st[1]:+.0f} nm; metric {np.degrees(w):.4f} deg")
