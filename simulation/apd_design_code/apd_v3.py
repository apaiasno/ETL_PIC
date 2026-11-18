"""APD re-solve on Anusha's corrected n_eff table, with path offset and 2 deg trim.

Changes from the Oct-22 round:
  * NEW TABLE     StraightWaveguide_neff.csv - 0.05 um width grid, 2.3-6.8 um.
                  n_eff is lower below 3.4 um (-9.3e-5 at 2.4 um, tapering to
                  zero at 3.4 um); unchanged at and above 3.4 um.
  * TRIM CAP      trim_max now 0.035 rad (~2 deg) in apd_lattice_tools, so the
                  design must be within a couple of degrees of 180 as drawn.
  * PATH OFFSET   offset_col on: the LP may add a signed extra path dL in one arm
                  (|dL| <= offset_max), built into the layout rather than taken
                  from the heaters.  offset_col[b,j] = 2*pi/lam_j *
                  n_eff(3.4 um + bias_b, lam_j).  Positive dL = extra path in the
                  TOP arm (widths listed first).

Unchanged: cubic not-a-knot in width, exact raised-cosine taper integral,
biases -50/0/+50 nm applied to every drawn width including the access guide,
taper lengths from boundary_tapers(), L_total caps SECTION length only.
"""
import csv
import numpy as np
from scipy.interpolate import CubicSpline

import apd_lattice_tools as alt

# ---------------------------------------------------------------- new table
_rows = list(csv.DictReader(open("StraightWaveguide_neff.csv")))
_wl = sorted({round(float(r["wavelength"]), 6) for r in _rows})
_wd = sorted({round(float(r["width"]), 6) for r in _rows})
LAM = np.array(_wl)
WGRID = np.array(_wd)
NEFF = np.zeros((len(LAM), len(WGRID)))
MODE = {}
_iw = {w: i for i, w in enumerate(_wd)}
_il = {l: i for i, l in enumerate(_wl)}
for r in _rows:
    j, k = _il[round(float(r["wavelength"]), 6)], _iw[round(float(r["width"]), 6)]
    NEFF[j, k] = float(r["neff"])
    MODE[(j, k)] = r["mode"]

_SPL = [CubicSpline(WGRID, NEFF[j]) for j in range(len(LAM))]   # not-a-knot
K = 2 * np.pi / LAM
GUIDE = alt.GUIDE_UM                 # 3.4 um access guide
BIASES = np.array([-0.05, 0.0, 0.05])
NOMINAL = 1

# usable drawn widths: +/-50 nm of bias must stay inside the measured table.
# The new table starts at 2.3 um, so 2.35 um is now the lowest safe drawn width.
W_LO, W_HI = 2.35, 6.75
IDX = [k for k, w in enumerate(WGRID) if W_LO - 1e-9 <= w <= W_HI + 1e-9]


def neff(w):
    return np.array([s(w) for s in _SPL])


_NC = {}
for _b in BIASES:
    for _k in IDX:
        _NC[(_k, round(float(_b), 6))] = neff(WGRID[_k] + _b)
_NG = {round(float(b), 6): neff(GUIDE + b) for b in BIASES}


def neff_idx(k, b):
    return _NC[(k, round(float(b), 6))]


# ---------------------------------------------------------------- taper integral
_GLX, _GLW = np.polynomial.legendre.leggauss(16)
_S = 0.5 * (_GLX + 1.0)
_SW = 0.5 * _GLW
_TC = {}


def taper_mean_neff(w0, w1):
    """Length-averaged n_eff through a raised-cosine taper, split at grid nodes."""
    key = (round(w0, 6), round(w1, 6))
    v = _TC.get(key)
    if v is not None:
        return v
    if abs(w1 - w0) < 1e-12:
        v = neff(w0)
    else:
        lo, hi = min(w0, w1), max(w0, w1)
        nodes = [w for w in WGRID if lo + 1e-9 < w < hi - 1e-9]
        if w1 < w0:
            nodes = nodes[::-1]
        gs = [0.0] + [np.arccos(np.clip(1 - 2 * (w - w0) / (w1 - w0), -1, 1)) / np.pi
                      for w in nodes] + [1.0]
        acc = np.zeros(len(LAM))
        for a, c in zip(gs[:-1], gs[1:]):
            if c - a < 1e-14:
                continue
            for sw, gq in zip(_SW, _S):
                s = a + (c - a) * gq
                acc += (c - a) * sw * neff(w0 + (w1 - w0) * (0.5 - 0.5 * np.cos(np.pi * s)))
        v = acc
    _TC[key] = v
    return v


def taper_phase(top, bot, tapers, bias):
    t = [GUIDE + bias] + [w + bias for w in top] + [GUIDE + bias]
    b = [GUIDE + bias] + [w + bias for w in bot] + [GUIDE + bias]
    opd = np.zeros(len(LAM))
    for i, Lt in enumerate(tapers):
        if Lt > 0:
            opd += Lt * (taper_mean_neff(t[i], t[i + 1]) - taper_mean_neff(b[i], b[i + 1]))
    return K * opd


def build(it, ib, tapers):
    """A, T and the path-offset column, in the form solve_lengths() expects."""
    nb, nl, ns = len(BIASES), len(LAM), len(it)
    A = np.zeros((nb, nl, ns))
    T = np.zeros((nb, nl))
    OFF = np.zeros((nb, nl))
    top = [WGRID[IDX[k]] for k in it]
    bot = [WGRID[IDX[k]] for k in ib]
    for bi, bv in enumerate(BIASES):
        for i in range(ns):
            A[bi, :, i] = K * (neff_idx(IDX[it[i]], bv) - neff_idx(IDX[ib[i]], bv))
        T[bi] = taper_phase(top, bot, tapers, bv)
        OFF[bi] = K * _NG[round(float(bv), 6)]      # rad per um of extra top-arm path
    return A, T, OFF


def err_exact(top, bot, L, tapers, dL, target):
    """err[b, j] at arbitrary widths, including the path offset."""
    out = []
    for bv in BIASES:
        dn = np.zeros(len(LAM))
        for i in range(len(L)):
            dn = dn + L[i] * (neff(top[i] + bv) - neff(bot[i] + bv))
        ph = K * dn + taper_phase(top, bot, tapers, bv) + dL * K * neff(GUIDE + bv)
        out.append(ph - target)
    return np.array(out)


def evaluate(it, ib, L_device, mode, target=np.pi, L_min=60.0, allowance_deg=0.3,
             offset_max=1.0):
    top = [WGRID[IDX[k]] for k in it]
    bot = [WGRID[IDX[k]] for k in ib]
    tapers = alt.boundary_tapers(top, bot)
    tap_total = float(sum(tapers))
    L_sections = L_device - tap_total
    if L_sections < len(it) * L_min:
        return None
    A, T, OFF = build(it, ib, tapers)
    L, worst, status = alt.solve_lengths(A, T, LAM, L_sections, mode=mode,
                                         allowance_deg=allowance_deg, L_min=L_min,
                                         nominal=NOMINAL, target=target,
                                         offset_col=OFF, offset_max=offset_max)
    if L is None:
        return None
    dL = float(status[1]) if isinstance(status, tuple) else 0.0
    err = err_exact(top, bot, L, tapers, dL, target)
    return dict(it=list(it), ib=list(ib), L=np.asarray(L), tapers=tapers, dL=dL,
                tap_total=tap_total, sec_total=float(np.sum(L)),
                device=float(np.sum(L)) + tap_total, metric=float(worst),
                worst_trim=float(alt.worst_after_trim(err, LAM)),
                blind=float(alt.blind_residual(err, LAM)), err=err,
                mode=mode, target=target)


def branch_target(it, ib, L, tapers, dL=0.0):
    top = [WGRID[IDX[k]] for k in it]
    bot = [WGRID[IDX[k]] for k in ib]
    e = err_exact(top, bot, L, tapers, dL, 0.0)
    mid = e[NOMINAL][len(LAM) // 2]
    return float(np.pi * (2 * np.round((mid - np.pi) / (2 * np.pi)) + 1))


def search(it0, ib0, L_device, mode, rounds=6, L_min=60.0, target=None, verbose=True):
    it, ib = list(it0), list(ib0)
    if target is None:
        b0 = evaluate(it, ib, L_device, "standard", L_min=L_min)
        target = branch_target(it, ib, b0["L"], b0["tapers"], b0["dL"]) if b0 else np.pi
    best = evaluate(it, ib, L_device, mode, target=target, L_min=L_min)
    if best is None:
        return None
    for rnd in range(rounds):
        improved = False
        for vec in (it, ib):
            for i in range(len(vec)):
                cur = vec[i]
                for cand in range(len(IDX)):
                    if cand == cur:
                        continue
                    vec[i] = cand
                    r = evaluate(it, ib, L_device, mode, target=target, L_min=L_min)
                    if r is not None and r["metric"] < best["metric"] - 1e-9:
                        best, cur, improved = r, cand, True
                    else:
                        vec[i] = cur
                vec[i] = cur
        if verbose:
            print(f"    round {rnd+1}: {mode} {np.degrees(best['metric']):.4f} deg, "
                  f"device {best['device']:.0f} um, dL {1000*best['dL']:+.0f} nm", flush=True)
        if not improved:
            break
    return best


def load_design(path):
    rows = list(csv.DictReader(open(path)))
    gi = lambda w: min(range(len(IDX)), key=lambda k: abs(WGRID[IDX[k]] - w))
    return ([float(r["length_um"]) for r in rows],
            [gi(float(r["top_width_um"])) for r in rows],
            [gi(float(r["bottom_width_um"])) for r in rows])


def report(r, label, csv_out=None):
    top = [WGRID[IDX[k]] for k in r["it"]]
    bot = [WGRID[IDX[k]] for k in r["ib"]]
    L = np.round(r["L"], 3)
    print(f"\n=== {label} ===")
    print(f"  mode {r['mode']}, branch {r['target']/np.pi:+.0f}pi")
    print(f"  sections {r['sec_total']:.1f} + tapers {r['tap_total']:.0f} "
          f"= APD {r['device']:.1f} um")
    print(f"  path offset dL = {1000*r['dL']:+.1f} nm "
          f"({'extra path in TOP arm' if r['dL'] >= 0 else 'extra path in BOTTOM arm'})")
    print(f"  worst_after_trim {np.degrees(r['worst_trim']):.4f} deg   "
          f"blind_residual {np.degrees(r['blind']):.4f} deg")
    print(f"  {'sec':>4} {'length_um':>11} {'w_top':>7} {'w_bot':>7}")
    for i in range(len(L)):
        print(f"  {i+1:>4} {L[i]:>11.3f} {top[i]:>7.2f} {bot[i]:>7.2f}")
    print("  tapers: " + ", ".join(f"{t:.0f}" for t in r["tapers"]))
    print(f"  {'bias':>8} {'worst_after_trim':>18} {'blind_residual':>16}")
    for bi, bv in enumerate(BIASES):
        e = r["err"][bi:bi + 1]
        print(f"  {bv*1000:>+7.0f}  {np.degrees(alt.worst_after_trim(e, LAM)):>17.4f} "
              f"{np.degrees(alt.blind_residual(e, LAM)):>16.4f}")
    if csv_out:
        with open(csv_out, "w", newline="\n") as f:
            f.write("section,length_um,top_width_um,bottom_width_um\n")
            for i in range(len(L)):
                f.write(f"{i+1},{L[i]:.3f},{top[i]:.2f},{bot[i]:.2f}\n")
        t = [GUIDE] + top + [GUIDE]
        b = [GUIDE] + bot + [GUIDE]
        with open(csv_out.replace(".csv", "_tapers.csv"), "w", newline="\n") as f:
            f.write("boundary,taper_length_um,profile,top_from_um,top_to_um,"
                    "bot_from_um,bot_to_um\n")
            for i, Lt in enumerate(r["tapers"]):
                f.write(f"{i},{Lt:.1f},raised_cosine,{t[i]:.2f},{t[i+1]:.2f},"
                        f"{b[i]:.2f},{b[i+1]:.2f}\n")
        with open(csv_out.replace(".csv", "_phase.csv"), "w", newline="\n") as f:
            f.write("wavelength_um,err_bias_-50nm_deg,err_bias_0nm_deg,"
                    "err_bias_+50nm_deg\n")
            for j, lam in enumerate(LAM):
                f.write(f"{lam:.3f},{np.degrees(r['err'][0,j]):.6f},"
                        f"{np.degrees(r['err'][1,j]):.6f},"
                        f"{np.degrees(r['err'][2,j]):.6f}\n")
        print(f"  wrote {csv_out} + _tapers.csv + _phase.csv")
    return r
