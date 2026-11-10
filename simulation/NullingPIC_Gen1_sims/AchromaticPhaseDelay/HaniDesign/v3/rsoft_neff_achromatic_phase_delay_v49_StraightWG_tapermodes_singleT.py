#!/usr/bin/env python
# rsoft_neff_achromatic_phase_delay_v49_StraightWG_tapermodes_singleT.py
#
# V49: achromatic two-arm pi phase shifter, single temperature (20 C), with the
# two taper treatments Nem asked for.
#
# ===========================================================================
# THE POINT OF THIS VERSION
# ===========================================================================
# In V46/V47/V48 the tapers sat INLINE and differed between the two arms, so
# their phase and loss did NOT cancel in the differential quantities. That had
# two consequences:
#
#   1. The differential phase depended on how neff is interpolated inside a
#      taper, i.e. on an approximation off the 0.1 um width grid.
#   2. The taper loss asymmetry became a design variable the optimiser had to
#      fight, and it dominated the imbalance budget.
#
# V49 removes both by construction. Set TAPER_MODE to one of:
#
#   "symmetrized"  Every window of each arm is COPIED onto the end of the
#                  opposite arm. Each arm then traverses the union of both
#                  arms' window content, so
#                      integral n(w_top) dx |windows == integral n(w_bot) dx |windows
#                  and likewise for loss and for the slope-loss term. Taper
#                  phase and taper loss cancel EXACTLY in the difference. The
#                  differential phase is then a pure sum over the straight
#                  plateaus, with NO taper neff approximation in it at all.
#                  Geometry is complete and the total stays 4500 um.
#
#   "none"         No tapers in the model or the emitted geometry: abrupt width
#                  steps between straight plateaus. Use this if you want to
#                  insert tapers by hand afterwards. The script tells you, by
#                  re-evaluating with tapers inserted, exactly how much a given
#                  inline taper length would perturb the result -- so you can
#                  pick a taper length with the penalty in front of you.
#
#   "inline"       The V47/V48 behaviour, for comparison only.
#
# WHY THIS MAKES THE MATH EXACT
#   In "symmetrized" and "none" the differential optical path is
#       D(lam) = sum_i L_i * [ neff(a_i, lam) - neff(b_i, lam) ]
#   over straight plateaus only. Every term is the table's neff at a single
#   width -- no quadrature, no taper shape assumption. D is LINEAR in the
#   plateau lengths, so for fixed widths the minimax problem
#       min max_lam |2*pi/lam * D(lam) - pi|
#   is a linear program with a GLOBAL optimum. The imbalance is likewise
#       dLoss(lam) = s(lam) * sum_i L_i * [ q(a_i) - q(b_i) ]
#   a single wavelength shape (the taper slope-loss shape has cancelled), so the
#   power constraint is ONE linear constraint instead of a two-shape fight.
#
# THE COST, STATED UP FRONT
#   Cancelling the tapers also removes the taper-asymmetry compensator that
#   V48 used to meet a tight imbalance cap. Measured global bounds at a 4e-5
#   imbalance cap, compensator unavailable:
#       KW = 4e-7 (V46 model) -> 0.191 deg
#       KW = 1e-7             -> 0.180 deg
#       KW = 0  (width-neutral) -> 0.0157 deg
#   So with a width-DEPENDENT surrogate loss you trade phase for the
#   cancellation. With phase alone (no imbalance cap) these modes are excellent,
#   because the taper perturbation that limited V47/V48 is gone.
#   Both operating points are produced and reported; see ACCEPT_POLICY.
#
# LOSS / POWER WARNING, UNCHANGED FROM V47/V48
#   neff_imag in the supplied table is numerically zero (max 6.8e-13), so there
#   is NO measured loss in this data. Every loss and power number is a
#   SURROGATE, kept numerically identical to V46's for comparability, and its
#   width term makes WIDE guides lossy, which is backwards for weakly-guiding
#   multimode ridges. Replace SurrogateLoss.width_term() with BPM/FDTD/FemSIM/
#   RSoft transmission data before treating any power number as a spec.
#
# Run:
#   python rsoft_neff_achromatic_phase_delay_v49_StraightWG_tapermodes_singleT.py

from __future__ import annotations

import csv
import time
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline, PchipInterpolator, CubicSpline
from scipy.optimize import linprog

try:
    import gdstk
except Exception:
    gdstk = None
try:
    import gdspy
except Exception:
    gdspy = None


# ===========================================================================
# 1. USER CONTROLS
# ===========================================================================

TAPER_MODE = "symmetrized"          # "symmetrized" | "none" | "inline"
ALSO_RUN_MODE = "none"              # a second mode to run and tabulate, or None

OUTDIR = Path(f"v49_straightwg_{TAPER_MODE}_singleT")

NEFF_CSV_CANDIDATES = [
    Path("StraightWaveguide_neff_long.csv"),
    Path(r"C:\Users\hnejadri\Downloads\StraightWaveguide_neff_long.csv"),
    Path("..") / "StraightWaveguide_neff_long.csv",
]

T_OPT_C = 20.0                      # one temperature only

ACTIVE_LENGTH_UM = 3600.0
TRIM_LENGTH_UM = 900.0
TOTAL_LENGTH_UM = ACTIVE_LENGTH_UM + TRIM_LENGTH_UM
CENTER_TO_CENTER_SEPARATION_UM = 296.278
INPUT_OUTPUT_WIDTH_UM = 3.4
GDS_LAYER, GDS_DATATYPE = 1, 0

# loss model: KW is the width-dependence coefficient. DESIGN_KW is what the
# optimiser uses; LEGACY_KW (V46's) is carried for continuity scoring only.
DESIGN_KW = 1.0e-7
LEGACY_KW = 4.0e-7

# imbalance caps to sweep. None = phase only, no power constraint.
POWER_CAPS = [None, 4.0e-4, 1.0e-4, 4.0e-5]

LAMBDA_MIN_UM, LAMBDA_MAX_UM = 1.45, 1.65
N_LAMBDA_OPT = 81
N_LAMBDA_VERIFY = 401
N_LAMBDA_REPORT = 41

PHASE_ORDER_M = -1                  # raw target (2m+1)*pi; -pi wraps to +pi
TARGET_RAW_PHASE_RAD = (2 * PHASE_ORDER_M + 1) * np.pi

PHASE_GOAL_WORST_DEG = 0.05
PHASE_GOAL_RMS_DEG = 0.025
PHASE_LOOSE_WORST_DEG = 0.20

WIDTH_SOFT_MARGIN_UM = 0.06
MIN_PLATEAU_LENGTH_UM = 60.0
# Minimum transition-window length. The slope cap governs the big jumps anyway
# (it needs L >= pi*|dw|/(2*s_max) = 20.9*|dw|), so this is only the floor for
# small steps. In "symmetrized" mode every window is traversed TWICE per device,
# so each micron here costs two microns of straight-section budget -- keeping the
# floor modest is what buys back the phase performance symmetrizing costs.
WINDOW_LENGTH_UM = 40.0
MAX_ABS_WIDTH_SLOPE = 0.075         # |dw/dx| cap, ~1:13

# For "none": taper lengths to try inserting afterwards, to quantify the penalty
MANUAL_TAPER_LENGTHS_UM = [20.0, 40.0, 60.0, 100.0]

ROBUST_MODELS = ("cubic", "bilin")
VERIFY_MODELS = ("cubic", "quintic", "bilin", "pchip", "linlam")
PRIMARY_MODEL = "cubic"

N_SECTION_OPTIONS = [4, 5, 6, 7, 8, 9, 10, 12]
MEASURE_LP_WIDTH_STEP_UM = 0.02
ATOM_MERGE_TOL_UM = 0.025
SLP_ITERS = 90
SLP_SECONDS = 40.0
N_GAUSS_TAPER = 24

# acceptance: A = phase goals met and the tightest cap met; B = phase goals met;
# C = beats V46 on all phase+power under both loss models; D = nothing better
ACCEPT_POLICY = ("A", "C", "B", "D")

LAMS_OPT = np.linspace(LAMBDA_MIN_UM, LAMBDA_MAX_UM, N_LAMBDA_OPT)
LAMS_VERIFY = np.linspace(LAMBDA_MIN_UM, LAMBDA_MAX_UM, N_LAMBDA_VERIFY)
LAMS_REPORT = np.linspace(LAMBDA_MIN_UM, LAMBDA_MAX_UM, N_LAMBDA_REPORT)

# V46 reference, verbatim from v46_selected_sections.csv
V46_SECTIONS = [
    (4.608238219088965, 5.343966755816174, 20.000000003927042),
    (4.599855751027265, 2.6000931653023165, 2279.6217690670587),
    (2.4, 5.401463349404745, 1640.0161208284544),
    (3.6015134333156373, 5.407536312189252, 240.3621100916251),
    (6.8, 5.395172111446474, 20.000000008935064),
]
V46_TRANSITION_LENGTH_UM = 50.0
V46_REPORTED = {
    "rms_phase_deg": 0.100589, "worst_phase_deg": 0.198082,
    "p95_phase_deg": 0.190453, "rms_power_imbalance": 1.54897636e-05,
    "worst_power_imbalance": 2.60349409e-05,
    "mean_surrogate_loss_np": 1.05364081e-02,
}

_GL_U, _GL_W = np.polynomial.legendre.leggauss(N_GAUSS_TAPER)
_GL_U = 0.5 * (_GL_U + 1.0)
_GL_W = 0.5 * _GL_W
_GL_SHAPE = 0.5 - 0.5 * np.cos(np.pi * _GL_U)


def cosine_shape(u):
    return 0.5 - 0.5 * np.cos(np.pi * np.asarray(u, dtype=float))


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


# ===========================================================================
# 2. NEFF TABLE AND THE INTERPOLANT ENSEMBLE
# ===========================================================================

class NeffTable:
    """The measured table plus five admissible interpolants.

    The optimiser minimises the worst case over ROBUST_MODELS and reports the
    worst case over VERIFY_MODELS, so a result cannot be an artifact of one
    interpolation scheme. Measured: a design fitted to the cubic spline alone
    reads 0.0138 deg under cubic but 0.477 deg under bilinear.
    """

    def __init__(self):
        rows = src = None
        for p in NEFF_CSV_CANDIDATES:
            try:
                if p.is_file():
                    rows = list(csv.DictReader(StringIO(p.read_text())))
                    src = str(p.resolve())
                    break
            except Exception:
                continue
        if rows is None:
            raise SystemExit(
                "StraightWaveguide_neff_long.csv not found. Put it beside this "
                "script or edit NEFF_CSV_CANDIDATES at the top.")
        self.source = src
        w = np.array([float(r["width"]) for r in rows])
        l = np.array([float(r["wavelength"]) for r in rows])
        n = np.array([float(r["neff"]) for r in rows])
        ni = np.array([float(r.get("neff_imag") or 0.0) for r in rows])

        self.widths = np.array(sorted(set(np.round(w, 9))))
        self.lams = np.array(sorted(set(np.round(l, 9))))
        self.N = np.full((len(self.widths), len(self.lams)), np.nan)
        self.NI = np.zeros_like(self.N)
        for wi, li, nv, niv in zip(w, l, n, ni):
            iw = int(np.argmin(np.abs(self.widths - wi)))
            il = int(np.argmin(np.abs(self.lams - li)))
            self.N[iw, il] = nv
            self.NI[iw, il] = niv
        if np.isnan(self.N).any():
            raise SystemExit("neff table is not a complete width x wavelength grid.")

        self.neff_imag_absmax = float(np.max(np.abs(self.NI)))
        self.neff_imag_nonzero = int(np.sum(self.NI != 0.0))
        self.neff_imag_is_meaningful = self.neff_imag_absmax > 1e-9

        self._sp3 = RectBivariateSpline(self.widths, self.lams, self.N, kx=3, ky=3, s=0)
        k = min(5, len(self.widths) - 1, len(self.lams) - 1)
        self._sp5 = RectBivariateSpline(self.widths, self.lams, self.N, kx=k, ky=k, s=0)
        self._pchip = [PchipInterpolator(self.widths, self.N[:, j])
                       for j in range(len(self.lams))]
        self._spi = RectBivariateSpline(self.widths, self.lams, self.NI, kx=1, ky=1, s=0)
        self._warned = False

    def check_range(self, w):
        w = np.atleast_1d(w)
        if (w.min() < self.widths[0] - 1e-9 or w.max() > self.widths[-1] + 1e-9) \
                and not self._warned:
            print(f"  ** WARNING: width {w.min():.4f}-{w.max():.4f} um leaves the "
                  f"table range {self.widths[0]:.3f}-{self.widths[-1]:.3f} um; "
                  f"values are EXTRAPOLATED.")
            self._warned = True

    def Tlin(self, lam):
        T = np.zeros((len(self.lams), len(lam)))
        for j, lv in enumerate(lam):
            i = int(np.clip(np.searchsorted(self.lams, lv) - 1, 0, len(self.lams) - 2))
            f = (lv - self.lams[i]) / (self.lams[i + 1] - self.lams[i])
            T[i, j] = 1.0 - f
            T[i + 1, j] = f
        return T

    def Tcub(self, lam):
        I = np.eye(len(self.lams))
        return np.array([CubicSpline(self.lams, I[k])(lam)
                         for k in range(len(self.lams))])

    def eval(self, kind, w, lam, Tlin=None, Tcub=None):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        nl = len(lam)
        if kind in ("cubic", "quintic"):
            sp = self._sp3 if kind == "cubic" else self._sp5
            return sp.ev(np.repeat(w, nl), np.tile(lam, len(w))).reshape(len(w), nl)
        if kind == "bilin":
            Aw = np.array([np.interp(w, self.widths, self.N[:, j])
                           for j in range(len(self.lams))]).T
            return Aw @ (self.Tlin(lam) if Tlin is None else Tlin)
        if kind == "pchip":
            Aw = np.array([p(w) for p in self._pchip]).T
            return Aw @ (self.Tcub(lam) if Tcub is None else Tcub)
        if kind == "linlam":
            n0 = len(self.lams)
            Aw = self._sp3.ev(np.repeat(w, n0),
                              np.tile(self.lams, len(w))).reshape(len(w), n0)
            return Aw @ (self.Tlin(lam) if Tlin is None else Tlin)
        raise KeyError(kind)

    def eval_imag(self, w, lam):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        return np.maximum(self._spi.ev(np.repeat(w, len(lam)),
                                       np.tile(lam, len(w))).reshape(len(w), len(lam)),
                          0.0)


class ModelBank:
    def __init__(self, table: NeffTable, lam):
        self.t = table
        self.lam = np.asarray(lam, dtype=float)
        self._Tlin = table.Tlin(self.lam)
        self._Tcub = table.Tcub(self.lam)

    def __call__(self, kind, w):
        return self.t.eval(kind, w, self.lam, Tlin=self._Tlin, Tcub=self._Tcub)

    def imag(self, w):
        return self.t.eval_imag(w, self.lam)


# ===========================================================================
# 3. LOSS MODEL -- SURROGATE. REPLACE width_term() WITH REAL DATA.
# ===========================================================================

class SurrogateLoss:
    """alpha(w,lam) = 4*pi*Im(neff)/lam                     <- real, but ~0 here
                    + (1.55/lam)*(A0 + KW*(w-w_io)^2 + KN/w^2)   <- SURROGATE
       plus KS*(1.55/lam)^1.5 * (dw/dx)^2 along tapers           <- SURROGATE

    Numerically identical in form to V46's so the comparison is like for like.
    In "symmetrized" and "none" modes the slope term cancels in the difference.
    """
    W_IO = INPUT_OUTPUT_WIDTH_UM

    def __init__(self, table, kw=DESIGN_KW, ks=0.010, a0=1.5e-7, kn=2.0e-7,
                 label="design"):
        self.table = table
        self.A0, self.KW, self.KN, self.KS = float(a0), float(kw), float(kn), float(ks)
        self.label = label
        self.use_material = table.neff_imag_is_meaningful

    def width_term(self, w, lam):
        w = np.atleast_1d(np.asarray(w, dtype=float))[:, None]
        lam = np.asarray(lam, dtype=float)[None, :]
        return (1.55 / lam) * (self.A0 + self.KW * (w - self.W_IO) ** 2
                               + self.KN / np.maximum(w, 0.8) ** 2)

    def material_term(self, bank, w):
        if not self.use_material:
            return None
        return 4.0 * np.pi * bank.imag(w) / bank.lam[None, :]

    def slope_coefficient(self, lam):
        return self.KS * (1.55 / np.asarray(lam, dtype=float)) ** 1.5

    def q_ratio(self):
        f = lambda w: self.A0 + self.KW * (w - self.W_IO) ** 2 + self.KN / w ** 2
        return f(6.8) / f(2.4)


# ===========================================================================
# 4. GEOMETRY -- mode aware
# ===========================================================================
#
#   plateaus  i = 0..N-1 : straight, width a_i (top) / b_i (bottom), length Lp_i
#   windows   j = 0..N   : transition between consecutive widths, length Ltw_j
#
# Ltw_j is DERIVED, not optimised: it is the shortest window that satisfies the
# slope cap for BOTH arms, floored at WINDOW_LENGTH_UM. A cosine taper of height
# dw over length L peaks at pi*|dw|/(2L), so the cap needs L >= pi*|dw|/(2*s_max).
#
# Length budget by mode:
#   "none"         sum(Lp)                  = 4500
#   "inline"       sum(Lp) +     sum(Ltw)   = 4500
#   "symmetrized"  sum(Lp) + 2 * sum(Ltw)   = 4500
#     because each arm carries its own windows AND a copy of the other arm's.

def window_multiplier(mode):
    return {"none": 0, "inline": 1, "symmetrized": 2}[mode]


class Design:
    __slots__ = ("a", "b", "Lp", "mode", "Ltw_fixed")

    def __init__(self, a, b, Lp, mode=None, Ltw_fixed=None):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.Lp = np.asarray(Lp, dtype=float)
        self.mode = mode or TAPER_MODE
        # Explicit window lengths, used to reconstruct a design that was built
        # with windows we would not choose ourselves -- V46's fixed 50 um, which
        # violates our slope cap. Without this the reconstruction silently gets
        # different windows and the comparison is meaningless.
        self.Ltw_fixed = None if Ltw_fixed is None else np.asarray(Ltw_fixed, float)

    @property
    def n(self):
        return len(self.a)

    def copy(self):
        return Design(self.a.copy(), self.b.copy(), self.Lp.copy(), self.mode,
                      None if self.Ltw_fixed is None else self.Ltw_fixed.copy())

    def nodes(self):
        io = INPUT_OUTPUT_WIDTH_UM
        return np.r_[io, self.a, io], np.r_[io, self.b, io]

    def Ltw(self):
        if self.Ltw_fixed is not None:
            return self.Ltw_fixed
        if self.mode == "none":
            return np.zeros(self.n + 1)
        A, B = self.nodes()
        dmax = np.maximum(np.abs(np.diff(A)), np.abs(np.diff(B)))
        need = np.pi * dmax / (2.0 * MAX_ABS_WIDTH_SLOPE)
        return np.maximum(WINDOW_LENGTH_UM, need)

    def plateau_budget(self):
        return TOTAL_LENGTH_UM - window_multiplier(self.mode) * self.Ltw().sum()

    def total_length(self):
        return float(self.Lp.sum()
                     + window_multiplier(self.mode) * self.Ltw().sum())

    def max_abs_slope(self):
        if self.mode == "none":
            return 0.0
        A, B = self.nodes()
        L = self.Ltw()
        st = (np.pi * np.abs(np.diff(A)) / (2.0 * L)).max()
        sb = (np.pi * np.abs(np.diff(B)) / (2.0 * L)).max()
        return float(max(st, sb))

    def width_excursion(self):
        A, B = self.nodes()
        return float(np.abs(np.diff(A)).sum() + np.abs(np.diff(B)).sum())

    def fit_budget(self):
        """Rescale plateau lengths to the budget, respecting the length floor."""
        d = self.copy()
        avail = d.plateau_budget()
        extra = avail - d.n * MIN_PLATEAU_LENGTH_UM
        if extra <= 0:
            return None
        slack = np.maximum(d.Lp - MIN_PLATEAU_LENGTH_UM, 0.0)
        if slack.sum() > 0:
            d.Lp = MIN_PLATEAU_LENGTH_UM + slack * (extra / slack.sum())
        else:
            d.Lp = np.full(d.n, avail / d.n)
        return d

    # ---- window content, for absolute loss and for the inline offsets -------
    def window_samples(self, which):
        """(widths, x-weights) sampling one arm's own windows."""
        if self.mode == "none":
            return np.zeros(0), np.zeros(0)
        A, B = self.nodes()
        X = A if which == "top" else B
        L = self.Ltw()
        w = (X[:-1][:, None] + np.diff(X)[:, None] * _GL_SHAPE[None, :]).ravel()
        q = (L[:, None] * _GL_W[None, :]).ravel()
        return w, q

    def slope_sq_integral(self, which):
        if self.mode == "none":
            return 0.0
        A, B = self.nodes()
        X = A if which == "top" else B
        return float(np.sum(np.pi ** 2 * np.diff(X) ** 2 / (8.0 * self.Ltw())))

    def arm_window_sets(self, which):
        """Which window sets this ARM physically traverses.

        inline / none : only its own.
        symmetrized   : its own PLUS a copy of the other arm's, which is what
                        makes taper phase and loss common-mode and therefore
                        cancel in every differential quantity.
        """
        if self.mode == "symmetrized":
            return ["top", "bot"]
        return [which]

    def dense_profile(self, pts_per_um=1.0):
        """Emitted geometry.

        For "symmetrized" the copied window chain is appended after the active
        region. That chain is continuous and starts and ends at the I/O width,
        because an arm's windows already form a path io -> ... -> io once the
        plateaus are removed, so it splices on without a discontinuity.
        """
        A, B = self.nodes()
        L = self.Ltw()
        xs = [0.0]
        tw = [INPUT_OUTPUT_WIDTH_UM]
        bw = [INPUT_OUTPUT_WIDTH_UM]
        state = {"x": 0.0}

        def seg(L_, fa, fb):
            if L_ <= 0:
                return
            m = max(6, int(np.ceil(L_ * pts_per_um)))
            u = np.linspace(0.0, 1.0, m + 1)[1:]
            xs.extend((state["x"] + L_ * u).tolist())
            tw.extend(np.atleast_1d(fa(u)).tolist())
            bw.extend(np.atleast_1d(fb(u)).tolist())
            state["x"] += L_

        def ramp(w0, w1):
            return lambda u: w0 + (w1 - w0) * cosine_shape(u)

        def flat(v):
            return lambda u: np.full(len(np.atleast_1d(u)), v)

        # --- active region: own windows interleaved with the plateaus ---------
        for j in range(self.n + 1):
            if self.mode != "none":
                seg(L[j], ramp(A[j], A[j + 1]), ramp(B[j], B[j + 1]))
            if j < self.n:
                seg(self.Lp[j], flat(self.a[j]), flat(self.b[j]))

        # --- symmetrized: append the OTHER arm's window chain to each arm -----
        if self.mode == "symmetrized":
            for j in range(self.n + 1):
                seg(L[j], ramp(B[j], B[j + 1]), ramp(A[j], A[j + 1]))

        xs = np.asarray(xs)
        tw = np.asarray(tw)
        bw = np.asarray(bw)
        xg = np.linspace(0.0, xs[-1], max(1001, int(round(xs[-1])) + 1))
        return xg, np.interp(xg, xs, tw), np.interp(xg, xs, bw)


# ===========================================================================
# 5. EVALUATION
# ===========================================================================

def _window_terms(design, bank, loss, kind):
    """Per-arm window contributions to (optical path, loss) versus wavelength.

    For "symmetrized" these come out identical between the arms BY
    CONSTRUCTION. The caller records the residual difference so the
    cancellation is verified numerically rather than assumed.
    """
    lam = bank.lam
    zero = np.zeros_like(lam)
    if design.mode == "none":
        return zero.copy(), zero.copy(), zero.copy(), zero.copy()

    cache = {}
    for which in ("top", "bot"):
        w, q = design.window_samples(which)
        nq = bank(kind, w)
        aq = loss.width_term(w, lam)
        mt = loss.material_term(bank, w)
        if mt is not None:
            aq = aq + mt
        cache[which] = (q @ nq, q @ aq,
                        loss.slope_coefficient(lam) * design.slope_sq_integral(which))

    opd, los = {}, {}
    for arm in ("top", "bot"):
        o = zero.copy()
        l = zero.copy()
        for s in design.arm_window_sets(arm):
            o = o + cache[s][0]
            l = l + cache[s][1] + cache[s][2]
        opd[arm], los[arm] = o, l
    return opd["top"], opd["bot"], los["top"], los["bot"]


def eval_design(design, bank, loss, kind=PRIMARY_MODEL, check_range=False):
    lam = bank.lam
    if check_range:
        A, B = design.nodes()
        bank.t.check_range(np.r_[A, B])

    nt = bank(kind, design.a)
    nb = bank(kind, design.b)
    ot, ob, lt_w, lb_w = _window_terms(design, bank, loss, kind)

    D = design.Lp @ (nt - nb) + (ot - ob)
    raw = 2.0 * np.pi / lam * D

    at = loss.width_term(design.a, lam)
    ab = loss.width_term(design.b, lam)
    mt = loss.material_term(bank, design.a)
    if mt is not None:
        at = at + mt
        ab = ab + loss.material_term(bank, design.b)
    Lt = design.Lp @ at + lt_w
    Lb = design.Lp @ ab + lb_w

    Pt, Pb = np.exp(-Lt), np.exp(-Lb)
    return {
        "lam": lam, "D": D, "raw_phase": raw,
        "phase_err_deg": np.rad2deg(raw - TARGET_RAW_PHASE_RAD),
        "top_loss": Lt, "bot_loss": Lb, "top_power": Pt, "bot_power": Pb,
        "imbalance": (Pt - Pb) / np.maximum(Pt + Pb, 1e-300),
        "mean_loss": 0.5 * (Lt + Lb),
        "window_opd_diff": float(np.max(np.abs(ot - ob))),
        "window_loss_diff": float(np.max(np.abs(lt_w - lb_w))),
    }


def metrics(design, table, loss, bank_ver, bank_rep, models=VERIFY_MODELS):
    per = {}
    for k in models:
        e = eval_design(design, bank_ver, loss, kind=k)["phase_err_deg"]
        per[k] = {"worst_phase_deg": float(np.max(np.abs(e))),
                  "rms_phase_deg": rms(e), "mean_phase_deg": float(np.mean(e))}
    rep = eval_design(design, bank_rep, loss, kind=PRIMARY_MODEL)
    e, imb, lam = rep["phase_err_deg"], rep["imbalance"], rep["lam"]
    slope = np.gradient(e, lam)
    allw = np.r_[design.a, design.b]
    lo = table.widths[0] + WIDTH_SOFT_MARGIN_UM
    hi = table.widths[-1] - WIDTH_SOFT_MARGIN_UM
    return {
        "per_model": per,
        "worst_phase_deg": max(v["worst_phase_deg"] for v in per.values()),
        "rms_phase_deg": max(v["rms_phase_deg"] for v in per.values()),
        "worst_phase_primary_deg": float(np.max(np.abs(e))),
        "rms_phase_primary_deg": rms(e),
        "p95_phase_deg": float(np.percentile(np.abs(e), 95)),
        "mean_phase_deg": float(np.mean(e)),
        "worst_power_imbalance": float(np.max(np.abs(imb))),
        "rms_power_imbalance": rms(imb),
        "mean_power_imbalance": float(np.mean(imb)),
        "mean_surrogate_loss_np": float(np.mean(rep["mean_loss"])),
        "worst_arm_loss_np": float(max(rep["top_loss"].max(), rep["bot_loss"].max())),
        "slope_rms_deg_per_um": rms(slope),
        "curvature_rms_deg_per_um2": rms(np.gradient(slope, lam)),
        "n_sections": design.n,
        "min_plateau_um": float(design.Lp.min()),
        "max_plateau_um": float(design.Lp.max()),
        "window_len_um": (float(design.Ltw().max()) if design.mode != "none" else 0.0),
        "width_min_um": float(allw.min()), "width_max_um": float(allw.max()),
        "width_excursion_um": design.width_excursion(),
        "max_abs_slope": design.max_abs_slope(),
        "n_widths_near_table_edge": int(np.sum((allw < lo) | (allw > hi))),
        "n_plateaus_at_min_length": int(np.sum(design.Lp <= MIN_PLATEAU_LENGTH_UM + 1e-6)),
        "total_length_um": design.total_length(),
        "window_opd_diff": rep["window_opd_diff"],
        "window_loss_diff": rep["window_loss_diff"],
        "mode": design.mode,
        "report": rep,
    }


# ===========================================================================
# 6. STAGE 0 -- GLOBAL BOUND FROM THE WIDTH MEASURE
# ===========================================================================
#
# Phase depends only on the width MEASURE of each arm, so
#     min over designs  max over lam |dphi - pi|
# is a linear program with a genuine global optimum. In "symmetrized"/"none"
# there is no taper-asymmetry compensator to model, because it cancels, so the
# LP is an honest bound rather than an optimistic one -- which is exactly the
# weakness that misled V48.

def stage0_measure_lp(table, loss, bank, power_guard=None, mass=None,
                      models=("cubic", "quintic", "bilin"),
                      width_step=MEASURE_LP_WIDTH_STEP_UM):
    lam = bank.lam
    nl = len(lam)
    mass = TOTAL_LENGTH_UM if mass is None else float(mass)
    wg = np.arange(table.widths[0], table.widths[-1] + 1e-9, width_step)
    nw = len(wg)
    target_opd = (2 * PHASE_ORDER_M + 1) * lam / 2.0

    c = np.zeros(2 * nw + 1)
    c[-1] = 1.0
    A_ub, b_ub = [], []
    O = np.ones((nl, 1))
    for k in models:
        M = (bank(k, wg) - bank(k, np.array([INPUT_OUTPUT_WIDTH_UM]))[0][None, :]) * 1e3
        A_ub += [np.hstack([M.T, -M.T, -O]), np.hstack([-M.T, M.T, -O])]
        b_ub += [target_opd, -target_opd]
    if power_guard is not None:
        qv = loss.width_term(wg, np.array([1.55]))[:, 0] * 1e3
        Q = (1.55 / lam)[:, None] * qv[None, :]
        Z = np.zeros((nl, 1))
        A_ub += [np.hstack([Q, -Q, Z]), np.hstack([-Q, Q, Z])]
        b_ub += [np.full(nl, 2.0 * power_guard), np.full(nl, 2.0 * power_guard)]

    A_eq = np.array([np.hstack([np.ones(nw), np.zeros(nw), [0.0]]),
                     np.hstack([np.zeros(nw), np.ones(nw), [0.0]])])
    r = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                A_eq=A_eq, b_eq=[mass * 1e-3, mass * 1e-3],
                bounds=[(0.0, mass * 1e-3)] * (2 * nw) + [(0.0, None)],
                method="highs")
    if not r.success:
        return None
    return {"widths": wg, "mu_top": r.x[:nw] * 1e3, "mu_bot": r.x[nw:2 * nw] * 1e3,
            "bound_deg": float(np.rad2deg(2.0 * np.pi / 1.55 * r.x[-1]))}


def _atoms(wg, mu, tol=ATOM_MERGE_TOL_UM, n_keep=None, mass=TOTAL_LENGTH_UM):
    out = []
    for i in np.where(mu > 1e-9)[0]:
        if out and (wg[i] - out[-1][0]) <= tol:
            w0, m0 = out[-1]
            out[-1] = ((w0 * m0 + wg[i] * mu[i]) / (m0 + mu[i]), m0 + mu[i])
        else:
            out.append((wg[i], float(mu[i])))
    out.sort(key=lambda t: -t[1])
    if n_keep is not None:
        out = out[:n_keep]
    out.sort(key=lambda t: t[0])
    tot = sum(m for _, m in out)
    return [[w, m * mass / tot] for w, m in out]


def _refine(at, ab):
    secs, it, ib = [], 0, 0
    rt, rb = at[0][1], ab[0][1]
    guard = 0
    while it < len(at) and ib < len(ab) and guard < 10000:
        guard += 1
        take = min(rt, rb)
        secs.append([at[it][0], ab[ib][0], take])
        rt -= take
        rb -= take
        if rt <= 1e-9:
            it += 1
            rt = at[it][1] if it < len(at) else 0.0
        if rb <= 1e-9:
            ib += 1
            rb = ab[ib][1] if ib < len(ab) else 0.0
    return secs


def _order(secs):
    """Order sections so both arms move as monotonically as possible, which
    keeps every |dw| small and therefore every window short."""
    io = INPUT_OUTPUT_WIDTH_UM

    def cost(o):
        pts = [(io, io)] + [(secs[i][0], secs[i][1]) for i in o] + [(io, io)]
        return sum(abs(pts[k + 1][0] - pts[k][0]) + abs(pts[k + 1][1] - pts[k][1])
                   for k in range(len(pts) - 1))

    rem, order, cur = list(range(len(secs))), [], (io, io)
    while rem:
        j = min(rem, key=lambda i: abs(secs[i][0] - cur[0]) + abs(secs[i][1] - cur[1]))
        order.append(j)
        rem.remove(j)
        cur = (secs[j][0], secs[j][1])
    improved = True
    while improved:
        improved = False
        base = cost(order)
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if cost(cand) < base - 1e-12:
                    order, base, improved = cand, cost(cand), True
    return [secs[i] for i in order]


def seed_design(lp_sol, n_target, mode):
    """LP measure -> an ordered Design with n_target plateaus."""
    nk = max(2, (n_target + 2) // 2)
    at = _atoms(lp_sol["widths"], lp_sol["mu_top"], n_keep=nk)
    ab = _atoms(lp_sol["widths"], lp_sol["mu_bot"], n_keep=nk)
    secs = _order(_refine(at, ab))
    secs = [s for s in secs if s[2] > 1e-6]
    if len(secs) > n_target:
        secs.sort(key=lambda s: -s[2])
        secs = _order(secs[:n_target])
    if len(secs) < 2:
        return None
    d = Design([s[0] for s in secs], [s[1] for s in secs],
               [max(s[2], MIN_PLATEAU_LENGTH_UM) for s in secs], mode)
    return d.fit_budget()


# ===========================================================================
# 7. EXACT LP IN THE PLATEAU LENGTHS
# ===========================================================================
#
# For fixed widths, phase and loss are LINEAR in the plateau lengths, so this is
# a global minimax (or a global minimum imbalance) -- not a local search. In
# "symmetrized"/"none" the window offsets below are exactly zero, so the LP is
# the whole story for the lengths.

def length_lp(design, bank, loss, mode="phase", cap=None, models=ROBUST_MODELS):
    lam = bank.lam
    nl = len(lam)
    n = design.n
    avail = design.plateau_budget()
    if avail <= n * MIN_PLATEAU_LENGTH_UM:
        return None, None

    ot, ob, lt_w, lb_w = _window_terms(design, bank, loss, PRIMARY_MODEL)
    P, off = {}, {}
    for k in models:
        o_t, o_b, _, _ = _window_terms(design, bank, loss, k)
        dn = bank(k, design.a) - bank(k, design.b)
        P[k] = (2.0 * np.pi / lam)[:, None] * dn.T
        off[k] = (2.0 * np.pi / lam) * (o_t - o_b)
    dq = (loss.width_term(design.a, lam) - loss.width_term(design.b, lam)).T
    dq_off = lt_w - lb_w

    O = np.ones((nl, 1))
    Z = np.zeros((nl, 1))
    A_ub, b_ub = [], []
    c = np.zeros(n + 1)
    c[-1] = 1.0

    if mode == "phase":
        for k in models:
            tgt = TARGET_RAW_PHASE_RAD - off[k]
            A_ub += [np.hstack([P[k], -O]), np.hstack([-P[k], -O])]
            b_ub += [tgt, -tgt]
        if cap is not None:
            A_ub += [np.hstack([dq, Z]), np.hstack([-dq, Z])]
            b_ub += [2.0 * cap - dq_off, 2.0 * cap + dq_off]
    elif mode == "power":
        capr = np.deg2rad(cap)
        for k in models:
            tgt = TARGET_RAW_PHASE_RAD - off[k]
            A_ub += [np.hstack([P[k], Z]), np.hstack([-P[k], Z])]
            b_ub += [tgt + capr, -tgt + capr]
        A_ub += [np.hstack([dq, -O]), np.hstack([-dq, -O])]
        b_ub += [-dq_off, dq_off]
    else:
        raise ValueError(mode)

    r = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                A_eq=np.array([np.r_[np.ones(n), 0.0]]), b_eq=[avail],
                bounds=[(MIN_PLATEAU_LENGTH_UM, avail)] * n + [(0.0, None)],
                method="highs")
    if not r.success:
        return None, None
    out = design.copy()
    out.Lp = r.x[:n]
    return out, float(np.rad2deg(r.x[-1]) if mode == "phase" else 0.5 * r.x[-1])


# ===========================================================================
# 8. SEQUENTIAL LP OVER THE WIDTHS (trust region)
# ===========================================================================

def _pack(d):
    return np.r_[d.a, d.b, d.Lp]


def _unpack(z, n, mode):
    return Design(z[:n], z[n:2 * n], z[2 * n:], mode)


def _resid(z, n, mode, bank, loss, models):
    d = _unpack(z, n, mode)
    es = [eval_design(d, bank, loss, kind=k)["phase_err_deg"] for k in models]
    r = eval_design(d, bank, loss, kind=PRIMARY_MODEL)
    return (np.concatenate(es), r["imbalance"],
            float(np.mean(r["mean_loss"])), d.total_length())


def slp(design, table, bank, loss, mode="phase", cap=None, models=ROBUST_MODELS,
        iters=SLP_ITERS, seconds=SLP_SECONDS):
    t0 = time.time()
    n = design.n
    tmode = design.mode
    z = _pack(design)
    m = len(z)
    lo = np.r_[np.full(2 * n, table.widths[0]), np.full(n, MIN_PLATEAU_LENGTH_UM)]
    hi = np.r_[np.full(2 * n, table.widths[-1]), np.full(n, TOTAL_LENGTH_UM)]
    base = np.r_[np.full(2 * n, 0.15), np.full(n, 120.0)]
    tr = base.copy()
    h = np.r_[np.full(2 * n, 1e-6), np.full(n, 1e-3)]

    def score(zz):
        e, p, ml, _ = _resid(zz, n, tmode, bank, loss, models)
        we, wp = float(np.max(np.abs(e))), float(np.max(np.abs(p)))
        if mode == "phase":
            return we, max(0.0, wp - (cap if cap is not None else np.inf))
        return wp, max(0.0, we - cap)

    best_obj, best_viol = score(z)
    best_z = z.copy()

    for _ in range(iters):
        if time.time() - t0 > seconds:
            break
        e0, p0, _, len0 = _resid(z, n, tmode, bank, loss, models)
        ne, npw = len(e0), len(p0)
        Je = np.zeros((ne, m))
        Jp = np.zeros((npw, m))
        Jl = np.zeros(m)
        for i in range(m):
            zp = z.copy()
            zp[i] += h[i]
            e1, p1, _, len1 = _resid(zp, n, tmode, bank, loss, models)
            Je[:, i] = (e1 - e0) / h[i]
            Jp[:, i] = (p1 - p0) / h[i]
            Jl[i] = (len1 - len0) / h[i]

        Oe = np.ones((ne, 1))
        Ze = np.zeros((ne, 1))
        Op = np.ones((npw, 1))
        Zp = np.zeros((npw, 1))
        c = np.r_[np.zeros(m), 1.0]
        if mode == "phase":
            rows = [np.hstack([Je, -Oe]), np.hstack([-Je, -Oe])]
            rhs = [-e0, e0]
            if cap is not None:
                rows += [np.hstack([Jp, Zp]), np.hstack([-Jp, Zp])]
                rhs += [cap - p0, cap + p0]
        else:
            rows = [np.hstack([Je, Ze]), np.hstack([-Je, Ze])]
            rhs = [cap - e0, cap + e0]
            rows += [np.hstack([Jp, -Op]), np.hstack([-Jp, -Op])]
            rhs += [-p0, p0]

        bd = []
        for i in range(m):
            a_ = max(lo[i] - z[i], -tr[i])
            b_ = min(hi[i] - z[i], tr[i])
            bd.append((min(a_, 0.0), max(b_, 0.0)))
        bd.append((0.0, None))

        # total length must stay at 4500: linearised, then projected exactly
        r = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                    A_eq=np.r_[Jl, 0.0][None, :],
                    b_eq=[TOTAL_LENGTH_UM - len0], bounds=bd, method="highs")
        if not r.success:
            tr *= 0.5
            if np.all(tr < base * 1e-4):
                break
            continue

        d_new = _unpack(np.clip(z + r.x[:m], lo, hi), n, tmode).fit_budget()
        if d_new is not None and d_new.max_abs_slope() <= MAX_ABS_WIDTH_SLOPE * 1.0001:
            zn = _pack(d_new)
            obj, viol = score(zn)
            if (viol < best_viol - 1e-15) or (viol <= max(best_viol, 1e-15)
                                              and obj < best_obj):
                z, best_obj, best_viol = zn, obj, viol
                best_z = z.copy()
                tr = np.minimum(tr * 1.5, base * 3.0)
                continue
        tr *= 0.45
        if np.all(tr < base * 1e-4):
            break

    return _unpack(best_z, n, tmode), best_obj, best_viol


# ===========================================================================
# 9. V46 REFERENCE, scored on this evaluator in "inline" mode
# ===========================================================================

def build_v46_design():
    a = np.array([s[0] for s in V46_SECTIONS])
    b = np.array([s[1] for s in V46_SECTIONS])
    Lp = np.array([s[2] for s in V46_SECTIONS])
    # V46's actual geometry: six fixed 50 um cosine transitions. Pin them, so
    # this is the design V46 really built and the comparison is like for like.
    d = Design(a, b, Lp, "inline",
               Ltw_fixed=np.full(len(a) + 1, V46_TRANSITION_LENGTH_UM))
    err = d.total_length() - TOTAL_LENGTH_UM
    if abs(err) > 1e-6:
        d.Lp = d.Lp - err / d.n
    return d


# ===========================================================================
# 10. "none" MODE -- what inserting tapers by hand will cost you
# ===========================================================================

def manual_taper_penalty(design, table, loss, bank_opt, bank_ver, bank_rep,
                         taper_lengths=MANUAL_TAPER_LENGTHS_UM):
    """Take a taper-free design and insert inline cosine tapers of each length.

    Reported two ways:
      "as is"    plateaus rescaled to keep the total at 4500 um, nothing else
                 retuned. This is the penalty if you insert tapers and change
                 nothing else.
      "relength" the plateau lengths re-solved by the exact LP afterwards. This
                 is the best achievable by retuning lengths only, no new widths.
    """
    global WINDOW_LENGTH_UM
    keep = WINDOW_LENGTH_UM
    rows = []
    for Lt in taper_lengths:
        WINDOW_LENGTH_UM = float(Lt)
        d = Design(design.a, design.b, design.Lp, "inline").fit_budget()
        if d is None:
            rows.append((Lt, None, None, None, None, None))
            continue
        m_as = metrics(d, table, loss, bank_ver, bank_rep)
        d2, _ = length_lp(d, bank_opt, loss, mode="phase")
        m_re = metrics(d2, table, loss, bank_ver, bank_rep) if d2 is not None else None
        rows.append((Lt, m_as["worst_phase_deg"], m_as["rms_phase_deg"],
                     (m_re["worst_phase_deg"] if m_re else None),
                     (m_re["rms_phase_deg"] if m_re else None),
                     (d2 if d2 is not None else d)))
    WINDOW_LENGTH_UM = keep
    return rows


# ===========================================================================
# 11. OUTPUT HELPERS
# ===========================================================================

def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def split_active_trim(x, tw, bw):
    xa = np.linspace(0.0, ACTIVE_LENGTH_UM, 1201)
    xt = np.linspace(0.0, TRIM_LENGTH_UM, 401)
    return (xa, np.interp(xa, x, tw), np.interp(xa, x, bw),
            xt, np.interp(ACTIVE_LENGTH_UM + xt, x, tw),
            np.interp(ACTIVE_LENGTH_UM + xt, x, bw))


def emit_design(prefix, d, MM):
    rr = MM["report"]
    ll = rr["lam"]
    save_csv(OUTDIR / f"{prefix}_wavelength_by_wavelength.csv",
             ["lambda_um", "phase_difference_rad", "phase_difference_deg",
              "phase_error_from_pi_deg", "phase_error_from_pi_rad",
              "top_power_surrogate", "bottom_power_surrogate",
              "normalized_power_imbalance_surrogate",
              "top_loss_np_surrogate", "bottom_loss_np_surrogate",
              "differential_optical_path_um"],
             [[ll[i], rr["raw_phase"][i], np.rad2deg(rr["raw_phase"][i]),
               rr["phase_err_deg"][i], np.deg2rad(rr["phase_err_deg"][i]),
               rr["top_power"][i], rr["bot_power"][i], rr["imbalance"][i],
               rr["top_loss"][i], rr["bot_loss"][i], rr["D"][i]]
              for i in range(len(ll))])

    A, B = d.nodes()
    L = d.Ltw()
    rows = []
    x = 0.0
    for j in range(d.n + 1):
        if d.mode != "none":
            rows.append([f"window_{j}", "", "", L[j], x, x + L[j],
                         A[j], A[j + 1], B[j], B[j + 1]])
            x += L[j]
        else:
            rows.append([f"step_{j}", "", "", 0.0, x, x,
                         A[j], A[j + 1], B[j], B[j + 1]])
        if j < d.n:
            rows.append([f"plateau_{j + 1}", d.a[j], d.b[j], d.Lp[j],
                         x, x + d.Lp[j], "", "", "", ""])
            x += d.Lp[j]
    if d.mode == "symmetrized":
        for j in range(d.n + 1):
            rows.append([f"mirror_window_{j}", "", "", L[j], x, x + L[j],
                         B[j], B[j + 1], A[j], A[j + 1]])
            x += L[j]
    save_csv(OUTDIR / f"{prefix}_selected_sections.csv",
             ["element", "top_width_um", "bottom_width_um", "length_um",
              "x_start_um", "x_end_um",
              "top_w_from", "top_w_to", "bot_w_from", "bot_w_to"], rows)

    # the plain straight-section list, which is all you need for mode "none"
    save_csv(OUTDIR / f"{prefix}_straight_sections_only.csv",
             ["section_index", "top_width_um", "bottom_width_um",
              "straight_length_um"],
             [[i + 1, d.a[i], d.b[i], d.Lp[i]] for i in range(d.n)])

    xx, twv, bwv = d.dense_profile()
    xav, tav, bav, xtv, ttv, btv = split_active_trim(xx, twv, bwv)
    for nm, (X, T, Bo, hdr) in {
        f"{prefix}_full_layout_profile.csv": (xx, twv, bwv, "x_um"),
        f"{prefix}_active_width_profile.csv": (xav, tav, bav, "x_um"),
        f"{prefix}_trim_width_profile.csv": (xtv, ttv, btv, "x_trim_um"),
    }.items():
        save_csv(OUTDIR / nm,
                 [hdr, "top_width_um", "bottom_width_um",
                  "center_to_center_separation_um"],
                 [[X[i], T[i], Bo[i], CENTER_TO_CENTER_SEPARATION_UM]
                  for i in range(len(X))])


def write_gds(d, name, cellname):
    if gdstk is None and gdspy is None:
        print(f"  neither gdstk nor gdspy installed; skipping {name}")
        print("  (pip install gdstk, then re-run; the profile CSV has the geometry)")
        return None
    x, tw, bw = d.dense_profile()
    yt = +0.5 * CENTER_TO_CENTER_SEPARATION_UM
    yb = -0.5 * CENTER_TO_CENTER_SEPARATION_UM

    def poly(xv, yc, w):
        return np.vstack([np.column_stack([xv, yc + 0.5 * w]),
                          np.column_stack([xv[::-1], yc - 0.5 * w[::-1]])])

    p = OUTDIR / name
    try:
        if gdstk is not None:
            lib = gdstk.Library(unit=1e-6, precision=1e-9)
            c = lib.new_cell(cellname)
            for yc, w in ((yt, tw), (yb, bw)):
                c.add(gdstk.Polygon(poly(x, yc, w), layer=GDS_LAYER,
                                    datatype=GDS_DATATYPE))
            lib.write_gds(str(p))
        else:
            lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
            c = lib.new_cell(cellname)
            for yc, w in ((yt, tw), (yb, bw)):
                c.add(gdspy.Polygon(poly(x, yc, w), layer=GDS_LAYER,
                                    datatype=GDS_DATATYPE))
            lib.write_gds(str(p))
    except Exception as exc:
        print(f"  GDS write failed for {name} ({type(exc).__name__}: {exc})")
        return None
    return p


def make_plots(M49, M46, cands, d49, bound_rows, prefix="v49"):
    lam = M49["report"]["lam"]
    r49, r46 = M49["report"], M46["report"]
    x, tw, bw = d49.dense_profile()
    xa, ta, ba, xt, tt, bt = split_active_trim(x, tw, bw)

    plt.figure(figsize=(11, 6))
    plt.axhline(0.0, color="k", lw=1.4, label="Target (exact pi)")
    for g, st in [(0.05, ":"), (0.10, "-."), (0.20, "--")]:
        plt.axhline(+g, color="0.55", ls=st, lw=1.0, label=f"+/-{g:.2f} deg")
        plt.axhline(-g, color="0.55", ls=st, lw=1.0)
    plt.plot(lam, r46["phase_err_deg"], "s-", color="tab:red", ms=4, lw=1.3,
             label=f"V46: {M46['worst_phase_primary_deg']:.4f} deg here, "
                   f"{M46['worst_phase_deg']:.4f} worst case")
    plt.plot(lam, r49["phase_err_deg"], "o-", color="tab:blue", ms=4, lw=1.6,
             label=f"V49 {M49['mode']}: {M49['worst_phase_primary_deg']:.4f} deg "
                   f"here, {M49['worst_phase_deg']:.4f} worst case")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Phase error from pi (deg)")
    plt.title(f"Differential phase error vs pi, T = {T_OPT_C:g} C, "
              f"TAPER_MODE = {M49['mode']}")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_phase_error_vs_wavelength.png", dpi=180)
    lim = max(0.06, 1.35 * M49["worst_phase_primary_deg"])
    plt.ylim(-lim, lim)
    plt.title("Phase error, zoomed to the V49 scale")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_phase_error_vs_wavelength_zoom.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    plt.semilogy(lam, np.abs(r46["imbalance"]) + 1e-18, "s-", color="tab:red",
                 ms=4, label=f"V46 ({M46['worst_power_imbalance']:.3e})")
    plt.semilogy(lam, np.abs(r49["imbalance"]) + 1e-18, "o-", color="tab:blue",
                 ms=4, label=f"V49 {M49['mode']} "
                             f"({M49['worst_power_imbalance']:.3e})")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("|normalised power imbalance|  (SURROGATE)")
    plt.title("SURROGATE power imbalance. In symmetrized/none mode the taper "
              "contribution cancels exactly,\nso this is purely the straight-"
              "section width asymmetry. NOT a measured transmission imbalance.")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_power_imbalance_vs_wavelength.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    plt.plot(lam, r49["top_power"], "o-", ms=4, label="V49 top arm")
    plt.plot(lam, r49["bot_power"], "s-", ms=4, label="V49 bottom arm")
    plt.plot(lam, r46["top_power"], "^--", ms=3, alpha=0.6, label="V46 top arm")
    plt.plot(lam, r46["bot_power"], "v--", ms=3, alpha=0.6, label="V46 bottom arm")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Surrogate output power (fraction)")
    plt.title("Top / bottom surrogate output power vs wavelength")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_output_power_vs_wavelength.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    plt.plot(lam, r49["top_loss"], "o-", ms=4, label="V49 top arm")
    plt.plot(lam, r49["bot_loss"], "s-", ms=4, label="V49 bottom arm")
    plt.plot(lam, r49["mean_loss"], "^-", ms=4, label="V49 mean")
    plt.plot(lam, r46["mean_loss"], "--", alpha=0.7, label="V46 mean")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Accumulated surrogate loss (nepers)")
    plt.title("Top / bottom accumulated SURROGATE loss over the full 4500 um")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_accumulated_loss_vs_wavelength.png", dpi=180)
    plt.close()

    for fn, (xx, t_, b_, ttl, xl, vline) in {
        f"{prefix}_active_width_profile.png":
            (xa, ta, ba, f"Active APS width profile (0 - {ACTIVE_LENGTH_UM:g} um)",
             "x in active APS section (um)", None),
        f"{prefix}_trim_width_profile.png":
            (xt, tt, bt, f"Physical trim width profile ({TRIM_LENGTH_UM:g} um, "
                         f"inside the phase and loss integrals)",
             "x in trim section (um)", None),
        f"{prefix}_full_width_profile.png":
            (x, tw, bw, f"Full {TOTAL_LENGTH_UM:g} um profile, "
                        f"TAPER_MODE = {M49['mode']}",
             "x along device (um)", ACTIVE_LENGTH_UM),
    }.items():
        plt.figure(figsize=(12, 5))
        plt.plot(xx, t_, lw=1.5, label="Top arm")
        plt.plot(xx, b_, lw=1.5, label="Bottom arm")
        plt.axhline(INPUT_OUTPUT_WIDTH_UM, color="k", ls="--", lw=1.0,
                    label=f"I/O width {INPUT_OUTPUT_WIDTH_UM:g} um")
        if vline is not None:
            plt.axvline(vline, color="0.4", ls=":", lw=1.2,
                        label="active / trim boundary")
        if M49["mode"] == "symmetrized" and vline is not None:
            xs_mirror = d49.Lp.sum() + d49.Ltw().sum()
            plt.axvline(xs_mirror, color="tab:green", ls="-.", lw=1.2,
                        label="start of mirrored taper block")
        plt.xlabel(xl)
        plt.ylabel("Waveguide width (um)")
        plt.title(ttl)
        plt.grid(True, alpha=0.35)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(OUTDIR / fn, dpi=180)
        plt.close()

    plt.figure(figsize=(9, 6))
    if bound_rows:
        gg = [r[0] for r in bound_rows if r[0] is not None]
        bb = [r[1] for r in bound_rows if r[0] is not None]
        if gg:
            plt.plot(gg, bb, "k-", lw=1.6, label="global LP bound")
    cx = [c["worst_power_imbalance"] for c in cands]
    cy = [c["worst_phase_deg"] for c in cands]
    plt.plot(cx, cy, "o", ms=6, color="tab:blue", label="V49 candidates")
    plt.plot([M46["worst_power_imbalance"]], [M46["worst_phase_deg"]], "s",
             ms=11, color="tab:red", label="V46")
    plt.plot([M49["worst_power_imbalance"]], [M49["worst_phase_deg"]], "*",
             ms=18, color="tab:green", label="V49 accepted")
    plt.axhline(PHASE_GOAL_WORST_DEG, color="0.5", ls=":",
                label=f"phase goal {PHASE_GOAL_WORST_DEG} deg")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("worst |surrogate power imbalance|")
    plt.ylabel("worst |phase error from pi| (deg)")
    plt.title(f"Pareto front, TAPER_MODE = {M49['mode']}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_pareto_front.png", dpi=180)
    plt.close()


# ===========================================================================
# 12. MAIN
# ===========================================================================

def run_mode(mode, table, loss, loss_legacy, bank_opt, bank_ver, bank_rep,
             verbose=True):
    """Full search for one taper mode. Returns (candidates, bound_rows)."""
    def M(d):
        return metrics(d, table, loss, bank_ver, bank_rep)

    def ML(d):
        return metrics(d, table, loss_legacy, bank_ver, bank_rep)

    # representative plateau mass for the bound: windows cost length
    probe = Design([2.4, 6.8], [6.8, 2.4], [100.0, 100.0], mode)
    mass_guess = TOTAL_LENGTH_UM - window_multiplier(mode) * 7.0 * WINDOW_LENGTH_UM
    mass_guess = max(mass_guess, 0.5 * TOTAL_LENGTH_UM)

    bound_rows = []
    if verbose:
        print(f"\n  STAGE 0 global bound, plateau mass ~{mass_guess:.0f} um")
        print(f"    {'imbalance cap':>14s}   {'global bound (deg)':>19s}")
    for cap in POWER_CAPS:
        s = stage0_measure_lp(table, loss, bank_opt, power_guard=cap,
                              mass=mass_guess)
        if s is None:
            continue
        bound_rows.append((cap, s["bound_deg"]))
        if verbose:
            lbl = "none (phase only)" if cap is None else f"{cap:.1e}"
            print(f"    {lbl:>14s}   {s['bound_deg']:19.5f}")

    cands = []
    for cap in POWER_CAPS:
        s = stage0_measure_lp(table, loss, bank_opt, power_guard=cap,
                              mass=mass_guess)
        if s is None:
            continue
        for N in N_SECTION_OPTIONS:
            seed = seed_design(s, N, mode)
            if seed is None:
                continue
            d, _ = length_lp(seed, bank_opt, loss, mode="phase", cap=cap)
            if d is None:
                d, _ = length_lp(seed, bank_opt, loss, mode="phase")
                if d is None:
                    continue
            d, _, _ = slp(d, table, bank_opt, loss, mode="phase", cap=cap)
            pol, _ = length_lp(d, bank_opt, loss, mode="phase", cap=cap)
            if pol is not None:
                d = pol
            mm, mml = M(d), ML(d)
            cands.append({"label": f"{mode}_N{d.n}_cap"
                                   + ("none" if cap is None else f"{cap:.0e}"),
                          "design": d, "M": mm, "ML": mml, "cap": cap,
                          "n_req": N})
            if verbose:
                ok = "" if (cap is None or mm["worst_power_imbalance"]
                            <= cap * 1.001) else "  << cap missed"
                print(f"    N={d.n:2d} cap={'none' if cap is None else f'{cap:.0e}':>7s}"
                      f"  worst {mm['worst_phase_deg']:9.5f} deg  "
                      f"rms {mm['rms_phase_deg']:8.5f}  "
                      f"imb {mm['worst_power_imbalance']:9.3e}  "
                      f"loss {mm['mean_surrogate_loss_np']:.3e}  "
                      f"Lmin={mm['min_plateau_um']:6.1f}{ok}")
    return cands, bound_rows


def main():
    t0 = time.time()
    OUTDIR.mkdir(exist_ok=True, parents=True)

    print("=" * 78)
    print(f"V49 achromatic two-arm pi phase shifter -- TAPER_MODE = {TAPER_MODE}")
    print("=" * 78)

    table = NeffTable()
    loss = SurrogateLoss(table, kw=DESIGN_KW, label="v49_design")
    loss_legacy = SurrogateLoss(table, kw=LEGACY_KW, label="v46_legacy")
    bank_opt = ModelBank(table, LAMS_OPT)
    bank_ver = ModelBank(table, LAMS_VERIFY)
    bank_rep = ModelBank(table, LAMS_REPORT)

    print(f"\nneff source: {table.source}")
    print(f"  widths      : {table.widths[0]:.3f} to {table.widths[-1]:.3f} um "
          f"({len(table.widths)} values, step "
          f"{np.median(np.diff(table.widths)):.3f} um)")
    print(f"  wavelengths : {table.lams[0]:.3f} to {table.lams[-1]:.3f} um "
          f"({len(table.lams)} values, step "
          f"{np.median(np.diff(table.lams)):.3f} um)")
    print(f"  max|neff_imag| = {table.neff_imag_absmax:.3e} "
          f"({table.neff_imag_nonzero} of {table.N.size} rows nonzero)")
    if not table.neff_imag_is_meaningful:
        print("  *** neff_imag is NUMERICALLY ZERO: no measured loss in this")
        print("      table. All loss/power numbers are SURROGATE. ***")
    print(f"  loss model for optimisation : KW = {loss.KW:.2e}  "
          f"q(6.8)/q(2.4) = {loss.q_ratio():.2f}")
    print(f"  loss model for V46 continuity: KW = {loss_legacy.KW:.2e}  "
          f"q(6.8)/q(2.4) = {loss_legacy.q_ratio():.2f}")

    def M(d):
        return metrics(d, table, loss, bank_ver, bank_rep)

    def ML(d):
        return metrics(d, table, loss_legacy, bank_ver, bank_rep)

    print("\n" + "-" * 78)
    print("V46 reference, re-scored on this evaluator (inline tapers, as built)")
    print("-" * 78)
    d46 = build_v46_design()
    M46, M46L = M(d46), ML(d46)
    print(f"  total length            : {M46['total_length_um']:.6f} um")
    print(f"  worst phase (ensemble)  : {M46['worst_phase_deg']:.6f} deg")
    print(f"  worst phase (bilinear)  : "
          f"{M46['per_model']['bilin']['worst_phase_deg']:.6f} deg   "
          f"[V46's own model; it reported {V46_REPORTED['worst_phase_deg']:.6f}]")
    print(f"  rms phase (report grid) : {M46['rms_phase_primary_deg']:.6f} deg")
    print(f"  worst imbalance         : {M46['worst_power_imbalance']:.6e} "
          f"[design model] / {M46L['worst_power_imbalance']:.6e} [legacy]")
    print(f"  taper phase asymmetry   : {M46['window_opd_diff']:.6e} um of OPD "
          f"-> this is what symmetrizing removes")

    print("\n" + "-" * 78)
    print(f"SEARCH, TAPER_MODE = {TAPER_MODE}")
    print("-" * 78)
    cands, bound_rows = run_mode(TAPER_MODE, table, loss, loss_legacy,
                                 bank_opt, bank_ver, bank_rep)
    if not cands:
        raise SystemExit("no candidate was produced; relax MIN_PLATEAU_LENGTH_UM "
                         "or N_SECTION_OPTIONS")

    # verify the cancellation claim numerically
    worst_cancel = max(c["M"]["window_opd_diff"] for c in cands)
    worst_cancel_l = max(c["M"]["window_loss_diff"] for c in cands)
    print(f"\n  taper cancellation check over all candidates:")
    print(f"    max |OPD_top - OPD_bot| over the windows  = {worst_cancel:.3e} um")
    print(f"    max |loss_top - loss_bot| over the windows = {worst_cancel_l:.3e} np")
    if TAPER_MODE in ("symmetrized", "none"):
        print("    both must be ~0 by construction; they are, so taper phase and")
        print("    taper loss are genuinely common-mode in this design.")

    # ---------------- acceptance ----------------
    print("\n" + "-" * 78)
    print("ACCEPTANCE")
    print("-" * 78)
    tightest = min([c for c in POWER_CAPS if c is not None], default=None)

    def goals_met(m):
        return (m["worst_phase_deg"] <= PHASE_GOAL_WORST_DEG
                and m["rms_phase_deg"] <= PHASE_GOAL_RMS_DEG)

    def beats46(m):
        return (m["worst_phase_deg"] < M46["worst_phase_deg"]
                and m["rms_phase_deg"] < M46["rms_phase_deg"]
                and m["worst_power_imbalance"] < M46["worst_power_imbalance"]
                and m["rms_power_imbalance"] < M46["rms_power_imbalance"])

    def beats46L(mL):
        return (mL["worst_power_imbalance"] < M46L["worst_power_imbalance"]
                and mL["rms_power_imbalance"] < M46L["rms_power_imbalance"])

    tiers = {
        "A": [c for c in cands if goals_met(c["M"]) and tightest is not None
              and c["M"]["worst_power_imbalance"] <= tightest * 1.001],
        "C": [c for c in cands if beats46(c["M"]) and beats46L(c["ML"])],
        "B": [c for c in cands if goals_met(c["M"])],
    }
    accepted, tier = None, "D"
    for t in ACCEPT_POLICY:
        if t == "D":
            break
        got = tiers.get(t, [])
        if got:
            key = ((lambda c: c["M"]["worst_power_imbalance"]) if t in ("A", "B")
                   else (lambda c: c["M"]["worst_phase_deg"]))
            accepted, tier = min(got, key=key), t
            break
    if accepted is None:
        accepted = {"label": "v46_seed_preserved", "design": d46, "M": M46,
                    "ML": M46L, "cap": None,
                    "note": "no V49 candidate beat V46 under either loss model"}
        print("  *** No V49 candidate beat V46 on every metric under either loss")
        print("      model. The V46 seed is preserved and emitted unchanged. ***")
    else:
        accepted["note"] = (f"cap=" + ("none" if accepted["cap"] is None
                                       else f"{accepted['cap']:.1e}"))
        print(f"  accepted tier {tier}: {accepted['label']}")
        if tier == "B":
            print("  NOTE tier B: phase goals met; the surrogate imbalance may")
            print("       regress versus V46. Check the table in the summary.")

    d49, M49, M49L = accepted["design"], accepted["M"], accepted["ML"]

    # flattest candidate, regardless of power, as the alternative
    flat = min(cands, key=lambda c: c["M"]["worst_phase_deg"])
    alt = flat if flat["label"] != accepted["label"] else None
    if alt is not None:
        print(f"  alternative (flattest phase) emitted as 'v49alt_*': "
              f"{alt['label']}  worst {alt['M']['worst_phase_deg']:.5f} deg, "
              f"imb {alt['M']['worst_power_imbalance']:.3e}")

    # ---------------- manual-taper penalty, for mode "none" ----------------
    pen = None
    if TAPER_MODE == "none":
        print("\n" + "-" * 78)
        print("WHAT INSERTING TAPERS BY HAND WILL COST (accepted design)")
        print("-" * 78)
        print(f"    {'taper um':>9s} {'as-is worst':>12s} {'as-is rms':>10s} "
              f"{'relength worst':>15s} {'relength rms':>13s}")
        pen = manual_taper_penalty(d49, table, loss, bank_opt, bank_ver, bank_rep)
        for Lt, w1, r1, w2, r2 in [(p[0], p[1], p[2], p[3], p[4]) for p in pen]:
            f = lambda v: "   --" if v is None else f"{v:.5f}"
            print(f"    {Lt:9.1f} {f(w1):>12s} {f(r1):>10s} {f(w2):>15s} "
                  f"{f(r2):>13s}")
        print("    'as-is' = tapers inserted, plateaus rescaled to keep 4500 um,")
        print("    nothing else retuned. 'relength' = plateau lengths re-solved")
        print("    by the exact LP afterwards, widths untouched.")

    # ---------------- second mode for comparison ----------------
    other = None
    if ALSO_RUN_MODE and ALSO_RUN_MODE != TAPER_MODE:
        print("\n" + "-" * 78)
        print(f"COMPARISON RUN, TAPER_MODE = {ALSO_RUN_MODE}")
        print("-" * 78)
        oc, ob = run_mode(ALSO_RUN_MODE, table, loss, loss_legacy,
                          bank_opt, bank_ver, bank_rep)
        if oc:
            best_goal = [c for c in oc if goals_met(c["M"])]
            other = (min(best_goal, key=lambda c: c["M"]["worst_power_imbalance"])
                     if best_goal else min(oc, key=lambda c: c["M"]["worst_phase_deg"]))
            print(f"  best for {ALSO_RUN_MODE}: {other['label']}  "
                  f"worst {other['M']['worst_phase_deg']:.5f} deg  "
                  f"rms {other['M']['rms_phase_deg']:.5f}  "
                  f"imb {other['M']['worst_power_imbalance']:.3e}")

    # ---------------- outputs ----------------
    print("\n" + "-" * 78)
    print("WRITING OUTPUTS to", OUTDIR.resolve())
    print("-" * 78)
    emit_design("v49", d49, M49)
    gds = write_gds(d49, "v49_achromatic_pi_layer1.gds", "APS_V49")
    if alt is not None:
        emit_design("v49alt", alt["design"], alt["M"])
        write_gds(alt["design"], "v49alt_achromatic_pi_layer1.gds", "APS_V49_ALT")
    if other is not None:
        emit_design(f"v49_{ALSO_RUN_MODE}", other["design"], other["M"])
        write_gds(other["design"],
                  f"v49_{ALSO_RUN_MODE}_achromatic_pi_layer1.gds",
                  f"APS_V49_{ALSO_RUN_MODE.upper()}")

    hk = ["label", "mode", "n_sections", "worst_phase_deg", "rms_phase_deg",
          "p95_phase_deg", "mean_phase_deg", "worst_power_imbalance",
          "rms_power_imbalance", "mean_surrogate_loss_np", "min_plateau_um",
          "window_len_um", "width_min_um", "width_max_um", "width_excursion_um",
          "max_abs_slope", "n_widths_near_table_edge", "n_plateaus_at_min_length",
          "total_length_um", "window_opd_diff", "window_loss_diff"]
    save_csv(OUTDIR / "v49_pareto_and_history.csv",
             hk + ["requested_cap", "legacy_worst_power_imbalance"],
             [[c["M"].get(k, c["M"].get(k)) for k in hk]
              + [("none" if c["cap"] is None else c["cap"]),
                 c["ML"]["worst_power_imbalance"]] for c in cands])
    save_csv(OUTDIR / "v49_global_lp_bound_curve.csv",
             ["imbalance_cap", "global_min_worst_phase_deg"],
             [[("none" if g is None else g), b] for g, b in bound_rows])
    save_csv(OUTDIR / "v49_all_candidate_sections.csv",
             ["candidate", "section_index", "top_width_um", "bottom_width_um",
              "straight_length_um"],
             [[c["label"], i + 1, c["design"].a[i], c["design"].b[i],
               c["design"].Lp[i]] for c in cands for i in range(c["design"].n)])

    make_plots(M49, M46, [c["M"] for c in cands], d49, bound_rows)

    # ---------------- summary ----------------
    L = []
    P = L.append
    P(f"V49 achromatic two-arm pi phase shifter -- TAPER_MODE = {TAPER_MODE}")
    P("=" * 76)
    P(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}, "
      f"runtime {time.time()-t0:.1f} s")
    P(f"accepted: {accepted['label']}  (tier {tier}, {accepted.get('note','')})")
    P("")
    P("WHAT THIS MODE DOES")
    if TAPER_MODE == "symmetrized":
        P("  Every transition window of each arm is COPIED onto the end of the")
        P("  opposite arm, so each arm traverses the union of both arms' window")
        P("  content. Taper phase and taper loss are therefore identical between")
        P("  the arms and cancel EXACTLY in every differential quantity.")
        P("  Verified numerically, not assumed:")
        P(f"    max |OPD_top - OPD_bot| over windows  = {M49['window_opd_diff']:.3e} um")
        P(f"    max |loss_top - loss_bot| over windows = {M49['window_loss_diff']:.3e} np")
        P("  Consequence: the differential phase is a pure sum over straight")
        P("  plateaus, so NO taper neff approximation enters it at all.")
        P(f"  Length budget: sum(plateaus) + 2*sum(windows) = "
          f"{TOTAL_LENGTH_UM:.1f} um, because each arm carries both window sets.")
    elif TAPER_MODE == "none":
        P("  No tapers in the model or the emitted geometry: abrupt width steps")
        P("  between straight plateaus. The differential phase is exactly")
        P("    D(lam) = sum_i L_i * [neff(a_i,lam) - neff(b_i,lam)]")
        P("  with neff taken at single widths, so there is no taper neff")
        P("  approximation and no taper shape assumption anywhere.")
        P("  Insert your own tapers afterwards; the penalty table below says")
        P("  what each taper length costs.")
    else:
        P("  Tapers inline and different between the arms: the V47/V48 behaviour.")
        P("  Taper phase and loss do NOT cancel, so the differential phase does")
        P("  depend on how neff is interpolated inside a taper.")
    P("")
    P("HOW neff IS ESTIMATED WHERE THE WIDTH IS NOT ON THE TABLE GRID")
    P(f"  The table has {len(table.widths)} widths on a "
      f"{np.median(np.diff(table.widths)):.2f} um grid and "
      f"{len(table.lams)} wavelengths on a "
      f"{np.median(np.diff(table.lams)):.3f} um grid.")
    P("  Straight plateaus: the optimiser is free to pick any width, so neff is")
    P("  interpolated in BOTH width and wavelength.")
    P("  Tapers (inline mode, or for the absolute loss in symmetrized mode):")
    P("    w(u) = w0 + (w1-w0)*(0.5 - 0.5*cos(pi*u)), u = x/L, and")
    P(f"    phase = (2*pi/lam) * L * sum_g omega_g * neff(w(u_g), lam) on")
    P(f"    {N_GAUSS_TAPER} Gauss-Legendre nodes. At 24 nodes the quadrature error is")
    P("    ~1e-12 rad, so it contributes nothing.")
    P("  The interpolation is the only real approximation, so FIVE admissible")
    P(f"  interpolants are evaluated {VERIFY_MODELS} and the optimiser minimises")
    P(f"  the WORST CASE over {ROBUST_MODELS}. Per-interpolant results here:")
    for k in VERIFY_MODELS:
        v = M49["per_model"][k]
        held = "" if k in ROBUST_MODELS else "   (held out of the fit)"
        P(f"    {k:<9s} worst {v['worst_phase_deg']:.6f} deg   "
          f"rms {v['rms_phase_deg']:.6f} deg{held}")
    P("  Every phase number quoted below is the worst case across that set.")
    P("  Run taper_neff_validation_twostraight_onetaper.py for a straight/taper/")
    P("  straight structure you can hand to BPM or FemSIM to check this directly.")
    P("")
    P("SEGMENT COUNT")
    P(f"  accepted design uses {M49['n_sections']} straight plateau segments")
    P(f"  per arm ({M49['n_sections']}+1 = {M49['n_sections']+1} transitions).")
    P(f"  N was swept over {N_SECTION_OPTIONS}; it is a searched parameter, not")
    P("  a fixed one. Candidates at every N are in v49_pareto_and_history.csv.")
    P("")
    P("GLOBAL BOUND (linear program over the width measure, not a local optimum)")
    P(f"  {'imbalance cap':>14s}   {'global bound (deg)':>19s}")
    for g, b in bound_rows:
        P(f"  {('none' if g is None else f'{g:.1e}'):>14s}   {b:19.5f}")
    P("")
    P("V46 vs V49 -- HEAD TO HEAD (same evaluator, same loss model)")
    P("-" * 76)
    P(f"  {'metric':<34s}{'V46':>15s}{'V49':>15s}   verdict")

    def row(nm, k, fmt="{:.6f}", lower=True):
        a_, b_ = M46[k], M49[k]
        eq = abs(b_ - a_) <= 1e-15 * max(1.0, abs(a_))
        better = (b_ < a_) if lower else (b_ > a_)
        v = "same" if eq else ("BETTER" if better else "WORSE")
        if not eq and a_ != 0 and b_ != 0:
            v += f"  ({a_/b_:.2f}x)" if better else f"  ({b_/a_:.2f}x)"
        P(f"  {nm:<34s}{fmt.format(a_):>15s}{fmt.format(b_):>15s}   {v}")

    row("RMS phase error (deg)", "rms_phase_deg")
    row("worst phase error (deg)", "worst_phase_deg")
    row("P95 phase error (deg)", "p95_phase_deg")
    P(f"  {'mean phase error (deg)':<34s}{M46['mean_phase_deg']:>15.6f}"
      f"{M49['mean_phase_deg']:>15.6f}   "
      f"{'BETTER' if abs(M49['mean_phase_deg'])<abs(M46['mean_phase_deg']) else 'WORSE'}"
      f" (closer to 0)")
    row("RMS power imbalance", "rms_power_imbalance", fmt="{:.6e}")
    row("worst power imbalance", "worst_power_imbalance", fmt="{:.6e}")
    row("mean surrogate loss (nepers)", "mean_surrogate_loss_np", fmt="{:.6e}")
    row("worst arm loss (nepers)", "worst_arm_loss_np", fmt="{:.6e}")
    P("")
    P(f"  under V46'S OWN legacy loss model (KW = {loss_legacy.KW:.1e}):")
    for nm, k in [("RMS power imbalance", "rms_power_imbalance"),
                  ("worst power imbalance", "worst_power_imbalance"),
                  ("mean surrogate loss (nepers)", "mean_surrogate_loss_np")]:
        a_, b_ = M46L[k], M49L[k]
        v = "BETTER" if b_ < a_ else ("same" if b_ == a_ else "WORSE")
        rr = (a_ / b_) if (b_ and b_ < a_) else ((b_ / a_) if a_ else 0.0)
        P(f"  {nm:<34s}{a_:>15.6e}{b_:>15.6e}   {v}  ({rr:.2f}x)")
    P("  Phase does not depend on the loss model, so it is not repeated.")
    P("")
    P("  geometry")
    for nm, k, f in [("straight segments", "n_sections", "{:.0f}"),
                     ("min straight length (um)", "min_plateau_um", "{:.3f}"),
                     ("max straight length (um)", "max_plateau_um", "{:.3f}"),
                     ("window length (um)", "window_len_um", "{:.3f}"),
                     ("min width (um)", "width_min_um", "{:.4f}"),
                     ("max width (um)", "width_max_um", "{:.4f}"),
                     ("total width excursion (um)", "width_excursion_um", "{:.3f}"),
                     ("max |dw/dx|", "max_abs_slope", "{:.5f}"),
                     ("widths near table edge", "n_widths_near_table_edge", "{:.0f}"),
                     ("straights at the length floor", "n_plateaus_at_min_length", "{:.0f}"),
                     ("total length (um)", "total_length_um", "{:.4f}")]:
        P(f"  {nm:<34s}{f.format(M46[k]):>15s}{f.format(M49[k]):>15s}")
    P("")
    P("GOALS")
    P(f"  worst phase <= {PHASE_GOAL_WORST_DEG:g} deg : "
      f"{'MET' if M49['worst_phase_deg'] <= PHASE_GOAL_WORST_DEG else 'not met'}"
      f" ({M49['worst_phase_deg']:.6f})")
    P(f"  rms phase   <= {PHASE_GOAL_RMS_DEG:g} deg : "
      f"{'MET' if M49['rms_phase_deg'] <= PHASE_GOAL_RMS_DEG else 'not met'}"
      f" ({M49['rms_phase_deg']:.6f})")
    P(f"  worst phase <= {PHASE_LOOSE_WORST_DEG:g} deg : "
      f"{'MET' if M49['worst_phase_deg'] <= PHASE_LOOSE_WORST_DEG else 'not met'}")
    if alt is not None:
        am = alt["M"]
        P("")
        P("ALTERNATIVE: FLATTEST PHASE (emitted as v49alt_*)")
        P(f"  {alt['label']}: worst {am['worst_phase_deg']:.6f} deg, "
          f"rms {am['rms_phase_deg']:.6f} deg,")
        P(f"  P95 {am['p95_phase_deg']:.6f} deg, worst imbalance "
          f"{am['worst_power_imbalance']:.6e}, {am['n_sections']} segments")
    if pen is not None:
        P("")
        P("COST OF INSERTING TAPERS BY HAND (this accepted taper-free design)")
        P(f"  {'taper um':>9s} {'as-is worst':>12s} {'as-is rms':>10s} "
          f"{'relength worst':>15s} {'relength rms':>13s}")
        for Lt, w1, r1, w2, r2, _ in pen:
            f = lambda v: "   --" if v is None else f"{v:.5f}"
            P(f"  {Lt:9.1f} {f(w1):>12s} {f(r1):>10s} {f(w2):>15s} {f(r2):>13s}")
        P("  as-is    : tapers inserted, straights rescaled to keep 4500 um")
        P("  relength : straight lengths re-solved by the exact LP, widths fixed")
        P("  Use 'relength' if you will let me retune lengths after you choose a")
        P("  taper length; use 'as-is' if the geometry must stay exactly as given.")
    if other is not None:
        om = other["M"]
        P("")
        P(f"COMPARISON: TAPER_MODE = {ALSO_RUN_MODE} "
          f"(emitted as v49_{ALSO_RUN_MODE}_*)")
        P(f"  {other['label']}: worst {om['worst_phase_deg']:.6f} deg, "
          f"rms {om['rms_phase_deg']:.6f} deg, "
          f"imb {om['worst_power_imbalance']:.6e}, {om['n_sections']} segments")
    P("")
    P("LOSS / POWER WARNING")
    P(f"  neff_imag max |.| = {table.neff_imag_absmax:.3e}: numerically zero, so")
    P("  there is NO measured loss in this table and every loss and power number")
    P("  here is a SURROGATE. Its width term makes WIDE guides lossy")
    P(f"  (q(6.8)/q(2.4) = {loss.q_ratio():.2f}), which is backwards for weakly-")
    P("  guiding multimode ridges. Replace SurrogateLoss.width_term() with real")
    P("  BPM/FDTD/FemSIM/RSoft transmission data before treating any power")
    P("  number as a spec. The PHASE numbers come from the measured neff table")
    P("  and do not depend on the loss model at all.")
    P("")
    P("FILES")
    for nm in sorted(p.name for p in OUTDIR.iterdir() if p.is_file()):
        P(f"  {nm}")
    if gds is None:
        P("  (no GDS: install gdstk or gdspy; the profile CSVs hold the geometry)")
    (OUTDIR / "summary_v49.txt").write_text("\n".join(L) + "\n")

    print(f"  summary_v49.txt written ({len(L)} lines)")
    print(f"\nDONE in {time.time()-t0:.1f} s. Outputs in {OUTDIR.resolve()}")
    print(f"  mode {TAPER_MODE}, accepted {accepted['label']} (tier {tier})")
    print(f"  worst phase {M49['worst_phase_deg']:.6f} deg  (V46 "
          f"{M46['worst_phase_deg']:.6f})")
    print(f"  rms   phase {M49['rms_phase_deg']:.6f} deg  (V46 "
          f"{M46['rms_phase_deg']:.6f})")
    print(f"  worst imbalance {M49['worst_power_imbalance']:.6e}  (V46 "
          f"{M46['worst_power_imbalance']:.6e})")
    print(f"  segments {M49['n_sections']}, min straight "
          f"{M49['min_plateau_um']:.1f} um, total "
          f"{M49['total_length_um']:.3f} um")


if __name__ == "__main__":
    main()
