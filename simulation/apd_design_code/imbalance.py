"""Power imbalance vs wavelength for the three designs, from Anusha's taper data.

Each boundary's taper is a different size in the top arm than in the bottom arm
(w_top,i -> w_top,i+1 versus w_bot,i -> w_bot,i+1), so their losses do not
cancel: the difference is the imbalance.

LOSS MODEL
  Anusha measured T_fund for 7 representative steps at 3 wavelengths and 3-4
  taper lengths (69 points).  Her data collapses well onto

      loss_dB(dw, L, lam) = k(lam) * dw^p / L^q

  which is fitted here in log space over all 69 points.  The fit is reported
  with its residual so the extrapolation is visible, and losses for the actual
  design transitions are then read off it.

  Her data also shows loss falling with wavelength for the big steps and rising
  for the small ones, so k is fitted per wavelength and interpolated.

WHAT THIS IS NOT
  Only the TAPERS are counted.  Propagation loss in the plateaus is not in her
  data (it would be common-mode to first order anyway, since both arms span the
  same length).  Bend loss, sidewall scattering and the 2 mm trim phase
  shifters are all excluded.
"""
import csv
import numpy as np

WL3 = np.array([1450.0, 1550.0, 1650.0])


def load_measured(path="../taper_runs_for_Anusha_filled.csv"):
    """(dw, crosses, L, lam) -> loss_dB, from the A-D cases only."""
    out = []
    for r in csv.DictReader(open(path)):
        if not r["T_fund_dB"] or r["case"][0] in "EF":
            continue
        a, b = float(r["w_start_um"]), float(r["w_end_um"])
        out.append(dict(dw=abs(b - a), crosses=min(a, b) < 5.6 < max(a, b),
                        L=float(r["taper_length_um"]), lam=float(r["wavelength_nm"]),
                        loss=-float(r["T_fund_dB"])))
    return out


def fit_model(meas):
    """log(loss) = log k(lam) + p log dw - q log L.  Returns (p, q, k_per_lam, rms)."""
    rowsA, rhs = [], []
    lam_idx = {w: i for i, w in enumerate(WL3)}
    for m in meas:
        if m["loss"] <= 0:
            continue
        r = np.zeros(2 + len(WL3))
        r[0] = np.log(m["dw"])
        r[1] = -np.log(m["L"])
        r[2 + lam_idx[m["lam"]]] = 1.0
        rowsA.append(r)
        rhs.append(np.log(m["loss"]))
    A = np.array(rowsA)
    b = np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    p, q = sol[0], sol[1]
    logk = sol[2:]
    pred = A @ sol
    rms_dex = float(np.sqrt(((pred - b) ** 2).mean()))
    return p, q, np.exp(logk), rms_dex


def loss_db(dw, L, lam_um, p, q, kv):
    """Taper loss in dB at one wavelength (um), interpolating k over wavelength."""
    if dw < 1e-9 or L <= 0:
        return 0.0
    k = np.interp(lam_um * 1000.0, WL3, kv)
    return float(k * dw ** p / L ** q)


def arm_taper_losses(widths, tapers, guide=3.4):
    """[(dw, L), ...] for one arm's boundaries."""
    e = [guide] + list(widths) + [guide]
    return [(abs(e[i + 1] - e[i]), tapers[i]) for i in range(len(tapers))]


def imbalance_curve(top, bot, tapers, lams, p, q, kv):
    """(loss_top, loss_bot, imbalance) in dB at each wavelength."""
    lt_spec = arm_taper_losses(top, tapers)
    lb_spec = arm_taper_losses(bot, tapers)
    lt, lb = [], []
    for lam in lams:
        lt.append(sum(loss_db(dw, L, lam, p, q, kv) for dw, L in lt_spec))
        lb.append(sum(loss_db(dw, L, lam, p, q, kv) for dw, L in lb_spec))
    lt = np.array(lt)
    lb = np.array(lb)
    return lt, lb, lt - lb


def null_floor_db(imb_db):
    """Null floor from amplitude imbalance alone (phase perfect):
    I/Imax = [(1-r)/(1+r)]^2 with r the amplitude ratio."""
    r = 10 ** (-np.abs(imb_db) / 20.0)
    return 20 * np.log10(np.clip((1 - r) / (1 + r), 1e-12, None))


if __name__ == "__main__":
    import apd_v3 as V

    meas = load_measured()
    p, q, kv, rms = fit_model(meas)
    print(f"loss model fitted to {len(meas)} of Anusha's points:")
    print(f"  loss_dB = k(lam) * dw^{p:.3f} / L^{q:.3f}")
    print(f"  k at 1450/1550/1650 nm = {kv[0]:.4f} / {kv[1]:.4f} / {kv[2]:.4f}")
    print(f"  fit residual {rms:.3f} dex (x{np.exp(rms):.2f}) -- order of magnitude only\n")

    designs = [("a_design4_standard_v3", 0.0998),
               ("b_design4_latticeblind_v3", 0.141805),
               ("c_design2_standard_v3", 0.0983)]
    summary = []
    for name, dL in designs:
        rows = list(csv.DictReader(open(name + ".csv")))
        top = [float(r["top_width_um"]) for r in rows]
        bot = [float(r["bottom_width_um"]) for r in rows]
        tp = [float(r["taper_length_um"]) for r in
              csv.DictReader(open(name + "_tapers.csv"))]
        lt, lb, imb = imbalance_curve(top, bot, tp, V.LAM, p, q, kv)
        with open(name + "_imbalance.csv", "w", newline="\n") as f:
            f.write("wavelength_um,loss_top_dB,loss_bot_dB,imbalance_dB,"
                    "null_floor_from_imbalance_dB\n")
            for j, lam in enumerate(V.LAM):
                f.write(f"{lam:.3f},{lt[j]:.5f},{lb[j]:.5f},{imb[j]:+.5f},"
                        f"{null_floor_db(imb[j]):.1f}\n")
        worst = float(np.abs(imb).max())
        summary.append((name, lt.mean(), lb.mean(), worst, null_floor_db(worst)))
        print(f"{name}")
        print(f"  taper loss: top {lt.mean():.4f} dB, bottom {lb.mean():.4f} dB (band mean)")
        print(f"  imbalance {imb.min():+.4f} to {imb.max():+.4f} dB, "
              f"worst |.| {worst:.4f} dB")
        print(f"  null floor from imbalance alone: {null_floor_db(worst):.1f} dB")
        print(f"  wrote {name}_imbalance.csv")

    with open("SUMMARY_imbalance.csv", "w", newline="\n") as f:
        f.write("design,taper_loss_top_dB,taper_loss_bot_dB,worst_imbalance_dB,"
                "null_floor_from_imbalance_dB\n")
        for n, a, b, w, nf in summary:
            f.write(f"{n},{a:.4f},{b:.4f},{w:.4f},{nf:.1f}\n")
    print("\nwrote SUMMARY_imbalance.csv")
