"""Phase difference vs wavelength for the three APD designs, BEFORE lattice and trim.

Two figures:
  APD_phase_raw.png      the phase difference as drawn, i.e. 180 deg + the error
                         in the *_phase.csv files. No trim, no lattice.
  APD_phase_detail.png   the same curves with each one's own mean removed, so the
                         wavelength dependence is visible (the offsets are 100x
                         the shape, which otherwise hides it).

Reads the delivered CSVs, so the plots cannot drift from the shipped numbers.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESIGNS = [
    ("a_design4_standard_v3", "(a) design 4, standard — 3.17 mm", "#2563eb"),
    ("b_design4_latticeblind_v3", "(b) design 4, lattice-blind — 1.96 mm", "#238b57"),
    ("c_design2_standard_v3", "(c) design 2, standard — 3.68 mm", "#cf3f4f"),
]
BIAS_COLS = [("err_bias_-50nm_deg", "-50 nm", ":"),
             ("err_bias_0nm_deg", "nominal", "-"),
             ("err_bias_+50nm_deg", "+50 nm", "--")]


def load(name):
    rows = list(csv.DictReader(open(f"{name}.csv".replace(".csv", "_phase.csv"))))
    lam = np.array([float(r["wavelength_um"]) for r in rows]) * 1000.0
    return lam, {c: np.array([float(r[c]) for r in rows]) for c, _, _ in BIAS_COLS}


# ---------------------------------------------------------------- figure 1: raw
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
for ax, (name, title, colour) in zip(axes, DESIGNS):
    lam, cols = load(name)
    for (c, lbl, ls) in BIAS_COLS:
        ax.plot(lam, 180.0 + cols[c], ls, color=colour, lw=1.8, label=lbl)
    ax.axhline(180.0, color="0.4", lw=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wavelength (nm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="width bias", title_fontsize=8)
axes[0].set_ylabel("phase difference between the arms (°)")
fig.suptitle("APD phase difference vs wavelength — as drawn, no trim, no lattice",
             fontsize=12)
fig.tight_layout()
fig.savefig("APD_phase_raw.png", dpi=160)
print("wrote APD_phase_raw.png")

# ------------------------------------------------- figure 2: shape, offset removed
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
for ax, (name, title, colour) in zip(axes, DESIGNS):
    lam, cols = load(name)
    for (c, lbl, ls) in BIAS_COLS:
        e = cols[c]
        ax.plot(lam, e - e.mean(), ls, color=colour, lw=1.8,
                label=f"{lbl} (offset {e.mean():+.2f}°)")
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wavelength (nm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
axes[0].set_ylabel("deviation from each curve's own mean (°)")
fig.suptitle("Same curves with the constant offset removed — this is the part "
             "the trim heater cannot fix", fontsize=12)
fig.tight_layout()
fig.savefig("APD_phase_detail.png", dpi=160)
print("wrote APD_phase_detail.png")

# ---------------------------------------------------------------- numbers
print(f"\n{'design':<28} {'bias':>8} {'mean':>9} {'min':>9} {'max':>9} {'pk-pk':>9}")
for name, title, _ in DESIGNS:
    lam, cols = load(name)
    for (c, lbl, _) in BIAS_COLS:
        e = cols[c]
        print(f"{name:<28} {lbl:>8} {e.mean():>+9.4f} {e.min():>+9.4f} "
              f"{e.max():>+9.4f} {e.max()-e.min():>9.4f}")
print("\n(all in degrees away from 180; 'pk-pk' is what survives a perfect "
      "constant trim)")
