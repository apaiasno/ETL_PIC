"""Design (c) - design 2, standard metric, corrected n_eff table (v3, 4 Nov 2026).

The long design: best on its own of the three.

    APD length              3675.3 um (sections 2555.3 + tapers 1120)
    path offset             +98.3 nm of extra path in the TOP arm
    worst error after trim  0.0119 deg   (1450-1650 nm, +/-50 nm width bias)
    lattice-blind residual  0.0112 deg
    trim per bias           +1.50 / +0.00 / -1.53 deg  at -50 / 0 / +50 nm

Run:  python make_c_design2_standard.py
Needs: numpy, gdstk  (pip install gdstk)
"""
from _apd_gds import write

SECTIONS = [
    (377.674, 3.20, 4.35),
    (352.729, 6.20, 3.90),
    (232.010, 6.45, 5.05),
    (576.290, 6.10, 4.30),
    (294.073, 2.95, 4.80),
    (130.241, 3.10, 2.40),
    (128.707, 2.90, 5.00),
    (463.585, 2.80, 2.35),
]

TAPERS = [80, 160, 80, 80, 160, 160, 160, 160, 80]

PATH_OFFSET_UM = 0.0983          # + = extra path in the TOP arm

if __name__ == "__main__":
    write("c_design2_standard_v3", SECTIONS, TAPERS, PATH_OFFSET_UM)
