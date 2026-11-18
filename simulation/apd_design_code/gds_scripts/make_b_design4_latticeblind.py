"""Design (b) - design 4, lattice-blind metric, corrected n_eff table (v3, 4 Nov 2026).

Optimised to leave behind only what an MZI lattice can remove (piston, slope,
curvature and the heater's 1/lambda term), so it looks poor on its own and very
good behind a lattice.

    APD length              1964.6 um (sections 804.6 + tapers 1160)
    path offset             +141.8 nm of extra path in the TOP arm
    worst error after trim  0.2293 deg   <- poor alone, by design
    lattice-blind residual  0.0054 deg   <- the number that matters here
    trim per bias           +2.13 / +0.05 / -2.08 deg  at -50 / 0 / +50 nm
    handed to the lattice   0.299 / 0.087 / 0.300 deg  (0.3 deg allowance)

Note this design is 59% taper by length (1160 of 1964.6 um). Estimated taper
loss ~0.02 dB, so that is not disqualifying, but it is worth knowing.

Run:  python make_b_design4_latticeblind.py
Needs: numpy, gdstk  (pip install gdstk)
"""
from _apd_gds import write

SECTIONS = [
    (60.000, 6.65, 2.45),
    (145.746, 5.45, 5.25),
    (158.747, 5.85, 4.50),
    (123.030, 5.90, 4.95),
    (69.133, 6.10, 5.20),
    (91.707, 2.65, 4.75),
    (60.000, 4.20, 4.05),
    (96.265, 3.00, 4.80),
]

TAPERS = [160, 160, 160, 60, 60, 160, 160, 80, 160]

PATH_OFFSET_UM = 0.141805        # + = extra path in the TOP arm

if __name__ == "__main__":
    write("b_design4_latticeblind_v3", SECTIONS, TAPERS, PATH_OFFSET_UM)
