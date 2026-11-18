"""Design (a) - design 4, standard metric, corrected n_eff table (v3, 4 Nov 2026).

    APD length              3173.6 um (sections 2153.6 + tapers 1020)
    path offset             +99.8 nm of extra path in the TOP arm
    worst error after trim  0.0142 deg   (1450-1650 nm, +/-50 nm width bias)
    lattice-blind residual  0.0136 deg
    trim per bias           +1.57 / +0.02 / -1.57 deg  at -50 / 0 / +50 nm

Run:  python make_a_design4_standard.py
Needs: numpy, gdstk  (pip install gdstk)
"""
from _apd_gds import write

# (section length um, top width um, bottom width um)
SECTIONS = [
    (232.753, 3.20, 2.35),
    (323.338, 6.65, 5.05),
    (266.617, 6.30, 4.35),
    (270.748, 6.70, 2.40),
    (255.204, 2.80, 5.00),
    (284.671, 2.85, 4.25),
    (218.867, 3.00, 4.00),
    (301.417, 3.05, 4.80),
]

# raised-cosine taper length at each of the 9 boundaries, access guides included.
# Rule: 160 um if the step is > 1.2 um or crosses 5.6 um; 80 um for 0.5-1.2 um;
# 60 um for <= 0.5 um. Both arms share each boundary's length.
TAPERS = [80, 160, 80, 160, 160, 80, 60, 80, 160]

# the pi bias, built into the layout instead of taken from the heaters
PATH_OFFSET_UM = 0.0998          # + = extra path in the TOP arm

if __name__ == "__main__":
    write("a_design4_standard_v3", SECTIONS, TAPERS, PATH_OFFSET_UM)
