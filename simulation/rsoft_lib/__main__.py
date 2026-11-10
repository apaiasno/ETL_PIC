"""python -m rsoft_lib <file.fld>: print peak/edge amplitude statistics of a field file."""
import sys

import numpy as np

from .fields import read_fld, edge_to_peak_ratio

if len(sys.argv) > 1:
    fd = read_fld(sys.argv[1])
    ratio = edge_to_peak_ratio(fd)
    print(f"Read '{sys.argv[1]}': nx={fd.nx}, ny={fd.ny}, wavelength={fd.wavelength}")
    print(f"Peak amplitude:     {fd.peak_amp():.5E}")
    print(f"Max edge amplitude: {fd.edge_amp_max():.5E}")
    print(f"Edge/peak ratio:    {ratio:.5E} ({20 * np.log10(ratio):.1f} dB)")
else:
    print(__doc__)
