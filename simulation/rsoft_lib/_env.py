"""RSoft installation paths and the RSoftCircuit import.

Set the RSOFT_ROOT environment variable to override the install location.
If RSoft's Python tools are not installed on this machine, `RSoftCircuit`
is None: the parsing/plotting parts of the library still work, and only
code that builds or edits circuits needs RSoft.
"""
import os
import sys

RSOFT_ROOT = os.environ.get(
    "RSOFT_ROOT", r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft"
)

_PATHS = [
    os.path.join(RSOFT_ROOT, "products", "most", "python"),
    os.path.join(RSOFT_ROOT, "bin"),
    os.path.join(RSOFT_ROOT, "products", "python", "python3.11", "Lib", "site-packages"),
]
for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from rstools import RSoftCircuit
except ImportError:
    RSoftCircuit = None
