"""BeamPROP monitor (.mon) files: a z-series of scalar values per monitor.

File layout: a 5-line header (Monitor #, Path, Type, Tilt, Mode) followed by
one row per z step:

    Monitor #:                1             2
    Monitor Path:             1             1
    Monitor Type:      WG_POWER  LAUNCH_PHASE
    Monitor Tilt:             1             1
    Monitor Mode:             0             0
     0.000000E+00  8.016629E-01  0.000000E+00
     1.000000E-01  8.017489E-01  4.039808E-02
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MonitorData:
    z: np.ndarray            # shape (n_z,)
    values: np.ndarray       # shape (n_z, n_monitors)
    number: List[int]
    path: List[int]
    type: List[str]
    tilt: List[int]
    mode: List[int]

    @property
    def n_monitors(self) -> int:
        return len(self.number)

    def get(self, path: Optional[int] = None, type: Optional[str] = None) -> np.ndarray:
        """Column(s) matching the given path and/or type.

        Shape (n_z,) for a single match, (n_z, n_matches) otherwise.
        """
        idx = [
            i for i in range(self.n_monitors)
            if (path is None or self.path[i] == path)
            and (type is None or self.type[i] == type)
        ]
        if not idx:
            raise KeyError(f"No monitor found with path={path}, type={type}")
        cols = self.values[:, idx]
        return cols[:, 0] if cols.shape[1] == 1 else cols

    def at(self, z: Optional[float] = None) -> np.ndarray:
        """All monitor values (n_monitors,) at the z sample nearest `z`.

        With z=None, returns the last sample (the output plane).
        """
        i = -1 if z is None else int(np.abs(self.z - z).argmin())
        return self.values[i]


def _header_values(line: str) -> List[str]:
    """'Monitor Type:   WG_POWER  LAUNCH_PHASE' -> ['WG_POWER', 'LAUNCH_PHASE']."""
    return line.partition(":")[2].split()


def read_monitor(path: str) -> MonitorData:
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    number = [int(v) for v in _header_values(lines[0])]
    mon_path = [int(v) for v in _header_values(lines[1])]
    mon_type = _header_values(lines[2])
    tilt = [int(v) for v in _header_values(lines[3])]
    mode = [int(v) for v in _header_values(lines[4])]
    n_monitors = len(number)

    data_lines = lines[5:]
    if data_lines:
        data = np.loadtxt(data_lines, ndmin=2)
        if data.shape[1] != 1 + n_monitors:
            raise ValueError(
                f"{path}: rows have {data.shape[1]} values, expected "
                f"{1 + n_monitors} (z + {n_monitors} monitors)."
            )
    else:
        data = np.empty((0, 1 + n_monitors))

    return MonitorData(
        z=data[:, 0], values=data[:, 1:],
        number=number, path=mon_path, type=mon_type, tilt=tilt, mode=mode,
    )
