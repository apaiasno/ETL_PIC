"""Running an RSoft simulation from Python: set symbols, run, record, clean up."""
from __future__ import annotations

import fnmatch
import glob
import subprocess
from pathlib import Path

from . import _env
from .symbols import export_symbols


def run_sim(ind_path, prefix, symbols=None, exe="bsimw32", args=(),
            keep_ext=(".mon", ".json", ".dat"), check=True, timeout=None):
    """Run one simulation and keep only the output files you need.

    Steps: (1) set `symbols` in the .ind file and save it, (2) run
    `exe -min <ind> <args> prefix=<prefix> wait=0` in the .ind file's folder,
    (3) write the symbol table to `<prefix>.json`, (4) delete every file
    starting with `prefix` whose extension is not in `keep_ext` (skipped
    when keep_ext is None, which keeps every output file).

    Parameters
    ----------
    ind_path : str or Path
        The .ind file. It is modified in place when `symbols` is given.
    prefix : str
        Output path prefix (folder + file stem), passed to RSoft as prefix=.
    symbols : dict, optional
        {symbol name: value}, e.g. {"free_space_wavelength": 1.55}.
    exe : str
        "bsimw32" for BeamPROP, "femsim" for FemSIM.
    args : sequence of str
        Extra command-line options, e.g. ["mode_method=1", "mode_set=1"].
    keep_ext : sequence of str or None
        Endings of the files to keep (case-insensitive). Shell-style wildcards
        are allowed, e.g. ".m[0-9][0-9]" keeps .m00 to .m99 and
        "_ex.m[0-9][0-9]" keeps only the Ex mode files. None keeps all files.
    check : bool
        Raise RuntimeError if the run produced no output files at all
        (a crash or an unanswered dialog), instead of failing later when
        the analysis looks for the files.
    timeout : float, optional
        Seconds before the run is aborted.

    Returns
    -------
    subprocess.CompletedProcess
    """
    ind_path = Path(ind_path)

    if symbols:
        if _env.RSoftCircuit is None:
            raise ImportError("RSoft's rstools is not available; cannot edit the .ind file.")
        c = _env.RSoftCircuit()
        c.read(str(ind_path))
        for name, value in symbols.items():
            c.set_symbol(name, value)
        c.write(str(ind_path))

    command = [exe, "-min", ind_path.name, *args, f"prefix={prefix}", "wait=0"]
    result = subprocess.run(command, cwd=ind_path.parent, capture_output=True,
                            text=True, timeout=timeout)

    p = Path(prefix)
    outputs = [f for f in p.parent.glob(glob.escape(p.name) + "*") if f.is_file()]
    if check and not outputs:
        raise RuntimeError(
            f"{exe} produced no files for prefix '{prefix}'.\n"
            f"return code: {result.returncode}\nstderr: {result.stderr[:500]}"
        )

    export_symbols(str(ind_path), f"{prefix}.json")

    if keep_ext is not None:
        patterns = ["*" + e.lower() for e in keep_ext]
        for f in p.parent.glob(glob.escape(p.name) + "*"):
            if f.is_file() and not any(fnmatch.fnmatchcase(f.name.lower(), pat) for pat in patterns):
                f.unlink()

    return result