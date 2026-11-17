"""Running an RSoft simulation from Python: set symbols, run, record, clean up; and estimating the phase and loss of the
pathways of a circuit from the n_eff tables, before running anything."""
from __future__ import annotations

import fnmatch
import glob
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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

    # Refuse unfilled placeholders such as {"<pade symbol>": "<value>"} before the .ind file is touched
    for name, value in (symbols or {}).items():
        if not re.fullmatch(r"[A-Za-z_]\w*", str(name)) or re.fullmatch(r"<.*>", str(value).strip()):
            raise ValueError(f"Invalid symbol {name!r} = {value!r}: it looks like an unfilled placeholder. "
                             f"The .ind file was not modified.")

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


# --------------------------------------------------------------------------- #
# Wavelength scans of a circuit written by Circuit.write, and their monitors
# --------------------------------------------------------------------------- #

def read_pathways(mon):
    """Power and phase of every pathway from a BeamPROP .mon file (or MonitorData) with the monitors that Circuit.write puts
    on each pathway (a FILE_POWER and a FILE_PHASE monitor). Returns {pathway: dict(z, power, phase_deg, phase_unwrapped_deg)}:
    z (um), the power in the pathway's mode, and the phase in degrees as BeamPROP writes it (wrapped) and unwrapped along z."""
    from .monitors import MonitorData, read_monitor
    m = mon if isinstance(mon, MonitorData) else read_monitor(str(mon))
    out = {}
    for k in sorted(set(m.path)):
        rec = dict(z=m.z)
        for i in range(m.n_monitors):
            if m.path[i] != k:
                continue
            if m.type[i] == "FILE_POWER":
                rec["power"] = m.values[:, i]
            elif m.type[i] == "FILE_PHASE":
                rec["phase_deg"] = m.values[:, i]
                rec["phase_unwrapped_deg"] = np.unwrap(m.values[:, i], period=360)
        out[k] = rec
    return out


_WAVELENGTH_KEYS = ("wavelength", "Wav", "free_space_wavelength")     # grid keys that mean the wavelength
_DEFAULT_LABELS = {k: "Wav" for k in _WAVELENGTH_KEYS}


def _label_value(value):
    """A parameter value as it appears in a file name: integers as they are (4 -> 4), other numbers as param_str (1.55 -> 1p550),
    anything else as text (underscores and other separators become -)."""
    from .utils import param_str
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return param_str(float(value))
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", str(value)).replace(".", "p")


def run_scan(ind_path, save_dir, name, grid, labels=None, width=3.4, mode_file=None, mode_dir=None, symbols=None,
             exe="bsimw32", keep_ext=(".mon", ".json"), skip_existing=True, timeout=None, check_symbols=True):
    """Run a circuit's .ind (written by Circuit.write) at every point of a parameter scan.

    grid: {symbol: values} for every combination of the values (the first symbol varies slowest), e.g.
          {"wavelength": [1.50, 1.55], "Step_z": [0.1, 0.2]}; or a list of {symbol: value} dicts for chosen points only.
          "wavelength" (or "Wav") stands for free_space_wavelength; when it is scanned, the launch and monitor mode file of every pathway
          (Launch_filename_k, Monitor_filename_k) is set to the mode of that wavelength as in scan_wavelengths.
    labels: {symbol: short name} for the file names (default "Wav" for the wavelength, the symbol name otherwise). Each run's
          prefix is <save_dir>/<name>_<label><value>_<label><value>..., e.g. ..._Wav1p550_Step_z0p100.
    width, mode_file, mode_dir: which mode file goes with a wavelength (see scan_wavelengths). symbols: symbols set in every run.
    skip_existing: do not rerun a point whose .mon file is already there (a long scan can be resumed).
    check_symbols: before the first run, check that every scanned symbol and every symbol in `symbols` is a symbol of the .ind
    and raise ValueError naming the others (a misspelled name would otherwise be added to the file and change nothing);
    check_symbols=False to set a symbol the file does not have yet.
    Returns a DataFrame with one row per point: one column per scanned symbol, then prefix, mon, seconds (0 if skipped), skipped."""
    import difflib
    import itertools
    import time
    import pandas as pd
    from .circuit import pathway_file_symbols
    from .modes import MODE_LIB_DIR, mode_field

    if isinstance(grid, dict):
        keys = list(grid)
        points = [dict(zip(keys, combo)) for combo in itertools.product(*[list(np.atleast_1d(grid[k])) for k in keys])]
    else:
        points = [dict(p) for p in grid]
        keys = list(dict.fromkeys(k for p in points for k in p))
    labels = {**_DEFAULT_LABELS, **(labels or {})}
    wav_keys = [k for k in keys if k in _WAVELENGTH_KEYS]
    if len(wav_keys) > 1:
        raise ValueError(f"The grid names the wavelength more than once: {wav_keys}")
    wav_key = wav_keys[0] if wav_keys else None
    if check_symbols:
        from .symbols import read_all_symbols
        known = read_all_symbols(str(ind_path))
        unknown = [k for k in [*keys, *(symbols or {})] if k not in _WAVELENGTH_KEYS and k not in known]
        if unknown:
            hints = "; ".join(f"{k!r}" + (f" (did you mean {', '.join(map(repr, close))}?)" if (close := difflib.get_close_matches(k, known, n=3, cutoff=0.5)) else "")
                              for k in unknown)
            raise ValueError(f"Not symbols of {Path(ind_path).name}: {hints}. Nothing was run. "
                             f"Use check_symbols=False to set a symbol the file does not have yet.")
    if wav_key is not None:
        if mode_file is None:
            mode_file = lambda w: mode_field(w, width, mode_dir=mode_dir or MODE_LIB_DIR)[0]
        files = {float(w): mode_file(float(w)) for w in {float(p[wav_key]) for p in points if wav_key in p}}   # fails early
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for point in points:
        point = {k: (v.item() if hasattr(v, "item") else v) for k, v in point.items()}
        tag = "_".join(f"{labels.get(k, k)}{_label_value(point[k])}" for k in keys if k in point)
        prefix = str(save_dir / f"{name}_{tag}")
        mon = prefix + ".mon"
        row = {**point, "prefix": prefix, "mon": mon}
        if skip_existing and Path(mon).exists():
            rows.append({**row, "seconds": 0.0, "skipped": True})
            continue
        run_symbols = {("free_space_wavelength" if k in _WAVELENGTH_KEYS else k): v for k, v in point.items()}
        if wav_key is not None and wav_key in point:
            f = files[float(point[wav_key])]
            run_symbols.update(pathway_file_symbols(str(ind_path), launch=f, monitor=f))
        run_symbols.update(symbols or {})
        t0 = time.time()
        run_sim(ind_path, prefix, run_symbols, exe=exe, keep_ext=keep_ext, timeout=timeout)
        rows.append({**row, "seconds": time.time() - t0, "skipped": False})
        print(f"{name} {tag}: done in {rows[-1]['seconds']:.0f} s")
    return pd.DataFrame(rows)


def scan_wavelengths(ind_path, save_dir, name, wavelengths, width=3.4, mode_file=None, mode_dir=None, symbols=None,
                     exe="bsimw32", keep_ext=(".mon", ".json"), skip_existing=True, timeout=None):
    """Run a circuit's .ind (written by Circuit.write) once per wavelength: run_scan with grid {"wavelength": wavelengths}.

    For each wavelength w it sets free_space_wavelength = w and the launch and monitor mode file of every pathway
    (Launch_filename_k, Monitor_filename_k) to the mode of that wavelength, then runs it with run_sim under the prefix
    <save_dir>/<name>_Wav<w> (e.g. ..._Wav1p550). The mode files are looked up for all wavelengths before anything runs, so a
    wavelength missing from the mode library fails at once.

    width: the guide width of the mode (default 3.4 um); mode_file: a function wav -> mode file name instead of the mode library
    (modes.mode_field(wav, width), in mode_dir); symbols: more symbols for every run (e.g. {"Step_z": 0.1});
    skip_existing: do not rerun a wavelength whose .mon file is already there (a long scan can be resumed).
    Returns a DataFrame with one row per wavelength: wavelength, prefix, mon (the .mon path), seconds (0 if skipped), skipped."""
    return run_scan(ind_path, save_dir, name, {"wavelength": [float(w) for w in wavelengths]}, width=width, mode_file=mode_file,
                    mode_dir=mode_dir, symbols=symbols, exe=exe, keep_ext=keep_ext, skip_existing=skip_existing, timeout=timeout)


def read_scan(scan):
    """The monitors of a scan: the DataFrame of run_scan or scan_wavelengths, or a dict {wavelength: .mon path}.
    Returns (table, traces): table, a DataFrame with one row per (scan point, pathway) at the end of the run: the scanned
    parameters, pathway, z_end, power_end, phase_end_deg, phase_end_unwrapped_deg; traces, {point: read_pathways(...)} with the full
    z dependence, where point is the value of the scanned parameter if there is one (e.g. the wavelength), else a tuple of the
    values in the order of the columns."""
    import pandas as pd
    if not hasattr(scan, "columns"):
        scan = pd.DataFrame([dict(wavelength=w, mon=m) for w, m in sorted(dict(scan).items())])
    params = [col for col in scan.columns if col not in ("prefix", "mon", "seconds", "skipped")]
    traces, rows = {}, []
    for _, point in scan.iterrows():
        tr = read_pathways(point["mon"])
        key = point[params[0]] if len(params) == 1 else tuple(point[p] for p in params)
        traces[key] = tr
        for k, r in tr.items():
            rows.append({**{p: point[p] for p in params}, "pathway": k, "z_end": float(r["z"][-1]),
                         "power_end": float(r["power"][-1]) if "power" in r else np.nan,
                         "phase_end_deg": float(r["phase_deg"][-1]) if "phase_deg" in r else np.nan,
                         "phase_end_unwrapped_deg": float(r["phase_unwrapped_deg"][-1]) if "phase_deg" in r else np.nan})
    return pd.DataFrame(rows), traces


# --------------------------------------------------------------------------- #
# Phase and loss of the pathways of a circuit, from the n_eff tables
# --------------------------------------------------------------------------- #

_TAPER_NAMES = {"cos", "sin", "exp", "sqrt", "pi", "abs", "z"}


def _taper_profile(expression):
    """The user-taper expression of a circuit (a function of z' from 0 to 1, trig functions in degrees) as a function of u."""
    code = compile(expression.replace("^", "**"), "<taper>", "eval")
    unknown = set(code.co_names) - _TAPER_NAMES
    if unknown:
        raise ValueError(f"Cannot evaluate the taper expression {expression!r}: unknown names {sorted(unknown)}")
    env = {"__builtins__": {}, "cos": lambda a: np.cos(np.radians(a)), "sin": lambda a: np.sin(np.radians(a)), "exp": np.exp,
           "sqrt": np.sqrt, "pi": np.pi, "abs": np.abs}
    return lambda u: np.broadcast_to(eval(code, env, {"z": u}), np.shape(u))


def _segment_geometry(s):
    """(length, radius) of a segment of a Circuit: radius is None for a line."""
    A, B = np.array(s["A"]), np.array(s["B"])
    if s["kind"] == "line":
        return float(np.hypot(*(B - A))), None
    if s.get("free"):
        radius, i0, i1 = s["free"]
        return abs(radius) * float(np.radians(abs(i1 - i0))), abs(radius)
    dx = B[0] - A[0]
    if abs(dx) < 1e-12:                                                       # an arc of no sideways extent is a line
        return float(np.hypot(*(B - A))), None
    rho = ((B - A) ** 2).sum() / (2 * dx)
    ra, rb = np.array([-rho, 0.0]), np.array([dx - rho, B[1] - A[1]])
    return abs(rho) * float(np.arctan2(abs(ra[0] * rb[1] - ra[1] * rb[0]), ra @ rb)), abs(float(rho))


@dataclass
class PathwayEstimate:
    """Estimated propagation phase and loss of one pathway. phase (rad), loss_dB and neff_mean (the phase-weighted mean n_eff) are
    floats for one wavelength and arrays for several. length, straight_length and bend_length are in um (along the centerline).
    segments: one dict per segment (segment, kind, length, radius, width, neff, phase, loss_dB). extrapolated_length: bend length
    (um) whose radius is outside the bend table; clamped_length: length (um) whose width is outside the straight table."""
    pathway: int
    wavelength: object
    length: float
    straight_length: float
    bend_length: float
    phase: object
    neff_mean: object
    loss_dB: object
    straight_loss_dB: object
    bend_loss_dB: object
    extrapolated_length: float = 0.0
    clamped_length: float = 0.0
    segments: list = field(default_factory=list, repr=False)


class PathwayEstimates(dict):
    """The PathwayEstimate of each pathway, keyed by the pathway number (1, 2, ...). print() shows a table."""

    def table(self, index=0):
        """A DataFrame with one row per pathway (for the wavelength with this index if there are several)."""
        import pandas as pd
        pick = lambda v: float(np.atleast_1d(v)[index])
        return pd.DataFrame([dict(pathway=k, length=e.length, straight=e.straight_length, bend=e.bend_length, phase_rad=pick(e.phase),
                                  phase_turns=pick(e.phase) / (2 * np.pi), n_eff_mean=pick(e.neff_mean), loss_dB=pick(e.loss_dB),
                                  bend_loss_dB=pick(e.bend_loss_dB)) for k, e in self.items()]).set_index("pathway")

    def difference(self, a, b):
        """Pathway a minus pathway b: dict(phase (rad), phase_wrapped (rad, within +-pi), loss_dB, length (um), optical_length (um,
        the phase difference as a path in vacuum: phase * wavelength / 2 pi)). Arrays if there are several wavelengths."""
        ea, eb = self[a], self[b]
        dphi = np.asarray(ea.phase) - np.asarray(eb.phase)
        return dict(phase=dphi[()], phase_wrapped=((dphi + np.pi) % (2 * np.pi) - np.pi)[()], loss_dB=(np.asarray(ea.loss_dB) - np.asarray(eb.loss_dB))[()],
                    length=ea.length - eb.length, optical_length=(dphi * np.asarray(ea.wavelength) / (2 * np.pi))[()])

    def __str__(self):
        if not self:
            return "no pathways"
        wl = np.atleast_1d(next(iter(self.values())).wavelength)
        head = f"wavelength {wl[0]:g} um" + (f" (first of {len(wl)}; the values are arrays)" if len(wl) > 1 else "")
        return head + "\n" + self.table().to_string(float_format=lambda v: f"{v:.6g}")


def estimate_pathways(circuit, wavelength, pathways=None, straight_dir=None, bend_csv=None, bend_dir=None, bend_root=None,
                      bend_table=None, bend_width=3.4, height=3.4, clamp=True, n_taper=32):
    """Estimate the propagation phase and loss of each pathway of a Circuit (rsoft_lib.circuit) from the n_eff tables.

    Each segment contributes along its centerline: a line, n_eff of the straight guide of its width (tapers are integrated along
    the width profile) from the straight mode library (modes.StraightModel); an arc, n_eff of the TE bend mode at its radius from
    the FemSIM bend scan (bend_modes.BendModel; for an arc of another width than bend_width the change of the straight guide's
    n_eff with width is added). phase = 2 pi / wavelength * sum(n_eff * length); loss = the bend modes' Im n_eff converted to dB
    (the straight guides' too, which is 0 unless the table has an imaginary part).

    Not included: the mismatch loss and phase at the joints between lines and arcs (offsets), coupling at MMIs (an MMI's slab is
    wider than the table, so with clamp=True it is counted with the widest guide of the table and flagged in clamped_length), and
    the phase of whatever the pathway does outside its segments. The bend n_eff is assumed to refer to the guide's center.

    wavelength: one value or an array (um). pathways: the pathway numbers (default all). straight_dir, bend_csv, bend_dir,
    bend_root: where the tables are (default the library's locations). bend_table: a DataFrame instead of reading the bend table.
    clamp: use the nearest tabulated width for widths outside the straight table instead of raising an error.
    Returns a PathwayEstimates (a dict, pathway number -> PathwayEstimate)."""
    from . import bend_modes, modes

    scalar = np.ndim(wavelength) == 0
    wl = np.atleast_1d(np.asarray(wavelength, float))
    straight = modes.straight_model(straight_dir or modes.MODE_LIB_DIR, height)
    if bend_table is not None:
        bend = bend_modes.BendModel(bend_table, straight, bend_width)
    else:
        bend = bend_modes.bend_model(bend_csv or bend_modes.BEND_NEFF_CSV, bend_dir or bend_modes.BEND_MODE_DIR,
                                     bend_root or bend_modes.BEND_MODE_ROOT, straight_dir or modes.MODE_LIB_DIR, bend_width)
    nodes, weights = np.polynomial.legendre.leggauss(n_taper)
    u, wu = (nodes + 1) / 2, weights / 2
    out = lambda v: v[0] if scalar else v
    k0 = 2 * np.pi / wl
    cache = {}
    n_ref = None                                                              # the straight guide of the bend table's width, once

    def evaluate(n):                                                          # one segment: its mean n_eff, flags
        s = circuit.segments[n - 1]
        L, R = _segment_geometry(s)
        wa, wb = s["wa"], s["wb"]
        varies = abs(wa - wb) > 1e-9
        if varies:
            f = _taper_profile(circuit._tapers[s["taper"] - 1][0]) if s.get("taper") is not None else (lambda x: x)
            widths = wa + (wb - wa) * f(u)
        else:
            widths, wu_ = np.array([wa]), np.array([1.0])
        w_ = wu if varies else wu_
        n_st, clamped = straight.neff(wl[:, None], widths[None, :], clamp)       # (n_wav, n_widths)
        n_mean = (n_st * w_).sum(axis=1)
        status = 0
        if R is not None:
            nonlocal n_ref
            n_bend, st = bend.neff(R, wl)
            if n_ref is None:
                n_ref = straight.neff(wl, bend_width, clamp)[0]
            n_mean = n_bend + (n_mean.real - n_ref.real)                         # the width correction (0 at the table's width)
            status = int(st.max()) if abs(int(st.max())) >= abs(int(st.min())) else int(st.min())
        return dict(segment=n, kind="line" if R is None else "arc", length=L, radius=R, width=(wa, wb), n_mean=n_mean,
                    clamped=bool(clamped.any()), status=status)

    from .bend_modes import loss_dB_per_cm_from_neff_imag
    result = PathwayEstimates()
    for number, pw in enumerate(circuit.pathways, start=1):
        if pathways is not None and number not in pathways:
            continue
        rows, total = [], dict(L=0.0, Ls=0.0, Lb=0.0, phase=0.0, loss=0.0, loss_s=0.0, loss_b=0.0, extra=0.0, clamp=0.0)
        for n in pw["segments"]:
            if n not in cache:
                cache[n] = evaluate(n)
            e = cache[n]
            L = e["length"]
            phase = k0 * e["n_mean"].real * L
            loss = loss_dB_per_cm_from_neff_imag(e["n_mean"].imag, wl) * L * 1e-4
            rows.append(dict(segment=n, kind=e["kind"], length=L, radius=e["radius"], width=e["width"], neff=out(e["n_mean"].real),
                             phase=out(phase), loss_dB=out(loss)))
            total["L"] += L
            total["Ls" if e["kind"] == "line" else "Lb"] += L
            total["phase"] = total["phase"] + phase
            total["loss"] = total["loss"] + loss
            total["loss_s" if e["kind"] == "line" else "loss_b"] = total["loss_s" if e["kind"] == "line" else "loss_b"] + loss
            total["extra"] += L if e["status"] != 0 else 0.0
            total["clamp"] += L if e["clamped"] else 0.0
        zero = np.zeros_like(wl)
        phase = np.asarray(total["phase"]) + zero
        result[number] = PathwayEstimate(
            pathway=number, wavelength=out(wl), length=total["L"], straight_length=total["Ls"], bend_length=total["Lb"],
            phase=out(phase), neff_mean=out(phase / (k0 * total["L"]) if total["L"] else zero), loss_dB=out(np.asarray(total["loss"]) + zero),
            straight_loss_dB=out(np.asarray(total["loss_s"]) + zero), bend_loss_dB=out(np.asarray(total["loss_b"]) + zero),
            extrapolated_length=total["extra"], clamped_length=total["clamp"], segments=rows)
    for number, est in result.items():
        if est.extrapolated_length:
            above = [r for r in est.segments if r["radius"] is not None and r["radius"] > bend.radii[-1]]
            warnings.warn(f"Pathway {number}: {est.extrapolated_length:.0f} um of bends have a radius outside the bend table "
                          f"({bend.radii[0]:g}-{bend.radii[-1]:g} um); above it the n_eff shift is continued as 1/R^2 and the loss "
                          f"log-linearly ({len(above)} arcs above, the rest below, where the value at {bend.radii[0]:g} um is used).")
        if est.clamped_length:
            warnings.warn(f"Pathway {number}: {est.clamped_length:.0f} um of the path are wider or narrower than the straight table "
                          f"({straight.widths[0]:g}-{straight.widths[-1]:g} um); the nearest width of the table was used there.")
    return result