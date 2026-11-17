"""Ideal polychromatic transfer matrices of standard photonic components, and the interference spectrum of a circuit built from them.

Use together with simulation.estimate_pathways: pathway(estimates, k) turns the phase and loss it estimated for pathway k into
a transfer matrix, the ideal components (y_splitter, mmi_2x2, thermo_optic_phase_shifter, ...) are the other blocks, and
chain / parallel connect them, for example an unbalanced Mach-Zehnder:

    est = rlib.estimate_pathways(c, wl)                       # wl: the wavelengths (um)
    T = rlib.transfer.mzi_response(wl, rlib.transfer.pathway(est, 1), rlib.transfer.pathway(est, 2))
    P = rlib.transfer.power(T)                                # (n_wavelengths, n_outputs)

Conventions
  * Wavelengths are in um. A field varies as exp(j w t - j beta z), so a length with phase phi (rad, positive) transmits
    exp(-j phi), and the cross port of a 2x2 coupler is -j (it lags by 90 degrees). Take the complex conjugate of every matrix
    for the other convention (the powers do not change).
  * A matrix has the shape (n_wavelengths, n_outputs, n_inputs): one matrix per wavelength, so a signal chain is a batched matrix
    product. Constant ("broadband") components are repeated for every wavelength so that all blocks have the same shape.
  * Losses are in dB of power and enter as the field amplitude 10 ** (-loss_dB / 20).
"""
from functools import reduce

import numpy as np

__all__ = [
    "y_splitter", "y_combiner", "mmi_2x2", "thermo_optic_phase_shifter", "thermo_optic_phase", "waveguide", "pathway",
    "chain", "parallel", "mzi_response", "power", "to_dB", "fsr",
]


def _wavelengths(wav):
    return np.atleast_1d(np.asarray(wav, float))


def _amplitude(loss_dB):
    return 10 ** (-np.asarray(loss_dB, float) / 20)


def _constant(matrix, wav, excess_loss_dB):
    """The same (out, in) matrix at every wavelength, scaled by the excess loss (a number or one value per wavelength)."""
    wav = _wavelengths(wav)
    out = np.broadcast_to(np.asarray(matrix, complex), (len(wav),) + np.shape(matrix)).copy()
    return out * np.reshape(_amplitude(excess_loss_dB) * np.ones(len(wav)), (-1, 1, 1))


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #

def y_splitter(wav, split=0.5, excess_loss_dB=0.0):
    """Ideal 1x2 splitter (a Y-junction; an ideal 1x2 MMI has the same matrix): shape (n, 2, 1). `split` is the power fraction that
    goes to output 2 (default a broadband 50/50), excess_loss_dB the loss of the device."""
    return _constant([[np.sqrt(1 - split)], [np.sqrt(split)]], wav, excess_loss_dB)


def y_combiner(wav, split=0.5, excess_loss_dB=0.0):
    """Ideal 2x1 combiner: shape (n, 1, 2), the transpose of y_splitter (reciprocity). It is not lossless: the part of the input
    that is not in the symmetric mode of the junction (for equal inputs, the part that is out of phase) is radiated, so the output
    power depends on the phase between the two inputs."""
    return np.swapaxes(y_splitter(wav, split, excess_loss_dB), 1, 2)


def mmi_2x2(wav, split=0.5, excess_loss_dB=0.0):
    """Ideal 2x2 MMI (paired interference): shape (n, 2, 2), [[sqrt(1-s), -j sqrt(s)], [-j sqrt(s), sqrt(1-s)]] with s = `split`, the
    power fraction that crosses to the other port (default a broadband 50/50, which is unitary for excess_loss_dB = 0)."""
    s = np.sqrt(split)
    c = np.sqrt(1 - split)
    return _constant([[c, -1j * s], [-1j * s, c]], wav, excess_loss_dB)


def thermo_optic_phase(wav, phase=None, wav0=1.55, delta_nL=None):
    """The phase (rad) of a thermo-optic phase shifter at each wavelength: delta_phi = 2 pi delta_n L / wav, so it scales as 1 / wav.
    Give either `phase`, the phase at wavelength wav0, or delta_nL, the index change times the heated length (um)."""
    wav = _wavelengths(wav)
    if (phase is None) == (delta_nL is None):
        raise ValueError("Give exactly one of phase (rad at wav0) and delta_nL (um)")
    return np.asarray(phase, float) * wav0 / wav if delta_nL is None else 2 * np.pi * np.asarray(delta_nL, float) / wav


def thermo_optic_phase_shifter(wav, phase=None, wav0=1.55, delta_nL=None, loss_dB=0.0):
    """Ideal thermo-optic phase shifter: shape (n, 1, 1), transmission exp(-j phi(wav)) with phi from thermo_optic_phase (phi = phase *
    wav0 / wav, or 2 pi delta_nL / wav). loss_dB: its loss, a number or one value per wavelength. For a heater, delta_nL =
    (dn_eff/dT) * (temperature rise) * (heated length)."""
    phi = thermo_optic_phase(wav, phase, wav0, delta_nL)
    return waveguide(phi * np.ones(len(_wavelengths(wav))), loss_dB)


def waveguide(phase, loss_dB=0.0):
    """A length of guide with this phase (rad, one value per wavelength) and loss (dB): shape (n, 1, 1), exp(-j phase) times the
    amplitude of the loss."""
    phase = np.atleast_1d(np.asarray(phase, float))
    amp = _amplitude(loss_dB) * np.ones(len(phase))
    return (amp * np.exp(-1j * phase)).reshape(-1, 1, 1)


def pathway(estimates, number):
    """The transfer matrix (n, 1, 1) of a pathway from simulation.estimate_pathways: its estimated phase and loss. `estimates` is the
    result of estimate_pathways (use an array of wavelengths to get a spectrum); `number` the pathway number (1, 2, ...)."""
    e = estimates[number]
    return waveguide(e.phase, e.loss_dB)


# --------------------------------------------------------------------------- #
# Connecting blocks
# --------------------------------------------------------------------------- #

def chain(*blocks):
    """Blocks in series, in the order the light passes them: chain(A, B, C) is the matrix C B A at each wavelength. A block with one
    wavelength is applied to all."""
    return reduce(lambda acc, b: np.matmul(b, acc), blocks[1:], np.asarray(blocks[0]))


def parallel(*blocks):
    """Blocks side by side (the arms of an interferometer): the block-diagonal matrix, shape (n, sum of outputs, sum of inputs)."""
    n = max(np.shape(b)[0] for b in blocks)
    rows, cols = sum(np.shape(b)[1] for b in blocks), sum(np.shape(b)[2] for b in blocks)
    out = np.zeros((n, rows, cols), complex)
    r = c = 0
    for b in blocks:
        _, ro, co = np.shape(b)
        out[:, r:r + ro, c:c + co] = b
        r, c = r + ro, c + co
    return out


def mzi_response(wav, arm_a, arm_b, splitter="y", combiner="mmi", split_in=0.5, split_out=0.5, excess_loss_dB=0.0):
    """A Mach-Zehnder interferometer: splitter, the two arms (transfer matrices of shape (n, 1, 1), for example from pathway,
    possibly in chain with a thermo_optic_phase_shifter), combiner. Returns the matrix (n, outputs, inputs).
    splitter: "y" (1 input, y_splitter), "mmi" (2 inputs, mmi_2x2) or a matrix; combiner: "mmi" (2 outputs, mmi_2x2), "y" (1 output,
    y_combiner) or a matrix. split_in and split_out: the power fractions of the splitter and the combiner, excess_loss_dB the loss of
    each of them."""
    wav = _wavelengths(wav)
    parts = {"y": (lambda: y_splitter(wav, split_in, excess_loss_dB), lambda: y_combiner(wav, split_out, excess_loss_dB)),
             "mmi": (lambda: mmi_2x2(wav, split_in, excess_loss_dB), lambda: mmi_2x2(wav, split_out, excess_loss_dB))}
    first = parts[splitter][0]() if isinstance(splitter, str) else splitter
    last = parts[combiner][1]() if isinstance(combiner, str) else combiner
    return chain(first, parallel(arm_a, arm_b), last)


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #

def power(T, input=0):
    """Output powers (n_wavelengths, n_outputs) when light enters input `input` (0-based) of the matrices T."""
    return np.abs(np.asarray(T)[:, :, input]) ** 2


def to_dB(p):
    """Power in dB (10 log10), with -inf for zero."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.asarray(p, float))


def fsr(wav, phase_difference):
    """The local free spectral range (um) of an interferometer from the phase difference of its arms (rad, one value per wavelength):
    2 pi / |d(phase difference) / d wavelength|. With the arms' phases from estimate_pathways, this includes the dispersion of the
    guides (group index): FSR = wav^2 / (n_g * delta_L)."""
    return 2 * np.pi / np.abs(np.gradient(np.asarray(phase_difference, float), np.asarray(wav, float)))