"""Small helpers shared across the library."""


def param_str(value, decimals=3):
    """Filename-safe string for a scan parameter: 1.45 -> '1p450', 500 -> '500p000'.

    This is the naming convention used for scan outputs (Wav1p450, Radius500p000, ...).
    """
    return f"{value:0.{decimals}f}".replace(".", "p")


def parse_param(text):
    """Inverse of param_str: '1p450' -> 1.45."""
    return float(str(text).replace("p", "."))


def interp_pchip(x, y, xq):
    """Shape-preserving (PCHIP) interpolation of y(x) at xq, without extrapolation: NaN where xq is outside [x[0], x[-1]].
    x must increase. With one point, a constant that is valid only at that x."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    out = np.full(xq.shape, np.nan)
    if len(x) == 0:
        return out
    tol = 1e-9 * max(1.0, abs(x[-1]))
    inside = (xq >= x[0] - tol) & (xq <= x[-1] + tol)
    if len(x) == 1:
        out[inside] = y[0]
    elif inside.any():
        out[inside] = PchipInterpolator(x, y)(np.clip(xq[inside], x[0], x[-1]))
    return out