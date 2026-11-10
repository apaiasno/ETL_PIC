"""Small helpers shared across the library."""


def param_str(value, decimals=3):
    """Filename-safe string for a scan parameter: 1.45 -> '1p450', 500 -> '500p000'.

    This is the naming convention used for scan outputs (Wav1p450, Radius500p000, ...).
    """
    return f"{value:0.{decimals}f}".replace(".", "p")
