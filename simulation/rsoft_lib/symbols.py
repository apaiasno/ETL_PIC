"""Reading the symbol table of an RSoft .ind file."""
import json
import re

_ASSIGN = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)")
# First line of an object block, e.g. 'material 1', 'segment 1', 'pathway 1'.
_BLOCK_START = re.compile(r"\s*[A-Za-z_]\w*\s+\d+\s*$")


def read_all_symbols(ind_file, header_only=True):
    """Return {name: value_string} for the `name = value` lines of an .ind file.

    With header_only=True (default) parsing stops at the first object block
    (the first 'material 1' / 'segment 1' / ... line). Everything before it is
    the symbol table and global settings. Without this, property lines inside
    blocks (extended, pathway, kappa_xx, E, alpha, nu, ...) are returned as if
    they were symbols and can shadow real ones.
    """
    symbols = {}
    with open(ind_file) as f:
        for line in f:
            line = line.split("#")[0]
            if header_only and _BLOCK_START.match(line):
                break
            m = _ASSIGN.match(line)
            if m:
                symbols[m.group(1)] = m.group(2).strip()
    return symbols


def read_symbol(ind_file, symbol_name, cast=None):
    """Return one symbol's value as a string, or cast(value) if `cast` is given.

    Example: Length = read_symbol(ind, "Length", cast=float)
    """
    value = read_all_symbols(ind_file)[symbol_name]
    return cast(value) if cast else value


def export_symbols(ind_file, out_file):
    """Write the symbol table to a JSON file (scan metadata) and return it."""
    symbols = read_all_symbols(ind_file)
    with open(out_file, "w") as f:
        json.dump(symbols, f, indent=2)
    return symbols
