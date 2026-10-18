"""Shared by the table tests: correction of a table row and formatting of the results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cr3bp import symmetric_point          # noqa: E402
from src.moser import Moser                    # noqa: E402
from src.correction import correct             # noqa: E402
from src.stability import stability            # noqa: E402,F401
from src.cz import conley_zehnder              # noqa: E402,F401


def orbit(C, x, z, py, pz, T, symmetry, primary, k=1, N=1, extended=False):
    """Fixed-C correction of the orbit through (x, 0, z, 0, p_y, p_z) whose prime period is T, with
    single (N = 1) or multiple shooting, in double or extended precision."""
    sys = Moser(C, primary, extended)
    return correct(sys, symmetric_point(x, z, py, pz), T, symmetry, N=N, tol=1e-20 if extended else 1e-10)


def fmt(v):
    v = complex(v)
    if abs(v.imag) > 1e-6 * max(1, abs(v)):
        return f'{v.real:.4g}{v.imag:+.4g}i'
    if abs(v.real) >= 1e4:
        return f'{v.real:.2e}'
    return f'{v.real:.4f}' if abs(v.real) < 10 else f'{v.real:.5g}'


def indices_text(indices):
    return ', '.join(f"{n} = {fmt(v)}" + (f" ({'+' if kr > 0 else '-'})" if kr else '') for n, v, kr in indices)


def header():
    print('rows marked * are computed in extended precision')
    print(f"{'row':>5} {'C':>12} {'|F|':>8} {'it':>2} {'T':>11}   {'stability indices (Krein)':<38} CZ")


def line(name, o, k, indices, cz, extended=False):
    s = indices_text(indices)
    cz = ' -> '.join(' / '.join(map(str, c)) if isinstance(c, tuple) else str(c) for c in cz)
    print(f"{name + ('*' if extended else ''):>5} {float(o['sys'].C):12.8f} {float(o['residual']):8.1e} {o['iterations']:>2} "
          f"{k * float(o['T']):11.6f}   {s:<38} {cz}", flush=True)


def header_collision():
    print('rows marked * are computed in extended precision')
    print(f"{'row':>3} {'C':>22} {'eta_1':>22} {'|F|':>8} {'it':>2} {'T':>11}   stability indices (Krein)")


def line_collision(name, o, indices, extended=False):
    s = indices_text(indices)
    print(f"{name + ('*' if extended else ''):>3} {str(o['C'])[:22]:>22} {str(o['eta1'])[:22]:>22} {float(o['residual']):8.1e} "
          f"{o['iterations']:>2} {float(o['T']):11.6f}   {s}", flush=True)
