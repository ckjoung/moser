"""Stability indices and Krein signatures of symmetric periodic orbits (Section 3.3).

The stability indices are computed from the non-trivial eigenvalues of M rather
than directly from the trace formulas in Section 3.3. More precisely, we
restrict M to the tangent space T of {g_1 = g_2 = 0, Q_c = mu/2} at the initial
point and quotient out the flow direction X_Q, yielding a four-dimensional
reduced monodromy matrix R on T/<X_Q>, whose eigenvalues are the non-trivial
Floquet multipliers; the symplectic form restricted to T/<X_Q> gives the Krein 
signatures.

For highly unstable non-planar orbits (|s| > 1e5), the multipliers are obtained
from the generalized eigenvalue problem of Remark 3.2 in Section 3.3, which
avoids explicitly inverting V. For non-planar orbits computed in extended
precision, the multipliers are computed from M using mpmath with 34 digits
of working precision.

The stability index of a multiplier pair is s = (lambda + 1/lambda)/2; for the k-fold cover it is
the Chebyshev polynomial T_k(s).  Planar orbits have the in-plane index s_p and the vertical index
s_v (decoupled blocks of M).
"""

import numpy as np
import mpmath as mp
from scipy.linalg import eig, null_space

from .correction import FIX, SYMMETRY

J8 = np.block([[np.zeros((4, 4)), -np.eye(4)], [np.eye(4), np.zeros((4, 4))]])
PLANAR, VERTICAL = [0, 1, 2, 4, 5, 6], [3, 7]


def monodromy(orbit):
    start, end, n = SYMMETRY[orbit['symmetry']]
    Si, Sj = np.diag(FIX[start]['sigma']), np.diag(FIX[end]['sigma'])
    V = np.asarray(orbit['V'], float)
    W = Si @ np.linalg.solve(V, Sj @ V)
    return W if n == 2 else W @ W


def generalized_multipliers(orbit):
    """Eigenvalues nu of Sj V x = nu V Si x (squared for quarter-period segments), trivial ones removed."""
    start, end, n = SYMMETRY[orbit['symmetry']]
    Si, Sj = np.diag(FIX[start]['sigma']), np.diag(FIX[end]['sigma'])
    V = np.asarray(orbit['V'], float)
    nu = eig(Sj @ V, V @ Si, right=False)
    nu = nu if n == 2 else nu ** 2
    return nu[np.argsort(np.abs(nu - 1))][4:]


def extended_multipliers(orbit):
    """Non-trivial eigenvalues of M from V in real128, evaluated with mpmath (34 digits)."""
    mp.mp.dps = 34
    start, end, n = SYMMETRY[orbit['symmetry']]
    V = mp.matrix([[mp.mpf(str(v)) for v in row] for row in orbit['V']])
    Si, Sj = mp.diag(FIX[start]['sigma']), mp.diag(FIX[end]['sigma'])
    W = Si * mp.inverse(V) * Sj * V
    nu = mp.eig(W if n == 2 else W * W, left=False, right=False)
    nu = sorted(nu, key=lambda v: abs(v - 1))[4:]
    return [complex(v) for v in nu]


def reduced(orbit, M, idx=tuple(range(8))):
    """Matrix of the map induced by M on T/<X_Q> (restricted to the coordinates idx) and the
    symplectic form on it."""
    sys, z, idx = orbit['sys'], np.asarray(orbit['z0'], float), list(idx)
    X_Q = np.asarray(sys.field(orbit['z0']), float)[idx]
    G = np.array([np.r_[2 * z[:4], np.zeros(4)], np.r_[z[4:], z[:4]], np.asarray(sys.grad_K(orbit['z0']), float)])[:, idx]
    M = M[np.ix_(idx, idx)]
    B = null_space(G[np.linalg.norm(G, axis=1) > 0])
    C = B @ null_space((B.T @ X_Q)[None, :])
    A = np.column_stack([C, X_Q])
    R = np.linalg.lstsq(A, M @ C, rcond=None)[0][:-1]
    return R, C.T @ J8[np.ix_(idx, idx)] @ C


def pairs(lams):
    """Stability indices (largest |s| first) from a symplectic spectrum: one s per pair (lambda, 1/lambda
    or lambda, conj(lambda)), computed from the larger multiplier; the partner m of lambda is the one
    with m lambda closest to 1, which is robust when the small multipliers are inaccurate."""
    lams, out = [complex(v) for v in lams], []
    while lams:
        lam = lams.pop(int(np.argmax(np.abs(lams))))
        if lams:
            lams.pop(int(np.argmin([abs(np.log(m * lam)) for m in lams])))
        out.append((lam + 1 / lam) / 2)
    return out


def krein(R, Om, s):
    """Krein signature of the elliptic pair with index s: sign of omega(a, R a) + omega(b, R b),
    a + ib an eigenvector, eq. (28); 0 if s is not elliptic (or within 1e-6 of +-1)."""
    if abs(s.imag) > 1e-6 or abs(s.real) >= 1 - 1e-6:
        return 0
    lam = s.real + 1j * np.sqrt(1 - s.real ** 2)
    w, v = eig(R)
    u = v[:, int(np.argmin(np.abs(w - lam)))]
    a, b = u.real, u.imag
    return int(np.sign(a @ Om @ R @ a + b @ Om @ R @ b))


def chebyshev(s, k):
    if k == 1:
        return s
    r = np.sqrt(complex(s) ** 2 - 1)
    lam = s + r if abs(s + r) >= abs(s - r) else s - r
    return (lam ** k + lam ** -k) / 2


def stability(orbit, k=1):
    """[(name, s, Krein sign)] of the k-fold cover of the orbit: (s_1, s_2), or (s_p, s_v) if planar."""
    M = monodromy(orbit)
    if orbit['symmetry'] == 'planar':
        out = []
        for name, idx in (('s_p', PLANAR), ('s_v', VERTICAL)):
            R, Om = reduced(orbit, M, idx) if idx is PLANAR else (M[np.ix_(idx, idx)], J8[np.ix_(idx, idx)])
            s = pairs(np.linalg.eigvals(R))[0]
            Rk = np.linalg.matrix_power(R, k)
            out.append((name, chebyshev(s, k), krein(Rk, Om, chebyshev(s, k))))
        return out
    R, Om = reduced(orbit, M)
    s = pairs(np.linalg.eigvals(R))
    if orbit['sys'].extended:
        s = pairs(extended_multipliers(orbit))
    elif max(abs(v) for v in s) > 1e5:
        s = pairs(generalized_multipliers(orbit))
    Rk = np.linalg.matrix_power(R, k)
    return [(f's_{i + 1}', chebyshev(v, k), krein(Rk, Om, chebyshev(v, k))) for i, v in enumerate(s)]
