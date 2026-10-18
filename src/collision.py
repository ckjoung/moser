"""Symmetric periodic collision orbits (Section 4, Lemma 4.2).

A collision with the regularized primary (mass m) is a point with xi = (1, 0, 0, 0) and |eta| = m.
The rho_xz-symmetric collision points are

    u_0(eta_1) = (1, 0, 0, 0, 0, eta_1, 0, eta_3),   eta_3 = +-sqrt(m^2 - eta_1^2),

and a collision orbit is a zero of F(eta_1, c) = pi_Y P^k_{c,+-}(u_0(eta_1)), where P_{c,+-} is the
Poincare map of the section xi_1 = 0 (or xi_3 = 0) restricted to crossings in the direction +-, and Y
are the coordinates (besides the section coordinate and eta_0) vanishing on the target fixed point set:

    section xi_1 = 0:  Y = (xi_3, eta_2) for Fix(rho_xz),  Y = (eta_2, eta_3) for Fix(rho_x)
    section xi_3 = 0:  Y = (xi_1, eta_2) for Fix(rho_xz)

Planar case: u_0 = (1, 0, 0, 0, 0, +-m, 0, 0) is fixed and F(c) = eta_2(P^k_c(u_0)) is solved for c.
"""
import numpy as np

from .moser import Moser
from .correction import solve

XI1, XI3, ETA2, ETA3 = 1, 3, 6, 7
Y_SETS = {('xi1', 'rho_xz'): [XI3, ETA2], ('xi1', 'rho_x'): [ETA2, ETA3], ('xi3', 'rho_xz'): [XI1, ETA2],
          ('xi1', 'rho'): [ETA2]}
SYMMETRY = {'rho': 'planar', 'rho_xz': 'II', 'rho_x': 'IV'}      # symmetry type of the collision orbit


def collision_orbit(primary, eta1, C, eta3_sign, k, direction, target='rho_xz', section='xi1',
                    extended=False, tol=1e-10, max_iter=20):
    """Newton's method of Lemma 4.2 from (eta_1, C).  eta3_sign = 0: planar.

    Returns dict(eta1, C, residual, iterations) and the orbit data (z0, s, V, T, symmetry, sys) of the
    segment from the collision point to the target fixed point set."""
    sys = Moser(C, primary, extended)
    eta1, c, m = sys.num(eta1), sys.c, sys.m
    if not eta3_sign:
        eta1 = m if eta1 > 0 else -m
    Y = Y_SETS[(section, 'rho' if not eta3_sign else target)]
    isec = XI1 if section == 'xi1' else XI3
    for it in range(max_iter + 1):
        sys = Moser(-2 * c, primary, extended)
        u0 = sys.array([1, 0, 0, 0, 0, 0, 0, 0])
        u0[5] = eta1
        du0 = sys.array(np.zeros(8))
        du0[5] = 1
        if eta3_sign:
            u0[ETA3] = eta3_sign * np.sqrt(m * m - eta1 * eta1)
            du0[ETA3] = -eta1 / u0[ETA3]
        r = sys.crossing(u0, isec, direction, k, variational=True)
        X = sys.field(r['z'])
        P = np.eye(8) - np.outer(X, np.eye(8)[isec]) / X[isec]          # derivative of the Poincare map
        F = r['z'][Y]
        DF = np.column_stack([P @ r['V'] @ du0, P @ r['dz_dc']])[Y]
        residual = np.sqrt(F @ F)
        if residual < tol or it == max_iter:
            break
        if eta3_sign:
            d = solve(DF, F)
            eta1, c = eta1 - d[0], c - d[1]
        else:
            c = c - F[0] / DF[0, 1]
    symmetry = SYMMETRY['rho' if not eta3_sign else target]
    n = {'planar': 2, 'II': 2, 'IV': 4}[symmetry]
    return dict(eta1=eta1, C=-2 * c, residual=residual, iterations=it, z0=u0, s=r['s'], V=r['V'],
                T=n * r['t'], symmetry=symmetry, sys=sys)
