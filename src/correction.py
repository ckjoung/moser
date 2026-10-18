"""Newton correction of symmetric periodic orbits at fixed Jacobi constant (Sections 3.2, 3.5, 3.6).

Fixed point sets of the regularized symmetries, eq. (15), with the Newton unknowns at the
initial point (besides the integration time s) and the end conditions Y, eq. (23):

    Fix(rho_xz) = {xi_1 = xi_3 = eta_0 = eta_2 = 0}:  Case 1, X = (xi_2, eta_3, s),  Y = (xi_1, xi_3, eta_2)
    Fix(rho_x)  = {xi_1 = eta_0 = eta_2 = eta_3 = 0}: Case 2, X = (xi_2, xi_3, s),  Y = (xi_1, eta_2, eta_3)

In both cases xi_0 is fixed by |xi| = 1 and eta_1 by the energy condition K_c = 0.  The remaining
condition eta_0 = 0 at the end point follows from g_2 = 0 and is checked a posteriori.

Symmetry types (Section 3.2): the orbit starts on the first fixed point set and reaches the second
after 1/n of its period.  Planar orbits are rho_xz-symmetric orbits with xi_3 = eta_3 = 0.
"""
import numpy as np

FIX = {'rho_xz': dict(zero=[1, 3, 4, 6], free=[2, 7], Y=[1, 3, 6], sigma=[1, -1, 1, -1, -1, 1, -1, 1]),
       'rho_x': dict(zero=[1, 4, 6, 7], free=[2, 3], Y=[1, 6, 7], sigma=[1, -1, 1, 1, -1, 1, -1, -1])}
SYMMETRY = {'planar': ('rho_xz', 'rho_xz', 2), 'I': ('rho_x', 'rho_x', 2), 'II': ('rho_xz', 'rho_xz', 2),
            'III': ('rho_x', 'rho_xz', 4), 'IV': ('rho_xz', 'rho_x', 4)}
ETA1 = 5


def solve(A, b):
    """Linear solve; Gaussian elimination with partial pivoting for real128 arrays."""
    if A.dtype != object and A.dtype.kind == 'f':
        return np.linalg.solve(A, b)
    A, b = A.copy(), b.copy()
    n = len(b)
    for i in range(n):
        p = i + int(np.argmax([abs(float(A[j, i])) for j in range(i, n)]))
        A[[i, p]], b[[i, p]] = A[[p, i]], b[[p, i]]
        for j in range(i + 1, n):
            r = A[j, i] / A[i, i]
            A[j, i:] -= r * A[i, i:]
            b[j] -= r * b[i]
    x = b.copy()
    for i in reversed(range(n)):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def lift(sys, z, fix, u=None):
    """Point on the fixed point set and on K_c = 0 with free coordinates u (default: those of z).
    Returns z and dz/du, shape (8, 2)."""
    F = FIX[fix]
    z = np.array(z)
    z[F['zero']] = 0
    if u is not None:
        z[F['free']] = u
    z[0] = np.sign(z[0]) * np.sqrt(1 - z[1] ** 2 - z[2] ** 2 - z[3] ** 2)
    for _ in range(30):
        d = sys.K(z) / sys.grad_K(z)[ETA1]
        z[ETA1] -= d
        if abs(d) <= 4 * abs(z[ETA1]) * (1e-33 if sys.extended else 1e-16):
            break
    g = sys.grad_K(z)
    D = np.zeros((8, 2), dtype=z.dtype)
    for k, i in enumerate(F['free']):
        D[i, k] = 1
        if i < 4:
            D[0, k] = -z[i] / z[0]
        D[ETA1, k] = -(g[i] + g[0] * D[0, k]) / g[ETA1]
    return z, D


def correct(sys, X, T, symmetry, N=1, tol=1e-10, max_iter=20):
    """Correct the orbit through the physical symmetric point X with period T (single shooting,
    or multiple shooting with N nodes on the reduced segment, Section 3.6).

    Returns dict(z0, s, V, T, residual, iterations, symmetry): z0 the corrected initial point, s
    the regularized time and V the linearized flow of the reduced segment, T the period."""
    start, end, n = SYMMETRY[symmetry]
    Y = FIX[end]['Y']
    z0, _ = lift(sys, sys.from_physical(X), start)
    s = sys.time_to_s(z0, sys.num(T) / n)
    nodes = [z0]
    for i in range(1, N):
        nodes.append(sys.flow(nodes[-1], s / N)['z'])
    x = np.concatenate([z0[FIX[start]['free']]] + nodes[1:] + [[s]])
    for it in range(max_iter + 1):
        z0, D = lift(sys, z0, start, x[:2])
        s = x[-1]
        zs, segs, F, DF = z0, [], [], np.zeros((len(x), len(x)), dtype=x.dtype)
        for i in range(N):
            r = sys.flow(zs if i == 0 else x[2 + 8 * (i - 1): 2 + 8 * i], s / N, variational=True)
            segs.append(r)
            rows = slice(8 * i, 8 * i + (8 if i < N - 1 else len(Y)))
            sel = slice(None) if i < N - 1 else Y
            if i == 0:
                DF[rows, :2] = (r['V'] @ D)[sel]
            else:
                DF[rows, 2 + 8 * (i - 1): 2 + 8 * i] = r['V'][sel]
            if i < N - 1:
                F.append(r['z'] - x[2 + 8 * i: 10 + 8 * i])
                DF[rows, 2 + 8 * i: 10 + 8 * i] = -np.eye(8)
            else:
                F.append(r['z'][Y])
            DF[rows, -1] = sys.field(r['z'])[sel] / N
        F = np.concatenate(F)
        z_end = segs[-1]['z']
        residual = np.sqrt(F @ F + z_end[4] ** 2)              # |Y| and the a posteriori eta_0
        if residual < tol or it == max_iter:
            break
        x = x - solve(DF, F)
    V = np.eye(8, dtype=x.dtype)
    for r in segs:
        V = r['V'] @ V
    return dict(z0=z0, s=s, V=V, T=n * sum(r['t'] for r in segs), residual=residual, iterations=it,
                symmetry=symmetry, sys=sys, N=N)
