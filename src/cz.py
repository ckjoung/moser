"""Transverse Conley-Zehnder index of a periodic orbit (Appendix A).

The linearized flow is integrated in Moser regularized coordinates and mapped to physical
variations; in the symplectic transverse frame (U_1, U_2, V_1, V_2) of Moreno et al. [49],

    U_1 = J_0 grad H / <grad H, J_0^T Omega K_0 grad H>,   V_1 = K_0 grad H,

with (U_2, V_2) obtained from e_pz, e_z by symplectic projection off <X_H, Z> (Z = grad H/|grad H|^2),
it gives a path Psi: [0, T] -> Sp(4) with Psi(0) = id.  The path is extended from Psi(T) to the base
point B_+ = -id or B_- = diag(2, 1/2) + (-id) without crossing the Maslov cycle, using the symplectic
normal form of Psi(T), and mu_CZ is the winding number of t -> det_C(rho(Psi(t)))^2, rho the
unitary part.

Numerics: the path is sampled at every Taylor step, subdivided until consecutive arguments of
det_C(rho) differ by less than 0.2; rho(S) of strongly expanding S uses the exact symplectic
relation between large and small singular vectors.
"""
import numpy as np
import heyoka as hy

from .cr3bp import MU, grad_hamiltonian
from .correction import SYMMETRY

OM6 = np.block([[np.zeros((3, 3)), -np.eye(3)], [np.eye(3), np.zeros((3, 3))]])
OM4 = np.block([[np.zeros((2, 2)), -np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
OM2 = np.array([[0., -1.], [1., 0.]])
J0, K0 = np.zeros((6, 6)), np.zeros((6, 6))
J0[0, 1], J0[1, 0], J0[3, 4], J0[4, 3] = -1., 1., 1., -1.
K0[0, 4], K0[1, 3], K0[3, 1], K0[4, 0] = -1., 1., -1., 1.


# ------------------------------------------------------------------------------------- frame
def omega(u, v):
    return np.einsum('...i,ij,...j->...', u, OM6, v)


def frame(X, mu=MU):
    """Symplectic transverse frame (U1, U2, V1, V2) at the points X (n, 6)."""
    g = grad_hamiltonian(X, mu)
    XH, Z = -g @ OM6.T, g / np.sum(g * g, axis=1)[:, None]
    U1 = (g @ J0.T) / np.einsum('ni,ij,nj->n', g, J0.T @ OM6 @ K0, g)[:, None]
    V1 = g @ K0.T

    def project(w):
        w = w - omega(w, XH)[:, None] * Z - omega(Z, w)[:, None] * XH
        return w - omega(w, V1)[:, None] * U1 - omega(U1, w)[:, None] * V1
    U2, V2 = project(np.tile(np.eye(6)[5], (len(g), 1))), project(np.tile(np.eye(6)[2], (len(g), 1)))
    return U1, U2 / omega(U2, V2)[:, None], V1, V2


def frame_coordinates(W, F):
    """Coordinates (u1, u2, v1, v2) of the vectors W (n, 6, m) in the frame F."""
    U1, U2, V1, V2 = F
    return np.stack([np.einsum('nim,ij,nj->nm', W, OM6, V1), np.einsum('nim,ij,nj->nm', W, OM6, V2),
                     np.einsum('ni,ij,njm->nm', U1, OM6, W), np.einsum('ni,ij,njm->nm', U2, OM6, W)], axis=1)


_JAC = {}


def _moser_jacobians():
    """D(to_physical) (6x8) and D(from_physical) (8x6) of the Moser map of the light primary."""
    if not _JAC:
        z = list(hy.make_vars('xi0', 'xi1', 'xi2', 'xi3', 'eta0', 'eta1', 'eta2', 'eta3'))
        m = hy.par[0]
        x = [z[k] / (1. - z[0]) for k in (1, 2, 3)]
        y = [z[4] * z[k] + (1. - z[0]) * z[4 + k] for k in (1, 2, 3)]
        phys = [y[0] + (1. - m), y[1], y[2], -x[0], -x[1] + (1. - m), -x[2]]
        X = list(hy.make_vars('x', 'y', 'z', 'px', 'py', 'pz'))
        xx, yy = [-X[3], -(X[4] - (1. - m)), -X[5]], [X[0] - (1. - m), X[1], X[2]]
        n2, xy = hy.sum([v ** 2 for v in xx]), hy.sum([a * b for a, b in zip(xx, yy)])
        reg = [(n2 - 1.) / (n2 + 1.)] + [2. * v / (n2 + 1.) for v in xx] + [xy] + [0.5 * (n2 + 1.) * yy[k] - xy * xx[k]
                                                                                 for k in range(3)]
        _JAC['to'] = hy.cfunc([hy.diff(f, v) for f in phys for v in z], vars=z)
        _JAC['from'] = hy.cfunc([hy.diff(f, v) for f in reg for v in X], vars=X)
    return _JAC['to'], _JAC['from']


# ------------------------------------------------------------------------------ unitary part
def unitary_part(S, sep=1e3):
    """rho(S) = (S S^T)^{-1/2} S; the SVD S = W Sigma V^T gives rho(S) = W V^T.
    Symplectic singular values occur in reciprocal pairs (sigma, 1/sigma).
    For sigma > sep, reconstruct the reciprocal singular vectors using
    S (Om v) = Om w / sigma, hence (v', w') = (Om v, Om w)."""
    W, sig, Vt = np.linalg.svd(S)
    V, m = Vt.T, S.shape[0]
    Om = OM4 if m == 4 else OM2
    for i in range(m // 2):
        if sig[i] > sep:
            V[:, m - 1 - i], W[:, m - 1 - i] = Om @ V[:, i], Om @ W[:, i]
    return W @ V.T


def cdet_rho(S):
    U = unitary_part(S)
    n = U.shape[0] // 2
    return np.linalg.det(U[:n, :n] + 1j * U[n:, :n])


def winding(mats):
    c = np.array([cdet_rho(M) for M in mats])
    return np.angle(c[1:] / c[:-1])


# --------------------------------------------------------------------------- symplectic path
def symplectic_path(orbit, k=1, max_dangle=0.2):
    """Psi(t) along the k-fold cover of a corrected orbit (moser.Moser flow, any precision)."""
    sys = orbit['sys']
    S_total = orbit['s'] * SYMMETRY[orbit['symmetry']][2] * k
    to_j, from_j = _moser_jacobians()
    R = -np.eye(6) if sys.primary == 'earth' else np.eye(6)
    pars = np.array([float(sys.m)])
    tofl = lambda a: np.array([float(v) for v in np.ravel(a)]).reshape(np.shape(a))
    X0 = tofl(sys.to_physical(orbit['z0']))
    F0 = frame(X0[None])
    dz0 = from_j(R @ X0, pars=pars).reshape(8, 6) @ R @ np.stack([f[0] for f in F0], axis=1)

    def psi(states):
        zs = np.array([tofl(d['z']) for d in states])
        Vs = np.array([tofl(d['V']) for d in states])
        Dto = to_j(np.ascontiguousarray(zs.T)).T.reshape(-1, 6, 8)
        W = np.einsum('ij,njk,nkl,lm->nim', R, Dto, Vs, dz0)
        return frame_coordinates(W, frame(tofl(sys.to_physical(zs))))

    path, c_prev, total = [np.eye(4)], 1. + 0j, 0.
    for s0, s1, dense in sys.steps(orbit['z0'], S_total, variational=True):
        nsub = 1
        while True:
            S = psi([dense(s0 + (s1 - s0) * (i + 1) / nsub) for i in range(nsub)])
            c = np.array([cdet_rho(M) for M in S])
            d = np.angle(np.concatenate([[c[0] / c_prev], c[1:] / c[:-1]]))
            if np.max(np.abs(d)) < max_dangle or nsub >= 4096:
                break
            nsub *= 4
        total, c_prev = total + d.sum(), c[-1]
        path += list(S)
    return np.array(path), total


# ---------------------------------------------------------------------------- normal form
def _sinv(P, Om):
    return -Om @ P.T @ Om


def _plane(B, om):
    """Normal form basis (e, f), om(e, f) = 1, of a 2x2 symplectic B and its kind."""
    d = np.linalg.det(B)
    if d > 0.5:
        B = B / np.sqrt(d)
    w, v = np.linalg.eig(B)
    if abs(np.trace(B)) < 2.:
        i = int(np.argmax(w.imag))
        a, b = v[:, i].real, v[:, i].imag
        c = om(a, b)
        e, f = (a, b) if c > 0 else (b, a)
        return 'ell', e / np.sqrt(abs(c)), f / np.sqrt(abs(c))
    i = int(np.argmax(np.abs(w)))
    e, f = v[:, i].real, v[:, 1 - i].real
    return ('hyp+' if np.trace(B) > 0 else 'hyp-'), e, f / om(e, f)


def normal_form(M):
    """Symplectic P and the blocks of N = P^{-1} M P: ('ell', k, angle), ('hyp+-', k, lambda) on the
    k-th symplectic plane, or [('cpx', r, phi)] for a complex quadruple.  The eigenvector of a large
    multiplier is taken from M, that of its inverse from the exact inverse -Om M^T Om."""
    m = M.shape[0]
    n, Om = m // 2, (OM4 if m == 4 else OM2)
    om = lambda u, v: u @ Om @ v
    Minv = _sinv(M, Om)
    w, vec = np.linalg.eig(M)
    i0 = int(np.argmax(np.abs(w)))
    lam = w[i0]
    big = abs(lam) > 1. + max(1e-6, 1e-12 * np.abs(M).max())
    planes, cpx = [], None
    if big and m == 4 and abs(lam.imag) > 1e-9 * abs(lam):
        wi, vi = np.linalg.eig(Minv)
        j = int(np.argmin(np.abs(wi - np.conj(lam))))
        G, E = np.column_stack([vi[:, j].real, vi[:, j].imag]), np.column_stack([vec[:, i0].real, vec[:, i0].imag])
        cpx = np.column_stack([E, G @ np.linalg.inv(E.T @ Om @ G)])
    elif big:
        wi, vi = np.linalg.eig(Minv)
        e, f = vec[:, i0].real, vi[:, int(np.argmin(np.abs(wi - lam.real)))].real
        planes.append(('hyp+' if lam.real > 0 else 'hyp-', e, f / om(e, f)))
    elif m == 2:
        planes.append(_plane(M, om))
    else:
        for i in [i for i in range(m) if w[i].imag > 1e-9]:
            a, b = vec[:, i].real, vec[:, i].imag
            c = om(a, b)
            e, f = (a, b) if c > 0 else (b, a)
            planes.append(('ell', e / np.sqrt(abs(c)), f / np.sqrt(abs(c))))
        if len(planes) != 2:
            raise ValueError(f'degenerate spectrum: {w}')
    if m == 4 and cpx is None and len(planes) == 1:
        # restriction of M to the symplectic complement of the first plane
        e1, f1 = planes[0][1:]
        sproj = lambda v: v - om(v, f1) * e1 + om(v, e1) * f1
        basis = []
        for v in np.eye(4):
            v = sproj(v)
            for u in basis:
                v = v - (v @ u) * u
            if np.linalg.norm(v) > 1e-8 and len(basis) < 2:
                basis.append(v / np.linalg.norm(v))
        c1, c2 = basis
        Cb = np.column_stack([c1, c2 / om(c1, c2)])
        MC = np.column_stack([sproj(M @ Cb[:, j]) for j in range(2)])
        B = np.array([[om(MC[:, j], Cb[:, 1]) for j in range(2)], [om(Cb[:, 0], MC[:, j]) for j in range(2)]])
        kind, e2, f2 = _plane(B, lambda u, v: u[0] * v[1] - u[1] * v[0])
        planes.append((kind, Cb @ e2, Cb @ f2))
    if cpx is not None:
        P = cpx
    else:
        P = np.zeros((m, m))
        for k, (_, e, f) in enumerate(planes):
            P[:, k], P[:, n + k] = e, f
        if n == 2:
            for c in (1, 3):
                P[:, c] = P[:, c] - om(P[:, c], P[:, 2]) * P[:, 0] + om(P[:, c], P[:, 0]) * P[:, 2]
            P[:, 3] /= om(P[:, 1], P[:, 3])
    P[:, n:] = -P[:, n:]                                  # P^T Om P = Om
    N = _sinv(P, Om) @ M @ P
    if cpx is not None:
        A = N[:2, :2]
        return P, [('cpx', np.sqrt(abs(np.linalg.det(A))), np.angle(A[0, 0] + 1j * A[1, 0]))]
    blocks = []
    for k, (kind, _, _) in enumerate(planes):
        B = N[np.ix_([k, n + k], [k, n + k])]
        blocks.append((kind, k, np.mod(np.arctan2(B[1, 0], B[0, 0]), 2 * np.pi) if kind == 'ell' else B[0, 0]))
    return P, blocks


def _assemble(blocks, m, params):
    n, N = m // 2, np.zeros((m, m))
    for (kind, k, _), p in zip(blocks, params):
        if kind == 'cpx':
            A = p[0] * np.array([[np.cos(p[1]), -np.sin(p[1])], [np.sin(p[1]), np.cos(p[1])]])
            N[:2, :2], N[2:, 2:] = A, np.linalg.inv(A).T
        elif kind == 'ell':
            N[np.ix_([k, n + k], [k, n + k])] = [[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]]
        else:
            N[np.ix_([k, n + k], [k, n + k])] = np.diag([p, 1. / p])
    return N


def extension(M, steps=400):
    """Path from M to a base point avoiding the Maslov cycle. Let M = P N P^{-1}, 
    with P symplectic and N in symplectic normal form.
    (Step 1) Extend to P N1 P^{-1}, normalizing hyperbolic moduli to 2.
    (Step 2) Extend to N1 by contracting P to the identity in Sp(2n).
    (Step 3) Extend N1 to the base point B_+ or B_- without crossing the Maslov cycle."""
    m = M.shape[0]
    n, Om = m // 2, (OM4 if m == 4 else OM2)
    P, blocks = normal_form(M)
    p0 = [b[2] if b[0] != 'cpx' else (b[1], b[2]) for b in blocks]
    p1 = [np.sign(p) * (2. if abs(p) >= 1 else .5) if b[0] in ('hyp+', 'hyp-') else
          ((2. if p[0] >= 1 else .5), p[1]) if b[0] == 'cpx' else p for b, p in zip(blocks, p0)]
    Pinv, path = _sinv(P, Om), []
    for tau in np.linspace(0, 1, steps):                                   # Step 1
        par = [np.sign(a) * abs(a) ** (1 - tau) * abs(b) ** tau if bl[0] in ('hyp+', 'hyp-') else
               (a[0] ** (1 - tau) * b[0] ** tau, a[1]) if bl[0] == 'cpx' else a for bl, a, b in zip(blocks, p0, p1)]
        path.append(P @ _assemble(blocks, m, par) @ Pinv)
    path[0] = M
    N1 = _assemble(blocks, m, p1)
    W, sv, Vt = np.linalg.svd(P)                                           # Step 2: P = U S, U unitary
    U, S = W @ Vt, Vt.T @ np.diag(sv) @ Vt
    ev, Qs = np.linalg.eigh(S)
    lu, Vu = np.linalg.eig(U[:n, :n] + 1j * U[n:, :n])
    Vu_inv = np.linalg.inv(Vu)
    for tau in np.linspace(1, 0, steps):
        u = Vu @ np.diag(np.exp(1j * tau * np.angle(lu))) @ Vu_inv
        Pt = np.block([[u.real, -u.imag], [u.imag, u.real]]) @ Qs @ np.diag(ev ** tau) @ Qs.T
        path.append(Pt @ N1 @ _sinv(Pt, Om))
    for tau in np.linspace(0, 1, steps):                                   # Step 3
        par = []
        for b, p in zip(blocks, p1):
            if b[0] == 'ell':
                par.append(p + tau * (np.pi - p))
            elif b[0] == 'hyp-':
                par.append(-(abs(p) ** (1 - tau)))
            elif b[0] == 'hyp+':
                par.append(p)
            else:
                ph = np.pi if p[1] >= 0 else -np.pi
                par.append((p[0], p[1] + min(1, 2 * tau) * (ph - p[1])) if tau <= .5 else (p[0] ** (2 - 2 * tau), ph))
        path.append(_assemble(blocks, m, par))
    if [b[0] for b in blocks].count('hyp+') == 2:
        # diag(2, 1/2) + diag(2, 1/2) = diag(A, A^-T) with A = 2 id: rotate A by pi, then shrink to -id
        for tau in np.linspace(0, 1, steps):
            phi, r = min(1, 2 * tau) * np.pi, (2. if tau <= .5 else 2. ** (2 - 2 * tau))
            A = r * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            path.append(np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), np.linalg.inv(A).T]]))
    return path


def index_of(path, total):
    """CZ index of the sampled path (path[0] = id), arg det_C rho changing by `total` along it."""
    d = winding(extension(path[-1]))
    if np.max(np.abs(d)) > 0.5:
        d = winding(extension(path[-1], steps=4000))
    return int(np.round((total + d.sum()) / np.pi))


def conley_zehnder(orbit, k=1, planar=False):
    """mu_CZ of the k-fold cover of a corrected orbit; planar=True also returns the in-plane and
    vertical indices from the decoupled blocks (u1, v1) and (u2, v2). The orbit is assumed to be 
    non-degenerate with distinct Floquet multipliers."""
    path, total = symplectic_path(orbit, k)
    cz = index_of(path, total)
    if not planar:
        return cz
    out = [cz]
    for ii in ([0, 2], [1, 3]):
        p2 = path[:, ii][:, :, ii]
        out.append(index_of(p2, winding(p2).sum()))
    return tuple(out)
