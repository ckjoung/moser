"""Moser regularized CR3BP (Section 3.1).

Regularized state z = (xi_0, xi_1, xi_2, xi_3, eta_0, eta_1, eta_2, eta_3) on T*S^3.  The light
primary (mass m = mu) is regularized with eqs. (3), (4).  The Earth is
regularized with the same formulas after the point reflection X -> -X of phase space, which maps
the CR3BP with mass ratio mu to the one with mass ratio 1 - mu and the Earth to the position of the
light primary (m = 1 - mu).

The flow is the constrained system (11) for Q_c, with the Lagrange multipliers divided by
|xi|^2.  The physical time t is integrated alongside, dt/ds = (1 - xi_0)|eta|.

extended=True integrates in quadruple precision (heyoka real128, about 34 digits); all arrays are
then numpy arrays of real128.
"""
import numpy as np
import heyoka as hy

from .cr3bp import MU

Q128 = hy.real128
_COMPILED = {}


def _system():
    xi = list(hy.make_vars('xi0', 'xi1', 'xi2', 'xi3'))
    et = list(hy.make_vars('eta0', 'eta1', 'eta2', 'eta3'))
    m, c = hy.par[0], hy.par[1]
    y = [et[0] * xi[k] + (1. - xi[0]) * et[k] for k in (1, 2, 3)]
    r_other = hy.sqrt((y[0] + 1.) ** 2 + y[1] ** 2 + y[2] ** 2)
    f = (1. - (1. - xi[0]) * (c + 0.5) + (1. - xi[0]) * (et[1] * xi[2] - et[2] * xi[1])
         - (1. - m) * (1. - xi[0]) * (1. / r_other + y[0] + 0.5 * (1. - m)))          # eq. (8)
    eta_norm = hy.sqrt(hy.sum([e ** 2 for e in et]))
    Q = f ** 2 * eta_norm ** 2 / (2. * m)                                             # eq. (9)
    K = f * eta_norm - m
    dQ_dxi, dQ_deta = [hy.diff(Q, v) for v in xi], [hy.diff(Q, v) for v in et]
    xi_sq = hy.sum([v ** 2 for v in xi])
    a = hy.sum([xi[j] * dQ_deta[j] for j in range(4)]) / xi_sq
    b = (hy.sum([et[j] * dQ_deta[j] for j in range(4)]) - hy.sum([xi[j] * dQ_dxi[j] for j in range(4)])) / xi_sq
    rhs = [dQ_deta[k] - xi[k] * a for k in range(4)] + [-dQ_dxi[k] - xi[k] * b + et[k] * a for k in range(4)]
    t = hy.make_vars('t')
    z = xi + et
    sys = list(zip(z, rhs)) + [(t, (1. - xi[0]) * eta_norm)]
    return z, sys, rhs, K


def _compile(extended):
    if extended not in _COMPILED:
        z, sys, rhs, K = _system()
        fp = Q128 if extended else float
        zeros = lambda n: np.zeros(n, dtype=fp)
        pars = np.array([MU, -1.5], dtype=fp)
        kw = dict(compact_mode=True, fp_type=fp) if extended else dict(compact_mode=True, high_accuracy=True)
        flow = hy.taylor_adaptive(sys, zeros(9), pars=pars, **kw)
        # linearized flow with respect to the initial point and to the energy level c = par[1]
        var = hy.taylor_adaptive(hy.var_ode_sys(sys, z + [hy.par[1]], order=1), zeros(9 + 81), pars=pars, **kw)
        cf = lambda e: hy.cfunc(e, vars=z, fp_type=fp)
        _COMPILED[extended] = dict(flow=flow, var=var, field=cf(rhs), K=cf([K]), grad_K=cf([hy.diff(K, v) for v in z]))
    return _COMPILED[extended]


class Moser:
    """Regularized CR3BP at Jacobi constant C, regularized at primary 'moon' or 'earth'."""

    def __init__(self, C, primary='moon', extended=False, mu=MU):
        self.extended = extended
        self.num = (lambda v: v if isinstance(v, Q128) else Q128(str(v))) if extended else float
        self.primary = primary
        self.m = self.num(mu) if primary == 'moon' else 1 - self.num(mu)
        self.C = self.num(C)
        self.c = -self.C / 2
        self.pars = np.array([self.m, self.c], dtype=Q128 if extended else float)
        self._f = _compile(extended)

    def array(self, a):
        return np.array([self.num(v) for v in np.ravel(a)], dtype=Q128 if self.extended else float).reshape(np.shape(a))

    # coordinate changes, eqs. (3), (4), (5)
    def from_physical(self, X):
        X = self.array(X)
        X = -X if self.primary == 'earth' else X
        m = self.m
        x = -np.array([X[3], X[4] - (1 - m), X[5]])
        y = np.array([X[0] - (1 - m), X[1], X[2]])
        xx, xy = x @ x, x @ y
        return np.concatenate([[(xx - 1) / (xx + 1)], 2 * x / (xx + 1), [xy], (xx + 1) / 2 * y - xy * x])

    def to_physical(self, z):
        z = np.asarray(z)
        m = self.m
        x = z[..., 1:4] / (1 - z[..., 0:1])
        y = z[..., 4:5] * z[..., 1:4] + (1 - z[..., 0:1]) * z[..., 5:8]
        X = np.concatenate([y[..., :1] + (1 - m), y[..., 1:], -x[..., :1], -x[..., 1:2] + (1 - m), -x[..., 2:]], axis=-1)
        return -X if self.primary == 'earth' else X

    # energy K_c = f|eta| - m and the regularized vector field
    def K(self, z):
        return self._f['K'](z, pars=self.pars)[0]

    def grad_K(self, z):
        return self._f['grad_K'](z, pars=self.pars)

    def field(self, z):
        return self._f['field'](z, pars=self.pars)

    # integration
    def _start(self, z0, variational):
        ta = self._f['var' if variational else 'flow']
        ta.time = self.num(0)
        ta.pars[:] = self.pars
        st = np.zeros(ta.state.shape, dtype=ta.state.dtype)
        st[:8] = z0
        if variational:
            st[9:] = np.diag([1] * 8 + [0]).astype(ta.state.dtype).ravel()    # dz/dz0 = I, dz/dc = 0, dt = 0
        ta.state[:] = st
        return ta

    @staticmethod
    def _unpack(state, variational):
        out = dict(z=state[:8].copy(), t=state[8])
        if variational:
            D = state[9:].reshape(9, 9)
            out.update(V=D[:8, :8].copy(), dz_dc=D[:8, 8].copy())
        return out

    def flow(self, z0, s, variational=False):
        """Integrate for regularized time s: dict(z, t) and, if variational, V = dz/dz0 and dz/dc."""
        ta = self._start(z0, variational)
        oc = ta.propagate_until(self.num(s))[0]
        if oc != hy.taylor_outcome.time_limit:
            raise RuntimeError(f'integration stopped: {oc}')
        return self._unpack(ta.state, variational)

    def steps(self, z0, s_end=None, variational=False):
        """Yield (s0, s1, dense) for each Taylor step; dense(s) is the state at s in [s0, s1]."""
        ta = self._start(z0, variational)

        def dense(s):
            ta.update_d_output(s)
            return self._unpack(ta.d_output, variational)
        while s_end is None or ta.time < s_end:
            s0 = ta.time
            kw = dict(write_tc=True) if s_end is None else dict(max_delta_t=s_end - ta.time, write_tc=True)
            oc = ta.step(**kw)[0]
            if oc not in (hy.taylor_outcome.success, hy.taylor_outcome.time_limit):
                raise RuntimeError(f'integration stopped: {oc}')
            yield s0, ta.time, dense

    def time_to_s(self, z0, T):
        """Regularized time s at which the orbit through z0 reaches physical time T."""
        T = self.num(T)
        for s0, s1, dense in self.steps(z0):
            if dense(s1)['t'] >= T:
                s = s1
                for _ in range(50):
                    d = dense(s)
                    ds = (d['t'] - T) / ((1 - d['z'][0]) * np.sqrt(d['z'][4:] @ d['z'][4:]))
                    s = s - ds
                    if abs(ds) < (1e-32 if self.extended else 1e-15) * max(1, abs(s)):
                        break
                return s

    def crossing(self, z0, index, direction, k, variational=False, s_min=1e-9):
        """State at the k-th crossing of the section z[index] = 0 in the given direction (+1, -1);
        the crossing time is located on the flow, then the linearized flow is integrated up to it."""
        n = 0
        for s0, s1, dense in self.steps(z0):
            grid = [s0 + (s1 - s0) * i / 8 for i in range(9)]
            vals = [dense(u)['z'][index] for u in grid]
            for a, va, vb in zip(grid[:-1], vals[:-1], vals[1:]):
                if va == 0 or (va > 0) == (vb > 0) or (vb > va) != (direction > 0) or a < s_min:
                    continue
                n += 1
                if n == k:
                    s = a
                    for _ in range(50):                 # Newton iteration for the crossing time
                        z = dense(s)['z']
                        ds = z[index] / self.field(z)[index]
                        s = s - ds
                        if abs(ds) < (1e-32 if self.extended else 1e-15):
                            break
                    return dict(self.flow(z0, s, variational), s=s)
