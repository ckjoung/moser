"""The spatial circular restricted three-body problem in rotating coordinates (Section 2).

X = (x, y, z, p_x, p_y, p_z); the Earth is at (-mu, 0, 0) and the Moon at (1 - mu, 0, 0).
"""
import numpy as np

MU = 1.215058560962404e-2


def hamiltonian(X, mu=MU):
    x, y, z, px, py, pz = np.moveaxis(np.asarray(X), -1, 0)
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    return 0.5 * (px ** 2 + py ** 2 + pz ** 2) - (1 - mu) / r1 - mu / r2 + px * y - py * x


def grad_hamiltonian(X, mu=MU):
    """Gradient of H for an array of states X of shape (n, 6)."""
    x, y, z, px, py, pz = np.asarray(X, float).T
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    a, b = (1 - mu) / r1 ** 3, mu / r2 ** 3
    return np.stack([a * (x + mu) + b * (x - 1 + mu) - py, (a + b) * y + px, (a + b) * z,
                     px + y, py - x, pz], axis=1)


def symmetric_point(x=0., z=0., py=0., pz=0.):
    """Initial point (x, 0, z, 0, p_y, p_z) on Fix(rho_xz) (p_z = 0) or Fix(rho_x) (z = 0)."""
    return np.array([x, 0., z, 0., py, pz])
