"""Table 2: periodic collision orbits appearing along continuations of period-multiplying bifurcations
of NRHOs.

Every row is computed by Newton's method of Lemma lem:collision_newton on F(eta_1, c) (planar
orbits: on c only) for the k-th crossing of the section xi3 = 0 in the given direction, to
|F| < 1e-10, and the stability indices of the collision orbit are printed.
"""
from common import stability, line_collision, header_collision
from src.collision import collision_orbit

# row: (regularized primary, eta_1, C, sign of eta_3 (0: planar), k, crossing direction, target fixed point set)
ROWS = {
    '1': ('moon', 0.002862412451196783, 2.9483643141079363, -1, 1, -1, 'rho_xz'),
    '2': ('moon', -0.0020506482048347737, 3.0842810797788567, -1, 2, 1, 'rho_xz'),
    '3': ('moon', -0.0015068301252738375, 3.1058906286637638, -1, 2, -1, 'rho_xz'),
    '4': ('moon', -0.001011736308949541, 3.1290385557978473, -1, 3, 1, 'rho_xz'),
    '5': ('moon', -0.0007239257684689272, 3.150741882104818, -1, 3, -1, 'rho_xz'),
}

# rows computed in extended precision (real128, to |F| < 1e-20)
EXTENDED = set()

header_collision()
for name, row in ROWS.items():
    ext = name in EXTENDED
    o = collision_orbit(*row, section='xi3', extended=ext, tol=1e-20 if ext else 1e-10)
    line_collision(name, o, stability(o), ext)
