"""Table 1: symmetric periodic collision orbits found along families related to the planar L1 and L2
Lyapunov families.

Every row is computed by Newton's method of Lemma lem:collision_newton on F(eta_1, c) (planar
orbits: on c only) for the k-th crossing of the section xi1 = 0 in the given direction, to
|F| < 1e-10, and the stability indices of the collision orbit are printed.
Rows 2-4 are computed in extended precision.
"""
from common import stability, line_collision, header_collision
from src.collision import collision_orbit

# row: (regularized primary, eta_1, C, sign of eta_3 (0: planar), k, crossing direction, target fixed point set)
ROWS = {
    '1': ('earth', -0.987849414390376, 1.4220430991677298, 0, 2, 1, 'rho_xz'),
    '2': ('earth', '-0.987784562508307699174063148679729542', '1.42077137482574713861765657235182653', -1, 5, 1, 'rho_xz'),
    '3': ('earth', '-0.987849413987455228484538667634540262', '1.42106440512223076647411747768869228', -1, 4, -1, 'rho_xz'),
    '4': ('earth', '-0.987784288116568216977220549008828969', '1.42076606218939205525743528974016548', 1, 3, -1, 'rho_x'),
    '5': ('moon', 0.01215058560962404, 2.7325778910704526, 0, 1, -1, 'rho_xz'),
    '6': ('moon', 0.004606224461736393, 2.7820369901748045, -1, 1, 1, 'rho_x'),
    '7': ('moon', 0.004268055187760349, 2.795788852484662, 1, 3, -1, 'rho_xz'),
    '8': ('moon', 0.004009329017755287, 2.8079202798880774, -1, 2, 1, 'rho_x'),
    '9': ('moon', 0.010691862769384565, 2.7402468921461147, 1, 1, -1, 'rho_xz'),
}

# rows computed in extended precision (real128, to |F| < 1e-20):
# very unstable Earth collision orbits (s_1 = 1.6e12 - 1.2e15); for Nos. 2 and 3 |F| < 1e-10 is out
# of reach in double precision
EXTENDED = {'2', '3', '4'}

header_collision()
for name, row in ROWS.items():
    ext = name in EXTENDED
    o = collision_orbit(*row, section='xi1', extended=ext, tol=1e-20 if ext else 1e-10)
    line_collision(name, o, stability(o), ext)
