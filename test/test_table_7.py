"""Table 7: the family emerging from the period-doubling bifurcations of the L2 Lyapunov family.

Every row is corrected from the initial data below by the fixed-C Newton scheme of Sections 3.5-3.6
(single shooting, or multiple shooting with N nodes) to |F| < 1e-10, and its stability indices
(with the Krein signs of elliptic indices) and Conley-Zehnder index are printed.  For a row printed
with an index change a -> b, the index is computed for the orbits listed in SIDES, on either side of
the degeneracy.
"""
from common import orbit, stability, conley_zehnder, header, line

# row: (C, x, z, p_y, p_z, T, symmetry type, regularized primary, k, N)
#   initial point (x, 0, z, 0, p_y, p_z); T is the period of the prime orbit and the row is its
#   k-fold cover; N is the number of shooting nodes
ROWS = {
    '1-1': (2.95572007, 1.2747305062233765, 0.0, 0.7369690997528039, 0.0, 5.7134901726334855, 'planar', 'moon', 2, 1),
    '1-2': (2.95347279771, 1.2735443126353203, 0.0, 0.7367032844654504, 0.044516474511377704, 11.411585773722805, 'III', 'moon', 1, 1),
    '1-3': (2.9444841, 1.268729310588503, 0.0, 0.735684119109117, 0.10012592362563666, 11.352114114340477, 'III', 'moon', 1, 1),
    '1-4': (2.93970471, 1.2661278337093171, 0.0, 0.7351713653334915, 0.11991891563819915, 11.322192412559389, 'III', 'moon', 1, 1),
    '1-5': (2.82239314427, 1.2073121952656858, 0.0, 0.7233961905029636, 0.3631856852602168, 11.194610245337003, 'III', 'moon', 1, 1),
    '1-6': (2.10838148, 1.1275453833790379, 0.0, 0.467695978426118, 0.7969096021661947, 12.273667429043618, 'III', 'moon', 1, 1),
    '1-7': (-1.01998825, 1.0359407111769687, 0.0, -1.0763568141657562, 0.14737180091246416, 11.917080434818685, 'III', 'moon', 1, 1),
    '1-8': (-1.02316693396, 1.033401426530437, 0.0, -1.087047780135876, 0.13373539397320952, 11.870048124904294, 'III', 'moon', 1, 1),
    '1-9': (-1.03930262247, 1.024167496923495, 0.0, -1.1351382661493332, 0.03464973226190556, 11.668334615135581, 'III', 'moon', 1, 1),
    '1-10': (-1.0405962337, 1.0236325310030607, 0.0, -1.1385395836949348, 0.0, 5.827410942614815, 'planar', 'moon', 2, 1),
}

# orbits on either side of the degeneracy (same columns)
SIDES = {
    '1-1': [(2.9557224367011723, 1.2747267557017508, 0.0, 0.7369712502388159, 0.0, 5.713409644630871, 'planar', 'moon', 2, 1),
            (2.955719774695694, 1.2747309742010322, 0.0, 0.7369688314258878, 0.0, 5.7135002204790215, 'planar', 'moon', 2, 1)],
    '1-4': [(2.9397048501418497, 1.2661279103623027, 0.0, 0.735171380072384, 0.11991837976844633, 11.322193269792326, 'III', 'moon', 1, 1),
            (2.9397045638045944, 1.2661277537452147, 0.0, 0.7351713499579574, 0.11991947465299918, 11.32219151829891, 'III', 'moon', 1, 1)],
    '1-7': [(-1.01998777371059, 1.0359411106845113, 0.0, -1.0763551942075575, 0.14737380629141458, 11.917087601502695, 'III', 'moon', 1, 1),
            (-1.0199887066789228, 1.0359403281237753, 0.0, -1.0763583674203172, 0.14736987809701713, 11.917073563247145, 'III', 'moon', 1, 1)],
    '1-10': [(-1.0405233471361177, 1.0236090068409553, 0.0, -1.138648444410339, 0.0, 5.826922789638774, 'planar', 'moon', 2, 1),
             (-1.0406326667573371, 1.0236442962333991, 0.0, -1.1384851840869308, 0.0, 5.827654907581455, 'planar', 'moon', 2, 1)],
}

# rows computed in extended precision (real128): very unstable orbits, for which the double
# precision integration of the full period leaves the periodic orbit (CZ index) and the smaller
# stability index is not resolved
EXTENDED = {'1-3', '1-5', '1-6'}

header()
for name, row in ROWS.items():
    ext = name in EXTENDED
    o = orbit(*row, extended=ext)
    indices = stability(o, k=row[8])
    sides = [(orbit(*r, extended=ext), r[8]) for r in SIDES[name]] if name in SIDES else [(o, row[8])]
    cz = [conley_zehnder(p, k) for p, k in sides]
    line(name, o, row[8], indices, cz, ext)
