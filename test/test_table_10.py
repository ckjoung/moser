"""Table 10: the families emerging from the near-collision single-turn and period-doubling bifurcations of the L2 Lyapunov family (first and second block).

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
    '1-1': (2.8025491026, 1.6107296224384955, 0.0, 0.5868557232300888, 0.0, 9.100501326043057, 'planar', 'moon', 1, 1),
    '1-2': (2.79037942563, 1.5958446412341751, 0.1899133256450207, 0.5885296416348693, 0.0, 9.107887589090083, 'II', 'moon', 1, 1),
    '1-3': (2.55474031536, 1.3443001046446623, 0.7941284090027905, 0.6113096877986433, 0.0, 9.235816716195977, 'II', 'moon', 1, 1),
    '1-4': (-0.82707915, -1.3184630871340985, 0.7348415216855089, 0.6547672781680522, 0.0, 9.47944127395125, 'II', 'moon', 1, 1),
    '1-5': (-1.1390268855820047, -1.4620726453405655, 0.0062766543267391475, 0.69413092268, 0.0, 9.482271752826502, 'II', 'moon', 1, 1),
    '1-6': (-1.13905411, -1.462068362856662, 0.0, 0.6941425198394232, 0.0, 9.482257591933964, 'planar', 'moon', 1, 1),
    '2-1': (2.83529544, 1.5442878808038918, 0.0, 0.6155392884341684, 0.0, 8.770542725361311, 'planar', 'moon', 2, 1),
    '2-2': (2.81235453671, 1.515432017417741, 0.24965360874768822, 0.6197586334461705, 0.0, 17.661489655259732, 'IV', 'moon', 1, 1),
    '2-3': (-0.60557576899, -1.1293592604429628, 0.9496708454535893, 0.670337046188542, 0.0, 18.971518545520645, 'IV', 'moon', 1, 1),
    '2-4': (-1.15459879752, -1.4670636742708516, 0.07365913425939938, 0.6946444216946981, 0.0, 19.1281490037883, 'IV', 'moon', 1, 1),
    '2-5': (-1.15850741, -1.4683478146377849, 0.0, 0.6952599958770841, 0.0, 9.564840372564115, 'planar', 'moon', 2, 1),
}

# orbits on either side of the degeneracy (same columns)
SIDES = {
    '1-1': [(2.80254927111854, 1.6107292955448362, 0.0, 0.5868558631414962, 0.0, 9.100499880930165, 'planar', 'moon', 1, 1),
            (2.8025489341415826, 1.610729949210604, 0.0, 0.586855583369515, 0.0, 9.100502770585019, 'planar', 'moon', 1, 1)],
    '1-6': [(-1.1390538235300576, -1.462068285494631, 0.0, 0.6941424952473072, 0.0, 9.482256383203621, 'planar', 'moon', 1, 1),
            (-1.139054390149767, -1.462068438511913, 0.0, 0.6941425438889803, 0.0, 9.48225877399828, 'planar', 'moon', 1, 1)],
    '2-1': [(2.8352955153772363, 1.5442877207934376, 0.0, 0.6155393580470261, 0.0, 8.770541830789162, 'planar', 'moon', 2, 1),
            (2.83529537184862, 1.5442880254767597, 0.0, 0.615539225494329, 0.0, 8.77054353419638, 'planar', 'moon', 2, 1)],
    '2-5': [(-1.1585071226372667, -1.4683477072527586, 0.0, 0.6952599871977068, 0.0, 9.564839145927383, 'planar', 'moon', 2, 1),
            (-1.1585076899655466, -1.4683479192585494, 0.0, 0.6952600043330399, 0.0, 9.564841567627715, 'planar', 'moon', 2, 1)],
}

# rows computed in extended precision (real128): very unstable orbits, for which the double
# precision integration of the full period leaves the periodic orbit (CZ index) and the smaller
# stability index is not resolved
EXTENDED = set()

header()
for name, row in ROWS.items():
    ext = name in EXTENDED
    o = orbit(*row, extended=ext)
    indices = stability(o, k=row[8])
    sides = [(orbit(*r, extended=ext), r[8]) for r in SIDES[name]] if name in SIDES else [(o, row[8])]
    cz = [conley_zehnder(p, k) for p, k in sides]
    line(name, o, row[8], indices, cz, ext)
