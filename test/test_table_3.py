"""Table 3: vertical self-resonant orbits of the L1 and L2 Lyapunov families (first and second block).

Every row is corrected from the initial data below by the fixed-C Newton scheme of Sections 3.5-3.6
(single shooting, or multiple shooting with N nodes) to |F| < 1e-10, and its stability indices
(with the Krein signs of elliptic indices) and Conley-Zehnder index are printed.  For a row printed
with an index change a -> b, the index is computed for the orbits listed in SIDES, on either side of
the degeneracy.

The CZ column gives CZ / CZ_p / CZ_v (in-plane and vertical indices of the planar orbit).
"""
from common import orbit, stability, conley_zehnder, header, line

# row: (C, x, z, p_y, p_z, T, symmetry type, regularized primary, k, N)
#   initial point (x, 0, z, 0, p_y, p_z); T is the period of the prime orbit and the row is its
#   k-fold cover; N is the number of shooting nodes
ROWS = {
    '1-1': (3.1743519536, 0.8233908984294251, 0.0, 0.9497173035754897, 0.0, 2.742994071298387, 'planar', 'moon', 1, 1),
    '1-2': (3.02139213, 0.781573930703945, 0.0, 1.2247717299090648, 0.0, 3.9499986630369954, 'planar', 'moon', 1, 1),
    '1-3': (2.97098397298, 0.7420095222325099, 0.0, 1.2880660029682713, 0.0, 5.0199149255579565, 'planar', 'moon', 1, 1),
    '1-4': (2.9592889, 0.7272973994664047, 0.0, 1.3059138755103343, 0.0, 5.339251782037598, 'planar', 'moon', 1, 1),
    '1-5': (2.94927519, 0.7128281074538094, 0.0, 1.323067153046069, 0.0, 5.618250922860665, 'planar', 'moon', 1, 1),
    '1-6': (-0.46396129896, 0.610644034217928, 0.0, -1.407668045516484, 0.0, 5.434120551345182, 'planar', 'moon', 1, 1),
    '1-7': (-0.4991421, 0.6425923019364013, 0.0, -1.357400515424418, 0.0, 5.46025170524376, 'planar', 'moon', 1, 1),
    '1-8': (-0.54391851, 0.6888341723997511, 0.0, -1.2905959471998791, 0.0, 5.497831824695964, 'planar', 'moon', 1, 1),
    '1-9': (-0.6123037633478706, 0.8222697867279527, 0.0, -1.1278444700000003, 0.0, 5.564366757442795, 'planar', 'moon', 1, 1),
    '2-1': (3.15211890291, 1.180898560195212, 0.0, 1.0250423070825192, 0.0, 3.4155308936155553, 'planar', 'moon', 1, 1),
    '2-2': (3.01376751, 1.2199775837848257, 0.0, 0.7924834699254834, 0.0, 4.310509211642093, 'planar', 'moon', 1, 1),
    '2-3': (2.97326314714, 1.250960047237797, 0.0, 0.7528111128512371, 0.0, 5.165816013955984, 'planar', 'moon', 1, 1),
    '2-4': (2.96368427, 1.2629581445639602, 0.0, 0.7441638210036103, 0.0, 5.451914094835152, 'planar', 'moon', 1, 1),
    '2-5': (2.95572007, 1.2747305062233765, 0.0, 0.7369690997528039, 0.0, 5.7134901726334855, 'planar', 'moon', 1, 1),
    '2-6': (2.8352966, 1.5442854183463814, 0.0, 0.6155403597285213, 0.0, 8.770528958325793, 'planar', 'moon', 1, 1),
    '2-7': (2.8025491026, 1.6107296224384955, 0.0, 0.5868557232300888, 0.0, 9.100501326043057, 'planar', 'moon', 1, 1),
    '2-8': (0.80895979, 2.9614734595451866, 0.0, 0.022411369446330848, 0.0, 12.39090896530975, 'planar', 'earth', 1, 1),
}

# orbits on either side of the degeneracy (same columns)
SIDES = {
    '1-1': [(3.1743621756180547, 0.8233952592441395, 0.0, 0.9496762840245345, 0.0, 2.7429547782047705, 'planar', 'moon', 1, 1),
            (3.1743417230935567, 0.8233865359190973, 0.0, 0.949758342112611, 0.0, 2.7430333996712664, 'planar', 'moon', 1, 1)],
    '1-2': [(3.021394343080541, 0.7815750371583068, 0.0, 1.224769089176434, 0.0, 3.9499636436875774, 'planar', 'moon', 1, 1),
            (3.021389916909526, 0.7815728242051412, 0.0, 1.2247743706483292, 0.0, 3.9500336834633503, 'planar', 'moon', 1, 1)],
    '1-9': [(-0.6121037633478706, 0.8223245156083115, 0.0, -1.1278462127572784, 0.0, 5.5641091423213584, 'planar', 'moon', 1, 1),
            (-0.6125037633478706, 0.8232935446085495, 0.0, -1.1278110844431937, 0.0, 5.564624395128041, 'planar', 'moon', 1, 1)],
    '2-1': [(3.152124936506746, 1.1808954756692196, 0.0, 1.0250617635086028, 0.0, 3.4155167147375387, 'planar', 'moon', 1, 1),
            (3.152112863258312, 1.1809016471522409, 0.0, 1.025022834078141, 0.0, 3.415545087668975, 'planar', 'moon', 1, 1)],
    '2-2': [(3.01376867010539, 1.219977020555898, 0.0, 0.7924847165043905, 0.0, 4.31049170429999, 'planar', 'moon', 1, 1),
            (3.013766358816327, 1.219978142695315, 0.0, 0.7924822329405067, 0.0, 4.31052658466656, 'planar', 'moon', 1, 1)],
    '2-7': [(2.80254927111854, 1.6107292955448362, 0.0, 0.5868558631414962, 0.0, 9.100499880930165, 'planar', 'moon', 1, 1),
            (2.8025489341415826, 1.610729949210604, 0.0, 0.586855583369515, 0.0, 9.100502770585019, 'planar', 'moon', 1, 1)],
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
    cz = [conley_zehnder(p, k, planar=True) for p, k in sides]
    line(name, o, row[8], indices, cz, ext)
