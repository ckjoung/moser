"""Table 4: the family emerging from the period-doubling bifurcations of the L1 Lyapunov family.

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
    '1-1': (2.94927519, 0.7128281074538094, 0.0, 1.323067153046069, 0.0, 5.618250922860665, 'planar', 'moon', 2, 1),
    '1-2': (2.94883823848, 0.7129976673859209, 0.0, 1.3224345814961274, 0.03277716271962747, 11.23623003792325, 'III', 'moon', 1, 1),
    '1-3': (2.93133031186, 0.7198529319136855, 0.0, 1.2971539236143426, 0.20805731152552576, 11.227155638415658, 'III', 'moon', 1, 1),
    '1-4': (2.92807004, 0.7211405839954377, 0.0, 1.2924616011407584, 0.22576682695273761, 11.225995758164917, 'III', 'moon', 1, 1),
    '1-5': (2.32891576526, 0.8288093781892657, 0.0, 0.7291762241577048, 0.9220732584853512, 12.173812526489625, 'III', 'moon', 1, 1),
    '1-6': (0.45426234, 0.877466635551242, 0.0, -0.39372573909954633, 1.0680591894624416, 12.531275891812326, 'III', 'moon', 1, 1),
    '1-7': (-0.7595722486340839, 0.8490128317574805, 0.0, -1.1172973716777004, 0.28857260008633784, 12.214578824680132, 'III', 'moon', 1, 1),
    '1-8': (-0.75951054566, 0.8474942337256703, 0.0, -1.1192318805886323, 0.28488481426717616, 12.20370112735817, 'III', 'moon', 1, 1),
    '1-9': (-0.75409690311, 0.8325787281390212, 0.0, -1.13634774311587, 0.256857100843218, 12.101498727822964, 'III', 'moon', 1, 1),
    '1-10': (-0.75306427, 0.8308498634339926, 0.0, -1.1381939186224121, 0.2543192933387111, 12.090125509323842, 'III', 'moon', 1, 1),
    '1-11': (-0.71792306823, 0.7909136660654303, 0.0, -1.1790020038074966, 0.21563021393992748, 11.846163032509217, 'III', 'moon', 1, 1),
    '1-12': (-0.46594015621, 0.6118041298775481, 0.0, -1.4057890862229585, 0.02447175380563137, 10.874913954303874, 'III', 'moon', 1, 1),
    '1-13': (-0.46396129726, 0.6106440327397242, 0.0, -1.407668047929059, 0.0, 5.43412055015525, 'planar', 'moon', 2, 1),
}

# orbits on either side of the degeneracy (same columns)
SIDES = {
    '1-1': [(2.9492754813190025, 0.7128285518037596, 0.0, 1.323066626163986, 0.0, 5.618242856689584, 'planar', 'moon', 2, 1),
            (2.949274901352964, 0.7128276671770594, 0.0, 1.323067675097962, 0.0, 5.61825891508704, 'planar', 'moon', 2, 1)],
    '1-4': [(2.9280701484232035, 0.7211405411294183, 0.0, 1.2924617570956949, 0.225766263155065, 11.225995793289377, 'III', 'moon', 1, 1),
            (2.9280699314788765, 0.7211406269001532, 0.0, 1.2924614450448992, 0.22576739125828424, 11.225995723014687, 'III', 'moon', 1, 1)],
    '1-7': [(-0.7595722486353464, 0.8490129306486227, 0.0, -1.1172972439786486, 0.288572846870892, 12.214579536269415, 'III', 'moon', 1, 1),
            (-0.7595722486353464, 0.8490127309275565, 0.0, -1.1172975018819429, 0.28857234845711766, 12.214578099134489, 'III', 'moon', 1, 1)],
    '1-10': [(-0.7530643442827294, 0.8308499841243866, 0.0, -1.1381937903584816, 0.25431946657029864, 12.090126300252672, 'III', 'moon', 1, 1),
             (-0.7530641851209172, 0.8308497255273568, 0.0, -1.1381940651830211, 0.25431909539576275, 12.090124605568699, 'III', 'moon', 1, 1)],
    '1-13': [(-0.46396059212817975, 0.6106434196064541, 0.0, -1.4076690486238843, 0.0, 5.434120056450651, 'planar', 'moon', 2, 1),
             (-0.463962004096073, 0.6106446473555636, 0.0, -1.4076670448160682, 0.0, 5.434121045054804, 'planar', 'moon', 2, 1)],
}

# rows computed in extended precision (real128): very unstable orbits, for which the double
# precision integration of the full period leaves the periodic orbit (CZ index) and the smaller
# stability index is not resolved
EXTENDED = {'1-3', '1-6'}

header()
for name, row in ROWS.items():
    ext = name in EXTENDED
    o = orbit(*row, extended=ext)
    indices = stability(o, k=row[8])
    sides = [(orbit(*r, extended=ext), r[8]) for r in SIDES[name]] if name in SIDES else [(o, row[8])]
    cz = [conley_zehnder(p, k) for p, k in sides]
    line(name, o, row[8], indices, cz, ext)
