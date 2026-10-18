# Moser Regularization of the CR3BP for Symmetric Orbits: Bifurcation Networks of Lunar Orbits

Cengiz Aydin, Chankyu Joung

**Abstract.** We develop a computational framework based on Moser regularization for the analysis of 
symmetric periodic orbits in the spatial Circular Restricted Three-Body Problem (CR3BP). The formulation 
enables the numerical continuation of periodic orbit families through collision with either primary, 
together with the computation of Floquet multipliers and stability indices entirely in regularized 
coordinates. We further provide a characterization and numerical detection scheme for symmetric 
periodic collision orbits. We apply this framework to the Earth--Moon CR3BP and uncover bifurcation 
networks connecting several classical lunar periodic orbit families. Higher-period vertical 
bifurcations of the planar $L_1$ Lyapunov family generate spatial bridges reconnecting different regions 
of the same family across its collision with the Earth, while corresponding bifurcations of the $L_2$ 
Lyapunov family form bridges to planar families associated with retrograde comet orbits. Furthermore, 
period-multiplying bifurcations of Near-Rectilinear Halo Orbit (NRHO) families produce a systematic 
network connecting to planar prograde and retrograde families. These bifurcation connections in the 
near-Moon dynamics retain the characteristic organization by covering multiplicities of the corresponding 
bifurcation network in the limiting Hill problem. Conley--Zehnder indices and related symplectic 
invariants complement the numerical continuation and provide a topological organization of the results.

## Purpose

This repository contains the implementation of the algorithms of the paper and one test script per
table, which recomputes every row of the table from its initial data:

* Tables 3–11: the fixed-C Newton correction of symmetric periodic orbits in Moser regularized
  coordinates (Sections 3.5, 3.6), the stability indices (Section 3.3), Krein signatures and the
  transverse Conley–Zehnder index (Appendix A).
* Tables 1 and 2: Newton's method for symmetric periodic collision orbits (Section 4), and the
  stability indices of the collision orbits.

The initial data in the test scripts are given with full double precision digits, so every row
converges directly; they round to the values printed in the paper.

## Structure

```
src/
  cr3bp.py        CR3BP Hamiltonian and its gradient (Section 2)
  moser.py        Moser regularization at the Moon or the Earth, regularized flow and its
                  linearization (Section 3.1)
  correction.py   fixed point sets of the symmetries and the fixed-C Newton correction with single
                  and multiple shooting (Sections 3.2, 3.5, 3.6)
  stability.py    monodromy, stability indices (Section 3.4), Krein signatures (Appendix A)
  cz.py           transverse Conley-Zehnder index (Appendix A)
  collision.py    symmetric periodic collision orbits (Section 4)
test/
  common.py               correction of a table row, output format
  test_table_1.py, test_table_2.py, ..., test_table_11.py
  test_table_*.txt        output of the test scripts
```

## Running the tests

```
pip install -r requirements.txt
cd test
python test_table_4.py
```

Each script prints, for every row of its table, the Jacobi constant C, the final residual |F| of the
Newton iteration and the number of iterations, the period T, the stability indices with the Krein
signs of elliptic indices, and the Conley–Zehnder index. For a row printed in the paper with an
index change a → b, the index is computed for two orbits of the family on either side of the
degeneracy, whose initial data are listed in the script (`SIDES`). The output of every script is
included as `test_table_*.txt`.

The algorithms were implemented in Python 3.9 and all experiments were conducted on a machine with a 
AMD Ryzen 7 5800X processor and 64 GB RAM, running Ubuntu 20.04.6 (64-bit). Running times on one core: 
Tables 1, 2, 3, 10: under 1 minute; 4, 7, 11: about 1 minute; 5, 8, 9: 3-5 minutes; 6: about 12 minutes.

## AI Disclosure

The numerical algorithms in this repository and all results reported in the paper (Tables 1–11) were developed, implemented, and computed by the authors.

Claude Opus 5.5 (Anthropic) was used in the final stages, for two tasks:

- **Repository preparation:** restructuring and documenting the code, and writing the scripts that regenerate each table entry from the supplied initial data.
- **Reproducibility checks:** rerunning these scripts and comparing their output with the published values.

The AI tool did not generate or modify any of the reported results. The authors reviewed and validated all AI-assisted changes to the code.