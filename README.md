# Data and code deposit
**"Radius-dependent looseness of far-field Calderón–Zygmund bounds in turbulent vortex stretching"**
Jesper Lyng Jensen — SRT Compute ApS, Denmark (2026)

This deposit contains the derived per-target data supporting every table in the
manuscript, and the reproduction code including the orientation-verification pipeline.
Raw DNS fields are **not** redistributed (database terms); exact re-fetch coordinates and
seeded selection make every number reproducible.

## Contents
```
data/
  cz_convergence_v2_targets.csv   per-target data for Tables comp & conv (TURB-Rot)  [ADD LOCALLY]
  cz_convergence_v2_results.txt   aggregates + built-in reproduction check + provenance
  cz_v5_targets.csv               per-target data for Tables iso, re, nearfar + components (JHTDB)
  cz_reynolds_v5_results.txt      aggregates + gates + cutout origins + provenance
  theta/
    rigidity_v5_events.csv        per-event data for Table theta                      [ADD LOCALLY]
    README_THETA.txt              column documentation
code/
  cz_convergence_v2.py            TURB-Rot radius campaign (exhaustive, seeded, with
                                  built-in reproduction check against the paper tables)
  cz_reynolds_v5.py               JHTDB pipeline: orientation gate, tightness (4 shells),
                                  components, near/far, per-target dump.  Requires a JHTDB
                                  auth token (placeholder in file).
  theta_coherence_v1.py           blockwise coherence Θ vs phase-randomised surrogates
legacy/
  README_LEGACY.md                note on superseded pre-verification ingestion
CHECKSUMS.sha256
```

## Table → file mapping
| Manuscript table | Data file | Producing script |
|---|---|---|
| comp (capacity & realised vs R_max) | cz_convergence_v2_targets.csv | cz_convergence_v2.py |
| conv (C_far vs R_max) | cz_convergence_v2_targets.csv | cz_convergence_v2.py |
| iso (C_far, 4 shells × 3 Re) | cz_v5_targets.csv | cz_reynolds_v5.py |
| re (tail/median ratios) | cz_v5_targets.csv | cz_reynolds_v5.py |
| nearfar (far-field share) + radius-resolved components | cz_v5_targets.csv | cz_reynolds_v5.py |
| theta (blockwise coherence) | theta/rigidity_v5_events.csv | theta_coherence_v1.py |

## Column documentation
**cz_convergence_v2_targets.csv** (4,800 rows = 2 snapshots × 2 percentiles × 200 targets × 6 radii):
`snapshot, percentile, ix, iy, iz, omega_mag, R_max, sigma_abs, sigma_signed, C_far`

**cz_v5_targets.csv** (48,000 rows = 3 datasets × 4 cutouts × 2 percentiles × 500 targets × 4 shells):
`dataset, Re_lambda, cutout, origin, percentile, R_eta, ix, iy, iz, omega_mag, sigma_total,
sigma_abs, sigma_signed, C_far`
The near/far share of the manuscript is `sigma_signed/|sigma_total|` on the `R_eta = 60` rows.

**theta/rigidity_v5_events.csv**: see `theta/README_THETA.txt`.

Units: kernel sums use the grid-unit convention; by the degree −3 homogeneity of the
kernel this equals the physical integral exactly. `sigma_total` is in physical units
(second-order central differences of velocity at the target). `C_far` is dimensionless.

## Datasets, access, and exact provenance
- **TURB-Rot** (Biferale et al. 2020, SMART-Turb portal): snapshots `velo_0.h5`,
  `velo_100.h5`, full 256³ periodic, spectral differentiation (divergence 4×10⁻¹³).
  Target selection seeded: `SEED = 42` (+100 for the second snapshot).
- **JHTDB** (Li et al. 2008; Perlman et al. 2007): `isotropic1024coarse` (Re_λ≈433,
  t=1), `isotropic4096` (Re_λ≈611, t=1), `isotropic8192` (Re_λ≈1300, t=1). Four 256³
  cutouts per dataset at seeded random origins (`SEED = 42 + Re_λ`):
  - 433:  (327,44,226) (515,228,77) (234,708,510) (214,481,225)
  - 611:  (680,1480,2812) (2304,2370,382) (1162,507,990) (2592,2031,3199)
  - 1300: (3169,7866,1219) (5478,1841,4529) (6102,7494,2881) (7231,4042,4043)

## Orientation verification
JHTDB cutouts are delivered with array axes ordered (z, y, x). Every cutout is verified
against the exact incompressibility identity ⟨|S|²⟩/⟨|ω|²⟩ = 1/2 over all 36
axis/component permutations (gate [0.45, 0.55]; full-domain check for periodic
snapshots). All twelve cutouts used in the manuscript passed (0.476–0.522). Ingestion
versions predating this gate read cutouts as (x, y, z) and are superseded — see
`legacy/README_LEGACY.md`.

## Reproduction
```
python code/cz_convergence_v2.py        # TURB-Rot; ~13 min; prints a row-by-row
                                        # reproduction check against the paper tables
python code/cz_reynolds_v5.py           # JHTDB; ~36 min; requires auth token
python code/theta_coherence_v1.py jhtdb_iso_re433.h5 ... velo_0.h5 velo_100.h5
```
TURB-Rot values reproduce exactly (seeded selection on a deterministic pipeline).

## License and contact
© 2026 Jesper Lyng Jensen · SRT Compute ApS (CVR 30203178), Jenslevvej 68,
4070 Kirke Hyllinge, Denmark. License: see LICENSE (to be added before release tag).
