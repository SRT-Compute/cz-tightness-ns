# cz-tightness-ns

Data and code for:
J. L. Jensen, "Radius-dependent looseness of far-field Calderon-Zygmund
bounds in turbulent vortex stretching" (2026).

Code: MIT. Data: CC BY 4.0. See LICENSE.

## Files

| Manuscript table | Data | Code |
|---|---|---|
| comp, conv | data/cz_convergence_v2_targets.csv | code/cz_convergence_v2.py |
| iso, re, nearfar | data/cz_v5_targets.csv | code/cz_reynolds_v5.py |
| theta | data/theta/rigidity_v5_events.csv | code/theta_coherence_v1.py |

Run logs (cutout origins, orientation-gate ratios, seeds, sampling checks):
data/cz_convergence_v2_results.txt, data/cz_reynolds_v5_results.txt.

## Column schemas

cz_convergence_v2_targets.csv:
snapshot, percentile, ix, iy, iz, omega_mag, R_max, sigma_abs, sigma_signed, C_far

cz_v5_targets.csv:
dataset, Re_lambda, cutout, origin, percentile, R_eta, ix, iy, iz, omega_mag,
sigma_total, sigma_abs, sigma_signed, C_far

data/theta/rigidity_v5_events.csv:
Theta, A, ommax, nblocks, Th_null_med, pnull, sigma_loc, file, ev, kind
(kind: real | surrogat)

## Reproduction

    python code/cz_convergence_v2.py          # velo_0.h5, velo_100.h5 in cwd
    python code/cz_reynolds_v5.py             # JHTDB API; token required
    python code/cz_reynolds_v5.py re433.h5 re611.h5 re1300.h5   # local mode
    python code/theta_coherence_v1.py <velocity .h5 files>

Target selection is seeded; cz_convergence_v2.py self-checks its pooled
aggregates against the deposited results file.

Raw DNS fields (JHTDB, TURB-Rot) are not redistributed; exact cutout
coordinates and time indices are in the run logs.

SHA-256: CHECKSUMS.sha256.
