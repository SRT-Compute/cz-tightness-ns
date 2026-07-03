# Radius-dependent looseness of far-field Calderón–Zygmund bounds in turbulent vortex stretching

Code and derived data accompanying the manuscript:

> **Radius-dependent looseness of far-field Calderón–Zygmund bounds in turbulent vortex stretching**
> Jesper Lyng Jensen, SRT Compute ApS, 2026.

This repository contains the analysis and figure-generation scripts, the cutout and
target manifests, and the per-block statistics underlying every figure in the paper.
The analysis operates on publicly available Johns Hopkins Turbulence Database (JHTDB)
velocity cutouts; the cutout origins and random seeds are recorded in `data/cutout_manifest.csv`.

## Main finding

Splitting the far-field Calderón–Zygmund stretching bound into the geometric **capacity**
(what the bound permits) and the **realised** stretching (what the flow produces), the
tightness ratio C_far = realised / capacity declines by 37–43% as the outer integration
radius grows from R/η = 25 to 60. The decline holds across Reynolds numbers and both
intensity classes, every confidence interval excludes zero, and it survives the robustness
checks contained in `figure_data/` (derivative-order convergence, inner-cutoff sensitivity,
pooled-threshold and sixth-order-gradient subsets, and block-bootstrap resampling).

## Repository layout

```
code/
  cz_reynolds_v5.py        Main analysis: downloads JHTDB cutouts, selects targets,
                           computes capacity, realised stretching, and C_far per block.
  cz_convergence_v2.py     Derivative-order convergence campaign on the 256^3 cutouts.
  make_figures.py          Regenerates all five paper figures from figure_data/.

data/
  cz_v5_targets.csv               Per-target records for the Reynolds campaign (48,000 rows).
  cz_reynolds_v5_results.txt      Console log of the Reynolds campaign (all gate checks).
  cz_convergence_v2_targets.csv   Per-target records for the convergence campaign (4,800 rows).
  cz_convergence_v2_results.txt   Console log of the convergence campaign.
  cutout_manifest.csv             Cutout origins, voxel coordinates, seeds, η, and grid metadata.
  target_manifest.csv             Target-selection manifest (per cutout: percentile, ω bounds, counts).

figure_data/                      Per-block statistics underlying each figure.
```

## Figure → data mapping

| Figure | File(s) in `figure_data/` |
|--------|---------------------------|
| Fig. 1 — capacity / realisation / tightness vs R | `capacity_realisation.csv`, `tightness_blockbootstrap.csv` |
| Fig. 2 — derivative-order convergence of \|ω\| | `fd_order_validation.csv` |
| Fig. 3 — fractional decline of C_far + slope forest | `endpoint_decline.csv`, `slope_ratio.csv`, `capacity_logfit.csv` |
| Fig. 4 — realised far-field share of stretching | `nearfar_share.csv` |
| Fig. 5 — per-shell angular cancellation factor A | `shell_cancellation.csv`, `shell_increments.csv` |

Robustness variants supporting the figures: `tail_median_ratio.csv`,
`inner_cutoff_sensitivity.csv`, and the `*_pooled.csv` / `*_fd6subset.csv` /
`nearfar_share_filtered.csv` files (alternative threshold definitions, sixth-order-gradient
subset, and drop-fraction-filtered far-field shares).

## Reproduction

1. Obtain a JHTDB access token from https://turbulence.pha.jhu.edu and set it in
   `code/cz_reynolds_v5.py` (replace `YOUR_JHTDB_TOKEN_HERE`).
2. Run `python code/cz_reynolds_v5.py` to regenerate `data/cz_v5_targets.csv` and the
   Reynolds-campaign statistics; `python code/cz_convergence_v2.py` for the convergence
   campaign. Cutout origins and seeds are fixed in `data/cutout_manifest.csv`, so the
   selection is reproducible.
3. Run `python code/make_figures.py` to regenerate the five figures from `figure_data/`.

The per-block CSVs in `figure_data/` are provided directly so the figures and reported
intervals can be reproduced without re-downloading the JHTDB cutouts.

## Verifying integrity

`CHECKSUMS.sha256` lists the SHA-256 hash of every file. Verify with:

```
sha256sum -c CHECKSUMS.sha256
```

## License and attribution

See `LICENSE`. © 2026 Jesper Lyng Jensen, SRT Compute ApS (CVR 30203178),
Jenslevvej 68, 4070 Kirke Hyllinge, Denmark.

The data were derived from the Johns Hopkins Turbulence Database, an open resource of the
Institute for Data Intensive Engineering and Science.
