# Radius-dependent looseness of the far-field Calderón–Zygmund bound in turbulent vortex stretching

Code and derived data accompanying the manuscript:

> **Radius-dependent looseness of the far-field Calderón–Zygmund bound in turbulent vortex stretching**
> Jesper Lyng Jensen, SRT Compute ApS, 2026.

The analysis operates on publicly available Johns Hopkins Turbulence Database (JHTDB)
velocity cutouts. Cutout origins and random seeds are recorded in `data/cutout_manifest.csv`;
target selection is recorded in `data/target_manifest.csv`.

## Repository layout

```
code/
  cz_01_download.py    Downloads the JHTDB velocity cutouts (12 blocks per Reynolds number).
  cz_02_analysis.py    Full analysis: capacity, realised stretching, tightness, robustness
                       variants. Writes data/ manifests and all figure_data/ statistics.
  make_figures.py      Regenerates the five paper figures from figure_data/.

data/
  cutout_manifest.csv  Cutout origins, seeds, eta, and grid metadata per block.
  target_manifest.csv  Target selection per cutout: percentile, omega bounds, counts.

figure_data/           Per-block statistics underlying each figure.
```

## Figure → data mapping

| Figure | File(s) in `figure_data/` |
|--------|---------------------------|
| Fig. 1 | `capacity_realisation.csv`, `tightness_blockbootstrap.csv` |
| Fig. 2 | `fd_order_validation.csv` |
| Fig. 3 | `endpoint_decline.csv`, `slope_ratio.csv`, `capacity_logfit.csv` |
| Fig. 4 | `nearfar_share.csv` |
| Fig. 5 | `shell_cancellation.csv`, `shell_increments.csv` |

Robustness variants: `tail_median_ratio.csv`, `inner_cutoff_sensitivity.csv`, and the
`*_pooled.csv`, `*_fd6subset.csv`, and `nearfar_share_filtered.csv` files.

## Reproduction

1. Obtain a JHTDB access token from https://turbulence.pha.jhu.edu and set it in
   `code/cz_01_download.py` and `code/cz_02_analysis.py`
   (replace `YOUR_JHTDB_TOKEN_HERE`).
2. `python code/cz_01_download.py` downloads the velocity cutouts listed in
   `data/cutout_manifest.csv`.
3. `python code/cz_02_analysis.py` recomputes the manifests and every file in
   `figure_data/`.
4. `python code/make_figures.py` regenerates the five figures from `figure_data/`.

The per-block statistics in `figure_data/` are provided directly, so the figures and
reported intervals can be reproduced without re-downloading the JHTDB cutouts.

## Verifying integrity

```
sha256sum -c CHECKSUMS.sha256
```

## License

See `LICENSE`. © 2026 Jesper Lyng Jensen, SRT Compute ApS (CVR 30203178),
Jenslevvej 68, 4070 Kirke Hyllinge, Denmark.

The data were derived from the Johns Hopkins Turbulence Database, an open resource of the
Institute for Data Intensive Engineering and Science.
