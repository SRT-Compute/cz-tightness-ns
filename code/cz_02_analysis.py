"""
cz_02_analysis_v3_robust.py

Calderon-Zygmund far-field stretching analysis on JHTDB isotropic cutouts.

This version extends cz_02_analysis_v2_fd4.py with four peer-review robustness
checks, each gated by a config flag near the top of the file and each writing to
its OWN new CSV file. With all four flags off, this script reproduces the v2
core outputs byte-for-byte; with them on, the core outputs are unchanged and the
following additional files are written to OUT_PATH:

  (1) POOLED_THRESHOLD_SENSITIVITY -> tightness_pooled.csv,
      endpoint_decline_pooled.csv, tail_median_ratio_pooled.csv,
      nearfar_share_pooled.csv
      The main run conditions percentiles on each cutout's OWN interior vorticity
      distribution (cutout-local). This recomputes tightness/decline/tail-ratio/
      share using a threshold POOLED across all blocks of a Reynolds number, to
      show how much the cutout-local definition matters.

  (2) FD6_TIGHTNESS_SUBSET -> tightness_fd6subset.csv,
      endpoint_decline_fd6subset.csv, tail_median_ratio_fd6subset.csv
      Recomputes the full tightness pipeline at SIXTH order on a subset of blocks
      per Re (FD6_SUBSET_BLOCKS_PER_RE), so the conclusions are shown not to be an
      artefact of the fourth-order gradient used for both field and target ranking.

  (3) SHARE_SIGMATOTAL_FILTER -> nearfar_share_filtered.csv
      Recomputes the far/total share after dropping, within each block, the
      targets whose |sigma_total| lies in the lowest SHARE_FILTER_DROP_FRACTION,
      to show the share is not driven by near-zero denominators.

  (4) SHELL_CANCELLATION -> shell_cancellation.csv
      A descriptive per-shell angular-cancellation factor
      A(R) = |sum_shell q| / sum_shell |q| in radial bands (SHELL_CANCELLATION_BANDS),
      i.e. C_far evaluated per shell rather than cumulatively, showing where in
      radius the cancellation arises.

It inherits everything from the v2 FD4 script:
  - the main capacity/realisation/tightness measurement runs at FOURTH order
    (MAIN_GRAD_ORDER = 4)
  - output is written to ./cz_results_fd4/
  - the FD-order validation computes its own FD2 baseline internally
  - dx_phys from HDF5 metadata; P50 as a percentile bin; the same target set for
    main run and inner-cutoff sweeps; no periodic wrapping on non-periodic
    cutouts; nested bootstrap over snapshots/cutouts/targets; log-slope CIs by
    resampling data blocks; shell increments in their own CSV; FD2/FD4/FD6
    convergence check (optional FD8 database-gradient check off by
default); and cutout/target manifests.

Run:
    conda activate <env>
    python cz_02_analysis_v3_robust.py

Input:
    ./cz_data/iso_re{Re}_s{snap}_c{cut}.h5

Output:
    ./cz_results_fd4/cutout_manifest.csv
    ./cz_results_fd4/target_manifest.csv
    ./cz_results_fd4/capacity_realisation.csv
    ./cz_results_fd4/tightness_blockbootstrap.csv
    ./cz_results_fd4/endpoint_decline.csv
    ./cz_results_fd4/shell_increments.csv
    ./cz_results_fd4/capacity_logfit.csv
    ./cz_results_fd4/slope_ratio.csv
    ./cz_results_fd4/inner_cutoff_sensitivity.csv
    ./cz_results_fd4/nearfar_share.csv
    ./cz_results_fd4/tail_median_ratio.csv
    ./cz_results_fd4/fd_order_validation.csv
    ./cz_results_fd4/db_gradient_validation.csv   (only if USE_DB_GRADIENT_VALIDATION)
"""

import csv
import glob
import re as _re
import gc
from pathlib import Path

import numpy as np
import h5py


DATA_PATH = "./cz_data"
OUT_PATH = "./cz_results_fd4"

RETA_INNER = 17.0
RETA_CUTS = [25.0, 35.0, 50.0, 60.0]
RETA_INNER_SWEEP = [12.0, 17.0, 22.0, 27.0]
PERCENTILES = [50.0, 99.9]

N_TARGETS_PER_BIN = 200
N_BOOT = 2000
SEED = 42
EPS = 1e-12
PERCENTILE_BIN_WIDTH = 1.0
DOMAIN_LENGTH_DEFAULT = 2.0 * np.pi

# Finite-difference order for the main capacity/realisation/tightness measurement.
# The FD-order validation (fd_order_validation.csv) showed the extreme-vorticity
# tail is under-resolved at second order on 256^3 cutouts, so the main analysis
# uses fourth order. Set to 2 to reproduce the second-order run.
MAIN_GRAD_ORDER = 4

# --- Robustness additions (peer-review responses) -----------------------------
# (1) Pooled-threshold sensitivity. The main run conditions percentiles on each
#     cutout's own interior vorticity distribution (cutout-local). When this is on,
#     a second target set is also selected using a POOLED threshold computed across
#     all blocks of a given Reynolds number, and the tightness, tail/median ratio,
#     and far-share are recomputed on those pooled-threshold targets and written to
#     *_pooled.csv. This quantifies how much the cutout-local definition matters.
POOLED_THRESHOLD_SENSITIVITY = True

# (2) FD6 tightness subset. Recompute the full C_far / decline / tail-ratio /
#     share pipeline using sixth-order gradients on a subset of blocks per Re, to
#     show the conclusions are not an artefact of the (fourth-order) gradient used
#     for both the field and the target ranking. Written to *_fd6subset.csv.
FD6_TIGHTNESS_SUBSET = True
FD6_SUBSET_BLOCKS_PER_RE = 3      # number of cutouts per Re to re-measure at FD6

# (3) Far-share denominator robustness. Recompute the far/total share after
#     dropping targets in the lowest q-fraction of |sigma_total|, to show the
#     share is not driven by near-zero denominators. Written to nearfar_share_filtered.csv.
SHARE_SIGMATOTAL_FILTER = True
SHARE_FILTER_DROP_FRACTION = 0.05   # drop targets with the smallest 5% of |sigma_total|

# (4) Shell-wise angular cancellation. For each target, compute the per-shell
#     signed-to-unsigned ratio A(R) = |sum_shell q| / sum_shell |q| in radial bands,
#     a descriptive diagnostic of where in radius cancellation occurs. This is
#     C_far evaluated per shell rather than cumulatively. Written to shell_cancellation.csv.
SHELL_CANCELLATION = True
SHELL_CANCELLATION_BANDS = [(17.0, 25.0), (25.0, 35.0), (35.0, 50.0), (50.0, 60.0)]
# ------------------------------------------------------------------------------

DATASETS = [
    {"Re": 433,  "title": "isotropic1024coarse", "N_full": 1024, "eta": 2.873e-3},
    {"Re": 611,  "title": "isotropic4096",       "N_full": 4096, "eta": 1.3844e-3},
    {"Re": 1300, "title": "isotropic8192",       "N_full": 8192, "eta": 5.00e-4},
]
DATASET_BY_RE = {d["Re"]: d for d in DATASETS}
DATASET_BY_TITLE = {d["title"]: d for d in DATASETS}


# ----------------------------------------------------------------------
# Metadata and IO
# ----------------------------------------------------------------------
def _decode_attr(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        return x
    if isinstance(x, np.generic):
        return x.item()
    return x


def load_velocity(path):
    with h5py.File(path, "r") as f:
        u = np.stack([f["u"][:], f["v"][:], f["w"][:]], axis=0).astype(np.float64)
        meta = {k: _decode_attr(v) for k, v in f.attrs.items()}
    return u, meta


def metadata_constants(meta, path=None):
    title = _decode_attr(meta.get("dataset", ""))
    if isinstance(title, bytes):
        title = title.decode("utf-8")

    re_val = meta.get("Re_lambda", None)
    if re_val is None and path is not None:
        m = _re.search(r"iso_re(\d+)_", str(path))
        if m:
            re_val = int(m.group(1))
    if re_val is None:
        raise ValueError(f"Could not infer Re_lambda for {path}")
    re_val = int(re_val)

    default = DATASET_BY_RE.get(re_val, DATASET_BY_TITLE.get(title, {}))
    if not default:
        raise ValueError(f"Unknown dataset/Re combination: Re={re_val}, dataset={title}")

    eta = float(meta.get("eta", default["eta"]))
    n_full = int(meta.get("N_full", default["N_full"]))
    domain_length = float(meta.get("domain_length", DOMAIN_LENGTH_DEFAULT))
    dx_phys = float(meta.get("dx_phys", domain_length / n_full))
    snap = int(meta.get("snapshot", -1))
    cut = int(meta.get("cutout", -1))
    return {
        "Re": re_val,
        "dataset": title if title else default.get("title", ""),
        "eta": eta,
        "N_full": n_full,
        "domain_length": domain_length,
        "dx_phys": dx_phys,
        "snapshot": snap,
        "cutout": cut,
    }


def origin_to_string(origin):
    if origin is None:
        return ""
    if isinstance(origin, str):
        return origin
    try:
        return ";".join(str(int(v)) for v in origin)
    except TypeError:
        return str(origin)


# ----------------------------------------------------------------------
# Field operators
# ----------------------------------------------------------------------
def gradients_fd2(u, dx):
    return [[np.gradient(u[i], dx, axis=j, edge_order=2) for j in range(3)] for i in range(3)]


def gradients_fd4(u, dx):
    # Full 3x3 velocity-gradient tensor at fourth order. Uses a +/-2 np.roll
    # stencil, so the two outermost grid layers of the cube are wrapped and
    # invalid; the target margin (>= ~31 grid points here) keeps every target
    # and its integration shell well inside the valid interior.
    return [[derivative_fd4_interior(u[i], dx, axis=j) for j in range(3)] for i in range(3)]


def gradients_fd6(u, dx):
    # Full 3x3 velocity-gradient tensor at sixth order (+/-3 stencil). Used only
    # for the FD6 tightness-subset robustness check.
    return [[derivative_fd6_interior(u[i], dx, axis=j) for j in range(3)] for i in range(3)]


def gradients_for_main(u, dx):
    if MAIN_GRAD_ORDER == 2:
        return gradients_fd2(u, dx)
    if MAIN_GRAD_ORDER == 4:
        return gradients_fd4(u, dx)
    raise ValueError(f"MAIN_GRAD_ORDER must be 2 or 4, got {MAIN_GRAD_ORDER}")


def derivative_fd4_interior(f, dx, axis):
    # Fourth-order central derivative. Values within two grid points of the
    # cutout boundary are wrapped by np.roll and must be ignored downstream.
    return (
        -np.roll(f, -2, axis=axis)
        + 8.0 * np.roll(f, -1, axis=axis)
        - 8.0 * np.roll(f, 1, axis=axis)
        + np.roll(f, 2, axis=axis)
    ) / (12.0 * dx)


def derivative_fd6_interior(f, dx, axis):
    # Sixth-order central derivative. Values within three grid points of the
    # cutout boundary are wrapped by np.roll and must be ignored downstream.
    return (
        np.roll(f, -3, axis=axis)
        - 9.0 * np.roll(f, -2, axis=axis)
        + 45.0 * np.roll(f, -1, axis=axis)
        - 45.0 * np.roll(f, 1, axis=axis)
        + 9.0 * np.roll(f, 2, axis=axis)
        - np.roll(f, 3, axis=axis)
    ) / (60.0 * dx)


def _vorticity_magnitude_from_derivative(u, dx, deriv):
    duz_dy = deriv(u[2], dx, axis=1)
    duy_dz = deriv(u[1], dx, axis=2)
    wx = duz_dy - duy_dz
    del duz_dy, duy_dz

    dux_dz = deriv(u[0], dx, axis=2)
    duz_dx = deriv(u[2], dx, axis=0)
    wy = dux_dz - duz_dx
    del dux_dz, duz_dx

    duy_dx = deriv(u[1], dx, axis=0)
    dux_dy = deriv(u[0], dx, axis=1)
    wz = duy_dx - dux_dy
    del duy_dx, dux_dy

    omag = np.sqrt(wx * wx + wy * wy + wz * wz)
    del wx, wy, wz
    return omag


def vorticity_magnitude_fd4(u, dx):
    return _vorticity_magnitude_from_derivative(u, dx, derivative_fd4_interior)


def vorticity_magnitude_fd6(u, dx):
    return _vorticity_magnitude_from_derivative(u, dx, derivative_fd6_interior)


def vorticity_strain_from_grad(du):
    wx = du[2][1] - du[1][2]
    wy = du[0][2] - du[2][0]
    wz = du[1][0] - du[0][1]
    omega = np.stack([wx, wy, wz], axis=0)
    omag = np.sqrt(wx * wx + wy * wy + wz * wz)

    Sxx, Syy, Szz = du[0][0], du[1][1], du[2][2]
    Sxy = 0.5 * (du[0][1] + du[1][0])
    Sxz = 0.5 * (du[0][2] + du[2][0])
    Syz = 0.5 * (du[1][2] + du[2][1])
    return omega, omag, (Sxx, Syy, Szz, Sxy, Sxz, Syz)


def sigma_total_at(S, xi, i, j, k):
    Sxx, Syy, Szz, Sxy, Sxz, Syz = S
    a, b, c = xi
    return (
        a * a * Sxx[i, j, k]
        + b * b * Syy[i, j, k]
        + c * c * Szz[i, j, k]
        + 2.0 * a * b * Sxy[i, j, k]
        + 2.0 * a * c * Sxz[i, j, k]
        + 2.0 * b * c * Syz[i, j, k]
    )


# ----------------------------------------------------------------------
# Shell integration
# ----------------------------------------------------------------------
def shell_offsets(r_in, r_out):
    rng = int(np.ceil(r_out))
    co = np.arange(-rng, rng + 1)
    dx, dy, dz = np.meshgrid(co, co, co, indexing="ij")
    dx, dy, dz = dx.ravel(), dy.ravel(), dz.ravel()
    r2 = dx * dx + dy * dy + dz * dz
    mask = (r2 >= r_in * r_in) & (r2 <= r_out * r_out)

    dx = dx[mask].astype(np.int64)
    dy = dy[mask].astype(np.int64)
    dz = dz[mask].astype(np.int64)
    r2 = r2[mask].astype(np.float64)
    r = np.sqrt(r2)
    r5 = np.maximum(r2 * r2 * r, EPS)

    order = np.argsort(r)
    return dx[order], dy[order], dz[order], r[order], r5[order]


def shell_contributions(omega, ix, iy, iz, xi, off_x, off_y, off_z, r5, n):
    jx = ix + off_x
    jy = iy + off_y
    jz = iz + off_z
    if jx.min() < 0 or jy.min() < 0 or jz.min() < 0 or jx.max() >= n or jy.max() >= n or jz.max() >= n:
        raise IndexError("Shell exceeded cutout boundary. Increase target margin or reduce R_eta.")

    ox = omega[0, jx, jy, jz]
    oy = omega[1, jx, jy, jz]
    oz = omega[2, jx, jy, jz]

    dot = ox * xi[0] + oy * xi[1] + oz * xi[2]
    opx = ox - dot * xi[0]
    opy = oy - dot * xi[1]
    opz = oz - dot * xi[2]

    rx = -off_x.astype(np.float64)
    ry = -off_y.astype(np.float64)
    rz = -off_z.astype(np.float64)

    xdr = xi[0] * rx + xi[1] * ry + xi[2] * rz
    cx = xi[1] * rz - xi[2] * ry
    cy = xi[2] * rx - xi[0] * rz
    cz = xi[0] * ry - xi[1] * rx
    fac = -3.0 * xdr / (4.0 * np.pi * r5)
    return fac * (cx * opx + cy * opy + cz * opz)


def band_sum(csum, start, end):
    if end <= start:
        return 0.0
    value = csum[end - 1]
    if start > 0:
        value -= csum[start - 1]
    return value


# ----------------------------------------------------------------------
# Target selection
# ----------------------------------------------------------------------
def select_targets(omag, margin, percentiles, n_targets, seed):
    n = omag.shape[0]
    if 2 * margin >= n:
        raise ValueError(
            f"integration radius margin={margin} exceeds cutout size N={n}; "
            "use a larger cutout or a smaller R_eta range"
        )

    ii, jj, kk = np.where(np.ones((n - 2 * margin, n - 2 * margin, n - 2 * margin), dtype=bool))
    ii = ii + margin
    jj = jj + margin
    kk = kk + margin
    vals = omag[ii, jj, kk]

    rng = np.random.default_rng(seed)
    targets = {}
    info = {}

    for p in percentiles:
        if p >= 99.0:
            lo = float(np.percentile(vals, p))
            hi = float("inf")
            mask = vals >= lo
            mode = "upper_tail"
        else:
            width = PERCENTILE_BIN_WIDTH
            lo = float(np.percentile(vals, max(0.0, p - 0.5 * width)))
            hi = float(np.percentile(vals, min(100.0, p + 0.5 * width)))
            mask = (vals >= lo) & (vals <= hi)
            mode = f"bin_width_{width:g}_percentile_points"
            # Fallback only if a very small cutout gives too few candidates.
            while int(np.count_nonzero(mask)) < n_targets and width < 10.0:
                width *= 2.0
                lo = float(np.percentile(vals, max(0.0, p - 0.5 * width)))
                hi = float(np.percentile(vals, min(100.0, p + 0.5 * width)))
                mask = (vals >= lo) & (vals <= hi)
                mode = f"bin_width_{width:g}_percentile_points"

        cand_all = np.flatnonzero(mask)
        n_candidates = int(len(cand_all))
        if n_candidates == 0:
            targets[p] = np.empty((0, 3), dtype=np.int64)
            selected = np.array([], dtype=np.int64)
        elif n_candidates > n_targets:
            selected = rng.choice(cand_all, n_targets, replace=False)
        else:
            selected = cand_all

        targets[p] = np.column_stack([ii[selected], jj[selected], kk[selected]]).astype(np.int64)
        info[p] = {
            "mode": mode,
            "lo": lo,
            "hi": hi,
            "n_candidates": n_candidates,
            "n_selected": int(len(selected)),
        }

    return targets, info, int(len(vals))


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
def measure_all_inners_from_fields(omega, omag, S, eta, dx_phys, cuts_by_inner, targets):
    n = omag.shape[0]
    inner_values = sorted(cuts_by_inner.keys())
    min_inner = min(inner_values)
    max_cut = max(max(cuts) for cuts in cuts_by_inner.values() if cuts)

    grid_min_inner = min_inner * eta / dx_phys
    grid_max = max_cut * eta / dx_phys
    off_x, off_y, off_z, r_sorted, r5_sorted = shell_offsets(grid_min_inner, grid_max)

    start_index = {inner: int(np.searchsorted(r_sorted, inner * eta / dx_phys, side="left"))
                   for inner in inner_values}
    end_index = {
        inner: [int(np.searchsorted(r_sorted, cut * eta / dx_phys, side="right")) for cut in cuts_by_inner[inner]]
        for inner in inner_values
    }

    result = {
        inner: {
            p: {ci: {"signed": [], "abs": [], "share": [], "sigtot": []} for ci in range(len(cuts_by_inner[inner]))}
            for p in PERCENTILES
        }
        for inner in inner_values
    }

    # Optional per-shell angular-cancellation accumulation (band signed/abs sums per target).
    # Keyed by percentile -> list over targets of (band_signed[], band_abs[]) for SHELL_CANCELLATION_BANDS.
    shell_bands = None
    if SHELL_CANCELLATION:
        band_idx = []
        for (blo, bhi) in SHELL_CANCELLATION_BANDS:
            s = int(np.searchsorted(r_sorted, blo * eta / dx_phys, side="left"))
            e = int(np.searchsorted(r_sorted, bhi * eta / dx_phys, side="right"))
            band_idx.append((s, e))
        shell_bands = {p: {"signed": [[] for _ in band_idx], "abs": [[] for _ in band_idx]} for p in PERCENTILES}

    for p in PERCENTILES:
        coords = targets.get(p, np.empty((0, 3), dtype=np.int64))
        for ix, iy, iz in coords:
            mag = omag[ix, iy, iz]
            if mag < EPS:
                continue
            xi = (
                omega[0, ix, iy, iz] / mag,
                omega[1, ix, iy, iz] / mag,
                omega[2, ix, iy, iz] / mag,
            )
            contrib = shell_contributions(omega, ix, iy, iz, xi, off_x, off_y, off_z, r5_sorted, n)
            csum = np.cumsum(contrib)
            cabs = np.cumsum(np.abs(contrib))
            sig_tot = sigma_total_at(S, xi, ix, iy, iz)
            denom_total = max(abs(sig_tot), EPS)

            for inner in inner_values:
                start = start_index[inner]
                for ci, end in enumerate(end_index[inner]):
                    sf = band_sum(csum, start, end)
                    af = band_sum(cabs, start, end)
                    result[inner][p][ci]["signed"].append(abs(sf))
                    result[inner][p][ci]["abs"].append(af)
                    result[inner][p][ci]["share"].append(abs(sf) / denom_total)
                    result[inner][p][ci]["sigtot"].append(abs(sig_tot))

            if SHELL_CANCELLATION:
                for bi, (s, e) in enumerate(band_idx):
                    bsig = band_sum(csum, s, e)
                    babs = band_sum(cabs, s, e)
                    shell_bands[p]["signed"][bi].append(abs(bsig))
                    shell_bands[p]["abs"][bi].append(babs)

    for inner in inner_values:
        for p in PERCENTILES:
            for ci in result[inner][p]:
                for name in ("signed", "abs", "share", "sigtot"):
                    result[inner][p][ci][name] = np.asarray(result[inner][p][ci][name], dtype=np.float64)

    if SHELL_CANCELLATION:
        for p in PERCENTILES:
            for name in ("signed", "abs"):
                shell_bands[p][name] = [np.asarray(b, dtype=np.float64) for b in shell_bands[p][name]]
        return result, shell_bands
    return result, None


# ----------------------------------------------------------------------
# Bootstrap helpers
# ----------------------------------------------------------------------
def common_nonempty_keys(*groups):
    sets = []
    for group in groups:
        sets.append(set(k for k, v in group.items() if v is not None and len(v) > 0))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def nested_resample_keys(keys, rng):
    by_snapshot = {}
    for k in keys:
        by_snapshot.setdefault(k[0], []).append(k)
    snaps = sorted(by_snapshot.keys())
    selected_keys = []
    selected_snaps = rng.choice(snaps, size=len(snaps), replace=True)
    for snap in selected_snaps:
        cut_keys = by_snapshot[snap]
        selected_cuts = rng.integers(0, len(cut_keys), len(cut_keys))
        selected_keys.extend(cut_keys[i] for i in selected_cuts)
    return selected_keys


def pooled_median(groups, keys):
    vals = [np.asarray(groups[k], dtype=np.float64) for k in keys if len(groups[k]) > 0]
    if not vals:
        return float("nan")
    return float(np.median(np.concatenate(vals)))


def nested_bootstrap_median(groups, seed):
    keys = common_nonempty_keys(groups)
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0, 0

    point = pooled_median(groups, keys)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(N_BOOT):
        vals = []
        for k in nested_resample_keys(keys, rng):
            v = groups[k]
            n = len(v)
            if n == 0:
                continue
            idx = rng.integers(0, n, n)
            vals.append(v[idx])
        if vals:
            reps.append(np.median(np.concatenate(vals)))
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
    n_total = int(sum(len(groups[k]) for k in keys))
    return point, float(lo), float(hi), len(keys), n_total


def nested_bootstrap_ratio(groups_num, groups_den, seed):
    keys = common_nonempty_keys(groups_num, groups_den)
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0, 0

    nums = []
    dens = []
    for k in keys:
        n = min(len(groups_num[k]), len(groups_den[k]))
        nums.append(groups_num[k][:n])
        dens.append(groups_den[k][:n])
    num = np.concatenate(nums)
    den = np.concatenate(dens)
    point = float(np.median(num / np.maximum(den, EPS)))

    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(N_BOOT):
        nums = []
        dens = []
        for k in nested_resample_keys(keys, rng):
            n = min(len(groups_num[k]), len(groups_den[k]))
            if n == 0:
                continue
            idx = rng.integers(0, n, n)
            nums.append(groups_num[k][idx])
            dens.append(groups_den[k][idx])
        if nums:
            num_b = np.concatenate(nums)
            den_b = np.concatenate(dens)
            reps.append(np.median(num_b / np.maximum(den_b, EPS)))
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
    n_total = int(sum(min(len(groups_num[k]), len(groups_den[k])) for k in keys))
    return point, float(lo), float(hi), len(keys), n_total


def fit_log_curve(radii, values):
    x = np.log(np.asarray(radii, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), [float(v) for v in (y - pred)]


def nested_bootstrap_curve_slope(groups_by_radius, radii, seed):
    keys = common_nonempty_keys(*groups_by_radius)
    if not keys:
        return float("nan"), float("nan"), float("nan"), float("nan"), []

    point_values = [pooled_median(g, keys) for g in groups_by_radius]
    slope, _, r2, residuals = fit_log_curve(radii, point_values)

    rng = np.random.default_rng(seed)
    reps = []
    n_r = len(groups_by_radius)
    for _ in range(N_BOOT):
        vals_by_r = [[] for _ in range(n_r)]
        for k in nested_resample_keys(keys, rng):
            n = min(len(groups_by_radius[ri][k]) for ri in range(n_r))
            if n == 0:
                continue
            idx = rng.integers(0, n, n)
            for ri in range(n_r):
                vals_by_r[ri].append(groups_by_radius[ri][k][idx])
        if all(vals_by_r):
            values = [np.median(np.concatenate(vals_by_r[ri])) for ri in range(n_r)]
            try:
                s, _, _, _ = fit_log_curve(radii, values)
                reps.append(s)
            except np.linalg.LinAlgError:
                pass
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
    return slope, float(lo), float(hi), r2, residuals


def nested_bootstrap_slope_ratio(groups_signed_by_radius, groups_abs_by_radius, radii, seed):
    keys = common_nonempty_keys(*(groups_signed_by_radius + groups_abs_by_radius))
    if not keys:
        return float("nan"), float("nan"), float("nan")

    sig_values = [pooled_median(g, keys) for g in groups_signed_by_radius]
    abs_values = [pooled_median(g, keys) for g in groups_abs_by_radius]
    sig_slope, _, _, _ = fit_log_curve(radii, sig_values)
    abs_slope, _, _, _ = fit_log_curve(radii, abs_values)
    point = float(sig_slope / abs_slope) if abs(abs_slope) > EPS else float("nan")

    rng = np.random.default_rng(seed)
    reps = []
    n_r = len(radii)
    for _ in range(N_BOOT):
        sig_by_r = [[] for _ in range(n_r)]
        abs_by_r = [[] for _ in range(n_r)]
        for k in nested_resample_keys(keys, rng):
            n = min(
                min(len(groups_signed_by_radius[ri][k]) for ri in range(n_r)),
                min(len(groups_abs_by_radius[ri][k]) for ri in range(n_r)),
            )
            if n == 0:
                continue
            idx = rng.integers(0, n, n)
            for ri in range(n_r):
                sig_by_r[ri].append(groups_signed_by_radius[ri][k][idx])
                abs_by_r[ri].append(groups_abs_by_radius[ri][k][idx])
        if all(sig_by_r) and all(abs_by_r):
            sig_vals = [np.median(np.concatenate(sig_by_r[ri])) for ri in range(n_r)]
            abs_vals = [np.median(np.concatenate(abs_by_r[ri])) for ri in range(n_r)]
            try:
                s_sig, _, _, _ = fit_log_curve(radii, sig_vals)
                s_abs, _, _, _ = fit_log_curve(radii, abs_vals)
                if abs(s_abs) > EPS:
                    reps.append(s_sig / s_abs)
            except np.linalg.LinAlgError:
                pass
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
    return point, float(lo), float(hi)


def nested_bootstrap_increments(groups_by_radius, radii, seed):
    keys = common_nonempty_keys(*groups_by_radius)
    out = []
    if not keys:
        for i in range(1, len(radii)):
            out.append((float("nan"), float("nan"), float("nan")))
        return out

    lnr = np.log(np.asarray(radii, dtype=np.float64))
    point_values = [pooled_median(g, keys) for g in groups_by_radius]

    rng = np.random.default_rng(seed)
    reps_by_step = [[] for _ in range(1, len(radii))]
    n_r = len(radii)
    for _ in range(N_BOOT):
        vals_by_r = [[] for _ in range(n_r)]
        for k in nested_resample_keys(keys, rng):
            n = min(len(groups_by_radius[ri][k]) for ri in range(n_r))
            if n == 0:
                continue
            idx = rng.integers(0, n, n)
            for ri in range(n_r):
                vals_by_r[ri].append(groups_by_radius[ri][k][idx])
        if all(vals_by_r):
            values = [np.median(np.concatenate(vals_by_r[ri])) for ri in range(n_r)]
            for i in range(1, n_r):
                reps_by_step[i - 1].append((values[i] - values[i - 1]) / (lnr[i] - lnr[i - 1]))

    for i in range(1, len(radii)):
        point = (point_values[i] - point_values[i - 1]) / (lnr[i] - lnr[i - 1])
        reps = reps_by_step[i - 1]
        lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
        out.append((float(point), float(lo), float(hi)))
    return out


def nested_bootstrap_endpoint_decline(groups_signed_by_radius, groups_abs_by_radius, radii, seed):
    first = 0
    last = len(radii) - 1
    needed = [
        groups_signed_by_radius[first], groups_abs_by_radius[first],
        groups_signed_by_radius[last], groups_abs_by_radius[last],
    ]
    keys = common_nonempty_keys(*needed)
    if not keys:
        return float("nan"), float("nan"), float("nan")

    c_values = []
    for ri in (first, last):
        sig = np.concatenate([groups_signed_by_radius[ri][k] for k in keys])
        ab = np.concatenate([groups_abs_by_radius[ri][k] for k in keys])
        c_values.append(float(np.median(sig / np.maximum(ab, EPS))))
    point = c_values[1] / c_values[0] - 1.0 if c_values[0] > EPS else float("nan")

    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(N_BOOT):
        s0 = []
        a0 = []
        s1 = []
        a1 = []
        for k in nested_resample_keys(keys, rng):
            n = min(
                len(groups_signed_by_radius[first][k]), len(groups_abs_by_radius[first][k]),
                len(groups_signed_by_radius[last][k]), len(groups_abs_by_radius[last][k]),
            )
            if n == 0:
                continue
            idx = rng.integers(0, n, n)
            s0.append(groups_signed_by_radius[first][k][idx])
            a0.append(groups_abs_by_radius[first][k][idx])
            s1.append(groups_signed_by_radius[last][k][idx])
            a1.append(groups_abs_by_radius[last][k][idx])
        if s0 and s1:
            c0 = np.median(np.concatenate(s0) / np.maximum(np.concatenate(a0), EPS))
            c1 = np.median(np.concatenate(s1) / np.maximum(np.concatenate(a1), EPS))
            if c0 > EPS:
                reps.append(c1 / c0 - 1.0)
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)


def nested_bootstrap_tailratio(gs_tail, ga_tail, gs_med, ga_med, seed):
    keys = common_nonempty_keys(gs_tail, ga_tail, gs_med, ga_med)
    if not keys:
        return float("nan"), float("nan"), float("nan")

    def c_median(gs, ga):
        sig = np.concatenate([gs[k] for k in keys])
        ab = np.concatenate([ga[k] for k in keys])
        return float(np.median(sig / np.maximum(ab, EPS)))

    c50 = c_median(gs_med, ga_med)
    c99 = c_median(gs_tail, ga_tail)
    point = c99 / c50 if c50 > EPS else float("nan")

    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(N_BOOT):
        s50 = []
        a50 = []
        s99 = []
        a99 = []
        for k in nested_resample_keys(keys, rng):
            n50 = min(len(gs_med[k]), len(ga_med[k]))
            n99 = min(len(gs_tail[k]), len(ga_tail[k]))
            if n50 == 0 or n99 == 0:
                continue
            i50 = rng.integers(0, n50, n50)
            i99 = rng.integers(0, n99, n99)
            s50.append(gs_med[k][i50])
            a50.append(ga_med[k][i50])
            s99.append(gs_tail[k][i99])
            a99.append(ga_tail[k][i99])
        if s50 and s99:
            m50 = np.median(np.concatenate(s50) / np.maximum(np.concatenate(a50), EPS))
            m99 = np.median(np.concatenate(s99) / np.maximum(np.concatenate(a99), EPS))
            if m50 > EPS:
                reps.append(m99 / m50)
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)


# ----------------------------------------------------------------------
# Result extraction and validation
# ----------------------------------------------------------------------
def groups_for_metric(per_group, percentile, ci, metric):
    out = {}
    for g, data in per_group.items():
        try:
            arr = data[percentile][ci][metric]
        except KeyError:
            continue
        if arr is not None and len(arr) > 0:
            out[g] = np.asarray(arr, dtype=np.float64)
    return out


def groups_share_filtered(per_group, percentile, ci, drop_fraction):
    """Per-target far/total share after dropping, within each group, the targets
    whose |sigma_total| lies in the lowest drop_fraction of that group. This tests
    that the share is not driven by near-zero denominators."""
    out = {}
    for g, data in per_group.items():
        try:
            share = np.asarray(data[percentile][ci]["share"], dtype=np.float64)
            sigtot = np.asarray(data[percentile][ci]["sigtot"], dtype=np.float64)
        except KeyError:
            continue
        if share.size == 0:
            continue
        if drop_fraction > 0.0 and share.size >= 5:
            thr = np.quantile(sigtot, drop_fraction)
            keep = sigtot > thr
            share = share[keep]
        if share.size > 0:
            out[g] = share
    return out


def groups_for_shellband(per_group_shellbands, percentile, band_index, metric):
    """Collect per-target band signed/abs sums across groups for one shell band."""
    out = {}
    for g, sb in per_group_shellbands.items():
        if sb is None:
            continue
        try:
            arr = sb[percentile][metric][band_index]
        except (KeyError, IndexError):
            continue
        if arr is not None and len(arr) > 0:
            out[g] = np.asarray(arr, dtype=np.float64)
    return out


def _rel_diff_stats(a, b):
    rel = np.abs(a - b) / np.maximum(np.abs(b), EPS)
    nrms = np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), EPS)
    corr = np.corrcoef(a, b)[0, 1]
    return float(np.median(rel)), float(np.mean(rel)), float(np.percentile(rel, 99.9)), float(nrms), float(corr)


def fd_order_validation_row(re_val, snap, cut, u, dx_phys):
    """Three-order finite-difference convergence check for the vorticity magnitude.

    Documents the resolution of the vorticity magnitude independently of which
    order the main analysis uses: it computes FD2, FD4 and FD6 on the same field.
    The headline comparison is FD2 vs FD6; the FD4 vs FD6 comparison shows whether
    the sequence has converged, so that agreement reflects resolution rather than
    two low-order schemes sharing the same error. This is what justifies running
    the main analysis at fourth order (MAIN_GRAD_ORDER).
    """
    omag_fd2 = _vorticity_magnitude_from_derivative(
        u, dx_phys, lambda f, d, axis: np.gradient(f, d, axis=axis, edge_order=2)
    )
    omag_fd4 = vorticity_magnitude_fd4(u, dx_phys)
    omag_fd6 = vorticity_magnitude_fd6(u, dx_phys)
    n = omag_fd2.shape[0]
    mar = 8  # exceeds the FD6 +/-3 stencil; boundary-wrapped values excluded
    sl = (slice(mar, n - mar), slice(mar, n - mar), slice(mar, n - mar))
    a2 = omag_fd2[sl].ravel()
    a4 = omag_fd4[sl].ravel()
    a6 = omag_fd6[sl].ravel()

    med26, mean26, p99926, nrms26, corr26 = _rel_diff_stats(a2, a6)
    med46, mean46, p99946, nrms46, corr46 = _rel_diff_stats(a4, a6)

    row = [
        re_val, snap, cut,
        f"{med26:.6g}", f"{mean26:.6g}", f"{p99926:.6g}", f"{nrms26:.6g}", f"{corr26:.8f}",
        f"{med46:.6g}", f"{p99946:.6g}", f"{nrms46:.6g}",
    ]
    del omag_fd2, omag_fd4, omag_fd6
    return row


# ----------------------------------------------------------------------
# Optional: validate against the JHTDB database gradient (off by default)
# ----------------------------------------------------------------------
# JHTDB exposes getVelocityGradient computed on the full spectrally-resolved
# field with selectable finite-difference order (FD4/FD6/FD8). Comparing the
# local cutout vorticity against the FD8 database gradient is a stronger,
# independent check than the local FD2/FD4/FD6 convergence above, because the
# reference is evaluated on the full field rather than the 256^3 cutout.
#
# This is disabled by default because the exact getData spatial-operator
# argument name varies between givernylocal versions. Before enabling, confirm
# the call against the demo notebook of the installed givernylocal version
# (look for getVelocityGradient / spatial_operator='gradient' / 'FD8'), then set
# USE_DB_GRADIENT_VALIDATION = True. The cutout origin and N are read from the
# HDF5 metadata so the queried grid points match the stored cutout exactly.
USE_DB_GRADIENT_VALIDATION = False


def db_gradient_validation_row(meta, omag_fd2):
    """Compare local FD2 vorticity against the JHTDB FD8 database gradient.

    Returns None if the database query is unavailable or the API call does not
    match the installed givernylocal version. Enable via USE_DB_GRADIENT_VALIDATION.
    """
    try:
        import numpy as _np
        from givernylocal.turbulence_dataset import turb_dataset
        from givernylocal.turbulence_toolkit import getData

        title = meta["dataset"]
        origin = meta.get("origin", None)
        n_cube = int(meta.get("N", meta.get("N_cube", omag_fd2.shape[0])))
        snap = int(meta.get("snapshot", 1))
        if origin is None:
            return None
        ox, oy, oz = (int(v) for v in origin)

        ds = turb_dataset(dataset_title=title, output_path="./cz_data",
                          auth_token="YOUR_JHTDB_TOKEN_HERE")

        # Build the exact integer grid of the stored cutout.
        xs = _np.arange(ox, ox + n_cube)
        ys = _np.arange(oy, oy + n_cube)
        zs = _np.arange(oz, oz + n_cube)
        gx, gy, gz = _np.meshgrid(xs, ys, zs, indexing="ij")
        points = _np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(_np.float64)

        # NOTE: confirm this signature against the installed givernylocal demo.
        # getData(dataset, variable, time, temporal_method, spatial_method,
        #         spatial_operator, points)
        grad = getData(ds, "velocity", snap, "none", "fd8noint", "gradient", points)
        grad = _np.asarray(grad).reshape(n_cube, n_cube, n_cube, 9)

        # gradient layout: [dux/dx,dux/dy,dux/dz, duy/dx,duy/dy,duy/dz, duz/dx,duz/dy,duz/dz]
        wx = grad[..., 7] - grad[..., 5]
        wy = grad[..., 2] - grad[..., 6]
        wz = grad[..., 3] - grad[..., 1]
        omag_db = _np.sqrt(wx * wx + wy * wy + wz * wz)

        mar = 8
        sl = (slice(mar, n_cube - mar),) * 3
        med, mean, p999, nrms, corr = _rel_diff_stats(omag_fd2[sl].ravel(), omag_db[sl].ravel())
        return [f"{med:.6g}", f"{mean:.6g}", f"{p999:.6g}", f"{nrms:.6g}", f"{corr:.8f}"]
    except Exception as exc:  # noqa: BLE001
        print(f"  db-gradient validation skipped ({type(exc).__name__}: {str(exc)[:80]})")
        return None


def local_cutout_stats(omag, margin):
    n = omag.shape[0]
    sl = (slice(margin, n - margin), slice(margin, n - margin), slice(margin, n - margin))
    vals = omag[sl].ravel()
    return {
        "interior_points": int(vals.size),
        "omega_mean": float(np.mean(vals)),
        "omega_median": float(np.median(vals)),
        "omega_p99": float(np.percentile(vals, 99.0)),
        "omega_p999": float(np.percentile(vals, 99.9)),
        "enstrophy_mean": float(np.mean(vals * vals)),
    }


# ----------------------------------------------------------------------
# Robustness re-measurement passes (pooled threshold; FD6 subset)
# ----------------------------------------------------------------------
def select_targets_with_threshold(omag, margin, thresholds, n_targets, seed):
    """Like select_targets, but uses externally supplied (lo, hi) thresholds per
    percentile class instead of computing them from this cutout. Used for the
    pooled-threshold sensitivity, where the threshold is computed across all blocks
    of a Reynolds number rather than locally."""
    n = omag.shape[0]
    ii, jj, kk = np.where(np.ones((n - 2 * margin, n - 2 * margin, n - 2 * margin), dtype=bool))
    ii = ii + margin; jj = jj + margin; kk = kk + margin
    vals = omag[ii, jj, kk]
    rng = np.random.default_rng(seed)
    targets = {}
    for p, (lo, hi) in thresholds.items():
        if np.isinf(hi):
            mask = vals >= lo
        else:
            mask = (vals >= lo) & (vals <= hi)
        cand = np.flatnonzero(mask)
        if len(cand) == 0:
            targets[p] = np.empty((0, 3), dtype=np.int64)
        elif len(cand) > n_targets:
            sel = rng.choice(cand, n_targets, replace=False)
            targets[p] = np.column_stack([ii[sel], jj[sel], kk[sel]]).astype(np.int64)
        else:
            targets[p] = np.column_stack([ii[cand], jj[cand], kk[cand]]).astype(np.int64)
    return targets


def pooled_thresholds_for_re(group_files, gradient_fn):
    """Pass 1 of the pooled-threshold sensitivity: read every block of a Reynolds
    number, pool the interior vorticity magnitudes (subsampled to bound memory),
    and return (lo, hi) thresholds per percentile class computed on the pooled
    sample. For the median class the bin edges are the pooled percentile points;
    for the tail class the lower edge is the pooled percentile and hi = inf."""
    pooled = []
    rng = np.random.default_rng(SEED + 777)
    for snap_n, cut_n, path in group_files:
        u, meta = load_velocity(path)
        mc = metadata_constants(meta, path)
        dx_phys = mc["dx_phys"]; eta = mc["eta"]
        du = gradient_fn(u, dx_phys)
        _, omag, _ = vorticity_strain_from_grad(du)
        del du
        n = omag.shape[0]
        max_r_grid = max(RETA_CUTS) * eta / dx_phys
        margin = int(np.ceil(max_r_grid)) + 2
        sl = (slice(margin, n - margin),) * 3
        v = omag[sl].ravel()
        # subsample to at most ~400k points per block to bound the pooled array
        if v.size > 400000:
            idx = rng.choice(v.size, 400000, replace=False)
            v = v[idx]
        pooled.append(v.astype(np.float64))
        del u, omag, v
        gc.collect()
    pooled = np.concatenate(pooled)
    thr = {}
    for p in PERCENTILES:
        if p >= 99.0:
            thr[p] = (float(np.percentile(pooled, p)), float("inf"))
        else:
            w = PERCENTILE_BIN_WIDTH
            thr[p] = (float(np.percentile(pooled, max(0.0, p - 0.5 * w))),
                      float(np.percentile(pooled, min(100.0, p + 0.5 * w))))
    del pooled
    gc.collect()
    return thr


def remeasure_re(re_val, group_files, gradient_fn, thresholds=None):
    """Re-measure a Reynolds number, returning per_group_main keyed by (snap,cut).
    If thresholds is given, targets are selected with the pooled thresholds;
    otherwise targets are selected locally (used for the FD6 subset, which keeps
    the local definition but changes the gradient order). gradient_fn picks the
    finite-difference order."""
    cuts_by_inner = {RETA_INNER: [c for c in RETA_CUTS if c > RETA_INNER]}
    per_group = {}
    for snap_n, cut_n, path in group_files:
        u, meta = load_velocity(path)
        mc = metadata_constants(meta, path)
        eta = mc["eta"]; dx_phys = mc["dx_phys"]
        snap = mc["snapshot"] if mc["snapshot"] >= 0 else snap_n
        cut = mc["cutout"] if mc["cutout"] >= 0 else cut_n
        max_r_grid = max(RETA_CUTS) * eta / dx_phys
        margin = int(np.ceil(max_r_grid)) + 2
        du = gradient_fn(u, dx_phys)
        omega, omag, S = vorticity_strain_from_grad(du)
        del du; gc.collect()
        if thresholds is not None:
            targets = select_targets_with_threshold(
                omag, margin, thresholds, N_TARGETS_PER_BIN,
                SEED + re_val + 1000 * snap + 17 * cut + 5000)
        else:
            targets, _, _ = select_targets(
                omag, margin, PERCENTILES, N_TARGETS_PER_BIN,
                SEED + re_val + 1000 * snap + 17 * cut)
        measured, _ = measure_all_inners_from_fields(omega, omag, S, eta, dx_phys, cuts_by_inner, targets)
        per_group[(snap, cut)] = measured[RETA_INNER]
        del omega, omag, S, measured, targets, u
        gc.collect()
    return per_group


def emit_tightness_decline_tailratio(re_val, per_group, tag_col, tag_val,
                                     tight_rows, decline_rows, tailratio_rows,
                                     share_rows=None):
    """Compute tightness, endpoint decline, tail/median ratio (and optionally
    share) from a re-measured per_group, appending to the supplied row lists.
    tag_col/tag_val record which robustness variant produced the row."""
    for ci, reta in enumerate(RETA_CUTS):
        for p in PERCENTILES:
            gs = groups_for_metric(per_group, p, ci, "signed")
            ga = groups_for_metric(per_group, p, ci, "abs")
            c_pt, c_lo, c_hi, ngrp, ntot = nested_bootstrap_ratio(
                gs, ga, SEED + re_val + ci + int(round(10 * p)) + 200000)
            tight_rows.append([re_val, reta, p, f"{c_pt:.8g}", f"{c_lo:.8g}", f"{c_hi:.8g}",
                               tag_val, ngrp, ntot])
            if share_rows is not None:
                gsh = groups_for_metric(per_group, p, ci, "share")
                s_pt, s_lo, s_hi, ng, nt = nested_bootstrap_median(
                    gsh, SEED + re_val + ci + int(round(10 * p)) + 210000)
                share_rows.append([re_val, reta, p, f"{s_pt:.8g}", f"{s_lo:.8g}", f"{s_hi:.8g}",
                                   tag_val, ng, nt])
        gs50 = groups_for_metric(per_group, 50.0, ci, "signed")
        ga50 = groups_for_metric(per_group, 50.0, ci, "abs")
        gs99 = groups_for_metric(per_group, 99.9, ci, "signed")
        ga99 = groups_for_metric(per_group, 99.9, ci, "abs")
        tr_pt, tr_lo, tr_hi = nested_bootstrap_tailratio(
            gs99, ga99, gs50, ga50, SEED + re_val + ci + 220000)
        tailratio_rows.append([re_val, reta, f"{tr_pt:.8g}", f"{tr_lo:.8g}", f"{tr_hi:.8g}", tag_val])

    for p in PERCENTILES:
        gs_by = [groups_for_metric(per_group, p, ci, "signed") for ci in range(len(RETA_CUTS))]
        ga_by = [groups_for_metric(per_group, p, ci, "abs") for ci in range(len(RETA_CUTS))]
        d_pt, d_lo, d_hi = nested_bootstrap_endpoint_decline(
            gs_by, ga_by, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 230000)
        decline_rows.append([re_val, p, RETA_CUTS[0], RETA_CUTS[-1],
                             f"{d_pt:.8g}", f"{d_lo:.8g}", f"{d_hi:.8g}", tag_val])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)

    files_by_re = {d["Re"]: [] for d in DATASETS}
    for path in sorted(glob.glob(f"{DATA_PATH}/iso_re*_s*_c*.h5")):
        m = _re.search(r"iso_re(\d+)_s(\d+)_c(\d+)", path)
        if not m:
            continue
        re_val = int(m.group(1))
        if re_val in files_by_re:
            files_by_re[re_val].append((int(m.group(2)), int(m.group(3)), path))

    cuts_by_inner = {inner: [c for c in RETA_CUTS if c > inner] for inner in sorted(set(RETA_INNER_SWEEP + [RETA_INNER]))}

    cutout_rows = [[
        "Re", "dataset", "snapshot", "cutout", "path", "origin", "N_cube", "N_full",
        "eta", "dx_phys", "max_R_eta", "max_R_grid", "margin", "interior_points",
        "omega_mean", "omega_median", "omega_p99", "omega_p999", "enstrophy_mean",
    ]]
    target_rows = [[
        "Re", "snapshot", "cutout", "percentile", "selection_mode", "omega_lo", "omega_hi",
        "n_candidates", "n_selected",
    ]]
    cap_rows = [[
        "Re", "R_eta", "percentile", "capacity_median", "capacity_ci_lo", "capacity_ci_hi",
        "realisation_median", "realisation_ci_lo", "realisation_ci_hi", "n_groups", "n_targets",
    ]]
    tight_rows = [["Re", "R_eta", "percentile", "Cfar", "ci_lo", "ci_hi", "n_groups", "n_targets"]]
    endpoint_rows = [["Re", "percentile", "R_eta_lo", "R_eta_hi", "endpoint_decline", "ci_lo", "ci_hi"]]
    shell_rows = [[
        "Re", "percentile", "R_eta_lo", "R_eta_hi",
        "capacity_increment_dlnR", "capacity_increment_ci_lo", "capacity_increment_ci_hi",
        "realisation_increment_dlnR", "realisation_increment_ci_lo", "realisation_increment_ci_hi",
    ]]
    logfit_rows = [[
        "Re", "percentile",
        "capacity_slope", "capacity_slope_ci_lo", "capacity_slope_ci_hi", "capacity_r2", "capacity_residuals",
        "realisation_slope", "realisation_slope_ci_lo", "realisation_slope_ci_hi", "realisation_r2", "realisation_residuals",
    ]]
    slope_ratio_rows = [["Re", "percentile", "realisation_over_capacity_slope", "ci_lo", "ci_hi"]]
    sens_rows = [["Re", "inner_R_eta", "R_eta", "percentile", "Cfar", "ci_lo", "ci_hi", "n_groups", "n_targets"]]
    share_rows = [["Re", "R_eta", "percentile", "far_total_share", "ci_lo", "ci_hi", "n_groups", "n_targets"]]
    tailratio_rows = [["Re", "R_eta", "tail_median_ratio", "ci_lo", "ci_hi"]]
    fd_rows = [[
        "Re", "snapshot", "cutout",
        "median_rel_diff_fd2_fd6", "mean_rel_diff_fd2_fd6", "p999_rel_diff_fd2_fd6", "nrms_diff_fd2_fd6", "corr_fd2_fd6",
        "median_rel_diff_fd4_fd6", "p999_rel_diff_fd4_fd6", "nrms_diff_fd4_fd6",
    ]]
    db_grad_rows = [[
        "Re", "snapshot", "cutout",
        "median_rel_diff_fd2_db", "mean_rel_diff_fd2_db", "p999_rel_diff_fd2_db", "nrms_diff_fd2_db", "corr_fd2_db",
    ]]

    # Robustness-addition outputs.
    share_filtered_rows = [["Re", "R_eta", "percentile", "drop_fraction",
                            "far_total_share", "ci_lo", "ci_hi", "n_groups", "n_targets"]]
    shell_cancel_rows = [["Re", "percentile", "band_R_eta_lo", "band_R_eta_hi",
                          "shell_cancellation_A", "ci_lo", "ci_hi", "n_groups", "n_targets"]]
    pooled_tight_rows = [["Re", "R_eta", "percentile", "Cfar", "ci_lo", "ci_hi",
                          "threshold_mode", "n_groups", "n_targets"]]
    pooled_decline_rows = [["Re", "percentile", "R_eta_lo", "R_eta_hi", "endpoint_decline",
                            "ci_lo", "ci_hi", "threshold_mode"]]
    pooled_tailratio_rows = [["Re", "R_eta", "tail_median_ratio", "ci_lo", "ci_hi", "threshold_mode"]]
    pooled_share_rows = [["Re", "R_eta", "percentile", "far_total_share", "ci_lo", "ci_hi",
                          "threshold_mode", "n_groups", "n_targets"]]
    fd6_tight_rows = [["Re", "R_eta", "percentile", "Cfar", "ci_lo", "ci_hi",
                       "grad_order", "n_groups", "n_targets"]]
    fd6_decline_rows = [["Re", "percentile", "R_eta_lo", "R_eta_hi", "endpoint_decline",
                         "ci_lo", "ci_hi", "grad_order"]]
    fd6_tailratio_rows = [["Re", "R_eta", "tail_median_ratio", "ci_lo", "ci_hi", "grad_order"]]

    for d in DATASETS:
        re_val = d["Re"]
        group_files = files_by_re[re_val]
        if not group_files:
            print(f"No files found for Re={re_val}")
            continue

        per_group_main = {}
        per_group_sweep = {inner: {} for inner in cuts_by_inner}
        per_group_shellbands = {}

        for snap_from_name, cut_from_name, path in group_files:
            print(f"Processing {path}")
            u, meta = load_velocity(path)
            mc = metadata_constants(meta, path)
            eta = mc["eta"]
            dx_phys = mc["dx_phys"]
            snap = mc["snapshot"] if mc["snapshot"] >= 0 else snap_from_name
            cut = mc["cutout"] if mc["cutout"] >= 0 else cut_from_name
            key = (snap, cut)

            max_r_grid = max(RETA_CUTS) * eta / dx_phys
            margin = int(np.ceil(max_r_grid)) + 2

            du = gradients_for_main(u, dx_phys)
            omega, omag, S = vorticity_strain_from_grad(du)
            del du
            gc.collect()

            stats = local_cutout_stats(omag, margin)
            cutout_rows.append([
                re_val, mc["dataset"], snap, cut, path, origin_to_string(meta.get("origin", None)),
                u.shape[1], mc["N_full"], f"{eta:.8g}", f"{dx_phys:.8g}",
                max(RETA_CUTS), f"{max_r_grid:.6f}", margin, stats["interior_points"],
                f"{stats['omega_mean']:.8g}", f"{stats['omega_median']:.8g}",
                f"{stats['omega_p99']:.8g}", f"{stats['omega_p999']:.8g}",
                f"{stats['enstrophy_mean']:.8g}",
            ])

            targets, target_info, _ = select_targets(
                omag, margin, PERCENTILES, N_TARGETS_PER_BIN,
                SEED + re_val + 1000 * snap + 17 * cut,
            )
            for p in PERCENTILES:
                ti = target_info[p]
                target_rows.append([
                    re_val, snap, cut, p, ti["mode"], f"{ti['lo']:.8g}",
                    "inf" if not np.isfinite(ti["hi"]) else f"{ti['hi']:.8g}",
                    ti["n_candidates"], ti["n_selected"],
                ])

            measured, shell_bands = measure_all_inners_from_fields(omega, omag, S, eta, dx_phys, cuts_by_inner, targets)
            per_group_main[key] = measured[RETA_INNER]
            for inner in cuts_by_inner:
                per_group_sweep[inner][key] = measured[inner]
            if SHELL_CANCELLATION and shell_bands is not None:
                per_group_shellbands[key] = shell_bands

            del omega, S, measured, targets
            gc.collect()

            fd_rows.append(fd_order_validation_row(re_val, snap, cut, u, dx_phys))
            if USE_DB_GRADIENT_VALIDATION:
                db_row = db_gradient_validation_row(meta, omag)
                if db_row is not None:
                    db_grad_rows.append([re_val, snap, cut] + db_row)
            del u, omag
            gc.collect()

        # Main capacity, realisation, tightness, tail ratio, share.
        for ci, reta in enumerate(RETA_CUTS):
            for p in PERCENTILES:
                gs = groups_for_metric(per_group_main, p, ci, "signed")
                ga = groups_for_metric(per_group_main, p, ci, "abs")

                cap_pt, cap_lo, cap_hi, ngrp, ntot = nested_bootstrap_median(ga, SEED + re_val + ci + int(round(10 * p)))
                sig_pt, sig_lo, sig_hi, _, _ = nested_bootstrap_median(gs, SEED + re_val + ci + int(round(10 * p)) + 10000)
                cap_rows.append([
                    re_val, reta, p,
                    f"{cap_pt:.8g}", f"{cap_lo:.8g}", f"{cap_hi:.8g}",
                    f"{sig_pt:.8g}", f"{sig_lo:.8g}", f"{sig_hi:.8g}", ngrp, ntot,
                ])

                c_pt, c_lo, c_hi, ngrp_c, ntot_c = nested_bootstrap_ratio(gs, ga, SEED + re_val + ci + int(round(10 * p)) + 20000)
                tight_rows.append([re_val, reta, p, f"{c_pt:.8g}", f"{c_lo:.8g}", f"{c_hi:.8g}", ngrp_c, ntot_c])

                gsh = groups_for_metric(per_group_main, p, ci, "share")
                sh_pt, sh_lo, sh_hi, ngrp_s, ntot_s = nested_bootstrap_median(gsh, SEED + re_val + ci + int(round(10 * p)) + 30000)
                share_rows.append([re_val, reta, p, f"{sh_pt:.8g}", f"{sh_lo:.8g}", f"{sh_hi:.8g}", ngrp_s, ntot_s])

            gs50 = groups_for_metric(per_group_main, 50.0, ci, "signed")
            ga50 = groups_for_metric(per_group_main, 50.0, ci, "abs")
            gs99 = groups_for_metric(per_group_main, 99.9, ci, "signed")
            ga99 = groups_for_metric(per_group_main, 99.9, ci, "abs")
            tr_pt, tr_lo, tr_hi = nested_bootstrap_tailratio(gs99, ga99, gs50, ga50, SEED + re_val + ci + 40000)
            tailratio_rows.append([re_val, reta, f"{tr_pt:.8g}", f"{tr_lo:.8g}", f"{tr_hi:.8g}"])

        # Curve-level tests.
        for p in PERCENTILES:
            ga_by_radius = [groups_for_metric(per_group_main, p, ci, "abs") for ci in range(len(RETA_CUTS))]
            gs_by_radius = [groups_for_metric(per_group_main, p, ci, "signed") for ci in range(len(RETA_CUTS))]

            cap_s, cap_lo, cap_hi, cap_r2, cap_resid = nested_bootstrap_curve_slope(
                ga_by_radius, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 50000
            )
            sig_s, sig_lo, sig_hi, sig_r2, sig_resid = nested_bootstrap_curve_slope(
                gs_by_radius, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 60000
            )
            logfit_rows.append([
                re_val, p,
                f"{cap_s:.8g}", f"{cap_lo:.8g}", f"{cap_hi:.8g}", f"{cap_r2:.8g}",
                ";".join(f"{r:.8g}" for r in cap_resid),
                f"{sig_s:.8g}", f"{sig_lo:.8g}", f"{sig_hi:.8g}", f"{sig_r2:.8g}",
                ";".join(f"{r:.8g}" for r in sig_resid),
            ])

            ratio_pt, ratio_lo, ratio_hi = nested_bootstrap_slope_ratio(
                gs_by_radius, ga_by_radius, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 70000
            )
            slope_ratio_rows.append([re_val, p, f"{ratio_pt:.8g}", f"{ratio_lo:.8g}", f"{ratio_hi:.8g}"])

            decline_pt, decline_lo, decline_hi = nested_bootstrap_endpoint_decline(
                gs_by_radius, ga_by_radius, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 80000
            )
            endpoint_rows.append([
                re_val, p, RETA_CUTS[0], RETA_CUTS[-1],
                f"{decline_pt:.8g}", f"{decline_lo:.8g}", f"{decline_hi:.8g}",
            ])

            cap_incs = nested_bootstrap_increments(ga_by_radius, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 90000)
            sig_incs = nested_bootstrap_increments(gs_by_radius, RETA_CUTS, SEED + re_val + int(round(10 * p)) + 100000)
            for step, (cap_inc, sig_inc) in enumerate(zip(cap_incs, sig_incs), start=1):
                shell_rows.append([
                    re_val, p, RETA_CUTS[step - 1], RETA_CUTS[step],
                    f"{cap_inc[0]:.8g}", f"{cap_inc[1]:.8g}", f"{cap_inc[2]:.8g}",
                    f"{sig_inc[0]:.8g}", f"{sig_inc[1]:.8g}", f"{sig_inc[2]:.8g}",
                ])

        # Inner-cutoff sensitivity. Uses exactly the same selected targets.
        for inner in sorted(cuts_by_inner):
            cuts = cuts_by_inner[inner]
            per_group_inner = per_group_sweep[inner]
            for ci, reta in enumerate(cuts):
                for p in PERCENTILES:
                    gs = groups_for_metric(per_group_inner, p, ci, "signed")
                    ga = groups_for_metric(per_group_inner, p, ci, "abs")
                    c_pt, c_lo, c_hi, ngrp, ntot = nested_bootstrap_ratio(
                        gs, ga, SEED + re_val + int(round(10 * inner)) + ci + int(round(10 * p)) + 110000
                    )
                    sens_rows.append([re_val, inner, reta, p, f"{c_pt:.8g}", f"{c_lo:.8g}", f"{c_hi:.8g}", ngrp, ntot])

        # (3) Far-share denominator robustness: drop lowest-|sigma_total| targets.
        if SHARE_SIGMATOTAL_FILTER:
            for ci, reta in enumerate(RETA_CUTS):
                for p in PERCENTILES:
                    gsh_f = groups_share_filtered(per_group_main, p, ci, SHARE_FILTER_DROP_FRACTION)
                    shf_pt, shf_lo, shf_hi, ngrp_f, ntot_f = nested_bootstrap_median(
                        gsh_f, SEED + re_val + ci + int(round(10 * p)) + 120000
                    )
                    share_filtered_rows.append([
                        re_val, reta, p, SHARE_FILTER_DROP_FRACTION,
                        f"{shf_pt:.8g}", f"{shf_lo:.8g}", f"{shf_hi:.8g}", ngrp_f, ntot_f,
                    ])

        # (4) Shell-wise angular cancellation A(R) per radial band.
        if SHELL_CANCELLATION:
            for bi, (blo, bhi) in enumerate(SHELL_CANCELLATION_BANDS):
                for p in PERCENTILES:
                    gb_s = groups_for_shellband(per_group_shellbands, p, bi, "signed")
                    gb_a = groups_for_shellband(per_group_shellbands, p, bi, "abs")
                    a_pt, a_lo, a_hi, ngrp_b, ntot_b = nested_bootstrap_ratio(
                        gb_s, gb_a, SEED + re_val + bi + int(round(10 * p)) + 130000
                    )
                    shell_cancel_rows.append([
                        re_val, p, blo, bhi,
                        f"{a_pt:.8g}", f"{a_lo:.8g}", f"{a_hi:.8g}", ngrp_b, ntot_b,
                    ])

        # (1) Pooled-threshold sensitivity: re-select targets using a threshold
        #     pooled across all blocks of this Reynolds number, then recompute
        #     tightness / decline / tail-ratio / share on those targets.
        if POOLED_THRESHOLD_SENSITIVITY:
            print(f"  pooled-threshold pass (Re={re_val})")
            thr = pooled_thresholds_for_re(group_files, gradients_for_main)
            pg_pooled = remeasure_re(re_val, group_files, gradients_for_main, thresholds=thr)
            emit_tightness_decline_tailratio(
                re_val, pg_pooled, "threshold_mode", "pooled",
                pooled_tight_rows, pooled_decline_rows, pooled_tailratio_rows,
                share_rows=pooled_share_rows)
            del pg_pooled
            gc.collect()

        # (2) FD6 tightness subset: recompute the full tightness/decline/tail-ratio
        #     pipeline at sixth order on a subset of blocks, to show the
        #     conclusions are not an artefact of the fourth-order gradient.
        if FD6_TIGHTNESS_SUBSET:
            subset = group_files[:FD6_SUBSET_BLOCKS_PER_RE]
            print(f"  FD6 subset pass (Re={re_val}, {len(subset)} blocks)")
            pg_fd6 = remeasure_re(re_val, subset, gradients_fd6, thresholds=None)
            emit_tightness_decline_tailratio(
                re_val, pg_fd6, "grad_order", 6,
                fd6_tight_rows, fd6_decline_rows, fd6_tailratio_rows, share_rows=None)
            del pg_fd6
            gc.collect()

    def write_csv(rows, name):
        out_file = Path(OUT_PATH) / name
        with out_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"wrote {out_file}")

    write_csv(cutout_rows, "cutout_manifest.csv")
    write_csv(target_rows, "target_manifest.csv")
    write_csv(cap_rows, "capacity_realisation.csv")
    write_csv(tight_rows, "tightness_blockbootstrap.csv")
    write_csv(endpoint_rows, "endpoint_decline.csv")
    write_csv(shell_rows, "shell_increments.csv")
    write_csv(logfit_rows, "capacity_logfit.csv")
    write_csv(slope_ratio_rows, "slope_ratio.csv")
    write_csv(sens_rows, "inner_cutoff_sensitivity.csv")
    write_csv(share_rows, "nearfar_share.csv")
    write_csv(tailratio_rows, "tail_median_ratio.csv")
    write_csv(fd_rows, "fd_order_validation.csv")
    if USE_DB_GRADIENT_VALIDATION and len(db_grad_rows) > 1:
        write_csv(db_grad_rows, "db_gradient_validation.csv")

    # Robustness-addition outputs.
    if SHARE_SIGMATOTAL_FILTER and len(share_filtered_rows) > 1:
        write_csv(share_filtered_rows, "nearfar_share_filtered.csv")
    if SHELL_CANCELLATION and len(shell_cancel_rows) > 1:
        write_csv(shell_cancel_rows, "shell_cancellation.csv")
    if POOLED_THRESHOLD_SENSITIVITY and len(pooled_tight_rows) > 1:
        write_csv(pooled_tight_rows, "tightness_pooled.csv")
        write_csv(pooled_decline_rows, "endpoint_decline_pooled.csv")
        write_csv(pooled_tailratio_rows, "tail_median_ratio_pooled.csv")
        write_csv(pooled_share_rows, "nearfar_share_pooled.csv")
    if FD6_TIGHTNESS_SUBSET and len(fd6_tight_rows) > 1:
        write_csv(fd6_tight_rows, "tightness_fd6subset.csv")
        write_csv(fd6_decline_rows, "endpoint_decline_fd6subset.csv")
        write_csv(fd6_tailratio_rows, "tail_median_ratio_fd6subset.csv")


if __name__ == "__main__":
    main()
