r"""
cz_01_download.py

Downloads isotropic-turbulence velocity cutouts from the Johns Hopkins
Turbulence Database for the Calderon-Zygmund far-field stretching analysis.

Re 611 (isotropic4096) has only ONE snapshot available in JHTDB, so temporal
bootstrap is not possible there. To still get 12 independent blocks at Re 611,
this version takes 1 snapshot x 12 spatial cutouts (not 3 snapshots x 4). The
other two Reynolds numbers keep 3 snapshots x 4 cutouts. All three therefore
have 12 blocks, but only Re 433 and Re 1300 have temporal (snapshot) robustness;
Re 611 has spatial-block robustness only.

    Re 433:  3 snapshots x 4 cutouts  = 12 blocks
    Re 611:  1 snapshot  x 12 cutouts = 12 blocks
    Re 1300: 3 snapshots x 4 cutouts  = 12 blocks

The cutout origins are drawn from a per-Re seeded RNG. Because the RNG stream is
sequential, drawing 12 origins for Re 611 reproduces the original 4 snapshot-1
origins as its first four entries, so the four existing Re-611 files (s1_c0..c3)
are recognised by filename, metadata-patched, and skipped; only the eight new
spatial cutouts (s1_c4..c11) are downloaded.

BEFORE RUNNING: if an earlier version produced Re-611 snapshot-2/3 files, delete
them first, because Re 611 must have a single snapshot:
    del cz_data\iso_re611_s2_c*.h5
    del cz_data\iso_re611_s3_c*.h5
(On Linux/macOS: rm -f ./cz_data/iso_re611_s2_c*.h5 ./cz_data/iso_re611_s3_c*.h5)

Metadata written/patched for every file:
  - N_full for each JHTDB dataset
  - domain_length = 2*pi
  - dx_phys = domain_length / N_full

Run:
    conda activate <env>
    python cz_01_download.py

Output:
    ./cz_data/iso_re433_s{1,2,3}_c{0..3}.h5
    ./cz_data/iso_re611_s1_c{0..11}.h5
    ./cz_data/iso_re1300_s{1,2,3}_c{0..3}.h5
"""

import sys
import subprocess
import time
import gc
from pathlib import Path

import numpy as np

try:
    import givernylocal  # noqa: F401
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "givernylocal"]
    )

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout
import h5py


AUTH_TOKEN = "YOUR_JHTDB_TOKEN_HERE"
OUTPUT_PATH = "./cz_data"
N_CUBE = 256
N_CUTOUTS_DEFAULT = 4
N_CUTOUTS_BY_RE = {
    433: 4,    # 3 snapshots x 4 cutouts  = 12 blocks
    611: 12,   # 1 snapshot  x 12 cutouts = 12 blocks (isotropic4096 has only 1 snapshot)
    1300: 4,   # 3 snapshots x 4 cutouts  = 12 blocks
}
SEED = 42
DOMAIN_LENGTH = 2.0 * np.pi

DATASETS = [
    {"title": "isotropic1024coarse", "Re": 433,  "N_full": 1024, "nu": 1.85e-4,
     "eta": 2.873e-3, "snapshots": [1, 2, 3]},
    {"title": "isotropic4096",       "Re": 611,  "N_full": 4096, "nu": 1.732e-4,
     "eta": 1.3844e-3, "snapshots": [1]},
    {"title": "isotropic8192",       "Re": 1300, "N_full": 8192, "nu": 4.385e-5,
     "eta": 5.00e-4,  "snapshots": [1, 2, 3]},
]


def get_dataset(title):
    return turb_dataset(dataset_title=title, output_path=OUTPUT_PATH, auth_token=AUTH_TOKEN)


def fetch_cube(ds, t, origin, size):
    x0, y0, z0 = (int(v) for v in origin)
    axes = np.array(
        [[x0, x0 + size - 1], [y0, y0 + size - 1], [z0, z0 + size - 1], [t, t]],
        dtype=np.int32,
    )
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    cutout = getCutout(ds, "velocity", axes, strides, verbose=False)

    if hasattr(cutout, "data_vars"):
        var = list(cutout.data_vars)[0]
        da = cutout[var]
        if {"zcoor", "ycoor", "xcoor", "values"}.issubset(set(da.dims)):
            arr = da.transpose("zcoor", "ycoor", "xcoor", "values").values
        else:
            arr = da.values
    else:
        arr = np.asarray(cutout)

    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[-1] != 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    return np.moveaxis(arr.astype(np.float64), -1, 0)  # (3, N, N, N)


def cutout_origins(n_full, size, n_cutouts, rng):
    return [
        tuple(int(v) for v in rng.integers(1, n_full - size + 1, size=3))
        for _ in range(n_cutouts)
    ]


def write_metadata(h5_file, d, snap, cutout_index, origin=None):
    """Write metadata needed for reproducible physical scaling."""
    n_full = int(d["N_full"])
    h5_file.attrs["Re_lambda"] = int(d["Re"])
    h5_file.attrs["dataset"] = d["title"]
    h5_file.attrs["snapshot"] = int(snap)
    h5_file.attrs["cutout"] = int(cutout_index)
    if origin is not None:
        h5_file.attrs["origin"] = tuple(int(v) for v in origin)
    h5_file.attrs["N"] = int(N_CUBE)
    h5_file.attrs["N_cube"] = int(N_CUBE)
    h5_file.attrs["N_full"] = n_full
    h5_file.attrs["domain_length"] = float(DOMAIN_LENGTH)
    h5_file.attrs["dx_phys"] = float(DOMAIN_LENGTH / n_full)
    h5_file.attrs["nu"] = float(d["nu"])
    h5_file.attrs["eta"] = float(d["eta"])
    h5_file.attrs["analysis_metadata_version"] = "2026-06-19-fixed-dx"


def patch_existing_file(outpath, d, snap, cutout_index):
    """Patch old HDF5 files that were downloaded before dx metadata existed."""
    with h5py.File(outpath, "a") as f:
        origin = f.attrs.get("origin", None)
        write_metadata(f, d, snap, cutout_index, origin=origin)


def main():
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for d in DATASETS:
        re_val, title, n_full = d["Re"], d["title"], d["N_full"]
        ds = get_dataset(title)
        rng = np.random.default_rng(SEED + re_val)

        for snap in d["snapshots"]:
            n_cutouts = N_CUTOUTS_BY_RE.get(re_val, N_CUTOUTS_DEFAULT)
            origins = cutout_origins(n_full, N_CUBE, n_cutouts, rng)
            for ci, origin in enumerate(origins):
                outpath = f"{OUTPUT_PATH}/iso_re{re_val}_s{snap}_c{ci}.h5"

                if Path(outpath).exists():
                    patch_existing_file(outpath, d, snap, ci)
                    print(f"patched metadata: re{re_val} s{snap} c{ci}")
                    continue

                u = fetch_cube(ds, snap, origin, N_CUBE)
                with h5py.File(outpath, "w") as f:
                    f.create_dataset("u", data=u[0], compression="gzip", compression_opts=4)
                    f.create_dataset("v", data=u[1], compression="gzip", compression_opts=4)
                    f.create_dataset("w", data=u[2], compression="gzip", compression_opts=4)
                    write_metadata(f, d, snap, ci, origin=origin)

                del u
                gc.collect()
                elapsed = time.time() - t_start
                print(f"re{re_val} s{snap} c{ci}  origin={origin}  ({elapsed:.0f}s)")

    print(f"done ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
