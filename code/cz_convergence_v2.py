"""
TURB-Rot radius campaign for the CZ-tightness study.

Exhaustive shell enumeration on two 256^3 periodic snapshots (velo_0.h5,
velo_100.h5), spectral derivatives. For R_max in {32,48,64,80,96,112}:
CZ capacity sigma_abs, realised |sigma_signed|, and tightness C_far at
percentile bins P50 and P99.9 (200 targets per bin per snapshot).

Deterministic: SEED = 42 (+100 for the second snapshot). Includes a
self-check that pooled aggregates match the deposited results file to
print precision.

Usage:   python cz_convergence_v2.py     (snapshots in working directory)
Outputs: cz_convergence_v2_results.txt, cz_convergence_v2_targets.csv.
"""

import numpy as np
import h5py
import time
import gc
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_0   = "velo_0.h5"
DATA_100 = "velo_100.h5"
OUT_FILE = "cz_convergence_v2_results.txt"

N   = 256
nu  = 0.012
EPS = 1e-12

R_INNER  = 8
R_MAX_LIST = [32, 48, 64, 80, 96, 112]   # capped < N/2 = 128
R_OUTER_MAX = max(R_MAX_LIST)            # 112: 2*112=224 < 256, ingen wrap-dobbelttaelling

# Vis konvergens for baade hale (P99.9) og moderat (P50)
PERCS_TEST = [50.0, 99.9]
N_PER_BIN  = 200          # publikationskvalitet
SEED       = 42
N_BOOTSTRAP = 2000

# ─────────────────────────────────────────────────────────────────────────────
lines_out = []
def out(s=""):
    print(s); lines_out.append(str(s))
def hdr(s):
    out("="*70); out(s); out("="*70)
def save_now():
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines_out))

# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE OFFSETS (ud til R_OUTER_MAX, med radius til cutoff-masking)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_offsets(r_inner, r_outer):
    rng_max = int(np.ceil(r_outer))
    coords = np.arange(-rng_max, rng_max+1)
    DX,DY,DZ = np.meshgrid(coords,coords,coords,indexing='ij')
    DX=DX.ravel(); DY=DY.ravel(); DZ=DZ.ravel()
    R2 = DX*DX+DY*DY+DZ*DZ
    mask = (R2 >= r_inner*r_inner) & (R2 <= r_outer*r_outer)
    DX=DX[mask].astype(np.int64); DY=DY[mask].astype(np.int64); DZ=DZ[mask].astype(np.int64)
    R2=R2[mask].astype(np.float64); R=np.sqrt(R2)
    R5=np.maximum(R2*R2*R,EPS)
    return DX,DY,DZ,R,R5

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + VORTICITY
# ─────────────────────────────────────────────────────────────────────────────
def load_and_compute(path):
    out(f"\nLoading: {path}")
    with h5py.File(path,'r') as f:
        u=np.zeros((3,N,N,N),dtype=np.float64)
        u[0]=f['PS3D/vx'][:]; u[1]=f['PS3D/vy'][:]; u[2]=f['PS3D/vz'][:]
    K1D=np.fft.fftfreq(N)*N
    KX,KY,KZ=np.meshgrid(K1D,K1D,K1D,indexing='ij')
    u_h=[np.fft.fftn(u[i]) for i in range(3)]
    omega=np.zeros_like(u)
    omega[0]=np.fft.ifftn(1j*KY*u_h[2]-1j*KZ*u_h[1]).real
    omega[1]=np.fft.ifftn(1j*KZ*u_h[0]-1j*KX*u_h[2]).real
    omega[2]=np.fft.ifftn(1j*KX*u_h[1]-1j*KY*u_h[0]).real
    omega_mag=np.sqrt(omega[0]**2+omega[1]**2+omega[2]**2)
    del u,u_h,KX,KY,KZ; gc.collect()
    out(f"  |omega| median={np.median(omega_mag):.4f}  max={np.max(omega_mag):.4f}")
    return omega, omega_mag

# ─────────────────────────────────────────────────────────────────────────────
# TARGET-SELEKTION (periodisk => INGEN margin)
# ─────────────────────────────────────────────────────────────────────────────
def select_targets(omega_mag, seed, n_per_bin, percentiles):
    rng = np.random.default_rng(seed)
    flat = omega_mag.ravel()
    thresholds = [np.percentile(flat,p) for p in percentiles]
    sorted_p = sorted(set([0.0]+percentiles+[100.0]))
    targets = {}
    for p in percentiles:
        th = np.percentile(flat, p)
        # hi-graense = naeste percentil i listen, ellers max
        higher = [q for q in percentiles if q > p]
        hi = np.percentile(flat, min(higher)) if higher else flat.max()*1.01
        mask = (omega_mag>=th) & (omega_mag<hi)
        cands = np.argwhere(mask)
        n = min(n_per_bin, len(cands))
        chosen = cands[rng.choice(len(cands), n, replace=False)]
        targets[p] = [(int(c[0]),int(c[1]),int(c[2])) for c in chosen]
        out(f"  P{p:.1f}: {len(cands):,} candidates -> {n} targets  (Phi>{th:.2f})")
    return targets

# ─────────────────────────────────────────────────────────────────────────────
# KERNEL-BIDRAG c[i] for alle offsets (een gang per target)
# ─────────────────────────────────────────────────────────────────────────────
def kernel_contributions(omega, ix0, iy0, iz0, xi0, DX, DY, DZ, R5):
    jx=(ix0+DX)%N; jy=(iy0+DY)%N; jz=(iz0+DZ)%N
    ox=omega[0][jx,jy,jz]; oy=omega[1][jx,jy,jz]; oz=omega[2][jx,jy,jz]
    dot=ox*xi0[0]+oy*xi0[1]+oz*xi0[2]
    opx=ox-dot*xi0[0]; opy=oy-dot*xi0[1]; opz=oz-dot*xi0[2]
    rx=-DX.astype(np.float64); ry=-DY.astype(np.float64); rz=-DZ.astype(np.float64)
    xdr=xi0[0]*rx+xi0[1]*ry+xi0[2]*rz
    cx=xi0[1]*rz-xi0[2]*ry; cy=xi0[2]*rx-xi0[0]*rz; cz=xi0[0]*ry-xi0[1]*rx
    fac=-3.0*xdr/(4.0*np.pi*R5)
    Tx=fac*cx; Ty=fac*cy; Tz=fac*cz
    return Tx*opx+Ty*opy+Tz*opz   # c[i] for each offset

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(vals, n_boot, seed):
    if len(vals)<5: return None,None,None
    rng=np.random.default_rng(seed); arr=np.array(vals); meds=[]
    for _ in range(n_boot):
        meds.append(np.median(rng.choice(arr,size=len(arr),replace=True)))
    return float(np.median(arr)), float(np.percentile(meds,2.5)), float(np.percentile(meds,97.5))

# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def convergence(label, data_path, DX, DY, DZ, R, R5, seed_offset=0):
    hdr(f"CONVERGENCE: {label}")
    omega, omega_mag = load_and_compute(data_path)
    targets = select_targets(omega_mag, SEED+seed_offset, N_PER_BIN, PERCS_TEST)

    # Precompute cutoff masks for each R_max
    masks = {rmax: (R <= rmax) for rmax in R_MAX_LIST}

    # For hvert target: c[i] een gang, derefter kumulative summer
    results = {p: {rmax: [] for rmax in R_MAX_LIST} for p in PERCS_TEST}
    res_abs = {p: {rmax: [] for rmax in R_MAX_LIST} for p in PERCS_TEST}
    res_sig = {p: {rmax: [] for rmax in R_MAX_LIST} for p in PERCS_TEST}

    for p in PERCS_TEST:
        tlist = targets[p]
        out(f"\n  Computing P{p:.1f} ({len(tlist)} targets, {len(DX):,} points each) ...")
        t1=time.time()
        for ti,(ix0,iy0,iz0) in enumerate(tlist):
            if ti%50==0 and ti>0:
                el=time.time()-t1
                out(f"    target {ti}/{len(tlist)} (ETA ~{el/ti*(len(tlist)-ti):.0f}s)")
            om0=np.array([omega[k,ix0,iy0,iz0] for k in range(3)])
            phi_t=np.linalg.norm(om0)
            if phi_t<EPS: continue
            xi0v=om0/phi_t
            c = kernel_contributions(omega,ix0,iy0,iz0,xi0v,DX,DY,DZ,R5)
            ac = np.abs(c)
            for rmax in R_MAX_LIST:
                m=masks[rmax]
                s_abs=np.sum(ac[m]); s_sig=abs(np.sum(c[m]))
                if s_abs>EPS:
                    results[p][rmax].append(s_sig/s_abs)
                    res_abs[p][rmax].append(float(s_abs))
                    res_sig[p][rmax].append(float(s_sig))
                    TARGET_ROWS.append((label,p,ix0,iy0,iz0,float(phi_t),rmax,
                                        float(s_abs),float(s_sig),float(s_sig/s_abs)))
        out(f"    done {time.time()-t1:.0f}s")
        save_now()

    del omega, omega_mag; gc.collect()
    return results, res_abs, res_sig

TARGET_ROWS=[]
PUB_CONV={32:(0.112,0.083),48:(0.099,0.069),64:(0.090,0.063),80:(0.086,0.057),96:(0.083,0.053),112:(0.080,0.051)}
PUB_COMP={32:(11.82,19.96,1.28,1.58),48:(15.36,23.74,1.50,1.62),64:(17.89,26.52,1.67,1.62),
          80:(19.87,28.51,1.71,1.61),96:(21.49,30.13,1.75,1.59),112:(22.87,31.45,1.78,1.57)}
def report_components(res_abs, res_sig, label):
    hdr(f"KOMPONENTER (tab:comp) — {label}")
    out(f"  {'R_max':>6} | {'abs P50':>8} {'abs P99.9':>9} | {'sig P50':>8} {'sig P99.9':>9}")
    for rmax in R_MAX_LIST:
        a50=np.median(res_abs[50.0][rmax]) if res_abs[50.0][rmax] else float('nan')
        a99=np.median(res_abs[99.9][rmax]) if res_abs[99.9][rmax] else float('nan')
        s50=np.median(res_sig[50.0][rmax]) if res_sig[50.0][rmax] else float('nan')
        s99=np.median(res_sig[99.9][rmax]) if res_sig[99.9][rmax] else float('nan')
        out(f"  {rmax:>6} | {a50:>8.2f} {a99:>9.2f} | {s50:>8.2f} {s99:>9.2f}")

def reproduction_check(res_comb, abs_comb, sig_comb):
    hdr("SELF-CHECK against deposited tab:comp/tab:conv aggregates (pooled)")
    ok=True
    for rmax in R_MAX_LIST:
        if rmax not in PUB_CONV: continue
        m50=np.median(res_comb[50.0][rmax]); m99=np.median(res_comb[99.9][rmax])
        p50,p99=PUB_CONV[rmax]
        fl = "OK" if (abs(m50-p50)<0.002 and abs(m99-p99)<0.002) else "DEVIATION"
        if fl!="OK": ok=False
        out(f"  tab:conv R={rmax:>3}: measured {m50:.4f}/{m99:.4f}  ref {p50:.3f}/{p99:.3f}  [{fl}]")
        a50=np.median(abs_comb[50.0][rmax]); a99=np.median(abs_comb[99.9][rmax])
        s50=np.median(sig_comb[50.0][rmax]); s99=np.median(sig_comb[99.9][rmax])
        pa50,pa99,ps50,ps99=PUB_COMP[rmax]
        fl2 = "OK" if (abs(a50-pa50)<0.05 and abs(a99-pa99)<0.05 and abs(s50-ps50)<0.02 and abs(s99-ps99)<0.02) else "DEVIATION"
        if fl2!="OK": ok=False
        out(f"  tab:comp R={rmax:>3}: abs {a50:.2f}/{a99:.2f} (ref {pa50}/{pa99})  sig {s50:.2f}/{s99:.2f} (ref {ps50}/{ps99})  [{fl2}]")
    out(f"\n  OVERALL: {'REPRODUCED — matches deposited aggregates' if ok else 'DEVIATION — investigate before use'}")

def report(results, label):
    hdr(f"C_far(R_max) — CONVERGENCE: {label}")
    out(f"\n  R_max    | shell-pts | " + " | ".join(f"P{p:.1f} C_far (95% CI)" for p in PERCS_TEST))
    out(f"  {'-'*8}-+-{'-'*9}-+-" + "-+-".join("-"*24 for _ in PERCS_TEST))

    # shell point counts per R_max
    for rmax in R_MAX_LIST:
        npts = int((4.0/3.0)*np.pi*(rmax**3 - R_INNER**3))  # approx
        cells = []
        for p in PERCS_TEST:
            vals = results[p][rmax]
            med,lo,hi = bootstrap_ci(vals, N_BOOTSTRAP, SEED+rmax+int(p))
            if med is None:
                cells.append(f"{'--':>24}")
            else:
                cells.append(f"{med:.4f} [{lo:.4f},{hi:.4f}]")
        out(f"  {rmax:>8} | ~{npts:>8,} | " + " | ".join(cells))

    # Convergence check: relative change between the last two R_max
    out(f"\n  CONVERGENCE CHECK (relative change R_max=96 -> 112):")
    for p in PERCS_TEST:
        v96 = results[p][96]; v112 = results[p][112]
        if v96 and v112:
            m96=np.median(v96); m112=np.median(v112)
            rel = abs(m112-m96)/max(m96,EPS)*100
            flag = "CONVERGED (<5%)" if rel<5 else "not converged"
            out(f"    P{p:.1f}: {m96:.4f} → {m112:.4f}  ({rel:.1f}% change)  [{flag}]")

    # Separation tail vs moderate at each R_max
    if 50.0 in PERCS_TEST and 99.9 in PERCS_TEST:
        out(f"\n  SEPARATION (P50 vs P99.9) at each R_max:")
        for rmax in R_MAX_LIST:
            v50=results[50.0][rmax]; v99=results[99.9][rmax]
            if v50 and v99:
                m50=np.median(v50); m99=np.median(v99)
                out(f"    R_max={rmax:>3}: P50={m50:.4f}, P99.9={m99:.4f}, "
                    f"ratio={m99/max(m50,EPS):.3f}")
    # ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
t_total=time.time()
hdr("CZ CONVERGENCE — TURB-Rot RADIUS CAMPAIGN")
out(f"  N={N}, nu={nu}, periodic domain (no margin)")
out(f"  R_inner={R_INNER}, R_max series={R_MAX_LIST}  (capped < N/2={N//2})")
out(f"  {N_PER_BIN} targets/bin, percentiler {PERCS_TEST}")
out(f"  Bootstrap {N_BOOTSTRAP} resamples")
save_now()

out(f"\nPrecomputing offsets R={R_INNER}-{R_OUTER_MAX} ...")
DX,DY,DZ,R,R5 = precompute_offsets(R_INNER, R_OUTER_MAX)
out(f"  {len(DX):,} points (reused via cutoff masking for all R_max)")
save_now()

res_0, abs_0, sig_0 = convergence("t=0",   DATA_0,   DX,DY,DZ,R,R5, seed_offset=0)
report(res_0, "t=0"); report_components(abs_0, sig_0, "t=0")
save_now()

res_100, abs_100, sig_100 = convergence("t=100", DATA_100, DX,DY,DZ,R,R5, seed_offset=100)
report(res_100, "t=100"); report_components(abs_100, sig_100, "t=100")
save_now()

# Kombineret
hdr("POOLED (t=0 + t=100)")
res_comb = {p:{rmax: res_0[p][rmax]+res_100[p][rmax] for rmax in R_MAX_LIST} for p in PERCS_TEST}
abs_comb = {p:{rmax: abs_0[p][rmax]+abs_100[p][rmax] for rmax in R_MAX_LIST} for p in PERCS_TEST}
sig_comb = {p:{rmax: sig_0[p][rmax]+sig_100[p][rmax] for rmax in R_MAX_LIST} for p in PERCS_TEST}
report(res_comb, "kombineret"); report_components(abs_comb, sig_comb, "kombineret (= tab:comp)")
reproduction_check(res_comb, abs_comb, sig_comb)
try:
    import csv as _csv
    with open("cz_convergence_v2_targets.csv","w",newline="") as fh:
        w=_csv.writer(fh)
        w.writerow(["snapshot","percentile","ix","iy","iz","omega_mag","R_max","sigma_abs","sigma_signed","C_far"])
        w.writerows(TARGET_ROWS)
    out(f"\nWrote cz_convergence_v2_targets.csv ({len(TARGET_ROWS)} rows).")
except Exception as e:
    out(f"WARNING: csv dump failed: {e}")

hdr(f"COMPLETE — {time.time()-t_total:.0f} s")
save_now()
print(f"\nSaved: {OUT_FILE}")
