"""
theta_coherence_v1.py — Blockwise phase coherence of the contracted far-field
stretching integrand at intense vorticity events, vs spectrum-matched
phase-randomised surrogates. (Code identical to the internal rigidity_test_v5.)

Theta = |sum_Q I_Q| / sum_Q |I_Q| over blocks Q (4 radial shells x 8 sign
octants) around local vorticity maxima. Includes per-cutout orientation
verification against the exact identity <|S|^2>/<|omega|^2> = 1/2 (gate
[0.45,0.55]; full-domain check for periodic snapshots). Outputs
rigidity_v5_events.csv with per-event Theta, the within-event random-sign
null (column pnull), and kind = real / surrogat.

Usage:  python theta_coherence_v1.py file1.h5 [file2.h5 ...]
Fields: velocity, 256^3, keys u/v/w or PS3D/vx,vy,vz.
"""
import sys, glob, numpy as np, h5py, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

N_EVENTS, SEP, RMIN, RMAX = 300, 16, 2.0, 22.0   # events, min-afstand, skal-graenser (celler)
NSHUF = 400                                       # blok-fortegns-shuffles per event (null)
SURR_REPS = 1                                     # antal fase-randomiserede surrogater per fil

def load_field(path):
    """Returnerer (omega[3,N,N,N]). Autodetekterer velocity vs vorticity, layouts."""
    with h5py.File(path, "r") as f:
        keys = []
        f.visit(lambda k: keys.append(k))
        dsets = {k: f[k] for k in keys if isinstance(f[k], h5py.Dataset)}
        print(f"  h5-indhold: {[(k, v.shape) for k, v in dsets.items()][:8]}")
        arr = None; names = None
        # (3,N,N,N) eller (N,N,N,3)
        for k, v in dsets.items():
            s = v.shape
            if len(s) == 4 and 3 in (s[0], s[-1]):
                a = v[...]
                arr = a if s[0] == 3 else np.moveaxis(a, -1, 0); names = k; break
        if arr is None:  # tre separate (N,N,N)
            comps = [k for k, v in dsets.items() if len(v.shape) == 3]
            for trio in (("u","v","w"), ("ux","uy","uz"), ("wx","wy","wz"), ("omx","omy","omz")):
                hit = [c for c in comps if c.split("/")[-1].lower() in trio]
                if len(hit) == 3:
                    hit = sorted(hit, key=lambda c: trio.index(c.split("/")[-1].lower()))
                    arr = np.stack([dsets[c][...] for c in hit]); names = ",".join(hit); break
            if arr is None and len(comps) == 3:
                arr = np.stack([dsets[c][...] for c in sorted(comps)]); names = ",".join(sorted(comps))
        if arr is None:
            raise RuntimeError("Kunne ikke finde 3-komponent felt - send h5-indholdslisten ovenfor.")
    arr = arr.astype(np.float32)
    isvel = not any(t in names.lower() for t in ("om", "vort", "w_x"))
    print(f"  felt: '{names}'  shape={arr.shape}  tolket som {'VELOCITY -> beregner curl' if isvel else 'VORTICITY'}")
    if isvel:
        g = lambda a, ax: np.gradient(a, axis=ax)
        u, v, w = arr
        om = np.stack([g(w,1)-g(v,2), g(u,2)-g(w,0), g(v,0)-g(u,1)])
        return om, arr          # (vorticity, velocity)
    return arr, None

def _ratio_of(vel):
    g = lambda a, ax: np.gradient(a, axis=ax)
    u, v, w = vel
    om = np.stack([g(w,1)-g(v,2), g(u,2)-g(w,0), g(v,0)-g(u,1)])
    S2 = 0.0
    for i in range(3):
        for j in range(3):
            Sij = 0.5*(g(vel[j], i) + g(vel[i], j)); S2 += float((Sij**2).mean()); del Sij
    return S2/float((om**2).sum(0).mean())

def calibrate_orientation(vel):
    """36 (akse-perm x komp-perm) paa subkube; vaelg ratio naermest 0.5; kraev [0.45,0.55]."""
    from itertools import permutations
    N = vel.shape[1]; c0 = N//2; h = min(48, c0-2)
    sub = vel[:, c0-h:c0+h, c0-h:c0+h, c0-h:c0+h].astype(np.float32)
    best = (None, None, 9e9)
    for ap in permutations((0,1,2)):
        subT = np.stack([np.transpose(sub[c], ap) for c in range(3)])
        for cp in permutations((0,1,2)):
            r = _ratio_of(subT[list(cp)])
            if abs(r-0.5) < abs(best[2]-0.5): best = (ap, cp, r)
    ap, cp, r = best
    ok = 0.45 <= r <= 0.55
    print(f"  ORIENTERING: akse={ap} komp={cp} subkube-ratio={r:.3f}  GATE {'OK' if ok else 'FEJLER'}")
    velC = np.stack([np.transpose(vel[c], ap) for c in range(3)])[list(cp)]
    return velC, ok

def make_surrogate(om, seed, vel=None):
    """Fase-randomiseret surrogat: samme energispektrum (per komponent og kryds),
    samme divergensfrihed (faelles fase per boelgevektor), Hermitisk via FFT af reel stoej.
    Randomiserer KUN kryds-k-faserne = den rumlige organisering."""
    rng = np.random.default_rng(seed)
    W = np.fft.fftn(rng.standard_normal(om.shape[1:]).astype(np.float32))
    W /= (np.abs(W) + 1e-30)
    out = np.empty_like(om)
    for c in range(3):
        out[c] = np.fft.ifftn(np.fft.fftn(om[c]) * W).real.astype(np.float32)
    vout = None
    if vel is not None:
        vout = np.empty_like(vel)
        for c in range(3):
            vout[c] = np.fft.ifftn(np.fft.fftn(vel[c]) * W).real.astype(np.float32)
    return out, vout

def find_peaks(om_mag, n, sep):
    flat = np.argsort(om_mag, axis=None)[::-1][: n * 60]
    pts, taken = [], []
    N = om_mag.shape[0]
    for idx in flat:
        p = np.array(np.unravel_index(idx, om_mag.shape))
        if p.min() >= RMAX + 1 and p.max() < N - RMAX - 1 and \
           all(np.abs(p - q).max() >= sep for q in taken):
            taken.append(p); pts.append(p)
            if len(pts) >= n: break
    return pts

def event_metrics(om, p, vel=None):
    R = int(RMAX) + 1
    sub = om[:, p[0]-R:p[0]+R+1, p[1]-R:p[1]+R+1, p[2]-R:p[2]+R+1]
    xi = sub[:, R, R, R].copy(); Phi = np.linalg.norm(xi)
    if Phi < 1e-12: return None
    xi /= Phi
    ax = np.arange(-R, R+1, dtype=np.float32)
    Z = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"))          # (3,n,n,n)
    r = np.sqrt((Z**2).sum(0)); m = (r >= RMIN) & (r <= RMAX)
    zh = Z / np.maximum(r, 1e-9)
    # CF93-integrand: (zh.xi) * ((zh x xi) . omega) / r^3   (celle-sum, konstant ligegyldig for Theta)
    zdxi = np.einsum("i,inml->nml", xi, zh)
    cross = np.stack([zh[1]*xi[2]-zh[2]*xi[1], zh[2]*xi[0]-zh[0]*xi[2], zh[0]*xi[1]-zh[1]*xi[0]])
    integ = zdxi * np.einsum("inml,inml->nml", cross, sub) / np.maximum(r, 1e-9)**3
    # blokke: dyadiske skaller x 8 oktanter i xi-rammen
    e1 = np.cross(xi, [0., 0., 1.]);  e1 = np.cross(xi, [0., 1., 0.]) if np.linalg.norm(e1) < 1e-6 else e1
    e1 /= np.linalg.norm(e1); e2 = np.cross(xi, e1)
    c1 = np.einsum("i,inml->nml", e1, zh) >= 0
    c2 = np.einsum("i,inml->nml", e2, zh) >= 0
    c3 = zdxi >= 0
    shell = np.digitize(r, [RMIN, 4, 8, 16, RMAX + 0.001])
    IQ = []
    for s in (1, 2, 3, 4):
        for o in range(8):
            bm = m & (shell == s) & (c1 == bool(o & 1)) & (c2 == bool(o & 2)) & (c3 == bool(o & 4))
            if bm.sum() >= 4: IQ.append(integ[bm].sum())
    IQ = np.array(IQ)
    if len(IQ) > 3 and np.abs(IQ).sum() > 0:
        Theta = abs(IQ.sum()) / np.abs(IQ).sum()
        # NULL: samme blok-magnituder, uafhaengige tilfaeldige fortegn.
        # Konservativ for at detektere UNDERTRYKKELSE (glathed korrelerer naboblokke
        # -> reel "tilfaeldig" turbulens ville ligge OVER denne null).
        nrng = np.random.default_rng(int(p[0])*73856093 ^ int(p[1])*19349663 ^ int(p[2])*83492791)
        signs = nrng.choice([-1.0, 1.0], size=(NSHUF, len(IQ)))
        th_null = np.abs(signs @ np.abs(IQ)) / np.abs(IQ).sum()
        Th_null_med = float(np.median(th_null))
        pnull = float((th_null >= Theta).mean())   # ~1: observeret UNDER null (aktiv defasning); ~0: laasning
    else:
        Theta, Th_null_med, pnull = np.nan, np.nan, np.nan
    # aksesymmetri-indeks A: variansandel forklaret af azimutalt middel i (rho,z)-bins
    zax = np.einsum("i,inml->nml", xi, Z); rho = np.sqrt(np.maximum(r**2 - zax**2, 0))
    erho = (Z - xi[:, None, None, None] * zax) / np.maximum(rho, 1e-9)
    ephi = np.stack([xi[1]*erho[2]-xi[2]*erho[1], xi[2]*erho[0]-xi[0]*erho[2], xi[0]*erho[1]-xi[1]*erho[0]])
    comps = [np.einsum("inml,inml->nml", b, sub) for b in
             (erho, ephi, np.broadcast_to(xi[:, None, None, None], sub.shape))]
    rb = np.digitize(rho, np.linspace(0, RMAX, 7)); zb = np.digitize(zax, np.linspace(-RMAX, RMAX, 9))
    # A = 1 - residualvarians(omkring azimutalt bin-middel) / totalvarians, over celler i gyldige bins
    valid = np.zeros_like(m)
    for rbi in range(1, 7):
        for zbi in range(1, 9):
            bm = m & (rb == rbi) & (zb == zbi)
            if bm.sum() >= 8: valid |= bm
    res = tot = 0.0
    for comp in comps:
        vm = comp[valid]; tot += ((vm - vm.mean())**2).sum()
        for rbi in range(1, 7):
            for zbi in range(1, 9):
                bm = valid & (rb == rbi) & (zb == zbi)
                if bm.sum(): v = comp[bm]; res += ((v - v.mean())**2).sum()
    A = max(0.0, 1.0 - res / max(tot, 1e-12))
    sigma_loc = np.nan
    if vel is not None:
        G = np.empty((3,3), np.float64)        # G_ij = du_j/dx_i (centrale differenser)
        for i in range(3):
            ip, im = p.copy(), p.copy(); ip[i] += 1; im[i] -= 1
            for j in range(3):
                G[i, j] = 0.5*(vel[j][tuple(ip)] - vel[j][tuple(im)])
        S = 0.5*(G + G.T)
        sigma_loc = float(xi @ S @ xi)
    return dict(Theta=Theta, A=A, ommax=float(Phi), nblocks=len(IQ),
                Th_null_med=Th_null_med, pnull=pnull, sigma_loc=sigma_loc)

rows = []
files = sum([glob.glob(a) for a in (sys.argv[1:] or ["*.h5"])], [])
def process(om, fname, kind, vel=None):
    mag = np.linalg.norm(om, axis=0)
    pts = find_peaks(mag, N_EVENTS, SEP)
    print(f"  [{kind}] events: {len(pts)}")
    for i, p in enumerate(pts):
        met = event_metrics(om, p, vel)
        if met and np.isfinite(met["Theta"]):
            met.update(file=fname, ev=i, kind=kind); rows.append(met)
GATES = {}
for path in files:
    print(f"== {path}")
    om, vel = load_field(path); fname = path.split("/")[-1].split("\\")[-1]
    if vel is not None:
        vel, gate_ok = calibrate_orientation(vel)
        GATES[fname] = gate_ok
        g = lambda a, ax: np.gradient(a, axis=ax)
        u, v, w = vel
        om = np.stack([g(w,1)-g(v,2), g(u,2)-g(w,0), g(v,0)-g(u,1)]).astype(np.float32)
        rfull = _ratio_of(vel)
        print(f"  KONTROL fuldfelt: <|S|2>/<|om|2> = {rfull:.3f}")
    else:
        print("  ADVARSEL: vorticity-fil — kan ikke orienterings-kalibrere via identiteten; GATE=ukendt")
        GATES[fname] = False
    process(om, fname, "real", vel)
    for s in range(SURR_REPS):
        om_s, vel_s = make_surrogate(om, seed=1234 + s, vel=vel)
        process(om_s, fname, "surrogat", vel_s)
        del om_s, vel_s
    del om, vel

df = pd.DataFrame(rows); df.to_csv("rigidity_v5_events.csv", index=False)
print(f"\nGemt rigidity_v5_events.csv  ({len(df)} events)")
def ranksum_p(a, b):   # Mann-Whitney, normal-approx, tosidet (uden scipy)
    a, b = np.asarray(a), np.asarray(b); n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5: return float("nan")
    allv = np.concatenate([a, b]); r = pd.Series(allv).rank().values
    U = r[:n1].sum() - n1*(n1+1)/2
    mu, sd = n1*n2/2, (n1*n2*(n1+n2+1)/12)**0.5
    from math import erf
    return 2*(1-0.5*(1+erf(abs((U-mu)/sd)/2**0.5)))
print("\n=== MEKANISME-TEST: Theta ~ sigma_loc/ommax (fasebudgettets to arme) ===")
def pcorr(d, x, y, z):
    rxy, rxz, ryz = d[x].corr(d[y]), d[x].corr(d[z]), d[y].corr(d[z])
    den = ((1-rxz**2)*(1-ryz**2))**0.5
    return (rxy - rxz*ryz)/den if den > 0 else float("nan")
for f, sub in df.groupby("file"):
    for kind in ("real", "surrogat"):
        s = sub[(sub.kind==kind) & sub.sigma_loc.notna()].copy()
        if len(s) < 10: continue
        s["ratio"] = s.sigma_loc/s.ommax
        print(f"{f} [{kind}]: corr(Th,sig/Phi)={s.Theta.corr(s.ratio):+.3f}  "
              f"partial(Th,sig|Phi)={pcorr(s,'Theta','sigma_loc','ommax'):+.3f}  "
              f"partial(Th,Phi|sig)={pcorr(s,'Theta','ommax','sigma_loc'):+.3f}  N={len(s)}")
print("Forudsigelse [fasebudget]: real: partial(Th,sig|Phi)>0 OG partial(Th,Phi|sig)<0 ; surrogat: begge ~0.")
print("\nGATE-STATUS:", GATES)
if not all(GATES.values()): print(">>> MINDST EN FIL FEJLEDE GATEN — dens raekker er suspenderet. <<<")
print("\n=== GAFLEN (v5, orienterings-korrigeret): real vs spektrum-matchet surrogat ===")
for f, sub in df.groupby("file"):
    re_, su_ = sub[sub.kind=="real"], sub[sub.kind=="surrogat"]
    if len(su_) == 0: continue
    p = ranksum_p(re_.Theta, su_.Theta)
    d = re_.Theta.median() - su_.Theta.median()
    verdict = "AKTIV defasning (real UNDER surrogat)" if (d < 0 and p < 0.05) else \
              "fase-ORGANISERING (real OVER surrogat)" if (d > 0 and p < 0.05) else "ingen forskel = spektrum forklarer alt"
    print(f"{f}: med Θ real={re_.Theta.median():.3f}  surrogat={su_.Theta.median():.3f}  Δ={d:+.3f}  p={p:.3g}  -> {verdict}")
def binom_p(k, n):  # tosidet sign-test, normal-approx (undgaar scipy)
    if n == 0: return float("nan")
    z = (k - n/2) / (0.5 * n**0.5)
    from math import erf
    return 2 * (1 - 0.5*(1+erf(abs(z)/2**0.5)))
for f, sub in df[df.kind=="real"].groupby("file"):
    qT, qD = sub.Theta.quantile(.9), (1 - sub.A).quantile(.9)
    corner = ((sub.Theta > qT) & ((1 - sub.A) > qD)).sum()
    nlow = int((sub.pnull > 0.5).sum())
    print(f"{f}: N={len(sub)}  corr(Theta,A)={sub.Theta.corr(sub.A):+.3f}  "
          f"HJOERNE: obs={corner}/{len(sub)*0.01:.1f}  |  "
          f"median pnull={sub.pnull.median():.3f}  andel UNDER null={nlow/len(sub):.1%}  sign-test p={binom_p(nlow, len(sub)):.2g}")
print("\nLAESNING: median pnull > 0.5 og andel>50% => turbulens-Theta ligger UNDER tilfaeldige fortegn = AKTIV defasning.")
print("          pnull ~ 0.5 => kun fravaer af laasning. pnull < 0.5 => delvis fase-organisering.")
fig, ax = plt.subplots(figsize=(7, 5))
for f, sub in df[df.kind=="real"].groupby("file"):
    ax.scatter(1 - sub.A, sub.Theta, s=10, alpha=.5, label=f.split(".")[0])
ax.set_xlabel("1 − A  (afstand fra aksesymmetri)"); ax.set_ylabel("Theta (kerne-koherens)")
ax.legend(fontsize=8); ax.set_title("Rigiditetstest: er hjoernet (hoej Theta, hoej 1−A) tomt?")
fig.tight_layout(); plt.savefig("rigidity_v5_scatter.png", dpi=130)
print("Gemt rigidity_v5_scatter.png — upload rigidity_v5_events.csv (+png) i chatten.")
