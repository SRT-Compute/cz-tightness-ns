"""
==============================================================================
FILNAVN:  cz_reynolds_v5.py  —  F-2/P0: ORIENTERINGS-KORRIGERET ARTIKEL-PIPELINE
GEM SOM:  C:/forskning/cz_reynolds_v5.py

F-2/P0 (laast foer data):
  v1-v3 laeste JHTDB-cutouts (z,y,x) som (x,y,z) (transpose-signaturen, ingen
  div/identitets-check). ALLE artiklens JHTDB-tal er dermed beregnet paa
  forvredne felter. v4 = v3 + orienterings-autokalibrering (identiteten
  <|S|2>/<|om|2>=0.5) efter HVER hentning + gate-suspension per cutout +
  LOKAL TILSTAND (koer paa lokale .h5 uden netvaerk:
      python cz_reynolds_v4.py jhtdb_iso_re433.h5 [re611] [re1300]).
  Maalekernen er uaendret fra v3 (artiklens protokol).

v5-UDVIDELSER (til artiklens resterende isotrope tabeller):
  1) sigma_total = xi0^T S(x0) xi0 ved target (centrale differenser paa u)
     -> near/far-andel |sigma_far|/|sigma_total| ved R/eta=60 (tab:nearfar).
  2) RETA_CUTOFFS udvidet med 60.0 (artiklens band [17,60]).
  3) Kapacitet (s_abs) og realiseret (|s_sig|) medianer per R/eta og percentil
     -> radius-oploest vaekst/saturation paa korrigerede data.
  PRAEREGISTRERING (nearfar): affected vaerdier var P50 0.32/0.34/0.30,
  P99.9 0.18/0.15/0.14. Near-field-dominans staar hvis korrigeret andel
  forbliver < 0.5; F-1-organisering kan HAEVE den signede tail-andel —
  retningen er aaben. Residual-vaekst: realiseret median vs R/eta
  monotont stigende => 'weak residual growth' genindsaettes; flad => saturation.

  SAMMENLIGNINGSGRUNDLAG (artiklens AFFECTED JHTDB-vaerdier, Re_lambda=433):
    Tabel 2 pooled: kapacitet 0.035, signed 0.0044, ratio 8.0, C_far 0.125.
    Abstract-baand: ratio 8-11 (nedre graense = JHTDB).
  PRAEREGISTREREDE BAAND for korrigeret C_far/ratio:
    A) C_far inden for +/-30% af 0.125 og ratio >= 5:
       JHTDB-benet kvantitativt robust -> rettede tabeller, konklusion uaendret.
    B) C_far hoejere (svagere cancellation), ratio 2-5:
       benet svaekkes, hovedkonklusionen (kapacitet >> realiseret) staar;
       abstract-baandet omskrives; proaktiv korrigeret version SKAL sendes.
    C) ratio < 2 eller C_far > 0.5:
       JHTDB degraderes fra validering til diskussion / artiklen hviler paa
       TURB-Rot alene; stoerre revision.
    TREND: hvis korrigeret C_far STIGER mod 1 ved hoej intensitet, modsiger
    det abstractets faldende trend og skal rettes uanset niveau.
  BEMAERK: baade eps_local og |omega|-percentil-targets var paavirket af
  fejlen (S2 inflateret, omega2 deflateret) — alle tal kan flytte sig.

ANACONDA PROMPT:
  cd C:\forskning
  python cz_reynolds_v3.py

KRÆVER:   internetadgang + JHTDB-token (verificeret virkende)
OUTPUT:   C:/forskning/cz_reynolds_v5_results.txt

EKSPERIMENT 4 v3 — TRE Re-PUNKTER (faktor-3 spaend)
====================================================
Tilfoejer isotropic8192 (Re_lambda~1300) til v2's 433 og 611.
Samme flowtype (forced isotropic), nu Re_lambda = 433 -> 611 -> 1300.

Datasaet (GLOBALE publicerede vaerdier):
  isotropic1024coarse: Re=433,  nu=1.85e-4,  eta=2.873e-3  (eta/dx=0.468)
  isotropic4096:        Re=611,  nu=1.732e-4, eta=1.384e-3  (eta/dx=0.902)
  isotropic8192:        Re~1300, nu=4.385e-5, eta=5.00e-4   (eta/dx=0.652)
                        snapshots 0-4 alle Re~1257-1301; bruger en af dem.

DESIGN (uaendret fra v2 — det troværdige grundlag):
  - 4 tilfaeldigt placerede 256^3 cutouts/datasaet, targets pooled
  - GLOBAL eta til fast R/eta-konvertering (ikke lokal cutout-eta)
  - FD-vorticity, udtømmende enumeration
  - C_far ved fast R/eta [25,35,50], inner R/eta=17
  - P50 og P99.9, bootstrap CI
  - eps-krydscheck + CI-overlap check

TIDSPROBE: isotropic8192 indekseres (som 4096 brugte t=1). Probe [1,2,3,4]
           — alle Re~1300 uanset 0/1-indeksering. Foerste der virker bruges.

Robust maal: ratio C_far(P99.9)/C_far(P50) over de tre Re. Hvis ~konstant
og CI'er overlapper => intensitets-faldet er Re-universelt over faktor-3.
==============================================================================
"""

import sys, subprocess, time, gc
import numpy as np
from pathlib import Path

LOCAL_FILES=[a for a in sys.argv[1:] if a.endswith(".h5")]
if not LOCAL_FILES:
    try:
        import givernylocal
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","--upgrade","givernylocal"])
        import givernylocal
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getCutout
else:
    turb_dataset=getCutout=None  # lokal tilstand: ingen netvaerk noedvendig
import h5py

# ── CONFIG ───────────────────────────────────────────────────────────────────
JHTDB_TOKEN = "YOUR_JHTDB_TOKEN_HERE"
OUT_FILE    = "C:/forskning/cz_reynolds_v5_results.txt"

DATASETS = [
    {"title":"isotropic1024coarse","Re_lambda":433,"nu":1.85e-4,"N_full":1024,
     "time":1,"eta_global":2.873e-3,"eps_global":0.0928},
    {"title":"isotropic4096",      "Re_lambda":611,"nu":1.732e-4,"N_full":4096,
     "time":1,"eta_global":1.3844e-3,"eps_global":1.41},
    {"title":"isotropic8192",      "Re_lambda":1300,"nu":4.385e-5,"N_full":8192,
     "time_candidates":[1,2,3,4],"eta_global":5.00e-4,"eps_global":1.35},
]

N_CUBE      = 256
N_CUTOUTS   = 4
RETA_INNER  = 17.0
RETA_CUTOFFS= [25.0, 35.0, 50.0, 60.0]
PERCS_TEST  = [50.0, 99.9]
N_PER_BIN_CUTOUT = 500
SEED        = 42
N_BOOTSTRAP = 2000
EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
lines_out=[]
def out(s=""):
    print(s); lines_out.append(str(s))
def hdr(s):
    out("="*70); out(s); out("="*70)
def save_now():
    Path(OUT_FILE).parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_FILE,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines_out))

# ─────────────────────────────────────────────────────────────────────────────
def get_dataset(title):
    return turb_dataset(dataset_title=title, output_path="C:/forskning",
                        auth_token=JHTDB_TOKEN)

def download_cube(ds, t, start, sz):
    x0,y0,z0=map(int,start)
    axes=np.array([[x0,x0+sz-1],[y0,y0+sz-1],[z0,z0+sz-1],[t,t]],dtype=np.int32)
    strides=np.array([1,1,1,1],dtype=np.int32)
    cutout=getCutout(ds,"velocity",axes,strides,verbose=False)
    if hasattr(cutout,"data_vars"):
        var=list(cutout.data_vars)[0];da=cutout[var]
        if set(["zcoor","ycoor","xcoor","values"]).issubset(set(da.dims)):
            arr=da.transpose("zcoor","ycoor","xcoor","values").values
        else: arr=da.values
    else: arr=np.asarray(cutout)
    arr=np.asarray(arr)
    if arr.ndim==5 and arr.shape[0]==1: arr=arr[0]
    if arr.shape[-1]!=3 and arr.shape[0]==3: arr=np.moveaxis(arr,0,-1)
    return np.moveaxis(arr.astype(np.float64),-1,0)

GATES=[]
TARGET_ROWS=[]   # deposit-grade per-target data -> cz_v5_targets.csv
def _ratio_of(vel):
    g=lambda a,ax: np.gradient(a,axis=ax)
    u,v,w=vel
    om=np.stack([g(w,1)-g(v,2), g(u,2)-g(w,0), g(v,0)-g(u,1)])
    S2=0.0
    for i in range(3):
        for j in range(3):
            Sij=0.5*(g(vel[j],i)+g(vel[i],j)); S2+=float((Sij**2).mean()); del Sij
    return S2/float((om**2).sum(0).mean())

def calibrate_orientation(vel):
    """36 (akse-perm x komp-perm) paa subkube; vaelg ratio naermest 0.5.
    Gate [0.45,0.55] paa subkube, fallback fuldfelt (jf. rigidity v5-laeren)."""
    from itertools import permutations
    N=vel.shape[1]; c0=N//2; h=min(48,c0-2)
    sub=vel[:,c0-h:c0+h,c0-h:c0+h,c0-h:c0+h].astype(np.float32)
    best=(None,None,9e9)
    for ap in permutations((0,1,2)):
        subT=np.stack([np.transpose(sub[c],ap) for c in range(3)])
        for cp in permutations((0,1,2)):
            r=_ratio_of(subT[list(cp)])
            if abs(r-0.5)<abs(best[2]-0.5): best=(ap,cp,r)
    ap,cp,r=best
    velC=np.stack([np.transpose(vel[c],ap) for c in range(3)])[list(cp)]
    ok=0.45<=r<=0.55
    if not ok:
        rf=_ratio_of(velC)
        ok=0.45<=rf<=0.55
        info=f"akse={ap} komp={cp} subkube={r:.3f} fuldfelt={rf:.3f}"
    else:
        info=f"akse={ap} komp={cp} subkube={r:.3f}"
    return velC,ok,info

def load_local(path):
    with h5py.File(path,"r") as f:
        ks=set()
        f.visit(lambda n: ks.add(n) if hasattr(f[n],"shape") else None)
        if {"u","v","w"}<=ks: names=["u","v","w"]
        elif {"PS3D/vx","PS3D/vy","PS3D/vz"}<=ks: names=["PS3D/vx","PS3D/vy","PS3D/vz"]
        else: raise ValueError(f"ukendte datasets: {sorted(ks)[:6]}")
        return np.stack([f[n][:].astype(np.float64) for n in names])

def probe_time(ds, title, candidates):
    """Find foerste gyldige tidsindeks ved at hente en lille 16^3 cutout."""
    out(f"  Tidsprobe for {title} (kandidater {candidates}) ...")
    for t in candidates:
        try:
            u=download_cube(ds,t,(100,100,100),16)
            out(f"    t={t}: OK (shape {u.shape})")
            del u; gc.collect()
            return t
        except Exception as e:
            msg=str(e)[:150]
            out(f"    t={t}: fejl ({msg})")
    return None

def vorticity_fd(u, dx):
    duz=np.gradient(u[2],dx,axis=(0,1,2))
    duy=np.gradient(u[1],dx,axis=(0,1,2))
    dux=np.gradient(u[0],dx,axis=(0,1,2))
    wx=duz[1]-duy[2]; wy=dux[2]-duz[0]; wz=duy[0]-dux[1]
    omega=np.stack([wx,wy,wz],axis=0)
    return omega, np.sqrt(wx*wx+wy*wy+wz*wz)

def precompute_offsets(r_inner,r_outer):
    rng_max=int(np.ceil(r_outer));coords=np.arange(-rng_max,rng_max+1)
    DX,DY,DZ=np.meshgrid(coords,coords,coords,indexing='ij')
    DX=DX.ravel();DY=DY.ravel();DZ=DZ.ravel();R2=DX*DX+DY*DY+DZ*DZ
    mask=(R2>=r_inner*r_inner)&(R2<=r_outer*r_outer)
    DX=DX[mask].astype(np.int64);DY=DY[mask].astype(np.int64);DZ=DZ[mask].astype(np.int64)
    R2=R2[mask].astype(np.float64);R=np.sqrt(R2);R5=np.maximum(R2*R2*R,EPS)
    return DX,DY,DZ,R,R5

def kernel_contributions(omega,ix0,iy0,iz0,xi0,DX,DY,DZ,R5,sz):
    jx=(ix0+DX)%sz;jy=(iy0+DY)%sz;jz=(iz0+DZ)%sz
    ox=omega[0][jx,jy,jz];oy=omega[1][jx,jy,jz];oz=omega[2][jx,jy,jz]
    dot=ox*xi0[0]+oy*xi0[1]+oz*xi0[2]
    opx=ox-dot*xi0[0];opy=oy-dot*xi0[1];opz=oz-dot*xi0[2]
    rx=-DX.astype(np.float64);ry=-DY.astype(np.float64);rz=-DZ.astype(np.float64)
    xdr=xi0[0]*rx+xi0[1]*ry+xi0[2]*rz
    cx=xi0[1]*rz-xi0[2]*ry;cy=xi0[2]*rx-xi0[0]*rz;cz=xi0[0]*ry-xi0[1]*rx
    fac=-3.0*xdr/(4.0*np.pi*R5)
    return fac*cx*opx+fac*cy*opy+fac*cz*opz

def bootstrap_ci(vals,n_boot,seed):
    if len(vals)<5: return None,None,None
    rng=np.random.default_rng(seed);arr=np.array(vals);meds=[]
    for _ in range(n_boot):
        meds.append(np.median(rng.choice(arr,size=len(arr),replace=True)))
    return float(np.median(arr)),float(np.percentile(meds,2.5)),float(np.percentile(meds,97.5))

# ─────────────────────────────────────────────────────────────────────────────
def analyse_dataset(spec):
    title=spec["title"];Re=spec["Re_lambda"];nu=spec["nu"];Nfull=spec["N_full"]
    eta_g=spec["eta_global"];eps_g=spec["eps_global"]
    hdr(f"DATASAET: {title}  (Re_lambda={Re})")
    dx=2.0*np.pi/Nfull
    eta_over_dx=eta_g/dx
    out(f"  GLOBAL eta={eta_g:.4e}  dx={dx:.4e}  eta/dx={eta_over_dx:.4f}")

    locals_=[f for f in LOCAL_FILES if str(Re) in f]
    if LOCAL_FILES and not locals_:
        out("  LOKAL TILSTAND: ingen matchende fil for dette datasaet — springer over.")
        return None
    ds=None if locals_ else get_dataset(title)

    # tidsindeks
    if locals_:
        t_use=None
    elif "time" in spec:
        t_use=spec["time"]
    else:
        t_use=probe_time(ds,title,spec["time_candidates"])
        if t_use is None:
            out(f"  INGEN gyldig tid fundet for {title}. Springer over.")
            return None
        out(f"  Bruger t={t_use}")

    r_in_grid=RETA_INNER*eta_over_dx
    r_out_grid=max(RETA_CUTOFFS)*eta_over_dx
    margin=int(np.ceil(r_out_grid))+4
    out(f"  R/eta [{RETA_INNER},{max(RETA_CUTOFFS)}] => grid [{r_in_grid:.1f},{r_out_grid:.1f}], margin={margin}")
    DX,DY,DZ,R,R5=precompute_offsets(r_in_grid,r_out_grid)
    cut_masks={X:(R<=X*eta_over_dx) for X in RETA_CUTOFFS}
    out(f"  shell-punkter: {len(DX):,}")

    rng=np.random.default_rng(SEED+Re)
    max_origin=Nfull-N_CUBE
    origins=[tuple(int(v) for v in rng.integers(0,max_origin+1,size=3)) for _ in range(N_CUTOUTS)]
    out(f"  {N_CUTOUTS} cutout-origins: {origins}")
    if locals_:
        origins=locals_
        out(f"  LOKAL TILSTAND: {len(origins)} fil(er) i stedet for API-cutouts: {origins}")

    pooled={p:{X:[] for X in RETA_CUTOFFS} for p in PERCS_TEST}
    pooled_abs={p:{X:[] for X in RETA_CUTOFFS} for p in PERCS_TEST}
    pooled_sig={p:{X:[] for X in RETA_CUTOFFS} for p in PERCS_TEST}
    pooled_nf={p:[] for p in PERCS_TEST}
    eps_list=[]

    for ci,origin in enumerate(origins):
        out(f"\n  --- Cutout {ci+1}/{N_CUTOUTS} @ {origin} ---")
        t0=time.time()
        try:
            u=load_local(origin) if locals_ else download_cube(ds,t_use,origin,N_CUBE)
        except Exception as e:
            out(f"    hent/laes FEJL: {str(e)[:200]}"); continue
        u,gate_ok,ginfo=calibrate_orientation(u)
        out(f"    ORIENTERING: {ginfo}  GATE {'OK' if gate_ok else 'FEJLER'}")
        GATES.append((title,str(origin),gate_ok))
        if not gate_ok:
            out("    GATE FEJLER -> cutout SUSPENDERET."); del u; gc.collect(); continue
        omega,omega_mag=vorticity_fd(u,dx)
        eps_local=nu*np.mean(omega_mag**2)
        eps_list.append(eps_local)
        out(f"    eps_local={eps_local:.4f} (global {eps_g})  "
            f"|omega| med={np.median(omega_mag):.3f}  ({time.time()-t0:.0f}s)")

        sz=omega.shape[1]
        valid=np.zeros((sz,sz,sz),dtype=bool)
        valid[margin:sz-margin,margin:sz-margin,margin:sz-margin]=True
        vflat=omega_mag[valid].ravel()
        crng=np.random.default_rng(SEED+Re+ci+1)
        for p in PERCS_TEST:
            th=np.percentile(vflat,p)
            higher=[q for q in PERCS_TEST if q>p]
            hi=np.percentile(vflat,min(higher)) if higher else vflat.max()*1.01
            mask=valid&(omega_mag>=th)&(omega_mag<hi)
            cands=np.argwhere(mask)
            n=min(N_PER_BIN_CUTOUT,len(cands))
            if n==0: continue
            chosen=cands[crng.choice(len(cands),n,replace=False)]
            for (ix0,iy0,iz0) in chosen:
                ix0,iy0,iz0=int(ix0),int(iy0),int(iz0)
                om0=np.array([omega[k,ix0,iy0,iz0] for k in range(3)])
                phi_t=np.linalg.norm(om0)
                if phi_t<EPS: continue
                xi0v=om0/phi_t
                # sigma_total = xi0^T S(x0) xi0, centrale differenser paa u
                G=np.empty((3,3))
                for a in range(3):
                    ip=[ix0,iy0,iz0]; im=[ix0,iy0,iz0]; ip[a]+=1; im[a]-=1
                    for b in range(3):
                        G[a,b]=(u[b][tuple(ip)]-u[b][tuple(im)])/(2.0*dx)
                Smat=0.5*(G+G.T)
                sig_tot=float(xi0v@Smat@xi0v)
                c=kernel_contributions(omega,ix0,iy0,iz0,xi0v,DX,DY,DZ,R5,sz)
                ac=np.abs(c)
                for X in RETA_CUTOFFS:
                    m=cut_masks[X]
                    s_abs=np.sum(ac[m]);s_sig=abs(np.sum(c[m]))
                    if s_abs>EPS:
                        pooled[p][X].append(s_sig/s_abs)
                        pooled_abs[p][X].append(s_abs)
                        pooled_sig[p][X].append(s_sig)
                        if X==60.0 and abs(sig_tot)>EPS:
                            pooled_nf[p].append(s_sig/abs(sig_tot))
                        TARGET_ROWS.append((title,Re,ci,str(origin),p,X,
                            ix0,iy0,iz0,float(phi_t),sig_tot,
                            float(s_abs),float(s_sig),
                            float(s_sig/s_abs) if s_abs>EPS else float('nan')))
        del omega,omega_mag,u; gc.collect()
        save_now()

    if eps_list:
        eps_mean=np.mean(eps_list)
        out(f"\n  KRYDSCHECK eps: middel over {len(eps_list)} cutouts = {eps_mean:.4f} "
            f"(global {eps_g})  ratio {eps_mean/eps_g:.2f}")
        out(f"    per cutout: {[f'{e:.4f}' for e in eps_list]}")
    return {"title":title,"Re":Re,"eta_over_dx":eta_over_dx,
            "pooled":pooled,"pooled_abs":pooled_abs,"pooled_sig":pooled_sig,
            "pooled_nf":pooled_nf,"eps_mean":np.mean(eps_list) if eps_list else None}

AFFECTED_NF={433:(0.322,0.182),611:(0.338,0.149),1300:(0.295,0.142)}
def report_v5(data):
    Re=data["Re"]
    hdr(f"v5-UDVIDELSER — {data['title']} (Re={Re})")
    out("  Kapacitet s_abs og realiseret |s_sig| (medianer, grid-enheds-integral):")
    out(f"   {'R/eta':>6} | {'abs P50':>9} {'abs P99.9':>10} | {'sig P50':>9} {'sig P99.9':>10} | {'C_far P50':>9}")
    sigP50=[]
    for X in RETA_CUTOFFS:
        a50=np.median(data['pooled_abs'][50.0][X]) if data['pooled_abs'][50.0][X] else float('nan')
        a99=np.median(data['pooled_abs'][99.9][X]) if data['pooled_abs'][99.9][X] else float('nan')
        s50=np.median(data['pooled_sig'][50.0][X]) if data['pooled_sig'][50.0][X] else float('nan')
        s99=np.median(data['pooled_sig'][99.9][X]) if data['pooled_sig'][99.9][X] else float('nan')
        c50=np.median(data['pooled'][50.0][X]) if data['pooled'][50.0][X] else float('nan')
        sigP50.append(s50)
        out(f"   {X:>6.0f} | {a50:>9.3f} {a99:>10.3f} | {s50:>9.4f} {s99:>10.4f} | {c50:>9.4f}")
    if all(np.isfinite(sigP50)) and len(sigP50)>=3:
        mono="MONOTONT STIGENDE" if all(sigP50[i]<sigP50[i+1] for i in range(len(sigP50)-1)) else "IKKE monotont (saturation/flad?)"
        out(f"  Residual-vaekst-check (realiseret P50 vs R/eta): {mono}  "
            f"(rel. aendring sidste trin: {(sigP50[-1]-sigP50[-2])/max(sigP50[-2],1e-12)*100:.1f}%)")
    out("")
    out("  NEAR/FAR-ANDEL |sigma_far|/|sigma_total| ved R/eta=60:")
    for p in PERCS_TEST:
        v=data['pooled_nf'][p]
        if v:
            med,lo,hi=bootstrap_ci(v,N_BOOTSTRAP,SEED+int(p*10))
            aff=AFFECTED_NF.get(Re,(None,None))[0 if p==50.0 else 1]
            cmp_=f"  (AFFECTED var {aff:.3f})" if aff else ""
            out(f"    P{p:g}: median={med:.3f} [{lo:.3f},{hi:.3f}]  N={len(v)}{cmp_}")
    out("  Dom: near-field-dominans staar hvis andel < 0.5; retning vs AFFECTED aaben (praereg. i header).")
    return None

def report(data):
    hdr(f"C_far ved fast R/eta — {data['title']} (Re={data['Re']})")
    out(f"  eta/dx={data['eta_over_dx']:.4f}")
    out(f"\n  {'R/eta':>6} | {'N':>5} | {'P50 C_far [95% CI]':>27} | {'P99.9 C_far [95% CI]':>27} | {'ratio':>7}")
    out(f"  {'-'*6}-+-{'-'*5}-+-{'-'*27}-+-{'-'*27}-+-{'-'*7}")
    summary={}
    for X in RETA_CUTOFFS:
        v50=data['pooled'][50.0][X];v99=data['pooled'][99.9][X]
        m50,l50,h50=bootstrap_ci(v50,N_BOOTSTRAP,SEED+int(X))
        m99,l99,h99=bootstrap_ci(v99,N_BOOTSTRAP,SEED+int(X)+1)
        if m50 and m99:
            ratio=m99/m50;summary[X]=(m50,m99,ratio,l99,h99)
            out(f"  {X:>6.0f} | {len(v99):>5} | {m50:.4f} [{l50:.4f},{h50:.4f}] | "
                f"{m99:.4f} [{l99:.4f},{h99:.4f}] | {ratio:>7.3f}")
    return summary

# ─────────────────────────────────────────────────────────────────────────────
t_total=time.time()
hdr("CZ Re-SKALERING v3 — TRE Re-PUNKTER (433, 611, 1300)")
out(f"  {N_CUTOUTS} cutouts/datasaet, {N_CUBE}^3, {N_PER_BIN_CUTOUT} targets/bin/cutout")
out(f"  R/eta inner={RETA_INNER}, cutoffs={RETA_CUTOFFS}, bootstrap {N_BOOTSTRAP}")
save_now()

summaries={}
for spec in DATASETS:
    try:
        data=analyse_dataset(spec)
        if data:
            summaries[spec["Re_lambda"]]=report(data)
            report_v5(data)
        save_now()
    except Exception as e:
        out(f"\n  FEJL i {spec['title']}: {type(e).__name__}: {str(e)[:300]}")
        save_now()

hdr("Re-SAMMENLIGNING: intensitets-fald over faktor-3 i Re")
out(f"\n  Ratio C_far(P99.9)/C_far(P50):")
Res=sorted(summaries.keys())
out(f"\n  {'R/eta':>6} | " + " | ".join(f"Re={Re:>5}" for Re in Res))
out(f"  {'-'*6}-+-" + "-+-".join("-"*8 for _ in Res))
for X in RETA_CUTOFFS:
    cells=[f"{summaries[Re][X][2]:.3f}" if X in summaries[Re] else "--" for Re in Res]
    out(f"  {X:>6.0f} | " + " | ".join(f"{c:>8}" for c in cells))

# Absolutte P99.9 C_far over Re (Re-universalitet)
out(f"\n  Absolut C_far(P99.9) ved fast R/eta over Re:")
out(f"\n  {'R/eta':>6} | " + " | ".join(f"Re={Re:>5}" for Re in Res))
for X in RETA_CUTOFFS:
    cells=[f"{summaries[Re][X][1]:.4f}" if X in summaries[Re] else "--" for Re in Res]
    out(f"  {X:>6.0f} | " + " | ".join(f"{c:>8}" for c in cells))

# Pairwise CI-overlap
out(f"\n  CI-OVERLAP (P99.9 C_far), pairwise:")
for X in RETA_CUTOFFS:
    out(f"    R/eta={X:.0f}:")
    for i in range(len(Res)):
        for j in range(i+1,len(Res)):
            Ra,Rb=Res[i],Res[j]
            if X in summaries[Ra] and X in summaries[Rb]:
                _,_,_,la,ha=summaries[Ra][X]; _,_,_,lb,hb=summaries[Rb][X]
                ov=max(la,lb)<=min(ha,hb)
                out(f"      Re{Ra} [{la:.4f},{ha:.4f}] vs Re{Rb} [{lb:.4f},{hb:.4f}] => "
                    f"{'OVERLAP' if ov else 'ADSKILT'}")

out(f"""
  FORTOLKNING (faktor-3 i Re_lambda: 433 -> 1300):
    - ratio < 1 ved alle Re => intensitets-faldet er reelt.
    - ratio ~konstant + CI-overlap over alle tre Re => Re-UNIVERSELT.
      Loefter til 'intensitets-betinget tendens robust over faktor-3 i Re'.
    - systematisk trend i ratio med Re => reel Re-afhaengighed.
    eps middel ~0.5x global ved alle (intermittens-undersampling), ens
    bias => fair sammenligning. Global eta brugt => korrekt R/eta.""")

try:
    import csv as _csv
    with open("C:/forskning/cz_v5_targets.csv","w",newline="") as fh:
        w=_csv.writer(fh)
        w.writerow(["dataset","Re_lambda","cutout","origin","percentile","R_eta",
                    "ix","iy","iz","omega_mag","sigma_total","sigma_abs","sigma_signed","C_far"])
        w.writerows(TARGET_ROWS)
    out(f"\nSkrev cz_v5_targets.csv ({len(TARGET_ROWS)} raekker) — deposit-grade per-target data.")
except Exception as e:
    out(f"\nADVARSEL: kunne ikke skrive targets-csv: {e}")
hdr("GATE-STATUS (alle cutouts)")
for g in GATES: out(f"  {g[0]} @ {g[1]}: {'OK' if g[2] else 'SUSPENDERET'}")
if GATES and not all(g[2] for g in GATES):
    out("  >>> Suspenderede cutouts indgaar IKKE i pooled-tal. <<<")
hdr("ARTIKEL-SAMMENLIGNING (Re_lambda=433)")
out("  AFFECTED (artiklen, Tabel 2): ratio 8.0, C_far 0.125")
out("  Korrigerede vaerdier: aflaes P50/P99.9 C_far pr. R/eta ovenfor og doem")
out("  efter de praeregistrerede baand A/B/C i headeren. Trend-checket:")
out("  stiger C_far mod 1 ved P99.9, modsiger det abstractets faldende trend.")
hdr(f"KOMPLET — {time.time()-t_total:.0f} sek")
save_now()
print(f"\nGemt: {OUT_FILE}")
