"""
make_figures.py

Generates the figures for the Calderon-Zygmund far-field looseness paper from
the FD4 analysis output (the CSVs written by cz_02_analysis.py).

Produces, in ./cz_figures/ :
  fig1_capacity_realisation_tightness.pdf  -- the main three-panel figure:
      (a) unsigned capacity vs ln R
      (b) realised |sigma_far| vs ln R          (note the much smaller scale)
      (c) tightness C_far vs R/eta
  fig2_fd_convergence.pdf                   -- supporting FD-order convergence:
      median and 99.9th-percentile relative difference of FD2 and FD4 against
      FD6, per Reynolds number, showing why the main analysis uses FD4

Each figure is vector PDF at a width suitable for a single journal column.
Run after the analysis, from the same working directory:
    conda activate <env>
    python make_figures.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

RESULTS_PATH = "./figure_data"
FIG_PATH = "./cz_figures"

RES = [433, 611, 1300]
# colour-blind-safe, print-safe palette (Wong); one colour per Reynolds number
COLOR = {433: "#0072B2", 611: "#D55E00", 1300: "#009E73"}
MARKER = {50.0: "o", 99.9: "s"}     # median vs extreme tail
RE_LABEL = {433: r"$\mathrm{Re}_\lambda=433$",
            611: r"$\mathrm{Re}_\lambda=611$",
            1300: r"$\mathrm{Re}_\lambda=1300$"}
CLASS_LABEL = {50.0: "median", 99.9: "tail"}

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "legend.fontsize": 6.6,
    "axes.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,   # embed TrueType so text stays editable/searchable
})


def read_csv(name):
    path = os.path.join(RESULTS_PATH, name)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_capacity_realisation():
    """Return data[Re][pct] = dict of arrays sorted by R_eta."""
    rows = read_csv("capacity_realisation.csv")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        re = int(r["Re"]); pct = float(r["percentile"])
        d = data[re][pct]
        d["R_eta"].append(float(r["R_eta"]))
        for k in ("capacity_median", "capacity_ci_lo", "capacity_ci_hi",
                  "realisation_median", "realisation_ci_lo", "realisation_ci_hi"):
            d[k].append(float(r[k]))
    for re in data:
        for pct in data[re]:
            d = data[re][pct]
            order = np.argsort(d["R_eta"])
            for k in list(d.keys()):
                d[k] = np.asarray(d[k])[order]
    return data


def load_tightness():
    rows = read_csv("tightness_blockbootstrap.csv")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        re = int(r["Re"]); pct = float(r["percentile"])
        d = data[re][pct]
        d["R_eta"].append(float(r["R_eta"]))
        d["Cfar"].append(float(r["Cfar"]))
        d["ci_lo"].append(float(r["ci_lo"]))
        d["ci_hi"].append(float(r["ci_hi"]))
    for re in data:
        for pct in data[re]:
            d = data[re][pct]
            order = np.argsort(d["R_eta"])
            for k in list(d.keys()):
                d[k] = np.asarray(d[k])[order]
    return data


def load_fd_validation():
    """Aggregate per-cutout FD validation to a mean per Reynolds number."""
    rows = read_csv("fd_order_validation.csv")
    agg = defaultdict(lambda: defaultdict(list))
    cols = ["median_rel_diff_fd2_fd6", "p999_rel_diff_fd2_fd6",
            "median_rel_diff_fd4_fd6", "p999_rel_diff_fd4_fd6"]
    for r in rows:
        re = int(r["Re"])
        for c in cols:
            agg[re][c].append(float(r[c]))
    out = {}
    for re in RES:
        out[re] = {c: float(np.mean(agg[re][c])) for c in cols}
    return out


def _ci_err(med, lo, hi):
    return np.vstack([med - lo, hi - med])


def make_fig1(capreal, tight):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    axa, axb, axc = axes

    # (a) unsigned capacity vs ln R, and (b) realised |sigma_far| vs ln R
    for ax, cap_key, lo_key, hi_key, title, ylab in (
        (axa, "capacity_median", "capacity_ci_lo", "capacity_ci_hi",
         r"(a) unsigned capacity", r"$\sigma^{\mathrm{abs}}_{\mathrm{far}}$"),
        (axb, "realisation_median", "realisation_ci_lo", "realisation_ci_hi",
         r"(b) realised contribution", r"$|\sigma_{\mathrm{far}}|$"),
    ):
        for re in RES:
            for pct in (50.0, 99.9):
                d = capreal[re][pct]
                x = d["R_eta"]
                y = d[cap_key]
                err = _ci_err(y, d[lo_key], d[hi_key])
                ax.errorbar(x, y, yerr=err, color=COLOR[re], marker=MARKER[pct],
                            markerfacecolor=("white" if pct == 99.9 else COLOR[re]),
                            markeredgecolor=COLOR[re], capsize=1.6, elinewidth=0.7,
                            linestyle=("-" if pct == 50.0 else "--"))
        ax.set_xscale("log")
        ax.set_xlabel(r"$R/\eta$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xticks([25, 35, 50, 60])
        ax.set_xticks([], minor=True)  # remove auto minor ticks/labels on log axis
        ax.set_xticklabels(["25", "35", "50", "60"])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # (c) tightness C_far vs R/eta
    for re in RES:
        for pct in (50.0, 99.9):
            d = tight[re][pct]
            err = _ci_err(d["Cfar"], d["ci_lo"], d["ci_hi"])
            axc.errorbar(d["R_eta"], d["Cfar"], yerr=err, color=COLOR[re],
                         marker=MARKER[pct],
                         markerfacecolor=("white" if pct == 99.9 else COLOR[re]),
                         markeredgecolor=COLOR[re], capsize=1.6, elinewidth=0.7,
                         linestyle=("-" if pct == 50.0 else "--"))
    axc.set_xlabel(r"$R/\eta$")
    axc.set_ylabel(r"$C_{\mathrm{far}} = |\sigma_{\mathrm{far}}|/\sigma^{\mathrm{abs}}_{\mathrm{far}}$")
    axc.set_title(r"(c) tightness", fontsize=8.5, loc="left")
    axc.set_xlim(22, 63)
    axc.yaxis.set_minor_locator(AutoMinorLocator())
    axc.xaxis.set_minor_locator(AutoMinorLocator())

    # combined legend: colour = Re, marker/linestyle = class
    from matplotlib.lines import Line2D
    re_handles = [Line2D([0], [0], color=COLOR[re], lw=1.4, label=RE_LABEL[re]) for re in RES]
    cls_handles = [
        Line2D([0], [0], color="0.3", marker="o", linestyle="-",
               markerfacecolor="0.3", label="median"),
        Line2D([0], [0], color="0.3", marker="s", linestyle="--",
               markerfacecolor="white", markeredgecolor="0.3", label="tail (P99.9)"),
    ]
    axa.legend(handles=re_handles, loc="upper left", fontsize=6.4)
    axc.legend(handles=cls_handles, loc="upper right", fontsize=6.4)

    fig.tight_layout(w_pad=1.1)
    out = os.path.join(FIG_PATH, "fig1_capacity_realisation_tightness.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_fig2(fd):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    x = np.arange(len(RES))
    w = 0.2

    med26 = [fd[re]["median_rel_diff_fd2_fd6"] * 100 for re in RES]
    p99926 = [fd[re]["p999_rel_diff_fd2_fd6"] * 100 for re in RES]
    med46 = [fd[re]["median_rel_diff_fd4_fd6"] * 100 for re in RES]
    p99946 = [fd[re]["p999_rel_diff_fd4_fd6"] * 100 for re in RES]

    ax.bar(x - 1.5 * w, med26, w, label="FD2 vs FD6, median", color="#999999")
    ax.bar(x - 0.5 * w, p99926, w, label="FD2 vs FD6, tail", color="#E69F00")
    ax.bar(x + 0.5 * w, med46, w, label="FD4 vs FD6, median", color="#56B4E9")
    ax.bar(x + 1.5 * w, p99946, w, label="FD4 vs FD6, tail", color="#0072B2")

    ax.set_xticks(x)
    ax.set_xticklabels([r"$433$", r"$611$", r"$1300$"])
    ax.set_xlabel(r"$\mathrm{Re}_\lambda$")
    ax.set_ylabel(r"relative difference in $|\omega|$ (\%)")
    ax.set_title(r"derivative-order convergence", fontsize=8.5, loc="left")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", fontsize=6.2)

    fig.tight_layout()
    out = os.path.join(FIG_PATH, "fig2_fd_convergence.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def load_simple(name, value_col, lo_col="ci_lo", hi_col="ci_hi", key_cols=("Re", "percentile")):
    """Generic loader: returns dict[Re][pct] -> list of (x_or_none, value, lo, hi).
    Used for slope_ratio, endpoint_decline, share, shell_cancellation."""
    rows = read_csv(name)
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        re = int(r["Re"]); pct = float(r["percentile"])
        out[re][pct].append(r)
    return out


def make_fig3_forest(decline_rows, slope_rows):
    """Forest plot: endpoint decline (left) and realisation/capacity slope ratio
    (right), each with CI, for all Re x both classes."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 2.6))

    # Build ordered rows: (Re, pct) top to bottom
    order = [(433, 50.0), (433, 99.9), (611, 50.0), (611, 99.9), (1300, 50.0), (1300, 99.9)]
    ylabels = [f"{RE_LABEL[re]}, {CLASS_LABEL[p]}" for (re, p) in order]
    ypos = np.arange(len(order))[::-1]  # top first

    # Left: endpoint decline (as percent)
    dmap = {(int(r["Re"]), float(r["percentile"])): r for r in decline_rows}
    for (re, p), y in zip(order, ypos):
        r = dmap[(re, p)]
        val = float(r["endpoint_decline"]) * 100
        lo = float(r["ci_lo"]) * 100
        hi = float(r["ci_hi"]) * 100
        axL.errorbar([val], [y], xerr=[[val - lo], [hi - val]], fmt=MARKER[p],
                     color=COLOR[re], markerfacecolor=("white" if p == 99.9 else COLOR[re]),
                     markeredgecolor=COLOR[re], capsize=2.4, elinewidth=0.9)
    axL.axvline(0.0, color="0.5", lw=0.7, linestyle=":")
    axL.set_yticks(ypos); axL.set_yticklabels(ylabels, fontsize=6.6)
    axL.set_xlabel(r"endpoint decline in $C_{\mathrm{far}}$, $R/\eta=25\!\to\!60$ (%)")
    axL.set_title(r"(a) tightness decline", fontsize=8.5, loc="left")
    axL.xaxis.set_minor_locator(AutoMinorLocator())

    # Right: realisation/capacity slope ratio (as percent)
    smap = {(int(r["Re"]), float(r["percentile"])): r for r in slope_rows}
    for (re, p), y in zip(order, ypos):
        r = smap[(re, p)]
        val = float(r["realisation_over_capacity_slope"]) * 100
        lo = float(r["ci_lo"]) * 100
        hi = float(r["ci_hi"]) * 100
        axR.errorbar([val], [y], xerr=[[val - lo], [hi - val]], fmt=MARKER[p],
                     color=COLOR[re], markerfacecolor=("white" if p == 99.9 else COLOR[re]),
                     markeredgecolor=COLOR[re], capsize=2.4, elinewidth=0.9)
    axR.set_yticks(ypos); axR.set_yticklabels([])
    axR.set_xlabel(r"realisation/capacity slope ratio (%)")
    axR.set_title(r"(b) slope separation", fontsize=8.5, loc="left")
    axR.set_xlim(0, None)
    axR.xaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout(w_pad=1.0)
    out = os.path.join(FIG_PATH, "fig3_decline_slope_forest.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def make_fig4_share(share_rows):
    """Far-field share of total stretching vs R/eta, P50 and P99.9, all Re."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7))
    by = defaultdict(lambda: defaultdict(list))
    for r in share_rows:
        by[int(r["Re"])][float(r["percentile"])].append(
            (float(r["R_eta"]), float(r["far_total_share"]), float(r["ci_lo"]), float(r["ci_hi"])))
    for re in RES:
        for p in (50.0, 99.9):
            d = sorted(by[re][p])
            x = np.array([t[0] for t in d]); y = np.array([t[1] for t in d])
            lo = np.array([t[2] for t in d]); hi = np.array([t[3] for t in d])
            ax.errorbar(x, y, yerr=_ci_err(y, lo, hi), color=COLOR[re], marker=MARKER[p],
                        markerfacecolor=("white" if p == 99.9 else COLOR[re]),
                        markeredgecolor=COLOR[re], capsize=1.6, elinewidth=0.7,
                        linestyle=("-" if p == 50.0 else "--"))
    ax.set_xlabel(r"$R/\eta$")
    ax.set_ylabel(r"far-field share $|\sigma_{\mathrm{far}}|/|\sigma_{\mathrm{total}}|$")
    ax.set_title(r"realised far-field share", fontsize=8.5, loc="left")
    ax.set_xlim(22, 63)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR[re], lw=1.4, label=RE_LABEL[re]) for re in RES]
    handles += [
        Line2D([0], [0], color="0.3", marker="o", linestyle="-", markerfacecolor="0.3", label="median"),
        Line2D([0], [0], color="0.3", marker="s", linestyle="--", markerfacecolor="white",
               markeredgecolor="0.3", label="tail (P99.9)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0)
    fig.tight_layout()
    out = os.path.join(FIG_PATH, "fig4_farfield_share.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def make_fig5_shellcancel(shell_rows):
    """Per-shell angular cancellation A(R) vs band midpoint, P50 and P99.9, all Re."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7))
    by = defaultdict(lambda: defaultdict(list))
    for r in shell_rows:
        mid = 0.5 * (float(r["band_R_eta_lo"]) + float(r["band_R_eta_hi"]))
        by[int(r["Re"])][float(r["percentile"])].append(
            (mid, float(r["shell_cancellation_A"]), float(r["ci_lo"]), float(r["ci_hi"])))
    for re in RES:
        for p in (50.0, 99.9):
            d = sorted(by[re][p])
            x = np.array([t[0] for t in d]); y = np.array([t[1] for t in d])
            lo = np.array([t[2] for t in d]); hi = np.array([t[3] for t in d])
            ax.errorbar(x, y, yerr=_ci_err(y, lo, hi), color=COLOR[re], marker=MARKER[p],
                        markerfacecolor=("white" if p == 99.9 else COLOR[re]),
                        markeredgecolor=COLOR[re], capsize=1.6, elinewidth=0.7,
                        linestyle=("-" if p == 50.0 else "--"))
    ax.set_xlabel(r"shell mid-radius $R/\eta$")
    ax.set_ylabel(r"shell cancellation $A=|\sum q|/\sum|q|$")
    ax.set_title(r"per-shell angular cancellation", fontsize=8.5, loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR[re], lw=1.4, label=RE_LABEL[re]) for re in RES]
    handles += [
        Line2D([0], [0], color="0.3", marker="o", linestyle="-", markerfacecolor="0.3", label="median"),
        Line2D([0], [0], color="0.3", marker="s", linestyle="--", markerfacecolor="white",
               markeredgecolor="0.3", label="tail (P99.9)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    out = os.path.join(FIG_PATH, "fig5_shell_cancellation.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(FIG_PATH, exist_ok=True)
    capreal = load_capacity_realisation()
    tight = load_tightness()
    fd = load_fd_validation()
    make_fig1(capreal, tight)
    make_fig2(fd)
    make_fig3_forest(read_csv("endpoint_decline.csv"), read_csv("slope_ratio.csv"))
    make_fig4_share(read_csv("nearfar_share.csv"))
    make_fig5_shellcancel(read_csv("shell_cancellation.csv"))
    print("done")


if __name__ == "__main__":
    main()
