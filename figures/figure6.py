"""Figure 6 panels: evoked responses to FAST targets and the paths from prestimulus alpha to RT."""

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as C

WINDOWS = {"P1": (80, 130), "N1": (150, 220)}
SITES = {"P1": ["O1", "Oz", "O2"], "N1": ["P7", "P8", "PO7", "PO8"]}


def panel_a():
    """Evoked responses at the P1 and N1 sites (site average per child, then mean +- SEM over children)."""
    d = C.read("Figure6_A")
    fig, axes = C.figure(68, 62, nrows=2, sharex=True)
    for ax, comp in zip(axes, ("P1", "N1")):
        ax.axvspan(*WINDOWS[comp], color=C.BAND, lw=0)
        for g in ("TD", "ASD"):
            r = d[(d.component == comp) & (d.group == g)]
            ax.fill_between(r.t_ms, r.mean_uv - r.sem_uv, r.mean_uv + r.sem_uv, color=C.GROUP[g], alpha=0.22, lw=0)
            ax.plot(r.t_ms, r.mean_uv, color=C.GROUP[g], lw=1.0, label=g)
        C.zero(ax)
        ax.set_xlim(-100, 500)
        ax.set_ylabel("µV")
        ax.text(0.01, 0.98, " · ".join(SITES[comp]), transform=ax.transAxes, fontsize=7, va="top")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, handlelength=1.2, columnspacing=1.0)
    axes[1].set_xlabel("Time from target onset (ms)")
    C.save(fig, "Figure6_A")


def panel_b():
    """Scalp distribution in the P1 and N1 windows by group and ASD - TD (rings: analysed sites)."""
    d = C.read("Figure6_B")
    v = {(r[0], r[1]): g.mean_uv.to_numpy() for r, g in d.groupby(["window", "map"], sort=False)}
    grp = [v[(w, k)] for w in WINDOWS for k in ("TD", "ASD")]
    dif = [v[(w, "diff")] for w in WINDOWS]
    vmax = float(5 * np.ceil(max(np.nanpercentile(np.abs(x), 98) for x in grp) / 5))
    dmax = float(np.ceil(max(np.nanpercentile(np.abs(x), 98) for x in dif)))
    fig, axes = C.figure(100, 66, nrows=2, ncols=3)
    for row, w in enumerate(WINDOWS):
        for col, k in enumerate(("TD", "ASD", "diff")):
            r = d[(d.window == w) & (d["map"] == k)]
            lim = dmax if k == "diff" else vmax
            im = C.topomap(axes[row, col], r.channel, r.mean_uv, (-lim, lim), ring=SITES[w])
            if k == "diff":
                imd = im
            else:
                img = im
            if row == 0:
                axes[row, col].set_title({"diff": "ASD − TD"}.get(k, k), color=C.GROUP.get(k, C.INK), loc="center")
        axes[row, 0].set_ylabel(f"{WINDOWS[w][0]}–{WINDOWS[w][1]} ms", fontsize=7.5, fontweight="bold")
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].set_yticks([])
    allg, alld = np.concatenate(grp), np.concatenate(dif)
    ext = lambda x, m: "both" if (x.min() < -m and x.max() > m) else "min" if x.min() < -m else "max" if x.max() > m else "neither"  # noqa: E731
    C.colorbar(fig, img, axes[:, 1], "µV", ticks=[-vmax, 0, vmax], extend=ext(allg, vmax))
    C.colorbar(fig, imd, axes[:, 2], "ASD − TD (µV)", ticks=[-dmax, 0, dmax], extend=ext(alld, dmax))
    C.save(fig, "Figure6_B")


def panel_c():
    """P1-site evoked response by within-child prestimulus alpha tercile."""
    d = C.read("Figure6_C")
    fig, axes = C.figure(62, 38, ncols=2, sharey=True)
    for ax, g in zip(axes, ("TD", "ASD")):
        ax.axvspan(*WINDOWS["P1"], color=C.BAND, lw=0)
        for t, col in zip(("low", "mid", "high"), (C.MUTED, "#5A5A5A", C.GROUP[g])):
            r = d[(d.group == g) & (d.alpha_tercile == t)]
            ax.plot(r.t_ms, r.mean_uv, color=col, lw=0.9, label=t)
        C.zero(ax)
        ax.set_xlim(-50, 250)
        ax.set_xticks([0, 100, 200])
        ax.set_title(g, color=C.GROUP[g], loc="center")
        ax.set_xlabel("ms")
    axes[0].set_ylabel("µV")
    axes[0].legend(title="alpha", loc="upper left", fontsize=6.5, title_fontsize=6.5, handlelength=1.0)
    C.save(fig, "Figure6_C")


def panel_d():
    """Paths from alpha to RT through P1 and N1 (baseline -50 to 0 ms) and the total path, by group.
    Solid: P < 0.05 (uncorrected)."""
    d = C.read("Figure6_D")
    nodes = {"alpha": (1.0, 3.0), "P1": (5.0, 5.2), "N1": (5.0, 0.8), "RT": (9.0, 3.0)}
    fig, axes = C.figure(96, 34, ncols=2)
    for ax, g in zip(axes, ("TD", "ASD")):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.4)
        ax.axis("off")
        for k, (x, y) in nodes.items():
            ax.add_patch(FancyBboxPatch((x - 0.95, y - 0.45), 1.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.3",
                                        fc="white", ec=C.GROUP[g] if k in ("alpha", "RT") else C.INK2, lw=0.8))
            ax.text(x, y, k, ha="center", va="center", fontsize=7.5, fontweight="bold")
        r = d[d.group == g].set_index("path")

        def edge(a, b, path, xy, nd):
            (x0, y0), (x1, y1) = nodes[a], nodes[b]
            est, p = r.loc[path, "estimate"], r.loc[path, "p"]
            col = C.INK if p < 0.05 else C.MUTED
            start = (x0 + np.sign(x1 - x0), y0 + (0.5 * np.sign(y1 - y0) if a == "alpha" else 0))
            end = (x1 - np.sign(x1 - x0), y1 + (0.5 * np.sign(y0 - y1) if b == "RT" else 0))
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, lw=1.0, color=col,
                                         ls="-" if p < 0.05 else (0, (2, 1.5))))
            ax.text(*xy, f"{est:+.{nd}f}", fontsize=7, ha="center", va="center", color=col)

        for comp, yt in (("P1", 4.75), ("N1", 1.25)):
            edge("alpha", comp, f"alpha -> {comp}", (2.35, yt), 3)
            edge(comp, "RT", f"{comp} -> RT", (7.65, yt), 2)
        edge("alpha", "RT", "alpha -> RT (total)", (5.0, 3.38), 2)
        ax.text(5.0, 2.62, "total", fontsize=7, ha="center", va="center", color=C.INK2)
        ax.text(0.05, 6.35, g, fontsize=7.5, fontweight="bold", color=C.GROUP[g], va="top")
    C.save(fig, "Figure6_D")


if __name__ == "__main__":
    for f in (panel_a, panel_b, panel_c, panel_d):
        f()
