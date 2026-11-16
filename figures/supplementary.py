"""Supplementary figure panels (Figures S1-S9)."""

import numpy as np
from matplotlib.lines import Line2D

import common as C


def figure_s1():
    """Per-child two-stage coupling slopes in each task (Welch P of the group difference)."""
    d = C.read("FigureS1")
    fig, axes = C.figure(180, 54, ncols=3, sharey=True)
    rng = np.random.default_rng(7)
    for ax, task in zip(axes, C.TASKS):
        t = d[d.task == task]
        for i, g in enumerate(("TD", "ASD")):
            C.raincloud(ax, t.slope_pct[t.group == g], i, C.GROUP[g], rng)
        C.zero(ax)
        ax.set_xticks([0, 1], ["TD", "ASD"])
        ax.set_xlim(-0.5, 1.45)
        ax.set_ylim(-20, 16)
        ax.set_title(f"{task}  ({C.fmt_p(t.welch_p.iloc[0])})")
    axes[0].set_ylabel(f"Per-child coupling ({C.UNIT})")
    C.save(fig, "FigureS1")


def figure_s2():
    """RT relative to the block mean by within-child alpha quintile in each task (mean +- 95% CI)."""
    d = C.read("FigureS2")
    fig, axes = C.figure(180, 58, ncols=3, sharey=True)
    for ax, task in zip(axes, C.TASKS):
        C.quintile_plot(ax, d[d.task == task])
        ax.set_ylim(-5.5, 6.5)
        ax.set_title(task)
    axes[0].legend(loc="upper left", handlelength=1.2)
    axes[0].set_ylabel("RT relative to block mean (%)")
    axes[1].set_xlabel("Alpha quintile (within child)")
    C.save(fig, "FigureS2")


def figure_s3_a():
    """Prestimulus alpha power over the scalp by group and ASD - TD, each task (rings: ROI)."""
    d = C.read("FigureS3_A")
    grp = d[d["map"] != "diff"].log10_power
    lo, hi = np.floor(grp.min() * 10) / 10, np.ceil(grp.max() * 10) / 10
    dl = float(np.ceil(d[d["map"] == "diff"].log10_power.abs().max() * 20) / 20)
    fig, axes = C.figure(110, 125, nrows=3, ncols=3)
    for row, task in enumerate(C.TASKS):
        for col, k in enumerate(("TD", "ASD", "diff")):
            r = d[(d.task == task) & (d["map"] == k)]
            im = C.topomap(axes[row, col], r.channel, r.log10_power, (-dl, dl) if k == "diff" else (lo, hi),
                           cmap=C.DIVERGING if k == "diff" else C.SEQUENTIAL, ring=C.ROI)
            if k == "diff":
                imd = im
            else:
                ims = im
            if row == 0:
                axes[row, col].set_title({"diff": "ASD − TD"}.get(k, k), color=C.GROUP.get(k, C.INK), loc="center")
        axes[row, 0].set_ylabel(task, fontsize=8.5, fontweight="bold")
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].set_yticks([])
    C.colorbar(fig, ims, axes[:, :2], "Alpha power, log$_{10}$ µV²/Hz", ticks=[lo, hi], orientation="horizontal")
    C.colorbar(fig, imd, axes[:, 2], "ASD − TD, log$_{10}$", ticks=[-dl, 0, dl], orientation="horizontal")
    C.save(fig, "FigureS3_A")


def figure_s3_b():
    """Prestimulus ROI power spectra by group, each task (mean +- 95% CI)."""
    d = C.read("FigureS3_B")
    fig, axes = C.figure(64, 120, nrows=3, sharex=True)
    for ax, task in zip(axes, C.TASKS):
        for g in ("TD", "ASD"):
            r = d[(d.task == task) & (d.group == g)]
            ax.fill_between(r.freq_hz, r.ci_lo, r.ci_hi, color=C.GROUP[g], alpha=0.2, lw=0)
            ax.plot(r.freq_hz, r.mean_log10_psd, color=C.GROUP[g], lw=1.0, label=g)
        ax.set_xlim(1, 30)
        ax.set_ylabel("log$_{10}$ µV²/Hz")
        ax.set_title(task)
    axes[0].legend(loc="upper right", handlelength=1.2)
    axes[-1].set_xlabel("Frequency (Hz)")
    C.save(fig, "FigureS3_B")


def figure_s4():
    """t maps behind the electrode cluster tests (one-sample t per group, Welch t for ASD - TD), each task."""
    d, cl = C.read("FigureS4"), C.read("FigureS4_clusters")
    fig, axes = C.figure(140, 140, nrows=3, ncols=3)
    for row, task in enumerate(C.TASKS):
        im = C.map_row(fig, axes[row], d, cl, task, value="t", vlim=(-5, 5), titles=(row == 0))
        axes[row, 0].set_ylabel(task, fontsize=8.5, fontweight="bold")
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].set_yticks([])
    t = d.t.to_numpy()
    ext = "both" if (t.min() < -5 and t.max() > 5) else "min" if t.min() < -5 else "max" if t.max() > 5 else "neither"
    C.colorbar(fig, im, axes, "t", ticks=[-5, 0, 5], extend=ext)
    C.save(fig, "FigureS4")


def figure_s5():
    """Coupling in TD children, unaffected siblings and autistic children (three-group models, 95% CI)."""
    d = C.read("FigureS5")
    fig, ax = C.figure(120, 66)
    for i, task in enumerate(C.TASKS):
        for g, dx, mk in (("TD", -0.22, "o"), ("SIB", 0.0, "s"), ("ASD", 0.22, "o")):
            r = d[(d.task == task) & (d.group == g)].iloc[0]
            C.ci_point(ax, i + dx, r.estimate_pct, r.ci_lo, r.ci_hi, C.GROUP[g], marker=mk, ms=4)
    C.zero(ax)
    n = d.set_index(["task", "group"])["n"]
    ax.set_xticks(range(3), [f"{t}\n{n[t, 'TD']} / {n[t, 'SIB']} / {n[t, 'ASD']}" for t in C.TASKS])
    ax.set_ylabel(f"Coupling ({C.UNIT})")
    ax.set_xlabel("n: TD / siblings / ASD")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color=C.GROUP[g], ms=4, label=C.GROUP_NAME[g])
                       for g, m in (("TD", "o"), ("SIB", "s"), ("ASD", "o"))], loc="upper right")
    C.save(fig, "FigureS5")


def figure_s6():
    """Coupling predicted at ages 9, 11 and 13 years by group (95% CI); P of the group difference in age slope."""
    d = C.read("FigureS6")
    fig, axes = C.figure(180, 54, ncols=3, sharey=True)
    for ax, task in zip(axes, C.TASKS):
        t = d[d.task == task]
        for g in ("TD", "ASD"):
            r = t[t.group == g]
            ax.fill_between(r.age, r.ci_lo, r.ci_hi, color=C.GROUP[g], alpha=0.14, lw=0)
            ax.plot(r.age, r.estimate_pct, "-", color=C.GROUP[g], lw=1.1, label=g)
            ax.plot(r.age, r.estimate_pct, "o", ms=3, mfc="white", mec=C.GROUP[g], mew=0.8)
        ax.set_title(f"{task}  ({C.fmt_p(t.p_age_slope_difference.iloc[0])})")
        C.zero(ax)
        ax.set_xticks([9, 11, 13])
        ax.set_ylim(-3.2, 4.4)
        ax.set_xlabel("Age (years)")
    axes[0].set_ylabel(f"Coupling ({C.UNIT})")
    axes[0].legend(loc="upper left")
    C.save(fig, "FigureS6")


def figure_s7_a():
    """FAST design: probability that the next image is a target after n non-targets since the last target."""
    d = C.read("FigureS7_A")
    fig, ax = C.figure(46, 46)
    ax.plot(d.n_nontargets, d.p_target, "-o", color=C.INK, ms=2.6, lw=0.9)
    ax.set_xlabel("Non-targets since last target")
    ax.set_ylabel("P(next is a target)")
    ax.set_ylim(0, 0.5)
    C.save(fig, "FigureS7_A")


def figure_s7_bcd():
    """Effect of the number of non-targets since the last target on RT (B), alpha (C) and coupling (D), by group."""
    labels = {"B": "% RT per non-target", "C": "Alpha (SD) per non-target", "D": "Coupling change (%)"}
    for letter, lab in labels.items():
        d = C.read(f"FigureS7_{letter}")
        fig, ax = C.figure(36, 46)
        for i, g in enumerate(("TD", "ASD")):
            r = d[d.group == g].iloc[0]
            C.ci_point(ax, i, r.estimate, r.ci_lo, r.ci_hi, C.GROUP[g], ms=3.6)
        C.zero(ax)
        ax.set_xticks([0, 1], ["TD", "ASD"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(lab)
        ax.set_title(f"ASD − TD {C.fmt_p(d.p_difference.iloc[0])}", fontweight="normal", fontsize=7)
        C.save(fig, f"FigureS7_{letter}")


def figure_s8():
    """Split-half reliability of per-child coupling (Spearman-Brown, 95% bootstrap CI)."""
    d = C.read("FigureS8")
    fig, ax = C.figure(110, 62)
    for i, task in enumerate(C.TASKS):
        a = d[(d.task == task) & (d.split == "first vs second half")].iloc[0]
        b = d[(d.task == task) & (d.split == "odd vs even")].iloc[0]
        C.ci_point(ax, i - 0.13, a.reliability, a.ci_lo, a.ci_hi, C.INK, ms=4)
        C.ci_point(ax, i + 0.13, b.reliability, max(b.ci_lo, -1.0), b.ci_hi, C.MUTED, ms=4, filled=False)
    C.zero(ax)
    ax.axhline(0.5, color=C.INK2, lw=0.6, ls=":")
    ax.set_xticks(range(3), C.TASKS)
    ax.set_ylim(-0.7, 1.3)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_ylabel("Split-half reliability")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C.INK, ms=4, label="first vs second half"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=C.MUTED, ms=4, label="odd vs even"),
                       Line2D([], [], color=C.INK2, lw=0.6, ls=":", label="reliability 0.5")],
              loc="upper left", ncols=2, columnspacing=1.0)
    C.save(fig, "FigureS8")


def figure_s9():
    """Frequency-resolved coupling in each task (A FAST, B IC, C AVSRT); grey: significant ASD - TD cluster."""
    cl = C.read("FigureS9_clusters")
    for letter, task in zip("ABC", C.TASKS):
        fig, (a1, a2) = C.figure(170, 42, ncols=2, width_ratios=[72, 64])
        C.spectrum_pair(a1, a2, task, f"FigureS9_{task}", cl, legend=(task == "FAST"))
        C.save(fig, f"FigureS9_{letter}")


if __name__ == "__main__":
    for f in (figure_s1, figure_s2, figure_s3_a, figure_s3_b, figure_s4, figure_s5, figure_s6, figure_s7_a,
              figure_s7_bcd, figure_s8, figure_s9):
        f()
