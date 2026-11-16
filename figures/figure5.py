"""Figure 5 panels: RT distribution and eye movements (FAST and all tasks)."""

from matplotlib.lines import Line2D

import common as C


def panel_a():
    """High minus low alpha tercile at each RT percentile (FAST, mean +- 95% CI); Holm-corrected group difference."""
    d = C.read("Figure5_A")
    fig, ax = C.figure(62, 52)
    for g, dx in (("TD", -1.2), ("ASD", 1.2)):
        r = d[d.group == g]
        ax.fill_between(r.rt_percentile + dx, r.ci_lo, r.ci_hi, color=C.GROUP[g], alpha=0.14, lw=0)
        ax.plot(r.rt_percentile + dx, r.mean_pct, "-o", color=C.GROUP[g], ms=3.2, mec="white", mew=0.5, lw=1.2,
                label=g)
    q = d.drop_duplicates("rt_percentile")
    for r in q[q.p_holm_difference < 0.05].itertuples():
        ax.text(r.rt_percentile, 7.2, f"Holm {C.fmt_p(r.p_holm_difference)}", ha="right", va="bottom", fontsize=7)
    C.zero(ax)
    ax.set_ylim(-6.5, 10.5)
    ax.set_xticks(q.rt_percentile, [f"{p}th" for p in q.rt_percentile])
    ax.set_xlabel("RT percentile")
    ax.set_ylabel("High − low alpha tercile (% RT)")
    ax.legend(loc="lower left", ncols=2, handlelength=1.3, columnspacing=0.8)
    C.save(fig, "Figure5_A")


def panel_b():
    """Saccade rate per child by task and group (median and interquartile range)."""
    d = C.read("Figure5_B")
    fig, ax = C.figure(52, 52)
    for i, task in enumerate(C.TASKS):
        for g, dx in (("TD", -0.15), ("ASD", 0.15)):
            r = d[(d.task == task) & (d.group == g)].iloc[0]
            ax.plot([i + dx] * 2, [r.q25, r.q75], color=C.GROUP[g], lw=2.2, solid_capstyle="butt")
            ax.plot(i + dx, r["median"], "o", ms=3.8, color="white", mec=C.GROUP[g], mew=1.0, zorder=4)
    ax.set_xticks(range(3), C.TASKS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("Saccades per minute")
    ax.legend(handles=[Line2D([], [], color=C.GROUP[g], lw=2.2, label=g) for g in ("TD", "ASD")], loc="upper left",
              handlelength=1.0)
    C.save(fig, "Figure5_B")


def panel_c():
    """ASD - TD difference in saccade rate against the ASD - TD difference in coupling, per task (95% CI)."""
    d = C.read("Figure5_C")
    ns = C.read("Figure5_C_no_saccade").iloc[0]
    fig, ax = C.figure(50, 52)
    offs = {"FAST": (0.35, -0.35, "left"), "IC": (-0.35, 0.35, "right"), "AVSRT": (0.0, -0.55, "center")}
    for r in d.itertuples():
        ax.plot([r.saccade_diff_lo, r.saccade_diff_hi], [r.coupling_diff_pct] * 2, color=C.INK2, lw=0.8)
        ax.plot([r.saccade_diff_per_min] * 2, [r.coupling_diff_lo, r.coupling_diff_hi], color=C.INK2, lw=0.8)
        ax.plot(r.saccade_diff_per_min, r.coupling_diff_pct, "D", ms=4, color=C.INK, mec="white", mew=0.6, zorder=4)
        dx, dy, ha = offs[r.task]
        ax.text(r.saccade_diff_per_min + dx, r.coupling_diff_pct + dy, r.task, fontsize=7.5, fontweight="bold",
                ha=ha, va="center")
    C.zero(ax)
    ax.set_xlabel("Saccades/min, ASD − TD")
    ax.set_ylabel("Coupling, ASD − TD (%)")
    ax.text(0.98, 0.04, f"FAST, no saccade trials:\n{ns.coupling_diff_pct:+.2f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7)
    C.save(fig, "Figure5_C")


if __name__ == "__main__":
    for f in (panel_a, panel_b, panel_c):
        f()
