"""Figure 3 panels: coupling in the three tasks."""

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import common as C


def panel_a():
    """Coupling by task and group from the group models (95% CI; grey: task SESOI); ASD - TD above each task."""
    d = C.read("Figure3_A")
    fig, ax = C.figure(78, 48)
    for i, task in enumerate(C.TASKS):
        s = C.SESOI[task]
        ax.add_patch(Rectangle((i - 0.42, -s), 0.84, 2 * s, fc=C.BAND, ec="none", zorder=0))
        for g, dx in (("TD", -0.13), ("ASD", 0.13)):
            r = d[(d.task == task) & (d.group == g)].iloc[0]
            C.ci_point(ax, i + dx, r.estimate_pct, r.ci_lo, r.ci_hi, C.GROUP[g], ms=4.6, lw=1.4)
        h = d[(d.task == task) & (d.group == "ASD-TD")].iloc[0]
        ax.text(i, 1.03, f"{h.estimate_pct:+.2f}%", ha="center", va="bottom", fontsize=7,
                fontweight="bold" if task == "FAST" else "normal", transform=ax.get_xaxis_transform())
    ax.text(1.0, 1.13, "ASD − TD", fontsize=7, color=C.INK2, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())
    C.zero(ax)
    ax.set_xticks(range(3), C.TASKS)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-2.4, 3.2)
    ax.set_ylabel(f"Coupling ({C.UNIT})")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C.GROUP[g], ms=4, label=g) for g in ("TD", "ASD")],
              loc="upper right", ncols=2, handletextpad=0.1, columnspacing=0.6)
    C.save(fig, "Figure3_A")


def panel_b():
    """Joint model of the three tasks: the ASD - TD difference per task (task SD units) and its task contrasts."""
    d = C.read("Figure3_B").set_index("term")
    fig, ax = C.figure(78, 44)
    for i, task in enumerate(C.TASKS):
        r = d.loc[f"ASD-TD {task}"]
        C.ci_point(ax, i, r.estimate, r.ci_lo, r.ci_hi, C.INK, marker="D", ms=4, lw=1.3)
    for a, b, term, y in ((0, 2, "FAST minus AVSRT", 0.082), (0, 1, "FAST minus IC", 0.05)):
        r = d.loc[term]
        ax.plot([a, a, b, b], [y - 0.007, y, y, y - 0.007], color=C.INK2, lw=0.6)
        ax.text((a + b) / 2, y + 0.004, f"{r.estimate:+.3f}, {C.fmt_p(r.p)}", ha="center", va="bottom", fontsize=7)
    r = d.loc["equal group difference in the three tasks"]
    ax.text(2.55, -0.127, f"χ²({int(r.df)}) = {r.chi2:.2f}, {C.fmt_p(r.p)}", fontsize=7, ha="right", va="bottom",
            color=C.INK2)
    C.zero(ax)
    ax.set_xticks(range(3), C.TASKS)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.13, 0.11)
    ax.set_ylabel("ASD − TD (task SD)")
    C.save(fig, "Figure3_B")


def panel_c():
    """Coupling at each electrode in IC and AVSRT (white dots: significant cluster; rings: ROI)."""
    d, cl = C.read("Figure3_C"), C.read("Figure3_C_clusters")
    fig, axes = C.figure(100, 100, nrows=2, ncols=3)
    for row, task in enumerate(("IC", "AVSRT")):
        im = C.map_row(fig, axes[row], d, cl, task, titles=(row == 0))
        axes[row, 0].set_ylabel(task, fontsize=8.5, fontweight="bold")
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].set_yticks([])
    C.colorbar(fig, im, axes, C.UNIT, ticks=[-2.5, 0, 2.5])
    C.save(fig, "Figure3_C")


if __name__ == "__main__":
    for f in (panel_a, panel_b, panel_c):
        f()
