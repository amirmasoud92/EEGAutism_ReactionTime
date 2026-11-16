"""Figure 2 panels: alpha-RT coupling in the visual detection stream (FAST)."""

import numpy as np
from matplotlib.ticker import MultipleLocator

import common as C


def panel_a():
    """RT relative to the block mean by within-child alpha quintile (mean +- 95% CI)."""
    fig, ax = C.figure(55, 52)
    C.quintile_plot(ax, C.read("Figure2_A"))
    ax.set_xlabel("Within-child quintile of alpha")
    ax.set_ylabel("RT relative to block mean (%)")
    ax.legend(loc="upper left", handlelength=1.4)
    C.save(fig, "Figure2_A")


def panel_b():
    """Group-specific coupling and the ASD - TD difference from the group model (95% CI; grey band: SESOI)."""
    d = C.read("Figure2_B").set_index("group")
    fig, (ax, axd) = C.figure(48, 52, ncols=2, sharey=True, width_ratios=[26, 14])
    s = C.SESOI["FAST"]
    for a in (ax, axd):
        a.axhspan(-s, s, color=C.BAND, lw=0, zorder=0)
        C.zero(a)
    ax.set_ylim(-2.9, 2.6)
    for i, g in enumerate(("TD", "ASD")):
        r = d.loc[g]
        C.ci_point(ax, i, r.estimate_pct, r.ci_lo, r.ci_hi, C.GROUP[g], ms=5.5, lw=1.6)
        ax.text(i + (0.14 if i == 0 else -0.14), r.estimate_pct, f"{r.estimate_pct:+.2f}%", fontsize=7.5,
                va="center", ha="left" if i == 0 else "right", color=C.GROUP[g])
    ax.set_xticks([0, 1], ["TD", "ASD"])
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylabel(f"Coupling ({C.UNIT})")
    ax.text(1.8, s, "SESOI", fontsize=7, color=C.MUTED, ha="right", va="bottom")
    r = d.loc["ASD-TD"]
    C.ci_point(axd, 0, r.estimate_pct, r.ci_lo, r.ci_hi, C.INK, marker="D", ms=4.8, lw=1.6)
    axd.text(0, 2.55, f"{r.estimate_pct:+.2f}%\n{C.fmt_p(r.p)}", fontsize=7, ha="center", va="top")
    axd.set_xlim(-0.6, 0.6)
    axd.set_xticks([0], ["ASD − TD"])
    axd.tick_params(axis="y", left=False)
    axd.spines["left"].set_visible(False)
    C.save(fig, "Figure2_B")


def panel_c():
    """Per-child two-stage coupling slopes in the alpha ROI."""
    d = C.read("Figure2_C")
    fig, ax = C.figure(52, 52)
    rng = np.random.default_rng(7)
    for i, g in enumerate(("TD", "ASD")):
        C.raincloud(ax, d.slope_pct[d.group == g], i, C.GROUP[g], rng)
    C.zero(ax)
    ax.set_xticks([0, 1], [f"{g} (n = {(d.group == g).sum()})" for g in ("TD", "ASD")])
    ax.set_xlim(-0.5, 1.45)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel(f"Per-child coupling ({C.UNIT})")
    C.save(fig, "Figure2_C")


def panel_d():
    """Coupling at each electrode: TD, ASD and ASD - TD (white dots: significant cluster; rings: ROI)."""
    fig, axes = C.figure(150, 52, ncols=3)
    im = C.map_row(fig, axes, C.read("Figure2_D"), C.read("Figure2_D_clusters"), "FAST")
    axes[0].text(0.02, 0.9, "L", transform=axes[0].transAxes, fontsize=7.5, color=C.INK2, ha="center")
    axes[0].text(0.98, 0.9, "R", transform=axes[0].transAxes, fontsize=7.5, color=C.INK2, ha="center")
    C.colorbar(fig, im, axes, C.UNIT, ticks=[-2.5, 0, 2.5])
    C.save(fig, "Figure2_D")


def panel_e():
    """Frequency-resolved coupling: groups by frequency and the ASD - TD difference aligned to the IAF."""
    fig, (a1, a2) = C.figure(170, 42, ncols=2, width_ratios=[72, 64])
    C.spectrum_pair(a1, a2, "FAST", "Figure2_E", C.read("Figure2_E_clusters"))
    C.save(fig, "Figure2_E")


if __name__ == "__main__":
    for f in (panel_a, panel_b, panel_c, panel_d, panel_e):
        f()
