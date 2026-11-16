"""Figure 4 panels: robustness of the FAST group difference."""

import numpy as np
from matplotlib.patches import Rectangle

import common as C

DIMENSIONS = ["Alpha measure", "Covariates", "Trials", "Children", "Model"]


def panel_a():
    """Specification curve: the ASD - TD difference in the main model and each analytic variant (95% CI), sorted by
    estimate, with the variant marked in the matrix below. Open grey: EOG indices that failed validation."""
    d = C.read("Figure4_A")
    n = len(d)
    order = d.sort_values("estimate_pct", kind="stable").index
    xpos = {i: x for x, i in enumerate(order)}
    fig, (ax, axm) = C.figure(130, 160, nrows=2, height_ratios=[30, 112], sharex=True)
    ax.axhspan(-C.SESOI["FAST"], C.SESOI["FAST"], color=C.BAND, lw=0, zorder=0)
    for i, r in d.iterrows():
        main, desc = r.role == "main", r.role == "descriptive"
        C.ci_point(ax, xpos[i], r.estimate_pct, r.ci_lo, r.ci_hi, C.MUTED if desc else C.INK,
                   marker="D" if main else "o", ms=4.4 if main else 2.6, lw=1.3 if main else 0.75, filled=not desc)
    C.zero(ax)
    ax.set_ylim(-3.6, 0.9)
    ax.set_ylabel(f"ASD − TD\n({C.UNIT})")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False)
    ax.text(n - 0.4, 0.68, "±SESOI", fontsize=7, color=C.MUTED, ha="right", va="bottom")
    n_excl = int((d.ci_hi < 0).sum())
    ax.text(-0.5, 0.72, f"All {n} CIs exclude 0" if n_excl == n else f"{n_excl} of {n} CIs exclude 0",
            fontsize=7, va="bottom")
    y, ys, heads = 0.0, {}, []
    for dim in DIMENSIONS[::-1]:
        for i in [i for i in order if d.loc[i, "dimension"] == dim][::-1]:
            ys[i] = y
            y += 1
        heads.append((y + 0.25, dim))
        y += 1.6
    for i in d.index[d.role == "main"]:
        ys[i] = y
    for i, r in d.iterrows():
        axm.plot(xpos[i], ys[i], "D" if r.role == "main" else "s", ms=2.6 if r.role == "main" else 2.2,
                 color=C.MUTED if r.role == "descriptive" else C.INK, mec="none", zorder=3)
    for x in range(n):
        axm.plot([x, x], [-0.6, y + 0.6], color=C.RULE, lw=0.3, zorder=0)
    for yh, dim in heads:
        axm.text(-1.2, yh, dim, fontsize=7, fontweight="bold", ha="right", va="center")
        axm.plot([-0.8, n - 0.2], [yh + 0.75, yh + 0.75], color=C.RULE, lw=0.5, zorder=0)
    axm.set_yticks([ys[i] for i in d.index], d.analysis, fontsize=7)
    axm.tick_params(axis="y", length=0, pad=1.5)
    for lab in axm.get_yticklabels():
        if lab.get_text() == d.analysis[d.role == "main"].iloc[0]:
            lab.set_fontweight("bold")
    axm.set_xlim(-0.8, n - 0.2)
    axm.set_ylim(-0.8, y + 0.8)
    axm.tick_params(axis="x", bottom=False, labelbottom=False)
    for s in ("bottom", "left"):
        axm.spines[s].set_visible(False)
    C.save(fig, "Figure4_A")


def panel_b():
    """Permutation of group labels: null distribution of the two-stage ASD - TD difference and the observed value."""
    h, o = C.read("Figure4_B"), C.read("Figure4_B_observed").iloc[0]
    fig, ax = C.figure(50, 34)
    ax.bar(h.bin_from, h["count"], width=h.bin_to - h.bin_from, align="edge", color=C.MUTED, alpha=0.5, lw=0)
    top = h["count"].max()
    ax.axvline(o.observed_pct, color=C.INK, lw=1.1)
    ax.text(o.observed_pct + 0.1, top * 1.02, "observed", fontsize=7, ha="left", va="bottom")
    ax.text(h.bin_to.iloc[-1], top * 1.02, C.fmt_p(max(o.p_two_sided, 1 / o.n_permutations)), fontsize=7,
            ha="right", va="bottom", color=C.INK2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(min(h.bin_from.iloc[0], o.observed_pct) - 0.35, h.bin_to.iloc[-1])
    ax.set_ylim(0, top * 1.45)
    ax.set_xlabel("ASD − TD (%)")
    C.save(fig, "Figure4_B")


def panel_c():
    """Cross-child pairing null (grey: 95% interval and mean of mispaired alpha and RT series) and observed values."""
    d = C.read("Figure4_C")
    fig, ax = C.figure(42, 34)
    for i, r in d.iterrows():
        col = {"TD": C.TD, "ASD": C.ASD}.get(r.row, C.INK)
        ax.add_patch(Rectangle((r.null_lo_pct, i - 0.25), r.null_hi_pct - r.null_lo_pct, 0.5, fc=C.BAND, ec="none"))
        ax.plot([r.null_mean_pct] * 2, [i - 0.25, i + 0.25], color=C.MUTED, lw=0.8)
        ax.plot(r.observed_pct, i, "D" if r.row == "ASD-TD" else "o", ms=3.6, color=col, mec="white", mew=0.5)
        if np.isfinite(r.p):
            pos = r.observed_pct > 0
            ax.text(2.5 if pos else -2.5, i - 0.32, C.fmt_p(r.p), fontsize=7, ha="right" if pos else "left",
                    va="bottom")
    ax.set_yticks(range(len(d)), [{"ASD-TD": "ASD − TD"}.get(r, r) for r in d.row])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(d) - 0.4, -0.6)
    C.zero(ax, "x")
    ax.set_xlim(-2.6, 2.6)
    ax.set_xlabel("Coupling (%)")
    C.save(fig, "Figure4_C")


def panel_d():
    """ASD - TD difference refitted in disjoint trial subsets (95% CI), with the P of the subset difference."""
    d = C.read("Figure4_D")
    fig, ax = C.figure(42, 42)
    ax.axhspan(-C.SESOI["FAST"], C.SESOI["FAST"], color=C.BAND, lw=0, zorder=0)
    short = {"face": "face", "object": "obj.", "first half": "1st\nhalf", "second half": "2nd\nhalf"}
    x, ticks, labels = 0.0, [], []
    for split in ("category", "session_half"):
        s = d[d.split == split]
        x0 = x
        for r in s.itertuples():
            C.ci_point(ax, x, r.estimate_pct, r.ci_lo, r.ci_hi, C.INK, marker="D", ms=3.2, lw=1.0)
            ticks.append(x)
            labels.append(short[r.subset])
            x += 1
        ax.text(x0 + 0.5, 0.75, C.fmt_p(s.p_difference_between_subsets.iloc[0]), ha="center", va="bottom",
                fontsize=7, color=C.INK2)
        x += 0.6
    C.zero(ax)
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.6, x - 0.9)
    ax.set_ylim(-3.6, 1.3)
    ax.set_ylabel("ASD − TD (%)")
    C.save(fig, "Figure4_D")


if __name__ == "__main__":
    for f in (panel_a, panel_b, panel_c, panel_d):
        f()
