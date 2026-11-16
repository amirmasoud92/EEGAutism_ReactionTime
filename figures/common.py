"""Colours, type and drawing helpers shared by the figure panels.

Colour meanings are the same in every panel: TD teal, ASD ochre, unaffected siblings violet; black for pooled values
and ASD - TD differences; blue-white-red for signed scalp maps (red = slower responses after higher alpha); a plum
scale for alpha power.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PANELS = HERE / "panels"
MM = 1 / 25.4

TD, ASD, SIB = "#1D6E73", "#C7782A", "#8E5BA8"
GROUP = {"TD": TD, "ASD": ASD, "SIB": SIB}
GROUP_NAME = {"TD": "Typically developing", "ASD": "Autistic", "SIB": "Unaffected siblings"}
INK, INK2, MUTED, RULE, BAND = "#1E1E1E", "#4F4F4F", "#8E8E8E", "#D6D6D6", "#ECEAE4"
DIVERGING = LinearSegmentedColormap.from_list(
    "coupling", ["#1B3F66", "#3E6E9E", "#8FB3D4", "#F6F4F0", "#E3A28E", "#B8513C", "#7A2418"], N=256)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "power", ["#F7F4EF", "#D9CDE0", "#A68CBF", "#6C4D93", "#3B2560"], N=256)
UNIT = "% RT per SD"
TASKS = ["FAST", "IC", "AVSRT"]
SESOI = {"FAST": 0.66, "IC": 0.84, "AVSRT": 1.02}   # smallest effect of interest, % RT per SD
ROI = ["PO7", "PO8", "PO3", "PO4", "POz", "O1", "O2", "Oz"]
TASK_LONG = {"FAST": "Visual detection stream (FAST)", "IC": "Illusory-contour stream (IC)",
             "AVSRT": "Audiovisual simple RT (AVSRT)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600,
})


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def figure(width_mm: float, height_mm: float, **kw):
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), layout="constrained", **kw)


_MINUS = re.compile(r"(?<![\w.)\]])-(?=\d|\.\d)")


def save(fig, name: str) -> None:
    """Write ``panels/<name>.pdf`` and ``.png`` (true minus signs in all text)."""
    from matplotlib.text import Text

    for tx in fig.findobj(Text):
        s = tx.get_text()
        if s and "-" in s and "$" not in s:
            tx.set_text(_MINUS.sub("−", s))
    PANELS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PANELS / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"


def zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK2, lw=0.6, zorder=0)


def ci_point(ax, x, est, lo, hi, color, marker="o", ms=4.5, lw=1.2, filled=True, horizontal=False):
    if horizontal:
        ax.plot([lo, hi], [x, x], color=color, lw=lw, zorder=3, solid_capstyle="butt")
        ax.plot(est, x, marker, ms=ms, mfc=color if filled else "white", mec=color, mew=0.9, zorder=4)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, zorder=3, solid_capstyle="butt")
        ax.plot(x, est, marker, ms=ms, mfc=color if filled else "white", mec=color, mew=0.9, zorder=4)


# ----------------------------------------------------------------------------------------------- shared panels
def quintile_plot(ax, d):
    """Mean RT deviation (+- 95% CI) by within-child alpha quintile, one line per group."""
    for g, dx in (("TD", -0.06), ("ASD", 0.06)):
        r = d[d.group == g]
        x, m, se = r.quintile + dx, r.mean_pct, r.se_pct
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=GROUP[g], alpha=0.15, lw=0)
        ax.plot(x, m, "-o", color=GROUP[g], ms=3.2, mec="white", mew=0.5, lw=1.2, label=g)
    zero(ax)
    ax.set_xticks(range(1, 6), ["1\nlow", "2", "3", "4", "5\nhigh"])


def raincloud(ax, v, x, color, rng):
    """Half density, jittered points and mean +- 95% CI."""
    from scipy.stats import gaussian_kde

    v = np.asarray(v, float)
    s = v.std(ddof=1)
    grid = np.linspace(v.min() - 3 * s, v.max() + 3 * s, 300)
    k = gaussian_kde(v)(grid)
    keep = k > 0.01 * k.max()
    k = 0.3 * k / k.max()
    ax.fill_betweenx(grid[keep], x - k[keep] - 0.04, x - 0.04, color=color, alpha=0.22, lw=0)
    ax.plot(x + rng.uniform(0.05, 0.28, len(v)), v, "o", ms=1.9, color=color, alpha=0.75, mec="none")
    m, se = v.mean(), s / np.sqrt(len(v))
    ci_point(ax, x - 0.04, m, m - 1.96 * se, m + 1.96 * se, INK, ms=3.2, lw=1.1)


_TOPO: dict = {}


def topomap(ax, channels, values, vlim, cmap=DIVERGING, ring=(), dots=()):
    """Scalp map of BioSemi-64 values; rings mark ``ring`` channels, white dots mark ``dots`` channels."""
    import mne
    from mne.channels.layout import _find_topomap_coords

    if not _TOPO:
        mont = mne.channels.make_standard_montage("biosemi64")
        info = mne.create_info(mont.ch_names, 512.0, "eeg")
        info.set_montage(mont)
        _TOPO.update(info=info, names=list(mont.ch_names),
                     pos=_find_topomap_coords(info, picks=list(range(64)), sphere=None))
    names, pos = _TOPO["names"], _TOPO["pos"]
    v = pd.Series(np.asarray(values, float), index=list(channels)).reindex(names).to_numpy()
    ok = np.isfinite(v)
    im, _ = mne.viz.plot_topomap(v[ok], mne.pick_info(_TOPO["info"], np.flatnonzero(ok)), axes=ax, cmap=cmap,
                                 vlim=vlim, contours=0, sensors=False, outlines="head", extrapolate="head",
                                 image_interp="cubic", res=160, border="mean", show=False)
    for ln in ax.lines:
        ln.set_linewidth(0.6)
        ln.set_color(INK2)
    ax.plot(pos[:, 0], pos[:, 1], ".", ms=1.1, color=INK2, alpha=0.55, zorder=5)
    for c in ring:
        ax.plot(*pos[names.index(c)], "o", ms=3.4, mfc="none", mec=INK, mew=0.55, zorder=6)
    for c in dots:
        ax.plot(*pos[names.index(c)], "o", ms=2.3, mfc="white", mec=INK, mew=0.5, zorder=7)
    return im


def colorbar(fig, im, ax, label, ticks, orientation="vertical", **kw):
    from matplotlib.ticker import FormatStrFormatter

    cb = fig.colorbar(im, ax=ax, orientation=orientation, ticks=ticks, shrink=0.7, aspect=18,
                      format=FormatStrFormatter("%g"), **kw)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.8, width=0.5, labelsize=7)
    cb.set_label(label, fontsize=7.5)
    return cb


def cluster_note(clusters, task, contrast):
    """'positive posterior / P = 0.002' for each significant cluster, else 'no cluster (P = smallest)'."""
    c = clusters[(clusters["task"] == task) & (clusters["map"] == contrast)].dropna(subset=["p"])
    sig = c[c.p < 0.05]
    if len(sig):
        return "\n".join(f"{r.sign} {r.region}\n{fmt_p(r.p)}" for r in sig.itertuples())
    return f"no cluster\n({fmt_p(c.p.min())})" if len(c) else "no cluster"


def map_row(fig, axes, d, clusters, task, contrasts=("TD", "ASD", "diff"), value="value_pct", vlim=(-2.5, 2.5),
            titles=True):
    """One scalp map per contrast with ROI rings, cluster dots and the cluster note underneath."""
    im = None
    for ax, k in zip(axes, contrasts):
        r = d[(d["task"] == task) & (d["map"] == k)]
        im = topomap(ax, r.channel, r[value], vlim, ring=ROI, dots=r.channel[r.in_cluster])
        if titles:
            ax.set_title({"diff": "ASD − TD"}.get(k, k), color=GROUP.get(k, INK), loc="center")
        ax.text(0.5, -0.03, cluster_note(clusters, task, k), transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color=INK2, linespacing=1.1)
    return im


def spectrum_pair(ax_abs, ax_iaf, task, prefix, clusters, legend=True):
    """Coupling by frequency per group (absolute frequency) and the ASD - TD difference against frequency - IAF.
    Grey bands: significant ASD - TD clusters."""
    for ax, axis in ((ax_abs, "absolute"), (ax_iaf, "iaf_aligned")):
        d = read(f"{prefix}_{axis}")
        cl = clusters[(clusters["task"] == task) & (clusters["axis"] == axis) & (clusters["p"] < 0.05)]
        for r in cl.itertuples():
            ax.axvspan(r.from_hz - 0.25, r.to_hz + 0.25, color=BAND, lw=0, zorder=0)
            ax.text((r.from_hz + r.to_hz) / 2, 1.0, fmt_p(r.p), fontsize=7, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2)
        if axis == "absolute":
            for g in ("TD", "ASD"):
                ax.fill_between(d.x, d[f"lo_{g}"], d[f"hi_{g}"], color=GROUP[g], alpha=0.15, lw=0)
                ax.plot(d.x, d[f"mean_{g}"], color=GROUP[g], lw=1.2, label=g)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Coupling (%)")
            ax.set_xlim(2, 40)
            if legend:
                ax.legend(loc="upper right", ncols=2, handlelength=1.2)
        else:
            ax.fill_between(d.x, d.lo_diff, d.hi_diff, color=INK, alpha=0.13, lw=0)
            ax.plot(d.x, d["diff"], color=INK, lw=1.2)
            ax.axvline(0, color=MUTED, lw=0.5, ls=":")
            ax.set_xlabel("Frequency − IAF (Hz)")
            ax.set_ylabel("ASD − TD (%)")
            ax.set_xlim(-6, 20)
        zero(ax)
