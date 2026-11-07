"""Optional plotting helper (requires matplotlib)."""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .solution import Solution


def plot_solution(inst: Instance, sol: Solution, ax=None, title: str | None = None):
    """Draw the zone network, charging stations and each vehicle's relocation.

    Zones need coordinates (``inst.coords``); ``random_instance`` provides them.
    Vehicle routes are drawn zone by zone; the label at each destination gives
    the initial -> final charge level (levels are 0-based).
    """
    import matplotlib.pyplot as plt

    if inst.coords is None:
        raise ValueError("instance has no zone coordinates to plot")
    xy = np.asarray(inst.coords)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    for i, j, _ in inst.edges:
        ax.plot(*xy[[i, j]].T, color="#d0d0d0", lw=0.8, zorder=1)
    size = 20 + 120 * inst.lam.sum(axis=1) / max(inst.lam.sum(axis=1).max(), 1e-9)
    ax.scatter(*xy.T, s=size, color="#9aa5b1", zorder=2, label="zone (size = demand)")
    st = inst.station_list
    ax.scatter(*xy[st].T, marker="s", s=140, facecolor="none", edgecolor="#d62728",
               lw=2, zorder=3, label="charging station")
    for j in st:
        ax.annotate(f"u={inst.stations[j]}", xy[j], xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color="#d62728")
    colors = plt.get_cmap("tab10").colors
    span = float(np.ptp(xy, axis=0).max()) or 1.0
    for f, r in enumerate(sol.routes):
        c = colors[f % len(colors)]
        zones = [r[0][0]] + [b[0] for a, b in zip(r[:-1], r[1:]) if b[0] != a[0]]
        # small per-vehicle offset so overlapping routes stay distinguishable
        off = 0.012 * span * (f - (len(sol.routes) - 1) / 2)
        pts = xy[zones] + off
        o, d = r[0], r[-1]
        st = sol.charging_station(f)
        if len(pts) > 1:
            ax.plot(*pts.T, color=c, lw=2, alpha=0.85, zorder=4)
            ax.annotate("", xy=pts[-1], xytext=pts[-2],
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2), zorder=5)
        ax.scatter(*xy[o[0]], marker="o", s=80, facecolor="white", edgecolor=c, lw=2, zorder=6)
        ax.scatter(*xy[d[0]], marker="*", s=200, color=c, edgecolor="k", lw=0.5, zorder=7)
        how = f" via {st}" if st is not None else ""
        ax.annotate(f"v{f}: L{o[1]}\u2192L{d[1]}{how}", xy[d[0]],
                    xytext=(7, 7 + 11 * (f % 3)), textcoords="offset points",
                    fontsize=8, color=c, zorder=8)
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="k", label="vehicle origin")
    ax.scatter([], [], marker="*", s=120, color="k", label="vehicle destination")
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    ax.set_title(title or f"{sol.method}: objective {sol.objective:.2f}")
    return ax
