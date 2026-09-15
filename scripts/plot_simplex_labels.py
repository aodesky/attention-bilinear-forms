#!/usr/bin/env python3
"""The simplex Delta^3 of weights, with its faces labeled by the fibers
of the profile map.

A reference picture for the introduction: no data and no accumulation
family, only the tetrahedron, its four vertices, and its edges, each labeled
by the fiber of π over it, as an equation in the decomposition
L = S + T (the theorem on the faces of the simplex).  The facets carry
no labels; the caption in attention.tex describes them.
Here H is a symmetric matrix whose spectrum is symmetric about zero and
P a positive semidefinite matrix, so that L = P reads "L is symmetric
positive semidefinite"; the caption in attention.tex says so.  A label
written in L rather than S marks a face on which T = 0.

The viewpoint is the one used by plot_simplex_scatter_pair.py, so this
picture and the scatter of the heads are seen from the same direction:
the negative semidefinite vertex d = 1 at the top, the
vertex b = 1 to the left, the positive semidefinite vertex c = 1 to the
right, and the antisymmetric vertex a = 1 at the bottom, nearest the
viewer.  The two faces meeting the viewer are shaded and the edge
running behind them is dashed.  That dashed edge carries no label: its
fiber is empty away from the endpoints, which the caption states.

Writing L = S + T with S symmetric and T antisymmetric: on the facets
{c = 0} and {d = 0} the antisymmetric part is unconstrained, since
a = ||T||^2 is free there.  The edges {a = c = 0} and {a = d = 0} are
the loci inside those facets where a = 0, and their labels are written
in L.

Usage:
    python plot_simplex_labels.py

Saves figures/qk_geometry/simplex_labels.{pdf,png}.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"

# Regular tetrahedron with the apex at d and the base a, b, c, viewed so
# that a is nearest and b and c fall to the left and to the right.  This
# is the projection of plot_simplex_scatter_pair.py, so this figure and
# the scatter of the heads are seen from the same direction.
BASE_AZIMUTH = 270.0              # screen azimuth of the vertex a
ELEVATION = np.deg2rad(20.0)      # viewer's elevation above the base plane
AZIMUTH = np.deg2rad(12.0)        # viewer's offset, to break the symmetry


def _vertices():
    """The four vertices a, b, c, d in space.

    The base carries b, a, c in that cyclic order, which is the
    reflection through the plane a = b of the arrangement with a, b, c.
    It puts the vertex b = 1 nearest the viewer and sends the
    facet {b = 0}, whose relative interior is empty, to the back."""
    ang = np.deg2rad(BASE_AZIMUTH - np.array([0.0, 120.0, 240.0]))
    base = np.stack([np.cos(ang), np.sin(ang), np.zeros(3)], axis=1)
    apex = np.array([0.0, 0.0, np.sqrt(2.0)])
    b, a, c = base                          # base slots: b nearest
    return np.vstack([a, b, c, apex])       # rows: a, b, c, d


def project(P):
    """Orthographic projection onto the screen of a viewer at azimuth
    AZIMUTH and elevation ELEVATION."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    ca, sa = np.cos(AZIMUTH), np.sin(AZIMUTH)
    xr = P[:, 0] * ca + P[:, 1] * sa
    yr = -P[:, 0] * sa + P[:, 1] * ca
    return np.stack([xr, yr * np.sin(ELEVATION)
                     + P[:, 2] * np.cos(ELEVATION)], axis=1)


VA, VB, VC, VD = project(_vertices())


def edge_angle(p, q):
    """Screen angle of the edge pq, folded into (-90, 90] so that text
    set at this angle always reads left to right."""
    u = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    ang = np.degrees(np.arctan2(u[1], u[0]))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return ang


def edge_label(ax, p, q, text, side, at=0.5, dist=0.05, fontsize=10):
    """Label the edge pq with the text set along the edge.

    The label is placed at the fraction `at` of the way from p to q and
    displaced by `dist` perpendicular to it, outward from the
    tetrahedron when side=+1."""
    p, q = np.asarray(p), np.asarray(q)
    u = q - p
    length = np.linalg.norm(u)
    u = u / length
    ang = edge_angle(p, q)
    nrm = side * np.array([-u[1], u[0]])
    pos = p + at * length * u + dist * nrm
    ax.text(pos[0], pos[1], text, ha="center", va="center",
            rotation=ang, rotation_mode="anchor", fontsize=fontsize)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))

    # the two facets facing the viewer, share the front edge BD:
    # {c = 0} = ABD on the left and {a = 0} = BCD on the right
    for tri, shade in (([VB, VD, VA], "#f4f4f4"), ([VB, VD, VC], "#e6e6e6")):
        ax.add_patch(Polygon(tri, closed=True, facecolor=shade,
                             edgecolor="none", zorder=0))
    for x, y in [(VB, VD), (VB, VA), (VB, VC), (VD, VA), (VD, VC)]:
        ax.plot([x[0], y[0]], [x[1], y[1]], color="black",
                linewidth=1.1, zorder=1)
    # AC = {b = d = 0} is the only edge hidden behind the solid
    ax.plot([VA[0], VC[0]], [VA[1], VC[1]], color="black",
            linewidth=0.9, linestyle="--", dashes=(4, 3), zorder=1)
    for v in (VA, VB, VC, VD):
        ax.plot(*v, marker="o", markersize=4, color="black", zorder=3)

    # vertices, each labeled by the fiber of π over it.  At a vertex
    # T = 0 except at a = 1, so the label is an equation for L itself.
    ax.text(VD[0], VD[1] + 0.035,
            r"$d = 1$: negative semidefinite, $L = -P$",
            ha="center", va="bottom", fontsize=11)
    ax.text(VB[0], VB[1] - 0.045,
            r"$b = 1$: $L=H$",
            ha="center", va="top", fontsize=11)
    ax.text(VA[0] - 0.035, VA[1],
            r"$a = 1$: antisymmetric, $L = T$",
            ha="right", va="center", fontsize=11)
    ax.text(VC[0] + 0.035, VC[1],
            r"$c = 1$: positive semidefinite, $L = P$",
            ha="left", va="center", fontsize=11)

    # edges.  The three edges through the b = 1 vertex B lie in
    # {a = 0} except BA itself, so their labels are equations for L.
    # BD is {a = c = 0}, BC is {a = d = 0}, AB is {c = d = 0}.
    edge_label(ax, VB, VD, r"$L = H - P$", side=-1, at=0.5, dist=0.035)
    edge_label(ax, VA, VB, r"$S = H$", side=-1, at=0.5)
    edge_label(ax, VB, VC, r"$L = H + P$", side=+1, at=0.5)
    # AD is {b = c = 0} and AC is {b = d = 0}, both through a = 1.
    edge_label(ax, VA, VD, r"$S = -P$", side=-1, at=0.5)
    edge_label(ax, VA, VC, r"$S = P$", side=+1, at=0.5, dist=0.03)

    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    pdf = OUT / "simplex_labels.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "simplex_labels.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
