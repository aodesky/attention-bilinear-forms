#!/usr/bin/env python3
"""Weights of the attention heads of two models on Delta^3, side by side.

Left: gpt2-large.  Right: Mistral-Small-24B-Base.  The weights
(a, b, c, d) and the three spectral types are read from the
symmetric-spectra analyses of the bilinear-form database under
data/bilinear-forms; nothing is recomputed here.  Each model's analysis
is resolved through its latest.json, so the figure follows whichever
analysis is current.

The negative semidefinite vertex d = 1 is at the top, the
vertex b = 1 at the bottom nearest the viewer, the antisymmetric vertex
a = 1 to the left and the positive semidefinite vertex c = 1 to the
right.  This sends the facet {b = 0}, whose relative interior is empty,
to the back.  The two facets facing the viewer are shaded and the
hidden edge is dashed.

The family Theta = {a = b, cd = 0} is drawn in red in each panel: it is the
pair of segments joining the midpoint (1/2, 1/2, 0, 0) of the edge
{c = d = 0} to the vertices c = 1 and d = 1.

The panels carry no legend: the marker shapes and colors are those of
the other figures, and the caption names them.  The per-type head
counts are printed on each run, for the caption to quote.

Usage:
    python plot_simplex_scatter_pair.py

Saves figures/qk_geometry/abcd_simplex_pair.{pdf,png}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "figures" / "qk_geometry"
ANALYSES = PROJECT / "data" / "bilinear-forms" / "analyses" / "symmetric_spectra"

# (directory name under the analyses tree, label for the panel)
PANELS = [
    ("gpt2-large", r"\texttt{gpt2-large}"),
    ("mistralai__Mistral-Small-24B-Base-2501",
     r"\texttt{Mistral-Small-24B-Base}"),
]

# The type names used in the analyses, in the order of CLUSTER_NAMES.
TYPE_NAMES = ["Type I+", "Type II", "Type I-"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_head_spectra import CLUSTER_NAMES, FAMILY_COLOR, marker_kwargs

# Regular tetrahedron with the apex at d and the base a, b, c, viewed so
# that a is nearest and b and c fall to the left and to the right.
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


def embed(a, b, c, d):
    return (np.outer(a, VA) + np.outer(b, VB)
            + np.outer(c, VC) + np.outer(d, VD))


def analysis_dir(model_dir):
    """The current analysis directory for a model, from its latest.json."""
    root = ANALYSES / model_dir
    latest = sorted(root.glob("*/latest.json"))
    if len(latest) != 1:
        raise FileNotFoundError(
            f"expected one latest.json under {root}, found {len(latest)}")
    meta = json.loads(latest[0].read_text())
    return latest[0].parent / meta["analysis_id"]


def load_weights(model_dir):
    """(a, b, c, d) and the type index 1, 2, 3 of every head."""
    d = analysis_dir(model_dir)
    spec = pd.read_csv(d / "head_spectra.csv")
    types = pd.read_csv(d / "weight_map_type_assignments_k3.csv")
    if not (spec.head_pk.values == types.head_pk.values).all():
        raise ValueError(f"{model_dir}: head_spectra.csv and the type "
                         "assignments are not aligned by head_pk")
    w = spec[["weight_a", "weight_b", "weight_c", "weight_d"]].to_numpy()
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError(f"{model_dir}: weights do not sum to one")
    name_to_index = {n: i + 1 for i, n in enumerate(TYPE_NAMES)}
    unknown = set(types.type) - set(name_to_index)
    if unknown:
        raise ValueError(f"{model_dir}: unexpected type labels "
                         f"{sorted(unknown)}")
    return w, types.type.map(name_to_index).to_numpy()


def draw_frame(ax):
    """The tetrahedron, its vertex labels, and the family Theta.

    The vertex b = 1 is nearest the viewer and d is on top, so
    the visible facets are {c = 0} = ABD on the left and {a = 0} = BCD
    on the right, sharing the front edge BD; the hidden edge is AC."""
    for tri, shade in (([VB, VD, VA], "#f4f4f4"), ([VB, VD, VC], "#e6e6e6")):
        ax.add_patch(Polygon(tri, closed=True, facecolor=shade,
                             edgecolor="none", zorder=0))
    for x, y in [(VB, VD), (VB, VA), (VB, VC), (VD, VA), (VD, VC)]:
        ax.plot([x[0], y[0]], [x[1], y[1]], color="black",
                linewidth=1.0, zorder=1)
    # AC = {b = d = 0} runs behind the solid
    ax.plot([VA[0], VC[0]], [VA[1], VC[1]], color="black",
            linewidth=0.8, linestyle="--", dashes=(4, 3), zorder=1)
    for v in (VA, VB, VC, VD):
        ax.plot(*v, marker="o", markersize=3.4, color="black", zorder=4)

    M = embed([0.5], [0.5], [0.0], [0.0])[0]
    for V in (VC, VD):
        ax.plot([M[0], V[0]], [M[1], V[1]], color=FAMILY_COLOR,
                linewidth=1.2, zorder=3, solid_capstyle="round")
    ax.plot(*M, marker="o", markersize=3.0, color=FAMILY_COLOR, zorder=4)
    lab = 0.55 * np.asarray(VD) + 0.45 * M
    ax.text(lab[0] - 0.03, lab[1], r"$\Theta$", color=FAMILY_COLOR,
            ha="right", va="center", fontsize=10)

    ax.text(VD[0], VD[1] + 0.03, r"$d = 1$", ha="center", va="bottom",
            fontsize=10)
    ax.text(VB[0], VB[1] - 0.035, r"$b = 1$", ha="center", va="top",
            fontsize=10)
    ax.text(VA[0] - 0.03, VA[1], r"$a = 1$", ha="right", va="center",
            fontsize=10)
    ax.text(VC[0] + 0.03, VC[1], r"$c = 1$", ha="left", va="center",
            fontsize=10)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    for ax, (model_dir, label) in zip(axes, PANELS):
        w, labels = load_weights(model_dir)
        draw_frame(ax)
        P = embed(w[:, 0], w[:, 1], w[:, 2], w[:, 3])
        for cl in (1, 2, 3):
            m = labels == cl
            ax.scatter(P[m, 0], P[m, 1], **marker_kwargs(cl, s=14),
                       zorder=2)
        ax.set_title(label, fontsize=11, pad=20)
        ax.set_aspect("equal")
        ax.axis("off")
        # keep each panel to its own tetrahedron, so nothing drawn in one
        # panel can spill into the other
        xs = [v[0] for v in (VA, VB, VC, VD)]
        ys = [v[1] for v in (VA, VB, VC, VD)]
        ax.set_xlim(min(xs) - 0.26, max(xs) + 0.20)
        ax.set_ylim(min(ys) - 0.14, max(ys) + 0.14)
        print(f"{model_dir}: {len(w)} heads; "
              + ", ".join(f"{CLUSTER_NAMES[c-1]} {int((labels==c).sum())}"
                          for c in (1, 2, 3)))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90,
                        wspace=0.04)
    pdf = OUT / "abcd_simplex_pair.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(OUT / "abcd_simplex_pair.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
