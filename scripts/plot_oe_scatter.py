#!/usr/bin/env python3
"""Scatter of the heads of a model in the (O, E) plane.

O and E are the sums of the odd and even moments of the normalized
spectrum of the symmetric part of each head's bilinear form:
O = sum lambda/(1 - lambda^2), E = sum lambda^2/(1 - lambda^2).
Marker coding matches the cluster figures (+ / o / - for Types
I+, II, I-).

Reads the cached spectra scripts/cluster_results/{tag}_sorted_spectra.npy;
run cluster_head_spectra.py --model {tag} first if the cache is missing.

Saves figures/qk_geometry/oe_scatter{_tag}.{pdf,png}.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

mpl.rcParams["text.usetex"] = True

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "scripts" / "cluster_results"
OUT = PROJECT / "figures" / "qk_geometry"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_head_spectra import MODELS, CLUSTER_NAMES, marker_kwargs


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="gpt2")
    tag = parser.parse_args().model
    suffix = "" if tag == "gpt2" else f"_{tag}"

    X = np.load(RESULTS / f"{tag}_sorted_spectra.npy")

    # paper clustering: k-means seed 0, renumbered by mean signed asymmetry
    _, raw = kmeans2(X, 3, minit="++", seed=0)
    imb = np.array([(r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum() for r in X])
    order = sorted(range(3), key=lambda cl: -imb[raw == cl].mean())
    labels = np.empty_like(raw)
    for new, old in enumerate(order, start=1):
        labels[raw == old] = new

    O = (X / (1 - X**2)).sum(axis=1)
    E = (X**2 / (1 - X**2)).sum(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for cl in (1, 2, 3):
        m = labels == cl
        ax.scatter(O[m], E[m], **marker_kwargs(cl, s=36),
                   label=f"{CLUSTER_NAMES[cl - 1]} ($n={m.sum()}$)",
                   zorder=2)
    omax = 1.05 * float(np.max(np.abs(O)))
    ax.set_xlim(-omax, omax)
    ax.set_xlabel("$O$")
    ax.set_ylabel("$E$")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    pdf = OUT / f"oe_scatter{suffix}.pdf"
    png = OUT / f"oe_scatter{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
