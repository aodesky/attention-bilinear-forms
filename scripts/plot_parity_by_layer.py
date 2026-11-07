#!/usr/bin/env python3
"""By-layer parity for all fourteen models in the full paper.

Each cell is the fraction of a layer's query heads assigned to one type.
Use the same symmetric-profile partition as the paper's parity table.
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_set as ms
from type_partition import assign, RADIUS


def main():
    mpl.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.5, "text.usetex": False})
    data = pd.read_csv(ms.PROJECT / "scripts/survey_results/head_statistics.csv")
    data["type"] = assign(data, RADIUS)
    infos = ms.all_model_info()
    n_rows = (len(infos) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(5.5, 1.1 * n_rows), squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.975, bottom=0.105,
                        hspace=0.94, wspace=0.40)
    for ax in list(axes.T.flat)[len(infos):]:
        ax.axis("off")
    records = []
    # Fill each column from top to bottom, preserving the paper's model order.
    for ax, info in zip(axes.T.flat, infos):
        rows = data[data.model == info.tag]
        counts = pd.crosstab(rows.layer, rows.type).reindex(
            index=range(info.n_layers), columns=[1, 2, 3], fill_value=0)
        if not (counts.sum(axis=1) == info.n_heads).all():
            raise ValueError(f"Incomplete layer counts for {info.tag}")
        fractions = counts.to_numpy() / info.n_heads
        im = ax.imshow(fractions.T, origin="upper", aspect="auto",
                       interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
                       extent=(-0.5, info.n_layers - 0.5, 2.5, -0.5))
        ax.set_yticks([0, 1, 2], [r"$\mathrm{I}_+$", r"$\mathrm{II}$", r"$\mathrm{I}_-$"])
        ax.set_xticks(np.unique(np.linspace(0, info.n_layers - 1, 5).round().astype(int)))
        ax.tick_params(axis="both", labelsize=7.5, length=2, width=0.5, pad=2)
        ax.set_title(info.tag, fontsize=8, loc="left", pad=4)
        ax.set_xlabel("layer", fontsize=7, labelpad=1)
        for layer, values in enumerate(fractions):
            records.append(dict(model=info.tag, layer=layer,
                                type_I_plus=values[0], type_II=values[1],
                                type_I_minus=values[2], heads=info.n_heads))
    bar_ax = fig.add_axes([0.31, 0.045, 0.43, 0.012])
    bar = fig.colorbar(im, cax=bar_ax, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.ax.set_xticklabels(["0", "25", "50", "75", "100"])
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    bar.set_label("percentage of the layer's heads", fontsize=8, labelpad=3)
    out = ms.PROJECT / "figures/qk_geometry"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "parity_by_layer.pdf",
                metadata={"Author": None, "Creator": None,
                          "CreationDate": None, "ModDate": None})
    fig.savefig(out / "parity_by_layer.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(records).to_csv(
        ms.PROJECT / "scripts/survey_results/parity_by_layer.csv", index=False)
    print(f"Wrote figures/qk_geometry/parity_by_layer.pdf ({len(ms.TAGS)} models)")


if __name__ == "__main__":
    main()
