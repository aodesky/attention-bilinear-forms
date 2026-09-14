#!/usr/bin/env python3
"""Build reusable symmetric-spectrum datasets and cluster analyses.

The input is a dataset produced by ``extract_qk_bilinear_db.py``. For every
selected model this script:

* computes the sorted eigenvalues of S=(L+L.T)/2 for every head;
* retains both raw and unit-length spectra, keyed to the extraction database;
* performs uncentered singular-value analysis as in Appendix D;
* assigns heads to the three spectral types using the profile map and the
  one-parameter accumulating family;
* records the assignments and whole-cluster and extreme-member average spectra; and
* renders the Appendix-D-style projection and spectrum figures.

Results are immutable and content-addressed by the source artifact checksums
and analysis settings. Re-running skips a completed matching analysis. Adding
another extracted model only creates a new package for that model.

Example:
    python scripts/analyze_bilinear_spectra.py \
        --dataset data/bilinear-forms --models facebook/opt-2.7b
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics import normalized_mutual_info_score, silhouette_samples, silhouette_score
from tqdm import tqdm


ANALYSIS_VERSION = 18
TYPE_IMBALANCE_TOLERANCE = 0.20
TYPE_II_EDGE_TOLERANCE = 0.01
DEFAULT_K = 3
DEFAULT_K_RANGE = (2, 3, 4, 5, 6)
DEFAULT_RESTARTS = 50
DEFAULT_SEED = 0
DEFAULT_EXTREME_HEADS = 30


@dataclass(frozen=True)
class ModelRecord:
    model_pk: int
    repo_id: str
    revision: str
    model_type: str
    positional_method: str
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    config: Mapping[str, Any]


@dataclass(frozen=True)
class HeadRecord:
    head_pk: int
    layer: int
    head: int
    artifact_path: str
    artifact_array: str
    artifact_index: int
    artifact_sha256: str


def safe_name(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def open_database(dataset: Path) -> sqlite3.Connection:
    path = dataset / "heads.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(f"Extraction database not found: {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def list_models(db: sqlite3.Connection, requested: Sequence[str] | None) -> list[ModelRecord]:
    rows = db.execute(
        """SELECT model_pk,repo_id,resolved_revision,model_type,positional_method,
                  d_model,n_layers,n_heads,head_dim,config_json
           FROM models ORDER BY repo_id,resolved_revision"""
    ).fetchall()
    models = [
        ModelRecord(
            int(row["model_pk"]), row["repo_id"], row["resolved_revision"],
            row["model_type"], row["positional_method"], int(row["d_model"]),
            int(row["n_layers"]), int(row["n_heads"]), int(row["head_dim"]),
            json.loads(row["config_json"]),
        )
        for row in rows
    ]
    if not requested or requested == ["all"]:
        return models
    wanted = set(requested)
    selected = [
        model for model in models
        if model.repo_id in wanted or safe_name(model.repo_id) in wanted
    ]
    found = {model.repo_id for model in selected} | {safe_name(model.repo_id) for model in selected}
    missing = sorted(wanted - found)
    if missing:
        available = ", ".join(model.repo_id for model in models) or "none"
        raise ValueError(f"Models not found in database: {missing}. Available: {available}")
    return selected


def head_records(db: sqlite3.Connection, model: ModelRecord) -> list[HeadRecord]:
    rows = db.execute(
        """SELECT h.head_pk,h.layer_index,h.head_index,h.artifact_array,
                  h.artifact_index,a.relative_path,a.sha256
           FROM heads h JOIN layers l ON l.layer_pk=h.layer_pk
           JOIN artifacts a ON a.artifact_pk=h.artifact_pk
           WHERE h.model_pk=? AND l.status='complete'
           ORDER BY h.layer_index,h.head_index""",
        (model.model_pk,),
    ).fetchall()
    records = [
        HeadRecord(
            int(row["head_pk"]), int(row["layer_index"]), int(row["head_index"]),
            row["relative_path"], row["artifact_array"], int(row["artifact_index"]),
            row["sha256"],
        )
        for row in rows
    ]
    expected = model.n_layers * model.n_heads
    if len(records) != expected:
        raise ValueError(
            f"{model.repo_id} is incomplete: found {len(records):,}/{expected:,} heads"
        )
    return records


def analysis_identity(
    model: ModelRecord,
    heads: Sequence[HeadRecord],
    settings: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    artifacts = sorted({(head.artifact_path, head.artifact_sha256) for head in heads})
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "model": {
            "repo_id": model.repo_id,
            "revision": model.revision,
            "model_pk": model.model_pk,
        },
        "source_artifacts": [
            {"relative_path": path, "sha256": digest} for path, digest in artifacts
        ],
        "settings": dict(settings),
    }
    digest = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    manifest["analysis_id"] = digest[:16]
    return digest[:16], manifest


def extract_spectra(
    dataset: Path,
    heads: Sequence[HeadRecord],
    expected_dim: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw = np.empty((len(heads), expected_dim), dtype=np.float64)
    normalized = np.empty_like(raw)
    summaries: list[dict[str, Any]] = []
    current_path: str | None = None
    current_npz: Any = None
    current_array: np.ndarray | None = None
    try:
        for row_index, record in enumerate(tqdm(heads, desc="eigenvalues", unit="head")):
            if record.artifact_path != current_path:
                if current_npz is not None:
                    current_npz.close()
                current_path = record.artifact_path
                current_npz = np.load(dataset / current_path, allow_pickle=False)
                current_array = current_npz[record.artifact_array]
            assert current_array is not None
            form = np.asarray(current_array[record.artifact_index], dtype=np.float64)
            symmetric = 0.5 * (form + form.T)
            antisymmetric = 0.5 * (form - form.T)
            values = np.linalg.eigvalsh(symmetric)
            if values.shape != (expected_dim,):
                raise ValueError(
                    f"Unexpected spectrum length {values.size} for L{record.layer}H{record.head}; "
                    f"expected {expected_dim}"
                )
            norm = float(np.linalg.norm(values))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"Degenerate spectrum for L{record.layer}H{record.head}")
            unit = values / norm
            raw[row_index] = values
            normalized[row_index] = unit
            positive = values[values > 0]
            negative = values[values < 0]
            weight_a, weight_b, weight_c, weight_d = simplex_weights_for_form(
                form, values, antisymmetric
            )
            summaries.append(
                {
                    "row_index": row_index,
                    "head_pk": record.head_pk,
                    "layer": record.layer,
                    "head": record.head,
                    "artifact_path": record.artifact_path,
                    "artifact_array": record.artifact_array,
                    "artifact_index": record.artifact_index,
                    "symmetric_norm": norm,
                    "trace": float(values.sum()),
                    "positive_count": int((values > 0).sum()),
                    "negative_count": int((values < 0).sum()),
                    "positive_energy_fraction": float(np.square(positive).sum() / norm**2),
                    "negative_energy_fraction": float(np.square(negative).sum() / norm**2),
                    "largest_eigenvalue": float(values[-1]),
                    "smallest_eigenvalue": float(values[0]),
                    "weight_a": weight_a,
                    "weight_b": weight_b,
                    "weight_c": weight_c,
                    "weight_d": weight_d,
                    "weight_sum": weight_a + weight_b + weight_c + weight_d,
                }
            )
    finally:
        if current_npz is not None:
            current_npz.close()
    return raw, normalized, summaries


def simplex_weights_for_form(
    form: np.ndarray,
    eigenvalues: np.ndarray | None = None,
    antisymmetric: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return the four profile coordinates of one bilinear form."""
    if antisymmetric is None:
        antisymmetric = 0.5 * (form - form.T)
    if eigenvalues is None:
        eigenvalues = np.linalg.eigvalsh(0.5 * (form + form.T))
    alpha = np.sort(eigenvalues[eigenvalues > 0])[::-1]
    beta = np.sort(-eigenvalues[eigenvalues < 0])[::-1]
    paired_dim = max(alpha.size, beta.size)
    alpha = np.pad(alpha, (0, paired_dim - alpha.size))
    beta = np.pad(beta, (0, paired_dim - beta.size))
    difference = alpha - beta
    form_energy = float(np.square(form).sum())
    return (
        float(np.square(antisymmetric).sum() / form_energy),
        float(2 * (alpha @ beta) / form_energy),
        float(np.square(difference[difference > 0]).sum() / form_energy),
        float(np.square(difference[difference < 0]).sum() / form_energy),
    )


def rope_inverse_frequencies(model: ModelRecord) -> np.ndarray:
    """Construct the checkpoint's base RoPE frequencies for one attention head."""
    rotary_dim = int(model.head_dim * float(
        model.config.get("partial_rotary_factor", model.config.get("rotary_pct", 1.0))
    ))
    rotary_dim -= rotary_dim % 2
    if rotary_dim <= 0:
        raise ValueError(f"{model.repo_id} has no even-dimensional rotary subspace")
    theta = float(model.config.get("rope_theta", model.config.get("rotary_emb_base", 10000.0)))
    frequencies = theta ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    scaling = model.config.get("rope_scaling")
    if scaling:
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type == "linear":
            frequencies /= float(scaling["factor"])
        elif rope_type not in {"default", None}:
            raise ValueError(
                f"RoPE scaling type {rope_type!r} needs an explicit frequency implementation"
            )
    return frequencies


def rotate_reduced_queries(
    queries: np.ndarray, displacement: int, frequencies: np.ndarray
) -> np.ndarray:
    """Apply split-half RoPE^displacement along the penultimate axis."""
    rotary_dim = 2 * frequencies.size
    half = frequencies.size
    angles = displacement * frequencies
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    first = queries[..., :half, :]
    second = queries[..., half:rotary_dim, :]
    rotated = queries.copy()
    rotated[..., :half, :] = first * cosine - second * sine
    rotated[..., half:rotary_dim, :] = second * cosine + first * sine
    return rotated


def simplex_weights_for_forms(forms: np.ndarray) -> np.ndarray:
    """Vectorized eigensolves followed by the profile map for a batch of forms."""
    symmetric = 0.5 * (forms + forms.swapaxes(-1, -2))
    antisymmetric = 0.5 * (forms - forms.swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    energies = np.square(forms).sum(axis=(-2, -1))
    result = np.empty((len(forms), 4), dtype=np.float64)
    result[:, 0] = np.square(antisymmetric).sum(axis=(-2, -1)) / energies
    for index, values in enumerate(eigenvalues):
        alpha = np.sort(values[values > 0])[::-1]
        beta = np.sort(-values[values < 0])[::-1]
        paired_dim = max(alpha.size, beta.size)
        alpha = np.pad(alpha, (0, paired_dim - alpha.size))
        beta = np.pad(beta, (0, paired_dim - beta.size))
        difference = alpha - beta
        result[index, 1] = 2 * (alpha @ beta) / energies[index]
        result[index, 2] = np.square(difference[difference > 0]).sum() / energies[index]
        result[index, 3] = np.square(difference[difference < 0]).sum() / energies[index]
    return result


def extract_rope_weight_trajectory(
    dataset: Path,
    model: ModelRecord,
    heads: Sequence[HeadRecord],
    powers: np.ndarray,
    weights_at_zero: np.ndarray,
) -> np.ndarray | None:
    """Compute profile coordinates across RoPE powers from compact factors."""
    if model.positional_method != "rope":
        return None
    frequencies = rope_inverse_frequencies(model)
    trajectory = np.empty((powers.size, len(heads), 4), dtype=np.float32)
    zero_rows = np.flatnonzero(powers == 0)
    if zero_rows.size:
        trajectory[zero_rows] = weights_at_zero
    grouped: dict[str, list[tuple[int, HeadRecord]]] = {}
    for row, head in enumerate(heads):
        grouped.setdefault(head.artifact_path, []).append((row, head))
    work = (powers.size - zero_rows.size) * len(heads)
    with tqdm(total=work, desc="RoPE powers", unit="form") as progress:
        for artifact_path, records in grouped.items():
            with np.load(dataset / artifact_path, allow_pickle=False) as artifact:
                if "Q_reduced" not in artifact or "K_reduced" not in artifact:
                    return None
                q_all = np.asarray(artifact["Q_reduced"], dtype=np.float64)
                k_all = np.asarray(artifact["K_reduced"], dtype=np.float64)
                artifact_indices = np.asarray(
                    [head.artifact_index for _, head in records], dtype=np.int64
                )
                output_rows = np.asarray([row for row, _ in records], dtype=np.int64)
                q_layer = q_all[artifact_indices]
                k_layer = k_all[artifact_indices]
                for power_row, power in enumerate(powers):
                    if power == 0:
                        continue
                    q = rotate_reduced_queries(q_layer, int(power), frequencies)
                    forms = np.matmul(k_layer.swapaxes(-1, -2), q)
                    trajectory[power_row, output_rows] = simplex_weights_for_forms(forms)
                    progress.update(len(records))
    return trajectory


def svd_coordinates(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, singular_values, vt = np.linalg.svd(spectra, full_matrices=False)
    coordinates = u * singular_values
    shares = singular_values**2 / np.square(singular_values).sum()
    if coordinates[:, 0].mean() < 0:
        coordinates[:, 0] *= -1
        vt[0] *= -1
    asymmetry = signed_asymmetry(spectra)
    probabilities = np.square(spectra)
    effective_rank = np.exp(
        -(probabilities * np.log(probabilities + np.finfo(float).tiny)).sum(axis=1)
    )
    for component, statistic in ((1, asymmetry), (2, effective_rank)):
        if component < coordinates.shape[1]:
            correlation = np.corrcoef(coordinates[:, component], statistic)[0, 1]
            if np.isfinite(correlation) and correlation < 0:
                coordinates[:, component] *= -1
                vt[component] *= -1
    return coordinates, shares, vt


def best_kmeans(
    spectra: np.ndarray, k: int, restarts: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(restarts):
        centroids, labels = kmeans2(spectra, k, minit="++", seed=seed + offset)
        if np.unique(labels).size != k:
            continue
        objective = float(np.square(spectra - centroids[labels]).sum())
        if best is None or objective < best[0]:
            best = objective, labels.copy(), centroids.copy()
    if best is None:
        raise RuntimeError(f"All {restarts} k-means attempts were degenerate for k={k}")
    return best[1], best[2], best[0]


def signed_asymmetry(spectra: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.square(row[row > 0]).sum() - np.square(row[row < 0]).sum()
            for row in spectra
        ]
    )


def ordered_labels(
    labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    asymmetry = signed_asymmetry(centroids)
    order = list(np.argsort(-asymmetry))
    relabeled = np.empty_like(labels)
    ordered_centroids = np.empty_like(centroids)
    for new, old in enumerate(order, start=1):
        relabeled[labels == old] = new
        ordered_centroids[new - 1] = centroids[old]
    return relabeled, ordered_centroids, order


def cluster_products(
    normalized: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    n_extreme: int,
    simplex_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    k = 3
    means = np.empty((k, normalized.shape[1]))
    raw_means = np.empty_like(means)
    extreme_means = np.empty_like(means)
    extreme_indices = np.full((k, n_extreme), -1, dtype=np.int64)
    symmetric_weights = simplex_weights[:, 1:] / (1.0 - simplex_weights[:, 0, None])
    for cluster in range(1, k + 1):
        indices = np.flatnonzero(labels == cluster)
        if indices.size == 0:
            means[cluster - 1] = np.nan
            raw_means[cluster - 1] = np.nan
            extreme_means[cluster - 1] = np.nan
            continue
        means[cluster - 1] = normalized[indices].mean(axis=0)
        raw_means[cluster - 1] = raw[indices].mean(axis=0)
        if cluster == 1:       # Type I+: closest to d=0
            scores = -symmetric_weights[indices, 2]
        elif cluster == 2:     # Type II: closest to the a-b edge
            scores = -(simplex_weights[indices, 2] + simplex_weights[indices, 3])
        else:                  # Type I-: closest to c=0
            scores = -symmetric_weights[indices, 1]
        representatives = indices[np.argsort(-scores)[: min(n_extreme, len(indices))]]
        extreme_indices[cluster - 1, : len(representatives)] = representatives
        extreme_means[cluster - 1] = normalized[representatives].mean(axis=0)
    return {
        "mean_normalized_spectrum": means,
        "mean_raw_spectrum": raw_means,
        "extreme_mean_normalized_spectrum": extreme_means,
        "extreme_row_indices": extreme_indices,
    }


TYPE_NAMES = {1: r"Type I$_+$", 2: "Type II", 3: r"Type I$_-$"}
INTERACTIVE_TYPE_NAMES = {1: "Type I₊", 2: "Type II", 3: "Type I₋"}


def cluster_name(cluster: int, k: int) -> str:
    return TYPE_NAMES.get(cluster, f"cluster {cluster}")


def weight_map_types(
    weights: np.ndarray,
    tolerance: float = TYPE_IMBALANCE_TOLERANCE,
    edge_tolerance: float = TYPE_II_EDGE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each head by explicit inequalities on its profile values.

    Type II contains the closed imbalance band and all heads sufficiently
    close to the a-b edge. Outside that edge neighborhood, Type I+ and I-
    are selected by the sign and magnitude of (c-d)/(c+d).
    """
    c, d = weights[:, 2], weights[:, 3]
    denominator = c + d
    imbalance = np.divide(c - d, denominator, out=np.zeros_like(c), where=denominator > 0)
    labels = np.full(len(weights), 2, dtype=np.int32)
    away_from_ab_edge = denominator > edge_tolerance
    labels[(imbalance > tolerance) & away_from_ab_edge] = 1
    labels[(imbalance < -tolerance) & away_from_ab_edge] = 3
    return labels, imbalance


def render_layer_heatmap(
    path_base: Path,
    model: ModelRecord,
    layers: np.ndarray,
    labels: np.ndarray,
) -> None:
    k = 3
    counts = np.zeros((model.n_layers, k), dtype=int)
    for layer, label in zip(layers, labels):
        counts[int(layer), int(label) - 1] += 1
    fractions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.22 * model.n_layers), 3.4))
    image = ax.imshow(fractions.T, aspect="auto", origin="lower", vmin=0, vmax=1,
                      cmap="Greys")
    ax.set_xlabel("layer")
    ax.set_ylabel("cluster")
    ax.set_yticks(np.arange(k), [cluster_name(i, k) for i in range(1, k + 1)])
    ax.set_xticks(np.arange(model.n_layers))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_title(f"{model.repo_id}: fraction of each layer in each cluster")
    fig.colorbar(image, ax=ax, label="fraction of layer's heads")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_simplex_weights(
    path_base: Path,
    model: ModelRecord,
    weights: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Projected tetrahedron used for Figure 11 of the paper."""
    from matplotlib.patches import Polygon

    vertices = np.asarray(
        [
            [0.42, 1.00],  # a: antisymmetric
            [0.00, 0.12],  # b: hyperbolic
            [0.78, 0.00],  # c: positive semidefinite
            [1.00, 0.48],  # d: negative semidefinite
        ]
    )
    va, vb, vc, vd = vertices
    points = weights @ vertices
    k = 3
    markers = ["+", "o", "_", "s", "x", "D", "^"]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.add_patch(Polygon([vb, vc, vd], closed=True, facecolor="#eeeeee",
                         edgecolor="none", zorder=0))
    for left, right in ((va, vb), (va, vc), (va, vd), (vb, vc), (vc, vd)):
        ax.plot([left[0], right[0]], [left[1], right[1]], color="black",
                linewidth=1.1, zorder=1)
    ax.plot([vb[0], vd[0]], [vb[1], vd[1]], color="black", linewidth=0.9,
            linestyle="--", dashes=(4, 3), zorder=1)
    for vertex in vertices:
        ax.plot(*vertex, marker="o", markersize=4, color="black", zorder=3)
    ax.text(va[0], va[1] + 0.03, "a = 1: antisymmetric",
            ha="center", va="bottom", fontsize=11)
    ax.text(vb[0] - 0.02, vb[1], "b = 1: hyperbolic",
            ha="right", va="center", fontsize=11)
    ax.text(vc[0], vc[1] - 0.04, "c = 1: positive semidefinite",
            ha="center", va="top", fontsize=11)
    ax.text(vd[0] + 0.02, vd[1], "d = 1: negative semidefinite",
            ha="left", va="center", fontsize=11)
    for cluster in range(1, k + 1):
        mask = labels == cluster
        marker = markers[(cluster - 1) % len(markers)]
        kwargs: dict[str, Any] = {
            "marker": marker, "color": "black", "s": 28, "linewidths": 0.7
        }
        if marker in {"o", "s", "D", "^"}:
            kwargs.update(facecolors="#dddddd", edgecolors="black")
        ax.scatter(points[mask, 0], points[mask, 1],
                   label=f"{cluster_name(cluster, k)} (n={mask.sum()})", zorder=2, **kwargs)
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{model.repo_id}: k={k}")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_symmetric_simplex(
    path_base: Path,
    model: ModelRecord,
    weights: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Projection to a=0, i.e. weights of the normalized symmetric part."""
    pb = np.asarray([0.5, np.sqrt(3) / 2])
    pc = np.asarray([0.0, 0.0])
    pd = np.asarray([1.0, 0.0])
    denominator = 1.0 - weights[:, 0]
    if np.any(denominator <= 0):
        raise ValueError("Cannot project a purely antisymmetric form to a=0")
    symmetric_weights = weights[:, 1:] / denominator[:, None]
    points = (
        np.outer(symmetric_weights[:, 0], pb)
        + np.outer(symmetric_weights[:, 1], pc)
        + np.outer(symmetric_weights[:, 2], pd)
    )
    k = 3
    markers = ["+", "o", "_", "s", "x", "D", "^"]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.plot([pb[0], pc[0]], [pb[1], pc[1]], color="black", linewidth=1.1)
    ax.plot([pb[0], pd[0]], [pb[1], pd[1]], color="black", linewidth=1.1)
    ax.plot([pc[0], pd[0]], [pc[1], pd[1]], color="black", linewidth=0.9,
            linestyle="--", dashes=(4, 3))
    for vertex in (pb, pc, pd):
        ax.plot(*vertex, marker="o", markersize=4, color="black", zorder=3)
    ax.text(pb[0], pb[1] + 0.035, "b = 1: hyperbolic",
            ha="center", va="bottom", fontsize=11)
    ax.text(pc[0] - 0.02, pc[1] - 0.035, "c = 1: positive semidefinite",
            ha="left", va="top", fontsize=11)
    ax.text(pd[0] + 0.02, pd[1] - 0.035, "d = 1: negative semidefinite",
            ha="right", va="top", fontsize=11)
    for cluster in range(1, k + 1):
        mask = labels == cluster
        marker = markers[(cluster - 1) % len(markers)]
        kwargs: dict[str, Any] = {
            "marker": marker, "color": "black", "s": 34, "linewidths": 0.75
        }
        if marker in {"o", "s", "D", "^"}:
            kwargs.update(facecolors="#dddddd", edgecolors="black")
        ax.scatter(points[mask, 0], points[mask, 1],
                   label=f"{cluster_name(cluster, k)} (n={mask.sum()})", zorder=2, **kwargs)
    ax.legend(fontsize=9, frameon=True, loc="upper left")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.92)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_interactive_simplex(
    path: Path,
    model: ModelRecord,
    weights: np.ndarray,
    labels: np.ndarray,
    heads: Sequence[HeadRecord],
    rope_powers: np.ndarray | None = None,
    rope_weights: np.ndarray | None = None,
) -> None:
    """Draggable 3-D rendering of the profiles, with optional RoPE displacement."""
    import plotly.graph_objects as go

    # Four alternating vertices of a cube form a regular tetrahedron.
    vertices = np.asarray(
        [
            [1.0, 1.0, 1.0],    # a
            [1.0, -1.0, -1.0],  # b
            [-1.0, 1.0, -1.0],  # c
            [-1.0, -1.0, 1.0],  # d
        ]
    )
    points = weights @ vertices
    fig = go.Figure()
    # Translucent faces establish depth while leaving interior heads visible.
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=[0, 0, 0, 1], j=[1, 1, 2, 2], k=[2, 3, 3, 3],
            color="lightgray", opacity=0.10, hoverinfo="skip", name="simplex",
            showlegend=False,
        )
    )
    for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        fig.add_trace(
            go.Scatter3d(
                x=vertices[[left, right], 0], y=vertices[[left, right], 1],
                z=vertices[[left, right], 2], mode="lines",
                line={"color": "black", "width": 4}, hoverinfo="skip",
                showlegend=False,
            )
        )
    vertex_text = [
        "a = 1: antisymmetric", "b = 1: hyperbolic",
        "c = 1: positive semidefinite", "d = 1: negative semidefinite",
    ]
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode="markers+text",
            marker={"size": 5, "color": "black"}, text=vertex_text,
            textposition="top center", hoverinfo="text", showlegend=False,
        )
    )
    # The empirical one-parameter family F={a=b, cd=0}. The theorem
    # parameter rho gives its canonical traversal from c=1 through the
    # midpoint of the a-b edge to d=1. Keep it hidden initially; its legend
    # entry is the requested show/hide control.
    rho_finite = np.logspace(-3, 3, 401)
    rho = np.concatenate(([0.0], rho_finite, [np.inf]))
    family_weights = np.zeros((rho.size, 4), dtype=float)
    finite = np.isfinite(rho)
    denominator = 1.0 + np.square(rho[finite])
    family_weights[finite, 0] = rho[finite] / denominator
    family_weights[finite, 1] = rho[finite] / denominator
    family_weights[finite, 2] = np.square(np.maximum(1.0 - rho[finite], 0.0)) / denominator
    family_weights[finite, 3] = np.square(np.maximum(rho[finite] - 1.0, 0.0)) / denominator
    family_weights[-1, 3] = 1.0
    family_points = family_weights @ vertices
    rho_text = np.asarray(
        ["infinity" if np.isinf(value) else f"{value:.5g}" for value in rho]
    )
    family_custom = np.column_stack((rho_text, family_weights.astype(object)))
    fig.add_trace(
        go.Scatter3d(
            x=family_points[:, 0], y=family_points[:, 1], z=family_points[:, 2],
            mode="lines", name="accumulating family",
            visible=True, line={"color": "#d62728", "width": 1.5},
            customdata=family_custom,
            hovertemplate=(
                "rho=%{customdata[0]}<br>a=%{customdata[1]:.5f}<br>"
                "b=%{customdata[2]:.5f}<br>c=%{customdata[3]:.5f}<br>"
                "d=%{customdata[4]:.5f}<extra>accumulating family</extra>"
            ),
        )
    )
    head_layers = np.asarray([head.layer for head in heads])
    head_indices = np.asarray([head.head for head in heads])
    def head_trace(
        cluster: int, frame_points: np.ndarray, frame_weights: np.ndarray,
        frame_labels: np.ndarray,
    ) -> go.Scatter3d:
        mask = frame_labels == cluster
        custom = np.column_stack((head_layers[mask], head_indices[mask], frame_weights[mask]))
        name = INTERACTIVE_TYPE_NAMES[cluster]
        common: dict[str, Any] = {
            "x": frame_points[mask, 0], "y": frame_points[mask, 1],
            "z": frame_points[mask, 2], "name": f"{name} (n={mask.sum()})",
            "customdata": custom,
            "hovertemplate": (
                "layer %{customdata[0]:.0f}, head %{customdata[1]:.0f}<br>"
                "a=%{customdata[2]:.5f}<br>b=%{customdata[3]:.5f}<br>"
                "c=%{customdata[4]:.5f}<br>d=%{customdata[5]:.5f}"
                f"<extra>{name}</extra>"
            ),
        }
        colors = {1: "#1f77b4", 2: "#e07a00", 3: "#5b5b5b"}
        return go.Scatter3d(
            mode="markers",
            marker={"size": 3, "symbol": "circle", "color": colors[cluster],
                    "opacity": 0.95},
            **common,
        )
    head_trace_start = len(fig.data)
    for cluster in range(1, 4):
        fig.add_trace(head_trace(cluster, points, weights, labels))
    if rope_powers is not None and rope_weights is not None:
        frames = []
        for power, frame_weights in zip(rope_powers, rope_weights):
            frame_labels, _ = weight_map_types(frame_weights)
            frame_points = frame_weights @ vertices
            traces = []
            for cluster in range(1, 4):
                traces.append(head_trace(cluster, frame_points, frame_weights, frame_labels))
            frames.append(go.Frame(
                name=str(int(power)), data=traces,
                traces=list(range(head_trace_start, head_trace_start + 3)),
                layout=go.Layout(title_text=(
                    f"{model.repo_id}: bilinear-form weights (RoPE power d = {int(power)})"
                )),
            ))
        fig.frames = frames
    fig.update_layout(
        title=(f"{model.repo_id}: bilinear-form weights"
               + (" (RoPE power d = 0)" if rope_weights is not None else
                  " (three profile types)")),
        scene={
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
            "zaxis": {"visible": False}, "aspectmode": "cube",
            # Start above the a-b edge, with d at the top, a at the bottom,
            # and b/c at the left/right, matching the paper-facing view.
            "camera": {
                "eye": {"x": 0.75, "y": 0.75, "z": 2.1},
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
            },
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    powers_json = json.dumps(
        [int(power) for power in rope_powers] if rope_powers is not None else []
    )
    script = f"const powers = {powers_json};\n"
    controls = """
<style>
  .plot-control {font:15px system-ui,sans-serif; padding:10px 20px 0;
                 display:flex; align-items:center; gap:14px; color:#222}
  .plot-control label {white-space:nowrap; font-weight:600}
  .plot-control input {width:min(420px,55vw); accent-color:#333}
  .plot-control output {min-width:10.5em; font-variant-numeric:tabular-nums}
</style>
"""
    if rope_powers is not None and rope_weights is not None:
        script += """
const slider = document.getElementById('rope-power-slider');
const output = document.getElementById('rope-power-value');
function showPower(index) {
  const power = powers[Number(index)];
  output.textContent = power;
  Plotly.animate('simplex-plot', [String(power)], {
    mode: 'immediate', frame: {duration: 0, redraw: true},
    transition: {duration: 0}
  });
}
slider.addEventListener('input', event => showPower(event.target.value));
"""
        controls += f"""
<div class="plot-control">
  <label for="rope-power-slider">Relative displacement</label>
  <input id="rope-power-slider" type="range" min="0" max="{len(rope_powers)-1}"
         value="0" step="1" aria-label="RoPE power">
  <output>d = <span id="rope-power-value">0</span></output>
</div>
"""
    html = fig.to_html(
        include_plotlyjs=True, full_html=True, div_id="simplex-plot",
        post_script=script, auto_play=False,
    )
    path.write_text(html.replace("<body>", "<body>" + controls, 1))


def render_projection(
    path_base: Path,
    model: ModelRecord,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> None:
    k = 3
    markers = ["+", "o", "_", "s", "x", "D", "^"]
    fig = plt.figure(figsize=(12.5, 5.1))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    for cluster in range(1, k + 1):
        mask = labels == cluster
        marker = markers[(cluster - 1) % len(markers)]
        label = f"{cluster_name(cluster, k)} (n={mask.sum()})"
        kwargs = {"marker": marker, "color": "black", "s": 24, "linewidths": 0.7}
        if marker in {"o", "s", "D", "^"}:
            kwargs.update(facecolors="#dddddd", edgecolors="black")
        ax3.scatter(coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2], **kwargs)
        ax2.scatter(
            coordinates[mask, 1], coordinates[mask, 2], label=label, **kwargs
        )
    ax3.set_xlabel("component 1")
    ax3.set_ylabel("component 2")
    ax3.set_zlabel("component 3")
    ax3.view_init(elev=15, azim=-35)
    ax2.set_xlabel("component 2")
    ax2.set_ylabel("component 3")
    ax2.legend(frameon=True, fontsize=8)
    fig.suptitle(model.repo_id)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_average_spectra(
    path_base: Path,
    model: ModelRecord,
    means: np.ndarray,
    labels: np.ndarray,
) -> None:
    k = means.shape[0]
    x = np.arange(means.shape[1])
    fig, axes = plt.subplots(1, k, figsize=(5.0 * k, 3.6), sharey=True)
    for cluster, ax in enumerate(np.atleast_1d(axes), start=1):
        values = means[cluster - 1]
        ax.plot(x, values, color="black", linewidth=1.2)
        ax.fill_between(x, 0, values, where=values < 0, color="#1f77b4", alpha=0.45)
        ax.fill_between(x, 0, values, where=values > 0, color="#d62728", alpha=0.45)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{cluster_name(cluster, k)} (n={(labels == cluster).sum()})")
        ax.set_xlabel("ordered eigenvalue")
    np.atleast_1d(axes)[0].set_ylabel("mean normalized eigenvalue")
    fig.suptitle(model.repo_id)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_extreme_histograms(
    path_base: Path,
    model: ModelRecord,
    extreme_means: np.ndarray,
    labels: np.ndarray,
    extreme_indices: np.ndarray,
) -> None:
    k = extreme_means.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(5.2 * k, 3.6), sharex=True, sharey=True)
    finite_values = extreme_means[np.isfinite(extreme_means)]
    horizontal_limit = (
        1.05 * float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    )
    for cluster, ax in enumerate(np.atleast_1d(axes), start=1):
        values = extreme_means[cluster - 1]
        finite = values[np.isfinite(values)]
        negative = finite[finite < 0]
        positive = finite[finite > 0]
        bins = np.linspace(-horizontal_limit, horizontal_limit, 81)
        ax.hist(negative, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.35)
        ax.hist(positive, bins=bins, color="#d62728", edgecolor="black", linewidth=0.35)
        if negative.size:
            negative_mean = float(negative.mean())
            ax.axvline(negative_mean, color="#1f77b4", linewidth=1.2,
                       linestyle="--", dashes=(4, 3),
                       label=f"negative-lobe mean: {negative_mean:.2f}")
        if positive.size:
            positive_mean = float(positive.mean())
            ax.axvline(positive_mean, color="#d62728", linewidth=1.2,
                       linestyle="--", dashes=(4, 3),
                       label=f"positive-lobe mean: {positive_mean:.2f}")
        count = int((extreme_indices[cluster - 1] >= 0).sum())
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xlim(-horizontal_limit, horizontal_limit)
        ax.set_title(
            f"{cluster_name(cluster, k)} (n={(labels == cluster).sum()}, {count} extreme)"
        )
        ax.set_xlabel("normalized eigenvalue")
        if negative.size or positive.size:
            ax.legend(loc="upper right", fontsize=8, frameon=True)
    np.atleast_1d(axes)[0].set_ylabel("count")
    fig.suptitle(model.repo_id)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_model(
    args: argparse.Namespace,
    dataset: Path,
    db: sqlite3.Connection,
    model: ModelRecord,
) -> dict[str, Any]:
    if model.model_type == "qwen3":
        raise ValueError(
            f"{model.repo_id} applies nonlinear Q/K normalization, so its deployed "
            "attention score is not a fixed bilinear form"
        )
    heads = head_records(db, model)
    n_extreme = args.n_extreme
    if n_extreme is None:
        n_extreme = DEFAULT_EXTREME_HEADS
    settings = {
        "classification": "per-head inequalities on (c-d)/(c+d)",
        "type_imbalance_tolerance": TYPE_IMBALANCE_TOLERANCE,
        "type_ii_ab_edge_tolerance": TYPE_II_EDGE_TOLERANCE,
        "n_extreme": n_extreme,
        "spectrum": "sorted eigenvalues of (L+L.T)/2, reduced nonzero subspace",
        "normalization": "unit Euclidean length per head",
        "dimension": 2 * model.head_dim,
        "svd_centered": False,
        "rope_power_max": args.rope_power_max,
        "rope_power_step": args.rope_power_step,
    }
    analysis_id, manifest = analysis_identity(model, heads, settings)
    model_root = (
        dataset / "analyses" / "symmetric_spectra" / safe_name(model.repo_id) / model.revision
    )
    output = model_root / analysis_id
    completion = output / "completion.json"
    if completion.is_file() and args.refresh_interactive_only:
        with np.load(output / "spectra.npz", allow_pickle=False) as cached:
            simplex_weights = cached["simplex_weights_abcd"]
        with np.load(output / "weight_map_types_k3.npz", allow_pickle=False) as cached:
            labels = cached["labels"]
        rope_powers = None
        rope_weights = None
        rope_cache = output / "rope_weight_trajectory.npz"
        if rope_cache.is_file():
            with np.load(rope_cache, allow_pickle=False) as cached:
                rope_powers = cached["powers"]
                rope_weights = cached["simplex_weights_abcd"]
        render_interactive_simplex(
            output / "simplex_weights_k3_interactive.html",
            model, simplex_weights, labels, heads, rope_powers, rope_weights,
        )
        print(f"[{model.repo_id}] refreshed interactive view in {output}")
        return json.loads(completion.read_text())
    if completion.is_file() and not args.force:
        print(f"[{model.repo_id}] unchanged; using {output}")
        latest = {"analysis_id": analysis_id, "relative_path": output.relative_to(dataset).as_posix()}
        atomic_json(model_root / "latest.json", latest)
        return json.loads(completion.read_text())

    output.mkdir(parents=True, exist_ok=True)
    print(f"\n[{model.repo_id}] extracting symmetric spectra from {len(heads):,} heads")
    started = time.monotonic()
    raw, normalized, head_summaries = extract_spectra(
        dataset, heads, 2 * model.head_dim
    )
    simplex_weights = np.asarray(
        [
            [row["weight_a"], row["weight_b"], row["weight_c"], row["weight_d"]]
            for row in head_summaries
        ],
        dtype=np.float64,
    )
    rope_powers: np.ndarray | None = None
    rope_weights: np.ndarray | None = None
    if model.positional_method == "rope":
        rope_powers = np.arange(
            0, args.rope_power_max + 1, args.rope_power_step, dtype=np.int32
        )
        if rope_powers[-1] != args.rope_power_max:
            rope_powers = np.append(rope_powers, args.rope_power_max)
        print(
            f"[{model.repo_id}] evaluating {len(rope_powers)} RoPE powers "
            f"from 0 to {args.rope_power_max}"
        )
        rope_cache = output / "rope_weight_trajectory.npz"
        if rope_cache.is_file():
            with np.load(rope_cache, allow_pickle=False) as cached:
                cached_powers = cached["powers"]
                cached_weights = cached["simplex_weights_abcd"]
            if np.array_equal(cached_powers, rope_powers) and cached_weights.shape == (
                len(rope_powers), len(heads), 4
            ):
                rope_weights = cached_weights
                print(f"[{model.repo_id}] using cached RoPE trajectory")
            else:
                rope_weights = extract_rope_weight_trajectory(
                    dataset, model, heads, rope_powers, simplex_weights
                )
        else:
            rope_weights = extract_rope_weight_trajectory(
                dataset, model, heads, rope_powers, simplex_weights
            )
        if rope_weights is None:
            print(
                f"[{model.repo_id}] compact RoPE factors are absent; "
                "the interactive view will remain fixed at d=0"
            )
            rope_powers = None
        else:
            np.savez_compressed(
                output / "rope_weight_trajectory.npz",
                powers=rope_powers,
                simplex_weights_abcd=rope_weights,
                head_pk=np.asarray([head.head_pk for head in heads], dtype=np.int64),
                layer=np.asarray([head.layer for head in heads], dtype=np.int32),
                head=np.asarray([head.head for head in heads], dtype=np.int32),
            )
    symmetric_simplex_weights = simplex_weights[:, 1:] / (
        1.0 - simplex_weights[:, 0, None]
    )
    coordinates, shares, right_vectors = svd_coordinates(normalized)
    asymmetry = signed_asymmetry(normalized)

    np.savez_compressed(
        output / "spectra.npz",
        eigenvalues_raw=raw,
        eigenvalues_unit=normalized,
        head_pk=np.asarray([head.head_pk for head in heads], dtype=np.int64),
        layer=np.asarray([head.layer for head in heads], dtype=np.int32),
        head=np.asarray([head.head for head in heads], dtype=np.int32),
        svd_coordinates=coordinates,
        singular_value_energy_shares=shares,
        right_singular_vectors=right_vectors,
        signed_asymmetry=asymmetry,
        simplex_weights_abcd=simplex_weights,
        symmetric_simplex_weights_bcd=symmetric_simplex_weights,
    )
    write_csv(output / "head_spectra.csv", list(head_summaries[0]), head_summaries)

    print(f"[{model.repo_id}] assigning three types from simplex weights")
    labels, weight_imbalance = weight_map_types(simplex_weights)
    distinct_types = np.unique(labels)
    if 1 < distinct_types.size < len(labels):
        samples = silhouette_samples(simplex_weights, labels)
        score = float(silhouette_score(simplex_weights, labels))
    else:
        samples = np.full(len(labels), np.nan)
        score = float("nan")
    sizes = [int((labels == cluster).sum()) for cluster in range(1, 4)]
    products = cluster_products(normalized, raw, labels, n_extreme, simplex_weights)
    layer_nmi = float(normalized_mutual_info_score(
        np.asarray([head.layer for head in heads]), labels
    ))
    print(f"  sizes={sizes}; weight-space silhouette={score:.3f}; layer association={layer_nmi:.3f}")
    np.savez_compressed(
        output / "weight_map_types_k3.npz",
        labels=labels,
        weight_imbalance=weight_imbalance,
        silhouette_per_head=samples,
        mean_normalized_spectrum=products["mean_normalized_spectrum"],
        mean_raw_spectrum=products["mean_raw_spectrum"],
        extreme_mean_normalized_spectrum=products["extreme_mean_normalized_spectrum"],
        extreme_row_indices=products["extreme_row_indices"],
    )
    assignment_rows = []
    for index, (head, label, sample) in enumerate(zip(heads, labels, samples)):
        assignment_rows.append({
            "row_index": index, "head_pk": head.head_pk, "layer": head.layer,
            "head": head.head, "type": TYPE_NAMES[int(label)].replace("$_+$", "+").replace("$_-$", "-"),
            "type_number": int(label), "weight_imbalance": float(weight_imbalance[index]),
            "weight_space_silhouette": float(sample),
            "signed_asymmetry": float(asymmetry[index]),
        })
    write_csv(output / "weight_map_type_assignments_k3.csv", list(assignment_rows[0]), assignment_rows)

    layer_rows = []
    for layer in range(model.n_layers):
        layer_mask = np.asarray([head.layer == layer for head in heads])
        row: dict[str, Any] = {"layer": layer, "total_heads": int(layer_mask.sum())}
        for cluster in range(1, 4):
            count = int(np.sum(layer_mask & (labels == cluster)))
            row[f"type_{cluster}_count"] = count
            row[f"type_{cluster}_fraction"] = count / max(int(layer_mask.sum()), 1)
        layer_rows.append(row)
    write_csv(output / "layer_type_counts_k3.csv", list(layer_rows[0]), layer_rows)

    render_projection(output / "projection_k3", model, coordinates, labels)
    render_average_spectra(
        output / "cluster_mean_spectra_k3", model,
        products["mean_normalized_spectrum"], labels,
    )
    render_extreme_histograms(
        output / "cluster_extreme_histograms_k3", model,
        products["extreme_mean_normalized_spectrum"], labels,
        products["extreme_row_indices"],
    )
    render_layer_heatmap(
        output / "layer_cluster_heatmap_k3", model,
        np.asarray([head.layer for head in heads]), labels,
    )
    render_simplex_weights(
        output / "simplex_weights_k3", model, simplex_weights, labels,
    )
    render_symmetric_simplex(
        output / "symmetric_simplex_weights_k3", model, simplex_weights, labels,
    )
    render_interactive_simplex(
        output / "simplex_weights_k3_interactive.html", model, simplex_weights, labels, heads,
        rope_powers, rope_weights,
    )
    elapsed = time.monotonic() - started
    summary = {
        **manifest,
        "status": "complete",
        "output_relative_path": output.relative_to(dataset).as_posix(),
        "head_count": len(heads),
        "spectrum_dimension": raw.shape[1],
        "positional_method": model.positional_method,
        "classification": "per-head inequalities on (c-d)/(c+d)",
        "type_imbalance_tolerance": TYPE_IMBALANCE_TOLERANCE,
        "type_ii_ab_edge_tolerance": TYPE_II_EDGE_TOLERANCE,
        "primary_k": 3,
        "primary_cluster_sizes": sizes,
        "weight_space_silhouette": score,
        "primary_cluster_mean_signed_asymmetry": [
            (float(asymmetry[labels == cluster].mean())
             if np.any(labels == cluster) else None)
            for cluster in range(1, 4)
        ],
        "primary_layer_normalized_mutual_information": float(
            normalized_mutual_info_score(
                np.asarray([head.layer for head in heads]), labels
            )
        ),
        "simplex_weight_means": {
            name: float(simplex_weights[:, index].mean())
            for index, name in enumerate(("a", "b", "c", "d"))
        },
        "simplex_weight_max_sum_error": float(
            np.max(np.abs(simplex_weights.sum(axis=1) - 1.0))
        ),
        "singular_value_energy_shares_first_three": [float(x) for x in shares[:3]],
        "first_coordinate_mean": float(coordinates[:, 0].mean()),
        "first_coordinate_std": float(coordinates[:, 0].std()),
        "elapsed_seconds": elapsed,
        "files": {
            "spectra": "spectra.npz",
            "head_metadata": "head_spectra.csv",
            "primary_clustering": "weight_map_types_k3.npz",
            "primary_assignments": "weight_map_type_assignments_k3.csv",
            **({"rope_weight_trajectory": "rope_weight_trajectory.npz"}
               if rope_weights is not None else {}),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(completion, summary)
    atomic_json(
        model_root / "latest.json",
        {"analysis_id": analysis_id, "relative_path": output.relative_to(dataset).as_posix()},
    )
    print(
        f"[{model.repo_id}] complete in {elapsed:.1f}s; three profile types; "
        f"results: {output}"
    )
    return summary


def parse_k_range(text: str) -> tuple[int, ...]:
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            left, right = part.split("-", 1)
            values.update(range(int(left), int(right) + 1))
        elif part:
            values.add(int(part))
    if not values or min(values) < 2:
        raise argparse.ArgumentTypeError("k range must contain integers >= 2")
    return tuple(sorted(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extraction dataset directory.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Repository IDs or 'all'.")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-extreme", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-interactive-only", action="store_true",
        help="Rebuild only the interactive HTML from cached analysis data.",
    )
    parser.add_argument("--rope-power-max", type=int, default=128,
                        help="Largest integer RoPE power in interactive sliders (default: 128).")
    parser.add_argument("--rope-power-step", type=int, default=1,
                        help="Spacing between precomputed RoPE powers (default: 1).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.restarts < 1:
        raise SystemExit("--restarts must be positive")
    if args.rope_power_max < 0 or args.rope_power_step < 1:
        raise SystemExit("RoPE power maximum must be nonnegative and step must be positive")
    dataset = Path(args.dataset).resolve()
    db = open_database(dataset)
    try:
        models = list_models(db, args.models)
        if not models:
            raise SystemExit("The extraction database contains no models")
        summaries = [analyze_model(args, dataset, db, model) for model in models]
    finally:
        db.close()
    print("\nSummary")
    for summary in summaries:
        shares = summary["singular_value_energy_shares_first_three"]
        print(
            f"  {summary['model']['repo_id']}: {summary['head_count']:,} heads; "
            f"three profile types; "
            f"first three components={100 * sum(shares):.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
