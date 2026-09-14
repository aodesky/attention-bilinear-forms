#!/usr/bin/env python3
"""The model set of the paper, and access to its extracted bilinear forms.

Every experiment of the paper's model survey reads its data through this
module, so that the figures, the tables and the quoted statistics all
describe the same set of heads.

The data come from the extraction database under ``data/bilinear-forms``
built by ``extract_qk_bilinear_db.py``: for each head it stores the
bilinear form ``L = W_K^T W_Q`` already restricted to the reduced
subspace ``im(W_K^T) + im(W_Q^T)``, so nothing here needs model weights
or a network connection.

``MODELS`` lists the fourteen models reported in the paper, in the order
of the models subsection.  The extraction database also holds the two
Qwen3 models; they are deliberately not part of the reported set, and
``MODELS`` is the single place that decides this.

For models with grouped-query attention (Mistral), several query heads
share one key head.  Each query head still has its own bilinear form
``W_K^T W_Q``, and the paper treats those forms as separate heads; the
key head a query head draws on is recorded as ``kv_head`` for reference.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "bilinear-forms"
DB_PATH = DATASET / "heads.sqlite3"
ANALYSES = DATASET / "analyses" / "symmetric_spectra"

# (short tag used in figures and file names, Hugging Face repo id), in the
# order of the models subsection of the paper.
MODELS: list[tuple[str, str]] = [
    ("distilgpt2", "distilgpt2"),
    ("gpt2", "gpt2"),
    ("gpt2-medium", "gpt2-medium"),
    ("gpt2-large", "gpt2-large"),
    ("opt-125m", "facebook/opt-125m"),
    ("opt-350m", "facebook/opt-350m"),
    ("opt-1.3b", "facebook/opt-1.3b"),
    ("opt-2.7b", "facebook/opt-2.7b"),
    ("opt-6.7b", "facebook/opt-6.7b"),
    ("opt-13b", "facebook/opt-13b"),
    ("bloom-3b", "bigscience/bloom-3b"),
    ("bloom-7b1", "bigscience/bloom-7b1"),
    ("pythia-70m-deduped", "EleutherAI/pythia-70m-deduped"),
    ("Mistral-Small-24B-Base", "mistralai/Mistral-Small-24B-Base-2501"),
]

TAGS = [tag for tag, _ in MODELS]
REPO_OF = dict(MODELS)


@dataclass(frozen=True)
class ModelInfo:
    """The architectural facts the paper quotes for a model."""
    tag: str
    repo_id: str
    n_layers: int
    n_heads: int          # query heads per layer
    n_kv_heads: int       # key/value heads per layer
    d_model: int          # the ambient dimension N
    head_dim: int         # the head dimension n
    positional_method: str
    n_heads_total: int


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"extraction database not found at {DB_PATH}; "
            "run scripts/extract_qk_bilinear_db.py first")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def model_info(tag: str) -> ModelInfo:
    """Architecture and head count of one model, from the database."""
    repo = REPO_OF[tag]
    with _connect() as con:
        row = con.execute(
            """select m.n_layers, m.n_heads, m.n_kv_heads, m.d_model,
                      m.head_dim, m.positional_method,
                      (select count(*) from heads h
                       where h.model_pk = m.model_pk) as n_heads_total
               from models m where m.repo_id = ?""", (repo,)).fetchone()
    if row is None:
        raise KeyError(f"{repo} is not in the extraction database")
    return ModelInfo(tag=tag, repo_id=repo, **dict(row))


def all_model_info() -> list[ModelInfo]:
    return [model_info(tag) for tag in TAGS]


def analysis_dir(tag: str) -> Path:
    """The current symmetric-spectra analysis directory for a model.

    Prefers the revision's ``latest.json`` when one is present, and falls
    back to the unique completed analysis otherwise.  Raises if the
    choice would be ambiguous, rather than picking one silently.
    """
    root = ANALYSES / REPO_OF[tag].replace("/", "__")
    if not root.is_dir():
        raise FileNotFoundError(f"no analysis for {tag} under {root}")

    latest = sorted(root.glob("*/latest.json"))
    if len(latest) == 1:
        meta = json.loads(latest[0].read_text())
        return latest[0].parent / meta["analysis_id"]

    required = {"head_spectra.csv", "weight_map_type_assignments_k3.csv",
                "spectra.npz", "completion.json"}
    complete = [d for rev in sorted(root.iterdir()) if rev.is_dir()
                for d in sorted(rev.iterdir())
                if d.is_dir() and required <= {f.name for f in d.iterdir()}]
    if len(complete) == 1:
        return complete[0]
    raise RuntimeError(
        f"{tag}: expected exactly one current analysis, found "
        f"{len(latest)} latest.json and {len(complete)} completed analyses")


def iter_forms(tag: str) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Yield ``(layer, head, kv_head, L)`` for every head of a model.

    ``L`` is the bilinear form of the head restricted to the reduced
    subspace, in float64.  Heads come in layer order, and within a layer
    in head order.
    """
    repo = REPO_OF[tag]
    with _connect() as con:
        rows = con.execute(
            """select a.relative_path, l.layer_index
               from artifacts a
               join models m using(model_pk)
               join layers l using(layer_pk)
               where m.repo_id = ? and a.kind = 'reduced_bilinear_forms'
               order by l.layer_index""", (repo,)).fetchall()
    if not rows:
        raise FileNotFoundError(f"no bilinear-form artifacts for {repo}")

    for row in rows:
        with np.load(DATASET / row["relative_path"]) as z:
            forms = z["L"]
            kv = z["kv_head_index"] if "kv_head_index" in z.files else None
            for head in range(forms.shape[0]):
                kv_head = int(kv[head]) if kv is not None else head
                yield (int(row["layer_index"]), head, kv_head,
                       forms[head].astype(np.float64))


def symmetric_antisymmetric(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The symmetric and antisymmetric parts of a bilinear form."""
    return 0.5 * (L + L.T), 0.5 * (L - L.T)


def sorted_lobes(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The lists lambda_+ and lambda_- of a symmetric form.

    The positive eigenvalues and the absolute values of the negative
    eigenvalues, each sorted decreasing and padded with zeros to a common
    length, as in the definition of the profile map.
    """
    ev = np.linalg.eigvalsh(S)
    pos = np.sort(ev[ev > 0])[::-1]
    neg = np.sort(-ev[ev < 0])[::-1]
    m = max(pos.size, neg.size)
    return (np.pad(pos, (0, m - pos.size)), np.pad(neg, (0, m - neg.size)))


def profile(L: np.ndarray) -> tuple[float, float, float, float]:
    """The profile π(L) = (a, b, c, d) of a nonzero bilinear form."""
    S, T = symmetric_antisymmetric(L)
    norm_sq = float((L ** 2).sum())
    if norm_sq == 0.0:
        raise ValueError("the profile of the zero form is undefined")
    lam_p, lam_m = sorted_lobes(S)
    u = lam_p - lam_m
    return (float((T ** 2).sum()) / norm_sq,
            float(2.0 * lam_p @ lam_m) / norm_sq,
            float((u[u > 0] ** 2).sum()) / norm_sq,
            float((u[u < 0] ** 2).sum()) / norm_sq)
