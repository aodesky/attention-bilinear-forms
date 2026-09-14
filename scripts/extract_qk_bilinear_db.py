#!/usr/bin/env python3
"""Stream attention Q/K weights into a restartable bilinear-form dataset.

This program is intended for large Hugging Face checkpoints that cannot be
loaded as a complete model.  It reads safetensors directly or memory-maps
weights-only PyTorch checkpoints, holds only the tensors needed for the current
layer, and stores one compact artifact per layer plus searchable provenance in
SQLite.

For a head with Wq, Wk in R^(d_head x d_model), the ambient bilinear form is

    L = Wk.T @ Wq.

Writing R for an orthonormal basis of span(Wk.T, Wq.T), the saved matrix is

    L_reduced = R.T @ L @ R = (Wk @ R).T @ (Wq @ R).

L is zero outside this at-most 2*d_head dimensional subspace, so L_reduced,
together with the recorded ambient dimension, is an exact orthogonal
representative of L (not a truncation).  The large basis R is deliberately not
saved: doing so would cost about as much disk as retaining the source weights.

Examples
--------
    python scripts/extract_qk_bilinear_db.py run \
        --models facebook/opt-6.7b facebook/opt-13b \
        --out /data/qk-bilinear

    python scripts/extract_qk_bilinear_db.py status --out /data/qk-bilinear

Safe interruption/restart
-------------------------
Artifacts are written to a temporary file and atomically renamed.  Only then
is the corresponding database transaction committed.  On restart, completed
layers whose files still match their recorded size and SHA-256 digest are
skipped.  A missing or damaged artifact is regenerated.

Supported layouts
-----------------
  * OPT: separate q_proj/k_proj
  * GPT-2 compatible (including Cerebras-GPT): fused c_attn
  * BLOOM: fused, head-interleaved query_key_value

RoPE checkpoints are rejected by default.  Pass --allow-rope explicitly to
override that guard.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from tqdm import tqdm


SCHEMA_VERSION = 1
FORMAT_VERSION = 2
# The models reported in the paper, in the order of its models
# subsection.  model_set.py is the single place this set is defined; the
# extraction database may also hold models outside it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_set import MODELS as _REPORTED_MODELS

DEFAULT_MODELS = [repo_id for _tag, repo_id in _REPORTED_MODELS]


@dataclass(frozen=True)
class LayerSpec:
    layer: int
    layout: str
    q_name: str
    k_name: str | None
    q_shard: str
    k_shard: str | None

    @property
    def shards(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(x for x in (self.q_shard, self.k_shard) if x))


@dataclass(frozen=True)
class Architecture:
    model_type: str
    layout: str
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    head_dim: int
    positional_method: str
    attention_bias: bool | None
    attention_dropout: float | None
    scale: float
    rope_theta: float | None
    rope_partial_factor: float
    rope_style: str | None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def parse_layer_selection(text: str | None, n_layers: int) -> set[int]:
    if text is None:
        return set(range(n_layers))
    selected: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(layer for layer in selected if layer < 0 or layer >= n_layers)
    if invalid:
        raise ValueError(f"Layer indices outside [0, {n_layers - 1}]: {invalid}")
    if not selected:
        raise ValueError("--layers selected no layers")
    return selected


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = FULL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            command_json TEXT NOT NULL,
            host TEXT NOT NULL,
            pid INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS models (
            model_pk INTEGER PRIMARY KEY,
            repo_id TEXT NOT NULL,
            requested_revision TEXT,
            resolved_revision TEXT NOT NULL,
            model_type TEXT NOT NULL,
            architecture TEXT NOT NULL,
            positional_method TEXT NOT NULL,
            d_model INTEGER NOT NULL,
            n_layers INTEGER NOT NULL,
            n_heads INTEGER NOT NULL,
            n_kv_heads INTEGER NOT NULL,
            head_dim INTEGER NOT NULL,
            parameter_count INTEGER,
            gated INTEGER,
            license TEXT,
            config_json TEXT NOT NULL,
            repo_metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(repo_id, resolved_revision)
        );
        CREATE TABLE IF NOT EXISTS layers (
            layer_pk INTEGER PRIMARY KEY,
            model_pk INTEGER NOT NULL REFERENCES models(model_pk) ON DELETE CASCADE,
            layer_index INTEGER NOT NULL,
            layout TEXT NOT NULL,
            q_tensor_name TEXT NOT NULL,
            k_tensor_name TEXT,
            source_shards_json TEXT NOT NULL,
            source_shapes_json TEXT,
            source_dtypes_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            error TEXT,
            UNIQUE(model_pk, layer_index)
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_pk INTEGER PRIMARY KEY,
            model_pk INTEGER NOT NULL REFERENCES models(model_pk) ON DELETE CASCADE,
            layer_pk INTEGER NOT NULL UNIQUE REFERENCES layers(layer_pk) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            format TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            storage_dtype TEXT NOT NULL,
            compressed INTEGER NOT NULL,
            byte_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            matrix_shape_json TEXT NOT NULL,
            array_metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS heads (
            head_pk INTEGER PRIMARY KEY,
            model_pk INTEGER NOT NULL REFERENCES models(model_pk) ON DELETE CASCADE,
            layer_pk INTEGER NOT NULL REFERENCES layers(layer_pk) ON DELETE CASCADE,
            artifact_pk INTEGER NOT NULL REFERENCES artifacts(artifact_pk) ON DELETE CASCADE,
            layer_index INTEGER NOT NULL,
            head_index INTEGER NOT NULL,
            kv_head_index INTEGER NOT NULL,
            artifact_array TEXT NOT NULL,
            artifact_index INTEGER NOT NULL,
            d_model INTEGER NOT NULL,
            head_dim INTEGER NOT NULL,
            reduced_dim INTEGER NOT NULL,
            q_norm REAL NOT NULL,
            k_norm REAL NOT NULL,
            bilinear_frobenius_norm REAL NOT NULL,
            trace REAL NOT NULL,
            symmetric_frobenius_norm REAL NOT NULL,
            antisymmetric_frobenius_norm REAL NOT NULL,
            source_scale REAL NOT NULL,
            UNIQUE(model_pk, layer_index, head_index)
        );
        CREATE INDEX IF NOT EXISTS idx_heads_model_layer
            ON heads(model_pk, layer_index);
        CREATE INDEX IF NOT EXISTS idx_layers_status ON layers(status);
        """
    )
    db.execute(
        "INSERT OR REPLACE INTO dataset_metadata(key,value_json) VALUES (?,?)",
        ("schema_version", json_text(SCHEMA_VERSION)),
    )
    db.execute(
        "INSERT OR IGNORE INTO dataset_metadata(key,value_json) VALUES (?,?)",
        (
            "bilinear_convention",
            json_text(
                {
                    "ambient": "L = W_K^T W_Q",
                    "saved": "L_reduced = R^T L R",
                    "basis": "columns of R orthonormally span im(W_K^T)+im(W_Q^T)",
                    "outside_saved_subspace": "zero",
                    "basis_saved": False,
                }
            ),
        ),
    )
    db.commit()
    return db


def config_int(config: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    raise ValueError(f"Configuration lacks all of: {', '.join(names)}")


def positional_method(config: Mapping[str, Any], model_type: str) -> str:
    attn_config = config.get("attn_config") or {}
    if isinstance(attn_config, Mapping) and bool(attn_config.get("alibi")):
        return "alibi"
    if config.get("rope_theta") is not None or config.get("rotary_pct") is not None:
        return "rope"
    position_type = str(config.get("position_embedding_type", "")).lower()
    if "rotary" in position_type or "rope" in position_type:
        return "rope"
    if bool(config.get("alibi")) or model_type == "bloom":
        return "alibi"
    if model_type in {"opt", "gpt2"} or config.get("n_positions") is not None:
        return "learned_absolute"
    return "unknown"


def architecture_from_config(config: Mapping[str, Any]) -> Architecture:
    model_type = str(config.get("model_type", "unknown")).lower()
    d_model = config_int(config, "hidden_size", "n_embd", "n_embed", "d_model")
    n_heads = config_int(config, "num_attention_heads", "n_head", "n_heads")
    n_layers = config_int(config, "num_hidden_layers", "n_layer", "n_layers")
    n_kv_heads = int(config.get("num_key_value_heads", n_heads))
    if d_model % n_heads:
        raise ValueError(f"hidden size {d_model} is not divisible by {n_heads} heads")
    head_dim = int(config.get("head_dim", d_model // n_heads))

    separate_qk_types = {
        "cohere", "gemma", "gemma2", "gemma3_text", "llama", "mistral",
        "mixtral", "olmo", "olmo2", "qwen2", "qwen3",
    }
    if model_type == "opt":
        layout = "separate_qk"
    elif model_type in separate_qk_types:
        layout = "separate_qk"
    elif model_type == "gpt2":
        layout = "gpt2_fused"
    elif model_type == "bloom":
        layout = "bloom_fused_interleaved"
    elif model_type == "gpt_neox":
        layout = "gpt_neox_fused_interleaved"
    elif model_type == "mpt":
        layout = "mpt_fused"
    else:
        # Cerebras-GPT configurations have historically identified as GPT-2;
        # fail closed for unfamiliar layouts rather than silently mis-split Q/K.
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Supported: opt, gpt2, bloom, "
            "gpt_neox, mpt, and Llama-style separate-Q/K architectures."
        )

    dropout = config.get("attention_dropout", config.get("attn_pdrop"))
    return Architecture(
        model_type=model_type,
        layout=layout,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        head_dim=head_dim,
        positional_method=positional_method(config, model_type),
        attention_bias=config.get("enable_bias", config.get("bias")),
        attention_dropout=float(dropout) if dropout is not None else None,
        scale=float(head_dim ** -0.5),
        rope_theta=(
            float(config.get("rope_theta", config.get("rotary_emb_base", 10000.0)))
            if positional_method(config, model_type) == "rope" else None
        ),
        rope_partial_factor=float(config.get("partial_rotary_factor", config.get("rotary_pct", 1.0))),
        rope_style=("split_half" if positional_method(config, model_type) == "rope" else None),
    )


def layer_number(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def discover_layers(
    weight_map: Mapping[str, str], architecture: Architecture
) -> list[LayerSpec]:
    found: dict[int, dict[str, tuple[str, str]]] = {}
    for name, shard in weight_map.items():
        layer = layer_number(name)
        if layer is None:
            continue
        slot = found.setdefault(layer, {})
        if architecture.layout == "separate_qk":
            if name.endswith(".self_attn.q_proj.weight"):
                slot["q"] = (name, shard)
            elif name.endswith(".self_attn.k_proj.weight"):
                slot["k"] = (name, shard)
        elif architecture.layout == "gpt2_fused":
            if name.endswith(".attn.c_attn.weight"):
                slot["qkv"] = (name, shard)
        elif architecture.layout == "bloom_fused_interleaved":
            if name.endswith(".self_attention.query_key_value.weight"):
                slot["qkv"] = (name, shard)
        elif architecture.layout == "gpt_neox_fused_interleaved":
            if name.endswith(".attention.query_key_value.weight"):
                slot["qkv"] = (name, shard)
        elif architecture.layout == "mpt_fused":
            if name.endswith(".attn.Wqkv.weight"):
                slot["qkv"] = (name, shard)

    specs: list[LayerSpec] = []
    for layer in sorted(found):
        slot = found[layer]
        if architecture.layout == "separate_qk" and {"q", "k"} <= slot.keys():
            specs.append(
                LayerSpec(layer, architecture.layout, slot["q"][0], slot["k"][0],
                          slot["q"][1], slot["k"][1])
            )
        elif "qkv" in slot:
            specs.append(
                LayerSpec(layer, architecture.layout, slot["qkv"][0], None,
                          slot["qkv"][1], None)
            )
    expected = set(range(architecture.n_layers))
    actual = {spec.layer for spec in specs}
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(
            f"Found {len(actual)}/{architecture.n_layers} attention layers; "
            f"missing {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )
    return specs


def reject_nonlinear_qk_normalization(weight_map: Mapping[str, str]) -> None:
    """Reject checkpoints whose deployed attention score is not bilinear."""
    q_norms = [name for name in weight_map if name.endswith(".self_attn.q_norm.weight")]
    k_norms = [name for name in weight_map if name.endswith(".self_attn.k_norm.weight")]
    if q_norms or k_norms:
        raise ValueError(
            "Checkpoint applies nonlinear Q/K normalization after q_proj/k_proj; "
            "its deployed attention scores are not fixed bilinear forms. "
            "Pre-normalization Q/K matrices are intentionally not extracted."
        )


def tensor_map_from_file(path: Path) -> dict[str, str]:
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {name: path.name for name in handle.keys()}
    state = load_pytorch_state(path)
    try:
        return {name: path.name for name in state}
    finally:
        del state


def download_file(
    repo_id: str, filename: str, revision: str, download_dir: Path, token: str | None
) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        local_dir=str(download_dir),
    )
    return Path(path)


def load_pytorch_state(path: Path) -> Mapping[str, torch.Tensor]:
    """Safely memory-map a weights-only PyTorch checkpoint.

    ``weights_only=True`` prevents execution of arbitrary pickled code.
    ``mmap=True`` prevents an unsharded multi-gigabyte file from being copied
    wholesale into RAM. PyTorch raises a clear error for legacy, non-zip files
    that cannot be memory-mapped; silently loading those is intentionally
    avoided.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError as exc:
        raise RuntimeError(
            "PyTorch >= 2.1 is required for safe memory-mapped .bin checkpoints"
        ) from exc
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dictionary in {path}, got {type(state).__name__}")
    return state


def load_tensor(path: Path, name: str) -> torch.Tensor:
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)
    if path.suffix == ".bin":
        state = load_pytorch_state(path)
        try:
            if name not in state:
                raise KeyError(f"Tensor {name!r} not found in {path.name}")
            # Keep the mapped storage alive through the returned tensor. The
            # other state-dictionary tensor objects can be released at once.
            return state[name]
        finally:
            del state
    raise ValueError(f"Unsupported checkpoint file type: {path.name}")


def split_qk(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor | None,
    architecture: Architecture,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q [n_q,h,d] and K [n_kv,h,d] in output-by-input convention."""
    d = architecture.d_model
    h = architecture.head_dim
    nq = architecture.n_heads
    nk = architecture.n_kv_heads

    if architecture.layout == "separate_qk":
        if k_raw is None:
            raise ValueError("Separate-Q/K layout requires a K tensor")
        q = q_raw
        k = k_raw
        if q.shape == (d, nq * h):
            q = q.T
        if k.shape == (d, nk * h):
            k = k.T
        if q.shape != (nq * h, d) or k.shape != (nk * h, d):
            raise ValueError(f"Unexpected separate Q/K shapes: {tuple(q.shape)}, {tuple(k.shape)}")
        return q.reshape(nq, h, d), k.reshape(nk, h, d)

    if architecture.layout == "gpt2_fused":
        fused = q_raw
        # GPT-2 Conv1D stores [input, output]. Some compatible checkpoints use
        # an ordinary Linear [output, input], so accept both unambiguously.
        if fused.shape == (d, 3 * d):
            fused = fused.T
        elif fused.shape != (3 * d, d):
            raise ValueError(f"Unexpected GPT-2 fused shape: {tuple(fused.shape)}")
        q, k, _v = fused.split(d, dim=0)
        return q.reshape(nq, h, d), k.reshape(nk, h, d)

    if architecture.layout == "mpt_fused":
        fused = q_raw
        if fused.shape == (d, 3 * d):
            fused = fused.T
        elif fused.shape != (3 * d, d):
            raise ValueError(f"Unexpected MPT fused shape: {tuple(fused.shape)}")
        q, k, _v = fused.split(d, dim=0)
        return q.reshape(nq, h, d), k.reshape(nk, h, d)

    if architecture.layout in {"bloom_fused_interleaved", "gpt_neox_fused_interleaved"}:
        fused = q_raw
        if fused.shape == (d, (nq + 2 * nk) * h):
            fused = fused.T
        if fused.shape != ((nq + 2 * nk) * h, d):
            raise ValueError(f"Unexpected interleaved fused shape: {tuple(fused.shape)}")
        if nq != nk:
            raise ValueError("Interleaved fused grouped-query attention is not yet supported")
        packed = fused.reshape(nq, 3, h, d)
        return packed[:, 0], packed[:, 1]

    raise AssertionError(architecture.layout)


def reduced_forms(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    architecture: Architecture,
    compute_dtype: torch.dtype,
    storage_dtype: np.dtype,
    progress: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[dict[str, Any]]]:
    reduced_dim = 2 * architecture.head_dim
    forms = np.empty(
        (architecture.n_heads, reduced_dim, reduced_dim), dtype=storage_dtype
    )
    q_factors = (
        np.empty((architecture.n_heads, architecture.head_dim, reduced_dim), dtype=storage_dtype)
        if architecture.positional_method == "rope" else None
    )
    k_factors = np.empty_like(q_factors) if q_factors is not None else None
    rows: list[dict[str, Any]] = []
    group_size = architecture.n_heads // architecture.n_kv_heads
    if architecture.n_heads % architecture.n_kv_heads:
        raise ValueError("Q head count must be divisible by K/V head count")

    for head in range(architecture.n_heads):
        kv_head = head // group_size
        wq = q_heads[head].to(dtype=compute_dtype)
        wk = k_heads[kv_head].to(dtype=compute_dtype)
        basis = torch.linalg.qr(torch.cat((wk.T, wq.T), dim=1), mode="reduced").Q
        kr = wk @ basis
        qr = wq @ basis
        form = kr.T @ qr
        forms[head] = form.cpu().numpy().astype(storage_dtype, copy=False)
        if q_factors is not None and k_factors is not None:
            q_factors[head] = qr.cpu().numpy().astype(storage_dtype, copy=False)
            k_factors[head] = kr.cpu().numpy().astype(storage_dtype, copy=False)
        symmetric = 0.5 * (form + form.T)
        antisymmetric = 0.5 * (form - form.T)
        rows.append(
            {
                "head_index": head,
                "kv_head_index": kv_head,
                "reduced_dim": int(form.shape[0]),
                "q_norm": float(torch.linalg.vector_norm(wq)),
                "k_norm": float(torch.linalg.vector_norm(wk)),
                "bilinear_frobenius_norm": float(torch.linalg.matrix_norm(form, ord="fro")),
                "trace": float(torch.trace(form)),
                "symmetric_frobenius_norm": float(torch.linalg.matrix_norm(symmetric, ord="fro")),
                "antisymmetric_frobenius_norm": float(torch.linalg.matrix_norm(antisymmetric, ord="fro")),
            }
        )
        if progress is not None:
            progress.update(1)
    return forms, q_factors, k_factors, rows


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_completed_layer(db: sqlite3.Connection, layer_pk: int, root: Path) -> bool:
    row = db.execute(
        """SELECT l.status, a.relative_path, a.byte_size, a.sha256
           FROM layers l LEFT JOIN artifacts a ON a.layer_pk=l.layer_pk
           WHERE l.layer_pk=?""",
        (layer_pk,),
    ).fetchone()
    if row is None or row["status"] != "complete" or row["relative_path"] is None:
        return False
    path = root / row["relative_path"]
    if not path.is_file() or path.stat().st_size != row["byte_size"]:
        return False
    return sha256_file(path) == row["sha256"]


def upsert_model(
    db: sqlite3.Connection,
    repo_id: str,
    requested_revision: str | None,
    resolved_revision: str,
    config: Mapping[str, Any],
    architecture: Architecture,
    repo_metadata: Mapping[str, Any],
) -> int:
    now = utc_now()
    db.execute(
        """
        INSERT INTO models(
          repo_id,requested_revision,resolved_revision,model_type,architecture,
          positional_method,d_model,n_layers,n_heads,n_kv_heads,head_dim,
          parameter_count,gated,license,config_json,repo_metadata_json,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(repo_id,resolved_revision) DO UPDATE SET
          requested_revision=excluded.requested_revision,
          positional_method=excluded.positional_method,
          parameter_count=excluded.parameter_count,
          gated=excluded.gated, license=excluded.license,
          config_json=excluded.config_json,
          repo_metadata_json=excluded.repo_metadata_json,
          updated_at=excluded.updated_at
        """,
        (
            repo_id, requested_revision, resolved_revision, architecture.model_type,
            architecture.layout, architecture.positional_method, architecture.d_model,
            architecture.n_layers, architecture.n_heads, architecture.n_kv_heads,
            architecture.head_dim, repo_metadata.get("parameter_count"),
            repo_metadata.get("gated"), repo_metadata.get("license"),
            json_text(config), json_text(repo_metadata), now, now,
        ),
    )
    row = db.execute(
        "SELECT model_pk FROM models WHERE repo_id=? AND resolved_revision=?",
        (repo_id, resolved_revision),
    ).fetchone()
    assert row is not None
    return int(row[0])


def ensure_layer(db: sqlite3.Connection, model_pk: int, spec: LayerSpec) -> int:
    db.execute(
        """
        INSERT INTO layers(model_pk,layer_index,layout,q_tensor_name,k_tensor_name,
                           source_shards_json,status)
        VALUES (?,?,?,?,?,?,'pending')
        ON CONFLICT(model_pk,layer_index) DO UPDATE SET
          layout=excluded.layout,q_tensor_name=excluded.q_tensor_name,
          k_tensor_name=excluded.k_tensor_name,source_shards_json=excluded.source_shards_json
        """,
        (model_pk, spec.layer, spec.layout, spec.q_name, spec.k_name, json_text(spec.shards)),
    )
    row = db.execute(
        "SELECT layer_pk FROM layers WHERE model_pk=? AND layer_index=?",
        (model_pk, spec.layer),
    ).fetchone()
    assert row is not None
    return int(row[0])


def record_layer(
    db: sqlite3.Connection,
    root: Path,
    model_pk: int,
    layer_pk: int,
    spec: LayerSpec,
    architecture: Architecture,
    path: Path,
    storage_dtype: str,
    compressed: bool,
    source_shapes: Mapping[str, Any],
    source_dtypes: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    relative = path.relative_to(root).as_posix()
    byte_size = path.stat().st_size
    digest = sha256_file(path)
    array_metadata = {
        "L": "R.T @ (W_K.T @ W_Q) @ R; first index is head_index",
        "kv_head_index": "K/V head used by each query head",
        "attention_scale": "architectural score multiplier 1/sqrt(head_dim); not applied to L",
    }
    if architecture.positional_method == "rope":
        array_metadata.update({
            "Q_reduced": "W_Q @ R; retained so relative-position forms can be reconstructed",
            "K_reduced": "W_K @ R; retained so relative-position forms can be reconstructed",
            "relative_form": "K_reduced.T @ RoPE(d) @ Q_reduced",
            "rope_theta": architecture.rope_theta,
            "rope_partial_factor": architecture.rope_partial_factor,
            "rope_style": architecture.rope_style,
        })
    now = utc_now()
    with db:
        db.execute("DELETE FROM artifacts WHERE layer_pk=?", (layer_pk,))
        cursor = db.execute(
            """
            INSERT INTO artifacts(model_pk,layer_pk,kind,relative_path,format,format_version,
              storage_dtype,compressed,byte_size,sha256,matrix_shape_json,array_metadata_json,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                model_pk, layer_pk, "reduced_bilinear_forms", relative, "npz",
                FORMAT_VERSION, storage_dtype, int(compressed), byte_size, digest,
                json_text([architecture.n_heads, 2 * architecture.head_dim,
                           2 * architecture.head_dim]),
                json_text(array_metadata), now,
            ),
        )
        artifact_pk = int(cursor.lastrowid)
        db.execute("DELETE FROM heads WHERE layer_pk=?", (layer_pk,))
        db.executemany(
            """
            INSERT INTO heads(model_pk,layer_pk,artifact_pk,layer_index,head_index,
              kv_head_index,artifact_array,artifact_index,d_model,head_dim,reduced_dim,
              q_norm,k_norm,bilinear_frobenius_norm,trace,symmetric_frobenius_norm,
              antisymmetric_frobenius_norm,source_scale)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    model_pk, layer_pk, artifact_pk, spec.layer, row["head_index"],
                    row["kv_head_index"], "L", row["head_index"], architecture.d_model,
                    architecture.head_dim, row["reduced_dim"], row["q_norm"], row["k_norm"],
                    row["bilinear_frobenius_norm"], row["trace"],
                    row["symmetric_frobenius_norm"], row["antisymmetric_frobenius_norm"],
                    architecture.scale,
                )
                for row in rows
            ],
        )
        db.execute(
            """UPDATE layers SET status='complete',completed_at=?,error=NULL,
               source_shapes_json=?,source_dtypes_json=? WHERE layer_pk=?""",
            (now, json_text(source_shapes), json_text(source_dtypes), layer_pk),
        )


def repository_metadata(info: Any) -> dict[str, Any]:
    card = getattr(info, "card_data", None)
    card_dict = card.to_dict() if hasattr(card, "to_dict") else (dict(card) if card else {})
    safetensors_info = getattr(info, "safetensors", None)
    parameters = getattr(safetensors_info, "parameters", None)
    parameter_count = sum(parameters.values()) if isinstance(parameters, dict) else None
    return {
        "sha": info.sha,
        "last_modified": str(getattr(info, "last_modified", None)),
        "gated": bool(getattr(info, "gated", False)),
        "private": bool(getattr(info, "private", False)),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or []),
        "license": card_dict.get("license"),
        "parameter_count": parameter_count,
    }


def choose_weight_files(info: Any) -> tuple[str | None, str | None]:
    names = {item.rfilename for item in info.siblings}
    if "model.safetensors.index.json" in names:
        return "model.safetensors.index.json", None
    if "model.safetensors" in names:
        return None, "model.safetensors"
    index = (
        "pytorch_model.bin.index.json"
        if "pytorch_model.bin.index.json" in names
        else None
    )
    single = "pytorch_model.bin" if "pytorch_model.bin" in names else None
    if not index and not single:
        raise ValueError("No supported safetensors or PyTorch weight checkpoint found")
    return index, single


def estimate_peak_shard_bytes(
    pending: Sequence[tuple[LayerSpec, int]], file_sizes: Mapping[str, int]
) -> int | None:
    """Estimate peak live checkpoint bytes under the sequential eviction policy."""
    if any(shard not in file_sizes for spec, _ in pending for shard in spec.shards):
        return None
    remaining: dict[str, int] = {}
    for spec, _ in pending:
        for shard in spec.shards:
            remaining[shard] = remaining.get(shard, 0) + 1
    live: set[str] = set()
    peak = 0
    for spec, _ in pending:
        live.update(spec.shards)
        peak = max(peak, sum(file_sizes[shard] for shard in live))
        for shard in spec.shards:
            remaining[shard] -= 1
            if remaining[shard] == 0:
                live.remove(shard)
    return peak


def preflight_disk_space(
    args: argparse.Namespace,
    architecture: Architecture,
    pending: Sequence[tuple[LayerSpec, int]],
    file_sizes: Mapping[str, int],
) -> None:
    bytes_per_value = np.dtype(args.storage_dtype).itemsize
    matrix_values = (
        len(pending) * architecture.n_heads * (2 * architecture.head_dim) ** 2
    )
    if architecture.positional_method == "rope":
        # Q_reduced and K_reduced together contain the same number of values as L.
        matrix_values *= 2
    # NPZ metadata is negligible beside the matrices. Compression can only
    # reduce this estimate, so the uncompressed size is a safe upper bound.
    artifact_upper_bound = matrix_values * bytes_per_value + len(pending) * 64 * 1024
    peak_download = estimate_peak_shard_bytes(pending, file_sizes)
    output_root = Path(args.out)
    download_root = Path(args.download_dir)
    output_free = shutil.disk_usage(output_root).free
    download_root.mkdir(parents=True, exist_ok=True)
    download_free = shutil.disk_usage(download_root).free
    same_device = output_root.stat().st_dev == download_root.stat().st_dev
    margin = int(args.disk_margin_gib * 1024 ** 3)
    artifact_gib = artifact_upper_bound / 1024 ** 3
    peak_text = "unknown" if peak_download is None else f"{peak_download / 1024 ** 3:.1f} GiB"
    print(
        f"Disk estimate: at most {artifact_gib:.1f} GiB new artifacts; "
        f"about {peak_text} peak temporary checkpoint data; "
        f"{output_free / 1024 ** 3:.1f} GiB free at output",
        flush=True,
    )
    if args.skip_disk_check or peak_download is None:
        return
    if same_device:
        required = artifact_upper_bound + peak_download + margin
        available = output_free
        location = output_root
    else:
        if artifact_upper_bound + margin > output_free:
            raise OSError(
                f"Insufficient output space: need about "
                f"{(artifact_upper_bound + margin) / 1024 ** 3:.1f} GiB, "
                f"have {output_free / 1024 ** 3:.1f} GiB"
            )
        required = peak_download + margin
        available = download_free
        location = download_root
    if required > available:
        raise OSError(
            f"Insufficient free space on {location}: need about "
            f"{required / 1024 ** 3:.1f} GiB including the safety margin, "
            f"have {available / 1024 ** 3:.1f} GiB. Choose another --download-dir/--out "
            f"or, if you have independently checked capacity, pass --skip-disk-check."
        )


def process_model(args: argparse.Namespace, db: sqlite3.Connection, repo_id: str) -> None:
    print(f"\n[{repo_id}] Resolving repository metadata", flush=True)
    api = HfApi(token=args.token)
    info = api.model_info(repo_id, revision=args.revision, files_metadata=True)
    resolved_revision = info.sha
    metadata = repository_metadata(info)
    model_download_dir = Path(args.download_dir) / safe_name(repo_id) / resolved_revision
    config_path = download_file(repo_id, "config.json", resolved_revision, model_download_dir, args.token)
    config = json.loads(config_path.read_text())
    architecture = architecture_from_config(config)
    if architecture.positional_method == "rope" and not args.allow_rope:
        raise ValueError("RoPE checkpoint rejected (use --allow-rope to override)")
    if architecture.positional_method == "unknown" and not args.allow_unknown_position:
        raise ValueError("Unknown positional method (use --allow-unknown-position to override)")

    model_pk = upsert_model(
        db, repo_id, args.revision, resolved_revision, config, architecture, metadata
    )
    index_name, single_name = choose_weight_files(info)
    if index_name:
        index_path = download_file(
            repo_id, index_name, resolved_revision, model_download_dir, args.token
        )
        index_json = json.loads(index_path.read_text())
        weight_map = index_json["weight_map"]
    else:
        assert single_name
        print(f"[{repo_id}] Downloading unsharded checkpoint to inspect tensor names", flush=True)
        single_path = download_file(
            repo_id, single_name, resolved_revision, model_download_dir, args.token
        )
        weight_map = tensor_map_from_file(single_path)

    reject_nonlinear_qk_normalization(weight_map)
    all_specs = discover_layers(weight_map, architecture)
    selected_layers = parse_layer_selection(args.layers, architecture.n_layers)
    specs = [spec for spec in all_specs if spec.layer in selected_layers]
    layer_rows = [(spec, ensure_layer(db, model_pk, spec)) for spec in specs]
    db.commit()
    pending: list[tuple[LayerSpec, int]] = []
    for spec, layer_pk in layer_rows:
        if valid_completed_layer(db, layer_pk, Path(args.out)):
            continue
        pending.append((spec, layer_pk))

    total_heads = len(specs) * architecture.n_heads
    pending_heads = len(pending) * architecture.n_heads
    print(
        f"[{repo_id}] selected {len(specs)}/{architecture.n_layers} layers x "
        f"{architecture.n_heads} heads "
        f"(d_model={architecture.d_model}, d_head={architecture.head_dim}, "
        f"position={architecture.positional_method}); "
        f"{total_heads - pending_heads:,}/{total_heads:,} heads already complete",
        flush=True,
    )
    if not pending:
        return

    file_sizes = {
        item.rfilename: int(item.size)
        for item in info.siblings
        if getattr(item, "size", None) is not None
    }
    preflight_disk_space(args, architecture, pending, file_sizes)

    remaining_uses: dict[str, int] = {}
    for spec, _ in pending:
        for shard in spec.shards:
            remaining_uses[shard] = remaining_uses.get(shard, 0) + 1

    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    storage_dtype = np.dtype(args.storage_dtype)
    model_artifact_dir = Path(args.out) / "bilinear_forms" / safe_name(repo_id) / resolved_revision
    with tqdm(total=pending_heads, desc=repo_id, unit="head", dynamic_ncols=True) as progress:
        for spec, layer_pk in pending:
            started = time.monotonic()
            db.execute(
                """UPDATE layers SET status='running',attempts=attempts+1,
                   started_at=?,completed_at=NULL,error=NULL WHERE layer_pk=?""",
                (utc_now(), layer_pk),
            )
            db.commit()
            try:
                shard_paths = {
                    shard: download_file(
                        repo_id, shard, resolved_revision, model_download_dir, args.token
                    )
                    for shard in spec.shards
                }
                q_raw = load_tensor(shard_paths[spec.q_shard], spec.q_name)
                k_raw = (
                    load_tensor(shard_paths[spec.k_shard], spec.k_name)
                    if spec.k_name is not None and spec.k_shard is not None
                    else None
                )
                source_shapes = {spec.q_name: list(q_raw.shape)}
                source_dtypes = {spec.q_name: str(q_raw.dtype)}
                if k_raw is not None and spec.k_name is not None:
                    source_shapes[spec.k_name] = list(k_raw.shape)
                    source_dtypes[spec.k_name] = str(k_raw.dtype)
                q_heads, k_heads = split_qk(q_raw, k_raw, architecture)
                del q_raw, k_raw
                forms, q_factors, k_factors, head_rows = reduced_forms(
                    q_heads, k_heads, architecture, compute_dtype, storage_dtype, progress
                )
                del q_heads, k_heads
                kv_indices = np.asarray(
                    [row["kv_head_index"] for row in head_rows], dtype=np.int32
                )
                artifact_path = model_artifact_dir / f"layer_{spec.layer:04d}.npz"
                artifact_arrays = {
                        "L": forms,
                        "kv_head_index": kv_indices,
                        "attention_scale": np.asarray(architecture.scale, dtype=np.float64),
                        "ambient_dimension": np.asarray(architecture.d_model, dtype=np.int32),
                        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
                    }
                if q_factors is not None and k_factors is not None:
                    artifact_arrays.update({
                        "Q_reduced": q_factors,
                        "K_reduced": k_factors,
                        "rope_theta": np.asarray(architecture.rope_theta, dtype=np.float64),
                        "rope_partial_factor": np.asarray(
                            architecture.rope_partial_factor, dtype=np.float64
                        ),
                    })
                atomic_save_npz(artifact_path, artifact_arrays, args.compress)
                del forms, q_factors, k_factors, artifact_arrays
                record_layer(
                    db, Path(args.out), model_pk, layer_pk, spec, architecture,
                    artifact_path, args.storage_dtype, args.compress, source_shapes,
                    source_dtypes, head_rows,
                )
                elapsed = time.monotonic() - started
                progress.set_postfix(layer=spec.layer, seconds=f"{elapsed:.1f}")
                for shard in spec.shards:
                    remaining_uses[shard] -= 1
                    if remaining_uses[shard] == 0 and not args.keep_downloads:
                        shard_paths[shard].unlink(missing_ok=True)
            except Exception:
                message = traceback.format_exc()
                db.execute(
                    "UPDATE layers SET status='failed',error=? WHERE layer_pk=?",
                    (message, layer_pk),
                )
                db.commit()
                print(f"\n[{repo_id}] FAILED layer {spec.layer}:\n{message}", file=sys.stderr)
                if not args.continue_on_error:
                    raise

    if not args.keep_downloads:
        # Remove empty directories and small hub bookkeeping files only after
        # all needed shards are gone. The durable config/index provenance is in SQLite.
        with contextlib.suppress(OSError):
            shutil.rmtree(model_download_dir)


def run_command(args: argparse.Namespace) -> int:
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.out = str(root)
    if args.download_dir is None:
        args.download_dir = str(root / ".downloads")
    else:
        args.download_dir = str(Path(args.download_dir).resolve())
    db = connect_db(root / "heads.sqlite3")
    run_id = str(uuid.uuid4())
    command = vars(args).copy()
    command.pop("function", None)
    if command.get("token"):
        command["token"] = "<redacted>"
    db.execute(
        "INSERT INTO runs(run_id,command_json,host,pid,started_at,status) VALUES (?,?,?,?,?,'running')",
        (run_id, json_text(command), platform.node(), os.getpid(), utc_now()),
    )
    db.commit()
    failures: list[str] = []
    try:
        for model in args.models:
            try:
                process_model(args, db, model)
            except Exception as exc:
                failures.append(f"{model}: {exc}")
                print(f"\nERROR [{model}]: {exc}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
        status = "complete" if not failures else "completed_with_errors"
        db.execute(
            "UPDATE runs SET status=?,finished_at=?,error=? WHERE run_id=?",
            (status, utc_now(), "\n".join(failures) or None, run_id),
        )
        db.commit()
    except (KeyboardInterrupt, SystemExit):
        db.execute(
            "UPDATE runs SET status='interrupted',finished_at=?,error=? WHERE run_id=?",
            (utc_now(), traceback.format_exc(), run_id),
        )
        db.commit()
        raise
    except Exception:
        db.execute(
            "UPDATE runs SET status='failed',finished_at=?,error=? WHERE run_id=?",
            (utc_now(), traceback.format_exc(), run_id),
        )
        db.commit()
        raise
    finally:
        db.close()
    print(f"\nDataset: {root / 'heads.sqlite3'}")
    return 1 if failures else 0


def status_command(args: argparse.Namespace) -> int:
    root = Path(args.out).resolve()
    db_path = root / "heads.sqlite3"
    if not db_path.exists():
        print(f"No database at {db_path}")
        return 1
    db = connect_db(db_path)
    rows = db.execute(
        """
        SELECT m.repo_id,substr(m.resolved_revision,1,12) AS revision,
          m.positional_method,m.d_model,m.head_dim,m.n_layers,m.n_heads,
          (SELECT COUNT(*) FROM layers l WHERE l.model_pk=m.model_pk
             AND l.status='complete') AS complete_layers,
          (SELECT COUNT(*) FROM layers l WHERE l.model_pk=m.model_pk
             AND l.status='failed') AS failed_layers,
          (SELECT COUNT(*) FROM heads h JOIN layers l ON l.layer_pk=h.layer_pk
             WHERE h.model_pk=m.model_pk AND l.status='complete') AS stored_heads,
          (SELECT COALESCE(SUM(a.byte_size),0) FROM artifacts a
             WHERE a.model_pk=m.model_pk) AS artifact_bytes
        FROM models m ORDER BY m.repo_id
        """
    ).fetchall()
    if not rows:
        print("Database contains no models.")
        return 0
    header = "model  revision  position  d_model/d_head  layers  heads  disk"
    print(header)
    print("-" * len(header))
    for row in rows:
        disk = float(row["artifact_bytes"]) / (1024 ** 3)
        print(
            f"{row['repo_id']}  {row['revision']}  {row['positional_method']}  "
            f"{row['d_model']}/{row['head_dim']}  "
            f"{row['complete_layers']}/{row['n_layers']} "
            f"({row['failed_layers']} failed)  {row['stored_heads']:,}/"
            f"{row['n_layers'] * row['n_heads']:,}  {disk:.2f} GiB"
        )
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restartable, disk-conscious extraction of attention bilinear forms."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Extract one or more models.")
    run.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    run.add_argument("--out", required=True, help="Dataset directory (database + artifacts).")
    run.add_argument("--download-dir", help="Temporary checkpoint directory (default: OUT/.downloads).")
    run.add_argument("--revision", help="Branch, tag, or commit; resolved commit is always recorded.")
    run.add_argument("--layers",
                     help="Optional layer selection such as 0-9,20,30-39 (default: all).")
    run.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token; defaults to HF_TOKEN.")
    run.add_argument("--compute-dtype", choices=("float32", "float64"), default="float64")
    run.add_argument("--storage-dtype", choices=("float32", "float64"), default="float32")
    run.add_argument("--compress", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--keep-downloads", action="store_true", help="Retain downloaded checkpoint shards.")
    run.add_argument("--disk-margin-gib", type=float, default=5.0,
                     help="Free-space safety margin used by the preflight check (default: 5).")
    run.add_argument("--skip-disk-check", action="store_true",
                     help="Proceed even when the conservative space estimate is too large.")
    run.add_argument("--allow-rope", action="store_true", help="Permit models detected as using RoPE.")
    run.add_argument("--allow-unknown-position", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    run.set_defaults(function=run_command)

    status = subparsers.add_parser("status", help="Show extraction progress from the database.")
    status.add_argument("--out", required=True, help="Dataset directory.")
    status.set_defaults(function=status_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
