#!/usr/bin/env python3
"""Check the extraction database against the raw weights, one layer per model.

For each model this reads the query and key projection tensors of one layer
directly from the checkpoint (a local safetensors file, or HTTP range requests
on the hub, so no shard is downloaded), slices the heads following the
Hugging Face modeling code of that architecture, forms ``L = W_K^T W_Q`` and
compares its singular values with those of the stored reduced form.  The
reduced form is an orthogonal representative of ``L``, so the nonzero
singular values must agree.

This is independent of ``extract_qk_bilinear_db.py``: the head slicing is
written out here from the modeling conventions rather than imported.

    nn.Linear  weight [out, in]: head i is rows  i*h:(i+1)*h   (OPT, Llama-style)
    GPT-2 Conv1D c_attn [in, 3*d]: q = x @ W[:, :d]; head i is columns i*h:(i+1)*h
    BLOOM / GPT-NeoX fused [3*d, in]: reshape to (heads, 3, h, in); q = [:, 0], k = [:, 1]

Usage:
    python verify_extraction.py                 # every model in the database
    python verify_extraction.py --layer 3 gpt2 facebook/opt-125m
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

TOL = 1e-5


def hub_tensor(repo: str, revision: str, names: tuple[str, ...]) -> np.ndarray:
    """One tensor by range requests on the repository's safetensors files.

    ``names`` are the candidate tensor names (checkpoints differ in whether
    they carry a ``transformer.`` prefix); exactly one must be present.
    """
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    r = requests.get(base + "model.safetensors.index.json")
    if r.status_code == 200:
        weight_map = r.json()["weight_map"]
        present = [n for n in names if n in weight_map]
        if len(present) != 1:
            raise KeyError(f"{repo}: expected one of {names} in the index, found {present}")
        name, shard = present[0], weight_map[present[0]]
        url = base + shard
        n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}).content)[0]
        header = json.loads(requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}).content)
    else:
        url = base + "model.safetensors"
        n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}).content)[0]
        header = json.loads(requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}).content)
        present = [n_ for n_ in names if n_ in header]
        if len(present) != 1:
            raise KeyError(f"{repo}: expected one of {names} in the checkpoint, found {present}")
        name = present[0]
    meta = header[name]
    a, b = meta["data_offsets"]
    raw = requests.get(url, headers={"Range": f"bytes={8 + n + a}-{8 + n + b - 1}"}).content
    if meta["dtype"] == "BF16":
        v = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif meta["dtype"] == "F16":
        v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif meta["dtype"] == "F32":
        v = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"{name}: unsupported dtype {meta['dtype']}")
    return v.reshape(meta["shape"]).astype(np.float64)


def bin_tensor(repo: str, revision: str, names: tuple[str, ...], sharded: bool) -> np.ndarray:
    """One tensor from a pytorch_model.bin checkpoint (the shard holding it is downloaded whole)."""
    import torch
    if sharded:
        with open(hf_hub_download(repo, "pytorch_model.bin.index.json", revision=revision)) as f:
            weight_map = json.load(f)["weight_map"]
        present = [n for n in names if n in weight_map]
        if len(present) != 1:
            raise KeyError(f"{repo}: expected one of {names} in the index, found {present}")
        path = hf_hub_download(repo, weight_map[present[0]], revision=revision)
    else:
        path = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    present = [n for n in names if n in state]
    if len(present) != 1:
        raise KeyError(f"{repo}: expected one of {names} in the checkpoint, found {present}")
    return state[present[0]].to(torch.float64).numpy()


def tensor(repo: str, revision: str, *names: str) -> np.ndarray:
    files = {s.rfilename for s in HfApi().model_info(repo, revision=revision).siblings}
    if any(f.endswith(".safetensors") for f in files):
        return hub_tensor(repo, revision, names)
    if "pytorch_model.bin" in files:
        return bin_tensor(repo, revision, names, sharded=False)
    if "pytorch_model.bin.index.json" in files:
        return bin_tensor(repo, revision, names, sharded=True)
    raise ValueError(f"{repo}: no safetensors or pytorch_model.bin")


def heads_from_raw(model_type: str, repo: str, revision: str, layer: int,
                   nq: int, nk: int, h: int, d: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Q [nq, h, d] and K [nk, h, d] sliced as the modeling code does."""
    if model_type == "opt":
        p = f"decoder.layers.{layer}.self_attn"                       # old .bin checkpoints drop "model."
        wq = tensor(repo, revision, f"model.{p}.q_proj.weight", f"{p}.q_proj.weight")
        wk = tensor(repo, revision, f"model.{p}.k_proj.weight", f"{p}.k_proj.weight")
        return wq.reshape(nq, h, d), wk.reshape(nk, h, d)
    if model_type in {"mistral", "qwen3", "llama", "glm4_moe"}:
        p = f"model.layers.{layer}.self_attn"
        wq = tensor(repo, revision, f"{p}.q_proj.weight")
        wk = tensor(repo, revision, f"{p}.k_proj.weight")
        return wq.reshape(nq, h, d), wk.reshape(nk, h, d)
    if model_type == "gpt2":
        w = tensor(repo, revision, f"h.{layer}.attn.c_attn.weight",
                   f"transformer.h.{layer}.attn.c_attn.weight")     # Conv1D [in, 3d]
        if w.shape != (d, 3 * d):
            raise ValueError(f"{repo}: c_attn has shape {w.shape}, expected {(d, 3 * d)}")
        q, k = w[:, :d].T, w[:, d:2 * d].T                            # [out, in]
        return q.reshape(nq, h, d), k.reshape(nk, h, d)
    if model_type == "bloom":
        w = tensor(repo, revision, f"h.{layer}.self_attention.query_key_value.weight",
                   f"transformer.h.{layer}.self_attention.query_key_value.weight")
        if w.shape != (3 * d, d):
            raise ValueError(f"{repo}: query_key_value has shape {w.shape}, expected {(3 * d, d)}")
        packed = w.reshape(nq, 3, h, d)
        return packed[:, 0], packed[:, 1]
    if model_type in {"deepseek_v2", "deepseek_v3"}:
        # Multi-head latent attention, written out from DeepSeek's modeling code:
        #   q_h = W_Q,h x; [c; k_R] = kv_a_proj_with_mqa x; k_nope,h = W_UK,h RMSNorm(c)
        # with the RMSNorm gain absorbed and the per-token scalar omitted.  The
        # singular values of K^T Q do not depend on how the coordinates are
        # ordered, so the rotary reordering done by the extractor is irrelevant here.
        n_nope, n_rope, r = (int(cfg[k]) for k in ("qk_nope_head_dim", "qk_rope_head_dim", "kv_lora_rank"))
        if h != n_nope + n_rope or nq != nk:
            raise ValueError(f"{repo}: head_dim {h} != {n_nope} + {n_rope}, or {nq} != {nk} kv heads")
        p = f"model.layers.{layer}.self_attn"
        wq = tensor(repo, revision, f"{p}.q_proj.weight").reshape(nq, h, d)
        kv_a = tensor(repo, revision, f"{p}.kv_a_proj_with_mqa.weight")
        gamma = tensor(repo, revision, f"{p}.kv_a_layernorm.weight")
        kv_b = tensor(repo, revision, f"{p}.kv_b_proj.weight")
        if kv_a.shape != (r + n_rope, d) or gamma.shape != (r,) or kv_b.shape[1] != r:
            raise ValueError(f"{repo}: unexpected MLA shapes {kv_a.shape}, {gamma.shape}, {kv_b.shape}")
        w_uk = kv_b.reshape(nq, -1, r)[:, :n_nope, :]
        k = np.concatenate((w_uk @ (gamma[:, None] * kv_a[:r]), np.broadcast_to(kv_a[r:], (nq, n_rope, d))), axis=1)
        return wq, k
    if model_type == "gpt_neox":
        w = tensor(repo, revision, f"gpt_neox.layers.{layer}.attention.query_key_value.weight")
        if w.shape != (3 * d, d):
            raise ValueError(f"{repo}: query_key_value has shape {w.shape}, expected {(3 * d, d)}")
        packed = w.reshape(nq, 3, h, d)
        return packed[:, 0], packed[:, 1]
    raise ValueError(f"unsupported model_type {model_type}")


def reduced_singular_values(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    basis = np.linalg.qr(np.concatenate((k.T, q.T), axis=1))[0]
    return np.linalg.svd((k @ basis).T @ (q @ basis), compute_uv=False)


def check(repo: str, layer: int) -> bool:
    with ms._connect() as con:
        m = con.execute("select * from models where repo_id=?", (repo,)).fetchone()
        if m is None:
            raise KeyError(repo)
        art = con.execute(
            """select a.relative_path from artifacts a join layers l using(layer_pk)
               where a.model_pk=? and l.layer_index=?""", (m["model_pk"], layer)).fetchone()
    if art is None:
        print(f"{repo:40s} layer {layer}: not extracted")
        return False
    with np.load(ms.DATASET / art["relative_path"]) as z:
        stored = z["L"].astype(np.float64)
    nq, nk, h, d = m["n_heads"], m["n_kv_heads"], m["head_dim"], m["d_model"]
    Q, K = heads_from_raw(m["model_type"], repo, m["resolved_revision"], layer, nq, nk, h, d,
                          json.loads(m["config_json"]))
    group = nq // nk
    errs = []
    for i in range(nq):
        s_raw = reduced_singular_values(Q[i], K[i // group])
        s_db = np.linalg.svd(stored[i], compute_uv=False)
        errs.append(np.linalg.norm(s_raw - s_db) / np.linalg.norm(s_raw))
    worst = max(errs)
    ok = worst < TOL
    print(f"{repo:40s} layer {layer:2d}  {nq:3d} heads  max rel. singular-value error {worst:.1e}  {'ok' if ok else 'MISMATCH'}",
          flush=True)
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repos", nargs="*", help="repository ids (default: every model in the database)")
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    repos = args.repos
    if not repos:
        with ms._connect() as con:
            repos = [r["repo_id"] for r in con.execute("select repo_id from models order by repo_id")]
    results = [check(r, args.layer) for r in repos]
    print(f"\n{sum(results)}/{len(results)} models agree with the raw weights")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
