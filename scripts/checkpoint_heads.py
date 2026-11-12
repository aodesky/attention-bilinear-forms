#!/usr/bin/env python3
"""Query and key matrices of the heads of a model loaded with transformers.

The survey's forms are extracted from checkpoint files by
``extract_qk_bilinear_db.py`` and read from the database by
``model_set.iter_forms``.  The supporting analyses that need the heads of a
model instantiated in memory, in particular a freshly initialized one,
use this module instead: ``load_model`` builds the model and
``iter_qk_heads`` yields the per-head matrices W_Q and W_K, each of shape
[head_dim, d_model], for the GPT-2, GPT-NeoX and OPT architectures.
``compute_reduced_spectrum`` gives the spectra of the symmetric and
antisymmetric parts of L = W_K^T W_Q on the joint row space of the two
matrices, as in the survey.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM


@dataclass
class HeadWeights:
    model_id: str
    model_type: str
    layer: int
    head: int
    kv_head: int
    Wq: torch.Tensor  # shape: [head_dim, d_model]
    Wk: torch.Tensor  # shape: [head_dim, d_model]


def projection_matrix(module: torch.nn.Module) -> torch.Tensor:
    """The matrix W of a projection module, so that output = W @ input.

    torch.nn.Linear stores its weight as [out, in]; the Conv1D module of
    GPT-2 stores it as [in, out], so that one is transposed.
    """
    cls = module.__class__.__name__
    if isinstance(module, torch.nn.Linear):
        return module.weight.detach().cpu().to(torch.float64)
    if cls == "Conv1D" and hasattr(module, "weight"):
        return module.weight.detach().cpu().to(torch.float64).T
    raise TypeError(f"Unsupported projection module type: {cls}")


def split_heads(W: torch.Tensor, n_heads: int, head_dim: Optional[int] = None) -> torch.Tensor:
    """Split W [n_heads * head_dim, d_model] into [n_heads, head_dim, d_model]."""
    out_dim, d_model = W.shape
    if head_dim is None:
        assert out_dim % n_heads == 0, (out_dim, n_heads)
        head_dim = out_dim // n_heads
    return W.reshape(n_heads, head_dim, d_model)


def iter_qk_heads(model, model_id: str) -> Generator[HeadWeights, None, None]:
    """Yield the per-head query and key matrices of a loaded model.

    Supported layouts:
      - GPT-2 / DistilGPT2: transformer.h.*.attn.c_attn
      - GPT-NeoX / Pythia: gpt_neox.layers.*.attention.query_key_value
      - OPT: model.decoder.layers.*.self_attn.q_proj/k_proj
    """
    cfg = model.config
    mt = getattr(cfg, "model_type", "unknown")

    # GPT-2 / DistilGPT2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = list(model.transformer.h)
        if blocks and hasattr(blocks[0].attn, "c_attn"):
            n_heads = int(getattr(cfg, "n_head"))
            d_model = int(getattr(cfg, "n_embd"))
            head_dim = d_model // n_heads
            for layer, block in enumerate(blocks):
                W = projection_matrix(block.attn.c_attn)  # [3*d_model, d_model]
                Wq, Wk, _Wv = W.split(d_model, dim=0)
                Q = split_heads(Wq, n_heads, head_dim)
                K = split_heads(Wk, n_heads, head_dim)
                for h in range(n_heads):
                    yield HeadWeights(model_id, mt, layer, h, h, Q[h], K[h])
            return

    # GPT-NeoX / Pythia: the fused projection is ordered by head, and
    # within each head as [query, key, value].
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = list(model.gpt_neox.layers)
        n_heads = int(getattr(cfg, "num_attention_heads"))
        d_model = int(getattr(cfg, "hidden_size"))
        head_dim = d_model // n_heads
        for layer, block in enumerate(layers):
            attn = block.attention
            W = projection_matrix(attn.query_key_value)  # [3*d_model, d_model]
            W_by_head = W.reshape(n_heads, 3 * head_dim, d_model)
            Q = W_by_head[:, :head_dim, :]
            K = W_by_head[:, head_dim:2 * head_dim, :]
            for h in range(n_heads):
                yield HeadWeights(model_id, mt, layer, h, h, Q[h], K[h])
        return

    # OPT
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = list(model.model.decoder.layers)
        if layers and hasattr(layers[0].self_attn, "q_proj"):
            n_heads = int(getattr(cfg, "num_attention_heads"))
            d_model = int(getattr(cfg, "hidden_size"))
            head_dim = d_model // n_heads
            for layer, block in enumerate(layers):
                attn = block.self_attn
                Q = split_heads(projection_matrix(attn.q_proj), n_heads, head_dim)
                K = split_heads(projection_matrix(attn.k_proj), n_heads, head_dim)
                for h in range(n_heads):
                    yield HeadWeights(model_id, mt, layer, h, h, Q[h], K[h])
            return

    raise ValueError(
        f"Could not find a supported Q/K layout for model_type={mt}. "
        "Add an extractor in iter_qk_heads()."
    )


def compute_reduced_spectrum(hw: HeadWeights) -> Tuple[np.ndarray, np.ndarray]:
    """The eigenvalues of S and the singular values of T on the joint row
    space of W_Q and W_K, for L = W_K^T W_Q = S + T."""
    Wq = hw.Wq.to(torch.float64)
    Wk = hw.Wk.to(torch.float64)
    U = torch.cat([Wk.T, Wq.T], dim=1)
    R = torch.linalg.qr(U, mode="reduced").Q
    KR = Wk @ R
    QR = Wq @ R
    Sred = 0.5 * (KR.T @ QR + QR.T @ KR)
    Ared = 0.5 * (KR.T @ QR - QR.T @ KR)
    evals = torch.linalg.eigvalsh(Sred).cpu().numpy()
    svals = torch.linalg.svdvals(Ared).cpu().numpy()
    return evals, svals


def load_model(model_id: str, trust_remote_code: bool = False):
    """Load a causal language model in float32 on the CPU."""
    print(f"\nLoading {model_id} ...")
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=False,
    )
