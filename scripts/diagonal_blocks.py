#!/usr/bin/env python3
"""The diagonal blocks of T at the heads, in the metric of S.

For each head the bilinear form L = W_K^T W_Q, already restricted to
the reduced subspace by the extraction database, is split as
L = S + T, with S = diag(A, -B) in an orthonormal eigenbasis of S
ordered by decreasing eigenvalue on V_+ and decreasing absolute value
on V_-, and T in blocks C, X, D.  The solutions of T S^{-1} T = S are
parametrized by

    C = A^{1/2} G A^{1/2},
    X = A^{1/2} (I + G^T G)^{1/2} O B^{1/2},
    D = B^{1/2} O^T G O B^{1/2},

with G antisymmetric and O orthogonal, and the normalized blocks
G = A^{-1/2} C A^{-1/2} and H = B^{-1/2} D B^{-1/2} have the same
singular values g_1 >= ... >= g_n.

For each head of the reported models the script computes

    delta   = (||C||^2 + ||D||^2) / ||T||^2,
    g1sq    = g_1^2, the largest squared singular value of G,
    ratio1  = ||T||^2 / 2<lambda_+, lambda_->,
    qshare  = the share of ||G||^2 carried by the entries G_ij with
              both i, j > n - floor(n/2), i.e. both indices among
              the floor(n/2) smallest eigenvalues of A
              (index-exchangeable G would give h(h-1)/(n(n-1)),
              h = floor(n/2)),

together with two consistency checks: the maximal difference between
the sorted singular values of G and of H (zero in exact arithmetic),
and the inequality (1 - delta) ||T||^2 <= 2 (1 + g1sq) <l+, l->.

For gpt2 it also draws random solutions from the parametrization --
G = s (Z - Z^T)/2 with Z standard Gaussian and s uniform on [0, 2.5],
O uniform (Haar) on the orthogonal group, with (A, B) taken from 8
randomly chosen heads -- and reports the distribution of delta both
unconditionally and conditioned on ratio1 in [0.8, 1.1].  Fixed seed.

The per-head values are written to
``scripts/survey_results/diagonal_blocks.csv`` and the printed report to
``scripts/survey_results/diagonal_blocks_summary.txt``.

Usage:
    python diagonal_blocks.py
    python diagonal_blocks.py --model gpt2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_set as ms

OUT = ms.PROJECT / "scripts" / "survey_results"


def reduced_blocks(L: np.ndarray):
    """(a, b, C, X, D) in the ordered eigenbasis of S.

    Raises if the signature of S is not (n, n), which the reduction of
    a nondegenerate head guarantees.
    """
    S, T = ms.symmetric_antisymmetric(L)
    w, V = np.linalg.eigh(S)
    ip, im_ = np.where(w > 0)[0], np.where(w < 0)[0]
    if len(ip) != len(im_) or len(ip) + len(im_) != len(w):
        raise ValueError(
            f"signature of S is ({len(ip)}, {len(im_)}) on a "
            f"{len(w)}-dimensional reduced subspace")
    ip = ip[np.argsort(-w[ip])]
    im_ = im_[np.argsort(w[im_])]
    Vp, Vm = V[:, ip], V[:, im_]
    return (w[ip], -w[im_],
            Vp.T @ T @ Vp, Vp.T @ T @ Vm, Vm.T @ T @ Vm)


def head_row(a, b, C, X, D) -> dict:
    n = len(a)
    sa, sb = np.sqrt(a), np.sqrt(b)
    G = C / np.outer(sa, sa)
    H = D / np.outer(sb, sb)
    sG = np.sort(np.linalg.svd(G, compute_uv=False))[::-1]
    sH = np.sort(np.linalg.svd(H, compute_uv=False))[::-1]
    mismatch = np.abs(sG - sH).max() / (1.0 + sG[0])
    nC2, nD2, nX2 = (C ** 2).sum(), (D ** 2).sum(), (X ** 2).sum()
    nT2 = nC2 + nD2 + 2 * nX2
    delta = (nC2 + nD2) / nT2
    pair = a @ b
    g1sq = sG[0] ** 2
    h = n // 2
    qshare = (G[n - h:, n - h:] ** 2).sum() / (G ** 2).sum()
    qbase = h * (h - 1) / (n * (n - 1))
    bound_ok = (1 - delta) * nT2 <= 2 * (1 + g1sq) * pair * (1 + 1e-9)
    return dict(n=n, delta=delta, g1sq=g1sq, ratio1=nT2 / (2 * pair),
                qshare=qshare, qbase=qbase, mismatch=mismatch,
                bound_ok=bound_ok)


def random_baseline(heads, rng, per_head=120, max_tries=60000):
    """Random solutions from the (G, O) parametrization with (A, B)
    taken from sampled heads; delta unconditionally and given
    ratio1 in [0.8, 1.1]."""
    def msqrt(M):
        w, V = np.linalg.eigh(M)
        return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T

    idx = rng.choice(len(heads), 8, replace=False)
    uncond, cond = [], []
    for hi in idx:
        a, b = heads[hi][0], heads[hi][1]
        n = len(a)
        sa, sb = np.sqrt(a), np.sqrt(b)
        pair = a @ b
        got = tries = 0
        while got < per_head and tries < max_tries:
            tries += 1
            s = rng.uniform(0, 2.5)
            Z = rng.standard_normal((n, n))
            G = s * (Z - Z.T) / 2
            q, r = np.linalg.qr(rng.standard_normal((n, n)))
            O = q * np.sign(np.diag(r))
            W = msqrt(np.eye(n) + G.T @ G) @ O
            C = np.outer(sa, sa) * G
            D = np.outer(sb, sb) * (O.T @ G @ O)
            X = np.outer(sa, sb) * W
            nT2 = (C ** 2).sum() + (D ** 2).sum() + 2 * (X ** 2).sum()
            delta = ((C ** 2).sum() + (D ** 2).sum()) / nT2
            r1 = nT2 / (2 * pair)
            uncond.append(delta)
            if 0.8 < r1 < 1.1:
                cond.append(delta)
                got += 1
    return np.array(uncond), np.array(cond)


def qtile(x) -> str:
    return (f"{np.median(x):.3f} "
            f"[{np.quantile(x, 0.1):.3f}, {np.quantile(x, 0.9):.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ms.TAGS, default=None)
    args = ap.parse_args()
    tags = [args.model] if args.model else list(ms.TAGS)

    lines: list[str] = []

    def out(msg: str) -> None:
        print(msg)
        lines.append(msg)

    all_rows = []
    gpt2_heads = None
    for tag in tags:
        blocks = []
        for layer, head, _kv, L in ms.iter_forms(tag):
            blocks.append((layer, head, reduced_blocks(L)))
        if tag == "gpt2":
            gpt2_heads = [bl for _, _, bl in blocks]
        rows = []
        for layer, head, bl in blocks:
            r = head_row(*bl)
            r.update(model=tag, layer=layer, head=head)
            rows.append(r)
        all_rows += rows
        R = {k: np.array([r[k] for r in rows]) for k in
             ("delta", "g1sq", "ratio1", "qshare", "qbase",
              "mismatch", "bound_ok")}
        out(f"{tag:24s} {len(rows):5d} heads")
        out(f"  delta    {qtile(R['delta'])}")
        out(f"  g1^2     {qtile(R['g1sq'])}")
        out(f"  ratio1   {qtile(R['ratio1'])}")
        out(f"  qshare   {qtile(R['qshare'])}"
            f"   (exchangeable-index value {np.median(R['qbase']):.3f})")
        out(f"  sigma(G)=sigma(H) max mismatch {R['mismatch'].max():.2e}"
            f"   bound holds {np.mean(R['bound_ok']) * 100:.0f}%")

    if gpt2_heads is not None:
        rng = np.random.default_rng(0)
        uncond, cond = random_baseline(gpt2_heads, rng)
        out("")
        out(f"random solutions, (A,B) from 8 gpt2 heads "
            f"({len(uncond)} draws, {len(cond)} with ratio1 in [0.8, 1.1]):")
        out(f"  delta unconditional          {qtile(uncond)}")
        out(f"  delta | ratio1 in [0.8,1.1]  {qtile(cond)}")
        gd = np.array([r["delta"] for r in all_rows if r["model"] == "gpt2"])
        out(f"  delta at the gpt2 heads      {qtile(gd)}")

    if args.model is None:
        OUT.mkdir(exist_ok=True)
        cols = ["model", "layer", "head", "n", "delta", "g1sq", "ratio1",
                "qshare", "qbase", "mismatch", "bound_ok"]
        pd.DataFrame(all_rows)[cols].to_csv(OUT / "diagonal_blocks.csv",
                                            index=False)
        (OUT / "diagonal_blocks_summary.txt").write_text(
            "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
