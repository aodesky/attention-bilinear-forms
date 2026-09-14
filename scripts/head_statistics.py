#!/usr/bin/env python3
"""Per-head statistics for every model of the paper's survey.

For each head this computes, from the bilinear form L stored by the
extraction database:

* the symmetric energy S, the balance score B and the pairing score P;
* the shape (a, b, c, d) of L and of its symmetric part;
* the ratio rho = ||lambda_-|| / ||lambda_+|| and the bound of the
  pairing lemma at that rho, with the gap between the two;
* the departure from proportional lobes,
  ||lambda_- - rho lambda_+|| / ||lambda_-||;
* the eigenvalue counts n_+, n_-, n_0 of the symmetric part; and
* the normalized trace |tr L| / ||L||.

The spectra themselves (the sorted lobes, the full eigenvalue
magnitudes, and the singular values of the antisymmetric part) are saved
alongside, so the spectral figures need not reopen the forms.

The result is written once and reused: every downstream script reads
``scripts/survey_results/head_statistics.csv`` rather than recomputing.

Usage:
    python head_statistics.py            # all models
    python head_statistics.py --model gpt2
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
TOL = 1e-9


def head_row(tag: str, layer: int, head: int, kv_head: int,
             L: np.ndarray) -> tuple[dict, dict]:
    """The statistics of one head, and its spectra."""
    S, T = ms.symmetric_antisymmetric(L)
    norm_sq = float((L ** 2).sum())
    if norm_sq == 0.0:
        raise ValueError(f"{tag} L{layer}H{head}: the form is zero")

    ev = np.linalg.eigvalsh(S)
    scale = max(abs(float(ev[0])), abs(float(ev[-1])))
    cut = TOL * scale
    n_pos = int((ev > cut).sum())
    n_neg = int((ev < -cut).sum())
    n_zero = ev.size - n_pos - n_neg

    lam_p, lam_m = ms.sorted_lobes(S)
    u = lam_p - lam_m
    sym_sq = float((S ** 2).sum())

    norm_p = float(np.linalg.norm(lam_p))
    norm_m = float(np.linalg.norm(lam_m))
    rho = norm_m / norm_p if norm_p > 0 else np.nan
    # the pairing score and the bound of the pairing lemma at this rho
    pairing = 1.0 - float(np.linalg.norm(lam_p - lam_m)) / np.sqrt(sym_sq)
    bound = 1.0 - abs(1.0 - rho) / np.sqrt(1.0 + rho ** 2)
    # departure from lambda_- = rho lambda_+
    lobes = (float(np.linalg.norm(lam_m - rho * lam_p)) / norm_m
             if norm_m > 0 else np.nan)

    sigma_T = np.linalg.svd(T, compute_uv=False)
    row = dict(
        model=tag, layer=layer, head=head, kv_head=kv_head,
        dim=L.shape[0],
        symmetric_energy=float((S ** 2).sum()) / norm_sq,
        balance=1.0 - abs(n_pos - n_neg) / (n_pos + n_neg),
        pairing=pairing, rho=rho, pairing_bound=bound,
        pairing_gap=bound - pairing, lobe_departure=lobes,
        n_pos=n_pos, n_neg=n_neg, n_zero=n_zero,
        a=float((T ** 2).sum()) / norm_sq,
        b=float(2.0 * lam_p @ lam_m) / norm_sq,
        c=float((u[u > 0] ** 2).sum()) / norm_sq,
        d=float((u[u < 0] ** 2).sum()) / norm_sq,
        b_sym=float(2.0 * lam_p @ lam_m) / sym_sq,
        c_sym=float((u[u > 0] ** 2).sum()) / sym_sq,
        d_sym=float((u[u < 0] ** 2).sum()) / sym_sq,
        trace_ratio=abs(float(np.trace(L))) / np.sqrt(norm_sq),
    )
    spectra = dict(eigenvalues=ev, sigma_T=sigma_T)
    return row, spectra


def run(tags: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for tag in tags:
        rows: list[dict] = []
        eigs: list[np.ndarray] = []
        sigs: list[np.ndarray] = []
        for layer, head, kv_head, L in ms.iter_forms(tag):
            row, spec = head_row(tag, layer, head, kv_head, L)
            rows.append(row)
            eigs.append(spec["eigenvalues"])
            sigs.append(spec["sigma_T"])

        info = ms.model_info(tag)
        if len(rows) != info.n_heads_total:
            raise RuntimeError(
                f"{tag}: iterated {len(rows)} heads, database says "
                f"{info.n_heads_total}")

        np.savez_compressed(
            OUT / f"{tag}_spectra.npz",
            eigenvalues=np.array(eigs),
            sigma_T=np.array(sigs),
            layer=np.array([r["layer"] for r in rows]),
            head=np.array([r["head"] for r in rows]))
        all_rows.extend(rows)

        df = pd.DataFrame(rows)
        print(f"{tag:26s} {len(rows):5d} heads   "
              f"balance min {df.balance.min():.3f}   "
              f"S median {df.symmetric_energy.median():.3f}   "
              f"P median {df.pairing.median():.3f}")

    out = pd.DataFrame(all_rows)
    path = OUT / "head_statistics.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(out)} heads, {out.model.nunique()} models)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=ms.TAGS, help="only this model")
    args = p.parse_args()
    run([args.model] if args.model else ms.TAGS)


if __name__ == "__main__":
    main()
