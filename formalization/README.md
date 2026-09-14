# Lean formalization of the appendices

This directory contains Lean 4 formalizations of the mathematical
results of the appendices of *On attention heads and bilinear forms*
(`attention.tex`), together with the two main-text results whose
proofs are given in the appendices.

## Requirements

- Lean toolchain: see `lean-toolchain` (elan installs it automatically)
- mathlib (pinned in `lake-manifest.json`; no other dependencies)

Build with

    lake exe cache get   # download the mathlib build cache
    lake build

## Layout

One file per result, organized in folders named after the appendices.

| File | Paper result |
|---|---|
| `Appendices/Common.lean` | Frobenius norm, symmetric/antisymmetric parts (shared definitions) |
| `Appendices/Weights.lean` | The lists α, β and the weights (a,b,c,d) of §"A simplex-valued map on bilinear forms" (shared definitions) |
| `Appendices/RandomMatrices/EnergyIdentities.lean` | Lemma (Symmetric and skew energy identities) |
| `Appendices/RandomMatrices/IidBaseline.lean` | Corollary (Iid square random matrix baseline) |
| `Appendices/RandomMatrices/QKProductBaseline.lean` | Corollary (Random QK-product baseline) |
| `Appendices/RandomMatrices/LowRankExpectedEnergy.lean` | Proposition (Expected symmetric energy for random low-rank QK products) |
| `Appendices/RandomMatrices/ProportionalLobes.lean` | Lemma (proportional lobes), together with its gamma-law instance used in the proof of Theorem (accumulation) |
| `Appendices/BalancedBimodalSpectra/Defs.lean` | The pairing-geometry definitions (Λ_m, λ±, ℐ, C, 𝒫, ι) |
| `Appendices/BalancedBimodalSpectra/BimodalIdentities.lean` | Lemma (identities for ℐ, C, 𝒫), with the note 0 ≤ 2C ≤ 1 |
| `Appendices/BalancedBimodalSpectra/BimodalInvolution.lean` | Lemma (the involution ι) |
| `Appendices/BalancedBimodalSpectra/SimplexFaces.lean` | Theorem (fibers of the weight map over the faces of Δ³; stated in the main text, proved in Appendix "Balanced bimodal spectra") |
| `Appendices/Moments/Defs.lean` | The statistics O, E, R (shared definitions) |
| `Appendices/Moments/OEEquivalences.lean` | Lemma (equivalent expressions for O, E, R) |
| `Appendices/Moments/GramForm.lean` | Lemma (computation of O, E, R from the key and query matrices) |

## Scope

Formalized: every lemma, proposition, corollary, and theorem stated
or proved in the appendices of `attention.tex` (Appendices "Random
matrices", "Balanced bimodal spectra", "Moments"), including the
main-text Theorem (simplex faces), whose proof appears there.  The remaining appendices
("Spectral clusters for the remaining models", "Code and experimental
data", "Figures for the score survey") contain figures, tables, and
experiment descriptions, and the appendix of `llm.tex` ("Summed
full-space heads") contains only definitions; none of these contain
mathematical results to formalize.  The remarks following the
Gram-form lemma are expository and are not formalized.

Encoding conventions are documented in the module docstring of each
file.  No `sorry`s, no axioms beyond the three standard ones
(`propext`, `Classical.choice`, `Quot.sound`), verified by
`scripts/check_axioms.sh`.
