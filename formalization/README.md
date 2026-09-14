# Lean formalization of the appendices

Lean 4 formalizations of the mathematical results of the appendices
of *On attention heads and bilinear forms*, together with the
definition and the main-text theorem on which Appendix B rests.

## Requirements

- Lean toolchain: see `lean-toolchain` (elan installs it automatically)
- mathlib (pinned in `lake-manifest.json`; no other dependencies)

Build with

    lake exe cache get   # download the mathlib build cache
    lake build

No `sorry`s, and no axioms beyond the three standard ones (`propext`,
`Classical.choice`, `Quot.sound`); `scripts/check_axioms.sh` checks
both.

## Layout

One file per result, in folders named after the appendices.  Results
are numbered as in the paper, where lemmas, propositions, corollaries
and theorems are numbered in separate sequences.

| File | Paper |
|---|---|
| `Appendices/Common.lean` | Frobenius norm; symmetric and antisymmetric parts (shared definitions) |
| `Appendices/Shape.lean` | Definition 6, the shape `s(L) = (a, b, c, d)` of a bilinear form (§6.1) |
| **Appendix A, Random matrices** | |
| `Appendices/RandomMatrices/EnergyIdentities.lean` | Lemma 6 (symmetric and skew energy identities) |
| `Appendices/RandomMatrices/IidBaseline.lean` | Corollary 1 (iid square random matrix baseline) |
| `Appendices/RandomMatrices/QKProductBaseline.lean` | Corollary 2 (random QK-product baseline) |
| `Appendices/RandomMatrices/LowRankExpectedEnergy.lean` | Proposition 4 (expected symmetric energy for random low-rank QK products) |
| `Appendices/RandomMatrices/ProportionalLobes.lean` | Lemma 7, the sorted lobes of two samples become proportional |
| **Appendix B, Balanced bimodal spectra** | |
| `Appendices/BalancedBimodalSpectra/Defs.lean` | The pairing geometry: `Λ_m`, `λ±`, `ℐ`, `C`, `𝒫`, `ι` |
| `Appendices/BalancedBimodalSpectra/BimodalIdentities.lean` | Lemma 8, the identities for `ℐ`, `C`, `𝒫`, with the note `0 ≤ 2C ≤ 1` |
| `Appendices/BalancedBimodalSpectra/BimodalInvolution.lean` | Lemma 9, the involution `ι` |
| `Appendices/BalancedBimodalSpectra/SimplexFaces.lean` | Theorem 5, the fibers of the shape map over the faces of `Δ₃` (stated in §6.1, proved in Appendix B) |
| **Appendix C, Moments** | |
| `Appendices/Moments/Defs.lean` | The statistics `O`, `E`, `R` |
| `Appendices/Moments/OEEquivalences.lean` | Lemma 10, `O` and `E` as sums of odd and even moments, partial fractions, and traces |
| `Appendices/Moments/GramForm.lean` | Lemma 11, computation of `O`, `E`, `R` from the key and query matrices |

## Scope

Every numbered result stated or proved in Appendices A, B and C is
formalized: Lemmas 6 to 11, Corollaries 1 and 2, Proposition 4, and
Theorem 5, whose statement is in the main text and whose proof is
Appendix B.  Definition 6 is formalized because Theorem 5 is stated in
its terms.

Not formalized: the two remarks following Lemma 11, which are
expository and stated without proof; and the results of the main
text, including Theorem 6, whose proof in the main text applies
Lemma 7.  Appendices D and E contain experiment descriptions, tables
and figures, with no mathematical results.

Each statement is formalized as stated in the paper; where a proof
needs a hypothesis the paper leaves implicit (a nonzero variance, a
positive dimension), or holds without one the paper carries, the
docstring of the theorem says so.  Encoding conventions are documented
in the module docstring of each file.
