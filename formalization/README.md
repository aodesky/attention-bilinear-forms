# Lean formalization

Lean 4 formalizations of the mathematical results of *On attention
heads and bilinear forms*: every numbered lemma, proposition,
corollary and theorem of the main text and of the appendices.

The [blueprint](https://aodesky.github.io/attention-bilinear-forms/blueprint/)
presents these statements and proof outlines with links to their Lean
declarations and a
[dependency graph](https://aodesky.github.io/attention-bilinear-forms/blueprint/dep_graph_document.html).
Its source and build instructions are in [`../blueprint/`](../blueprint/README.md).

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

One file per result, in `MainText/` and in folders named after the appendices.  Results
are numbered as in the paper, where lemmas, propositions, corollaries
and theorems are numbered in separate sequences.

| File | Paper |
|---|---|
| **Main text** | |
| `MainText/BiaffineDecomposition.lean` | Lemma 1 (biaffine decomposition) |
| `MainText/RoPELike.lean` | the definition of RoPE-like attention functions and Proposition 1 |
| `MainText/NearIdentity.lean` | Proposition 2 (transformers are near-identity) |
| `MainText/Homogenization.lean` | the definitions of attention functions, bilinear attention functions, heads and transformers; Proposition 3 (bilinear attention recovers QKV attention) |
| `MainText/PairingBound.lean` | the pairing score and Lemma 2 (pairing bound) |
| `MainText/RankBalance.lean` | Lemma 3 (rank balance) |
| `MainText/BalancedHeads.lean` | Theorem 3 (heads have balanced attention) |
| `MainText/ProfileScale.lean` | Lemma 4 (profile scale) |
| `MainText/ProfileInvariance.lean` | supporting facts: sorting depends only on the multiset of entries; the profile is invariant under orthogonal changes of coordinates |
| `MainText/RankOneLocus.lean` | Proposition 4 (profiles of rank-one matrices) |
| `MainText/Accumulation.lean` | Theorem 5 (accumulation), with the limit in `p(Θ)` and the edges it traces |
| `Appendices/Common.lean` | Frobenius norm; symmetric and antisymmetric parts (shared definitions) |
| `Appendices/Profile.lean` | Definition 6, the profile `π(L) = (a, b, c, d)` of a bilinear form (§5.1) |
| **Appendix A, Random matrices** | |
| `Appendices/RandomMatrices/EnergyIdentities.lean` | Lemma 5 (symmetric and skew energy identities) |
| `Appendices/RandomMatrices/IidBaseline.lean` | Corollary 1 (iid square random matrix baseline) |
| `Appendices/RandomMatrices/QKProductBaseline.lean` | Corollary 2 (random QK-product baseline) |
| `Appendices/RandomMatrices/LowRankExpectedEnergy.lean` | Proposition 5 (expected symmetric energy for random low-rank QK products) |
| `Appendices/RandomMatrices/ProportionalLobes.lean` | Lemma 6, the sorted lobes of two samples become proportional |
| **Appendix B, Balanced bimodal spectra** | |
| `Appendices/BalancedBimodalSpectra/Defs.lean` | The pairing geometry: `Λ_m`, `λ±`, `ℐ`, `C`, `𝒫`, `ι` |
| `Appendices/BalancedBimodalSpectra/BimodalIdentities.lean` | Lemma 7, the identities for `ℐ`, `C`, `𝒫`, with the note `0 ≤ 2C ≤ 1` |
| `Appendices/BalancedBimodalSpectra/BimodalInvolution.lean` | Lemma 8, the involution `ι` |
| `Appendices/BalancedBimodalSpectra/SimplexFaces.lean` | Theorem 4, the fibers of the profile map over the faces of `Δ₃` (stated in §5.1, proved in Appendix B) |
| **Appendix C, Moments** | |
| `Appendices/Moments/Defs.lean` | The statistics `O`, `E`, `R` |
| `Appendices/Moments/OEEquivalences.lean` | Lemma 9, `O` and `E` as sums of odd and even moments, partial fractions, and traces |
| `Appendices/Moments/GramForm.lean` | Lemma 10, computation of `O`, `E`, `R` from the key and query matrices |

## Scope

Every numbered result of the paper is formalized: in the main text,
Lemmas 1 to 4, Propositions 1 to 4 and Theorems 3 to 5 (Theorems 1
and 2 of the introduction restate Theorems 3 and 5); in the
appendices, Lemmas 5 to 10, Corollaries 1 and 2, Proposition 5, and
Theorem 4, whose statement is in the main text and whose proof is
Appendix B.  The definitions these results are stated in are
formalized with them.

Not formalized: the two remarks of the main text (the max potential,
and the remark following Lemma 3), which are stated without proof;
the observations, which are experimental; and Appendices D and E,
which contain experiment descriptions, tables and figures.

Each statement is formalized as stated in the paper; where a proof
needs a hypothesis the paper leaves implicit (a nonzero variance, a
positive dimension), or holds without one the paper carries, the
docstring of the theorem says so.  Encoding conventions are documented
in the module docstring of each file.
