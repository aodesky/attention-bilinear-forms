# Formalization handoff

Update 2026-09-04 (audit): `proportional_lobes` restated to match the
current paper (arbitrary positive law with finite nonzero second moment,
scales `u, v`); the gamma-law statement is kept as
`proportional_lobes_gamma`. `expected_symmetric_energy_low_rank_qk` and
`expected_skew_energy_low_rank_qk` now allow different variances for
`W_K` and `W_Q`; the common-variance versions are `…_common_variance`.

Status as of 2026-09-04 (end of session). All fifteen files are complete,
imported into the root `Appendices.lean`, and pass the project gate.

## Build protocol

- Toolchain: `leanprover/lean4:v4.27.0`, mathlib pinned in `lake-manifest.json`.
  `.lake` is present (APFS-cloned from the Residue project — do **not** run a
  fresh `lake build` from scratch, it triggers a multi-hour mathlib rebuild).
- **Per-file gate:** from `formalization/`, run
  `lake env lean Appendices/<path>/<File>.lean` (compiles that file only).
- **Project gate:** `lake build`, then `scripts/check_axioms.sh` (greps for
  `sorry`, runs a `CollectAxioms` check allowing only `propext`,
  `Classical.choice`, `Quot.sound`).
- If multiple agents run at once, only ONE should ever run `lake build`
  (lock contention); workers use `lake env lean` on their own file.

## What was done this session

1. `RandomMatrices/LowRankExpectedEnergy.lean` — imported and verified.
2. `BalancedBimodalSpectra/SimplexFaces.lean` — compile errors fixed; all
   parts of Theorem thm:simplex-faces proved (`weightB_eq_zero_iff`,
   `weightB_eq_one_iff`, `weightD_eq_zero_iff_forall`,
   `weightD_eq_zero_iff_decomp`, `weightD_eq_zero_decomp_commute`,
   `weightD_eq_one_iff`, and the `weightC` counterparts). Only hypothesis:
   `L ≠ 0`.
3. `Moments/GramForm.lean` — completed: `charpoly_roots_filter_ne_zero`
   (nonzero spectrum of S = spectrum of ½JG, as charpoly root multisets),
   `card_charpoly_roots_symPart`, `card_charpoly_roots_half_JG`,
   `IsSortedNormalizedNonzeroSpectrum` (+ existence, uniqueness),
   `memLambdaSet_of_isSortedNormalizedNonzeroSpectrum`, `statO_eq_trace`,
   `statE_eq_trace`, `statR_eq_trace`. Hypotheses: `M` surjective
   (`Function.Surjective (Matrix.mulVecLin (M WK WQ))`) and `1 ≤ n`, as in
   the paper.
4. `RandomMatrices/ProportionalLobes.lean` — completed: `proportional_lobes`
   states exactly the current paper lemma (sorting fact only; the stale
   "Consequently…" docstring paragraph was removed). Encoding: samples are
   sequences `X Y : ℕ → Ω → ℝ`, `α_n`/`β_n` are `sortDesc` of the first `n`
   terms, independence of everything is `iIndepFun (Sum.elim X Y) P`. Proof
   route: layer-cake identity + Fatou + SLLN (differs from the paper's
   generalized-inverse argument; the statement is unchanged).
5. Cleanup: `Appendices/Smoke.lean` deleted; duplicate
   `MemLambdaSet.m_pos` hoisted into `BalancedBimodalSpectra/Defs.lean`.

## Faithfulness reminder (user's standing rule)

Formalized statements must faithfully replicate the paper's statements. Do
not weaken/strengthen a theorem to ease the proof. If a statement/proof
issue is found, document the reasoning in an attendant file and STOP
formalizing that result rather than altering the math. Re-check each
target statement against the CURRENT `attention.tex` before trusting a
docstring, since the paper is being revised.
