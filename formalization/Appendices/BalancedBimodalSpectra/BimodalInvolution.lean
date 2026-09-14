/-
Appendix "Balanced bimodal spectra", Lemma (bimodal involution)
(label `lem:bimodal-involution`).

The map `ι` is an orthogonal, symmetric involution of `ℝ^n` and maps
`Λ_m` to `Λ_m`, with `λ₊(ιλ) = λ₋(λ)` and `λ₋(ιλ) = λ₊(λ)`.
Consequently `ℐ∘ι = -ℐ`, `C∘ι = C`, and `𝒫∘ι = 𝒫`.  Moreover, for
`λ ∈ Λ_m`,

    ‖λ - ιλ‖² = 2‖λ₊-λ₋‖²,  so  𝒫(λ) = 1 - (1/√2)‖λ - ιλ‖;

in particular `𝒫(λ) = 1` if and only if `ιλ = λ`.

"Orthogonal, symmetric involution" is rendered as: `ι ∘ ι = id`, `ι`
preserves the standard inner product `Σᵢ x i * y i`, and `ι` is
self-adjoint for it (`⟨ιx, y⟩ = ⟨x, ιy⟩`); linearity is recorded as
compatibility with addition and scalar multiplication.  Statements
that are identities not needing the hypothesis `λ ∈ Λ_m` that the
paper carries name it `_hx`.

This file is independent of `BimodalIdentities.lean`; the index-split
lemma is re-proved here privately.
-/
import Appendices.BalancedBimodalSpectra.Defs

namespace Appendices

open BigOperators

variable {m : ℕ} {x : Fin (2 * m) → ℝ}

/-! ### `ι` is an orthogonal, symmetric involution (a linear map) -/

/-- `ι ∘ ι = id`. -/
theorem iotaMap_iotaMap {n : ℕ} (x : Fin n → ℝ) :
    iotaMap (iotaMap x) = x := by
  funext i
  simp [iotaMap, Fin.rev_rev]

/-- `ι` is additive. -/
theorem iotaMap_add {n : ℕ} (x y : Fin n → ℝ) :
    iotaMap (x + y) = iotaMap x + iotaMap y := by
  funext i
  simp [iotaMap]
  ring

/-- `ι` is homogeneous. -/
theorem iotaMap_smul {n : ℕ} (c : ℝ) (x : Fin n → ℝ) :
    iotaMap (c • x) = c • iotaMap x := by
  funext i
  simp [iotaMap]

/-- `ι` is orthogonal: it preserves the standard inner product. -/
theorem sum_iotaMap_mul_iotaMap {n : ℕ} (x y : Fin n → ℝ) :
    ∑ i, iotaMap x i * iotaMap y i = ∑ i, x i * y i := by
  refine Fintype.sum_equiv Fin.revPerm _ _ fun i => ?_
  simp [iotaMap]

/-- `ι` is symmetric (self-adjoint): `⟨ιx, y⟩ = ⟨x, ιy⟩`. -/
theorem sum_iotaMap_mul {n : ℕ} (x y : Fin n → ℝ) :
    ∑ i, iotaMap x i * y i = ∑ i, x i * iotaMap y i := by
  refine Fintype.sum_equiv Fin.revPerm _ _ fun i => ?_
  simp only [iotaMap, Fin.revPerm_apply, Fin.rev_rev]
  ring

/-! ### `ι` maps `Λ_m` to `Λ_m` and exchanges `λ₊` with `λ₋` -/

/-- `λ₊(ιλ) = λ₋(λ)`. -/
theorem lamPlus_iotaMap (x : Fin (2 * m) → ℝ) :
    lamPlus (iotaMap x) = lamMinus x := by
  funext j
  simp [lamPlus, lamMinus, iotaMap, Fin.rev_rev]

/-- `λ₋(ιλ) = λ₊(λ)`. -/
theorem lamMinus_iotaMap (x : Fin (2 * m) → ℝ) :
    lamMinus (iotaMap x) = lamPlus x := by
  funext j
  simp [lamPlus, lamMinus, iotaMap]

/-- `ι` maps `Λ_m` to `Λ_m`. -/
theorem MemLambdaSet.iota_mem (hx : MemLambdaSet m x) :
    MemLambdaSet m (iotaMap x) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i j hij
    simp only [iotaMap, neg_le_neg_iff]
    exact hx.mono (Fin.rev_le_rev.mpr hij)
  · rw [← hx.unit]
    refine Fintype.sum_equiv Fin.revPerm _ _ fun i => ?_
    simp [iotaMap]
  · intro h
    have hrev : Fin.rev (⟨m - 1, by omega⟩ : Fin (2 * m))
        = (⟨m, by omega⟩ : Fin (2 * m)) := by
      ext
      simp [Fin.val_rev]
      omega
    show -x (Fin.rev ⟨m - 1, by omega⟩) < 0
    rw [hrev, neg_lt_zero]
    exact hx.middle_pos h
  · intro h
    have hrev : Fin.rev (⟨m, by omega⟩ : Fin (2 * m))
        = (⟨m - 1, by omega⟩ : Fin (2 * m)) := by
      ext
      simp [Fin.val_rev]
      omega
    show 0 < -x (Fin.rev ⟨m, by omega⟩)
    rw [hrev, neg_pos]
    exact hx.middle_neg h

/-- `ι(Λ_m) = Λ_m`: membership is preserved in both directions. -/
theorem memLambdaSet_iotaMap_iff :
    MemLambdaSet m (iotaMap x) ↔ MemLambdaSet m x :=
  ⟨fun h => iotaMap_iotaMap x ▸ h.iota_mem, MemLambdaSet.iota_mem⟩

/-! ### Transformation rules for `ℐ`, `C`, `𝒫` -/

/-- `ℐ ∘ ι = -ℐ`. -/
theorem imbalance_iotaMap (_hx : MemLambdaSet m x) :
    imbalance (iotaMap x) = -imbalance x := by
  rw [imbalance, imbalance, lamPlus_iotaMap, lamMinus_iotaMap]
  ring

/-- `C ∘ ι = C`. -/
theorem statC_iotaMap (_hx : MemLambdaSet m x) :
    statC (iotaMap x) = statC x := by
  rw [statC, statC, lamPlus_iotaMap, lamMinus_iotaMap]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- `𝒫 ∘ ι = 𝒫`. -/
theorem pairingScore_iotaMap (_hx : MemLambdaSet m x) :
    pairingScore (iotaMap x) = pairingScore x := by
  rw [pairingScore, pairingScore, lamPlus_iotaMap, lamMinus_iotaMap]
  congr 2
  exact Finset.sum_congr rfl fun j _ => by ring

/-! ### The displacement identity -/

private lemma sum_split' (g : Fin (2 * m) → ℝ) :
    ∑ i, g i
      = (∑ j : Fin m, g ⟨j, by omega⟩)
        + ∑ j : Fin m, g (Fin.rev ⟨j, by omega⟩) := by
  have hmm : m + m = 2 * m := by omega
  rw [← Fintype.sum_equiv (finCongr hmm) (fun i => g (finCongr hmm i)) g
    (fun i => rfl), Fin.sum_univ_add]
  congr 1
  refine Fintype.sum_equiv Fin.revPerm _ _ fun j => ?_
  have hj := j.isLt
  congr 1
  ext
  simp only [finCongr_apply, Fin.val_cast, Fin.val_natAdd, Fin.revPerm_apply,
    Fin.val_rev]
  omega

/-- `‖λ - ιλ‖² = 2‖λ₊-λ₋‖²`: the `j`-th and `(n+1-j)`-th coordinates
of `λ - ιλ` both equal `(λ₊)_j - (λ₋)_j` up to sign. -/
theorem sum_sq_sub_iotaMap (_hx : MemLambdaSet m x) :
    ∑ i, (x i - iotaMap x i) ^ 2
      = 2 * ∑ j, (lamPlus x j - lamMinus x j) ^ 2 := by
  rw [sum_split' (fun i => (x i - iotaMap x i) ^ 2)]
  have h1 : ∀ j : Fin m,
      (x ⟨j, by omega⟩ - iotaMap x ⟨j, by omega⟩) ^ 2
        = (lamPlus x j - lamMinus x j) ^ 2 := by
    intro j
    simp only [iotaMap, lamPlus, lamMinus, sub_neg_eq_add]
    ring
  have h2 : ∀ j : Fin m,
      (x (Fin.rev ⟨j, by omega⟩) - iotaMap x (Fin.rev ⟨j, by omega⟩)) ^ 2
        = (lamPlus x j - lamMinus x j) ^ 2 := by
    intro j
    simp only [iotaMap, lamPlus, lamMinus, sub_neg_eq_add, Fin.rev_rev]
  rw [Finset.sum_congr rfl fun j _ => h1 j,
    Finset.sum_congr rfl fun j _ => h2 j]
  ring

/-- `𝒫(λ) = 1 - (1/√2)‖λ - ιλ‖`. -/
theorem pairingScore_eq_dist_iotaMap (hx : MemLambdaSet m x) :
    pairingScore x
      = 1 - Real.sqrt (∑ i, (x i - iotaMap x i) ^ 2) / Real.sqrt 2 := by
  rw [pairingScore, sum_sq_sub_iotaMap hx,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), mul_comm, mul_div_assoc,
    div_self (ne_of_gt (Real.sqrt_pos.mpr two_pos)), mul_one]

/-- `𝒫(λ) = 1` if and only if `ιλ = λ`. -/
theorem pairingScore_eq_one_iff (hx : MemLambdaSet m x) :
    pairingScore x = 1 ↔ iotaMap x = x := by
  rw [pairingScore_eq_dist_iotaMap hx]
  constructor
  · intro h
    have h2 : Real.sqrt 2 ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr two_pos)
    have hq : Real.sqrt (∑ i, (x i - iotaMap x i) ^ 2) / Real.sqrt 2 = 0 := by
      linarith
    have hsqrt : Real.sqrt (∑ i, (x i - iotaMap x i) ^ 2) = 0 := by
      rcases div_eq_zero_iff.mp hq with h' | h'
      · exact h'
      · exact absurd h' h2
    have hnn : (0 : ℝ) ≤ ∑ i, (x i - iotaMap x i) ^ 2 :=
      Finset.sum_nonneg fun i _ => sq_nonneg _
    have hsum : ∑ i, (x i - iotaMap x i) ^ 2 = 0 := by
      have := Real.sqrt_eq_zero hnn
      exact this.mp hsqrt
    funext i
    have hterm := (Finset.sum_eq_zero_iff_of_nonneg
      (fun i _ => sq_nonneg (x i - iotaMap x i))).mp hsum i (Finset.mem_univ i)
    have hdiff : x i - iotaMap x i = 0 :=
      pow_eq_zero_iff two_ne_zero |>.mp hterm
    linarith
  · intro h
    rw [h]
    simp

end Appendices
