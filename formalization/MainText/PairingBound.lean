/-
Main text, §"Pairing score": the pairing score and Lemma `lemma:pairing-bound`.

    Write `S = ½(L + Lᵀ)`.  Let `λ₊` (resp. `λ₋`) denote the vector of
    positive (resp. absolute values of negative) eigenvalues of `S` sorted in
    decreasing order, with zeros appended so that they have the same length.
    The pairing score is `𝒫 = 1 - ‖λ₊ - λ₋‖/‖S‖`.

    Lemma.  Let `S` be a symmetric bilinear form with `λ₊ ≠ 0`, and let
    `ρ = ‖λ₋‖/‖λ₊‖`.  Then

        𝒫 = 1 - √(1 - 2⟨λ₊, λ₋⟩/‖S‖²)  ≤  1 - |1 - ρ|/√(1 + ρ²),

    with equality if and only if `λ₋ = ρ λ₊`.

Encoding.  The pairing score depends on the head only through `S`, and the
lemma is stated for a symmetric `S`, so both are stated for a real symmetric
matrix `S`.  As in `Appendices.Profile`, `λ₊` and `λ₋` are `alphaList S` and
`betaList S`, padded with zeros to length `N`; they are regarded as vectors of
`EuclideanSpace ℝ (Fin N)` so that `‖·‖` and `⟨·,·⟩` are the Euclidean norm
and inner product.  `‖S‖` is the Frobenius norm `√(frobSq S)`.  Appending
further zeros changes none of the quantities involved.

The proof follows the paper's:
  * `‖S‖² = ‖λ₊‖² + ‖λ₋‖²` (`frobSq_eq_norm_sq_add_norm_sq`), hence
    `‖λ₊ - λ₋‖² = ‖S‖² - 2⟨λ₊, λ₋⟩`, which gives the equality;
  * by the triangle inequality `‖λ₊ - λ₋‖ ≥ |‖λ₊‖ - ‖λ₋‖| = |1 - ρ| ‖λ₊‖`,
    with equality if and only if one of `λ±` is a nonnegative multiple of the
    other (`sameRay_iff_norm_sub`), that is, `λ₋ = ρ λ₊`;
  * dividing by `‖S‖ = √(1 + ρ²) ‖λ₊‖` gives the inequality.
-/
import Appendices.BalancedBimodalSpectra.SimplexFaces

namespace Appendices.MainText

open Matrix

variable {N : ℕ}

/-- `λ₊`, the positive eigenvalues of `S` in decreasing order padded with
zeros, as a Euclidean vector. -/
noncomputable def lamPlusVec (S : Matrix (Fin N) (Fin N) ℝ) (hS : S.IsHermitian) :
    EuclideanSpace ℝ (Fin N) :=
  WithLp.toLp 2 (alphaList S hS)

/-- `λ₋`, the absolute values of the negative eigenvalues of `S` in decreasing
order padded with zeros, as a Euclidean vector. -/
noncomputable def lamMinusVec (S : Matrix (Fin N) (Fin N) ℝ) (hS : S.IsHermitian) :
    EuclideanSpace ℝ (Fin N) :=
  WithLp.toLp 2 (betaList S hS)

/-- The pairing score `𝒫 = 1 - ‖λ₊ - λ₋‖/‖S‖`. -/
noncomputable def pairingScoreSym (S : Matrix (Fin N) (Fin N) ℝ) (hS : S.IsHermitian) : ℝ :=
  1 - ‖lamPlusVec S hS - lamMinusVec S hS‖ / Real.sqrt (frobSq S)

lemma norm_sq_euclidean (v : Fin N → ℝ) :
    ‖(WithLp.toLp 2 v : EuclideanSpace ℝ (Fin N))‖ ^ 2 = ∑ j, v j ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq]
  simp [Real.norm_eq_abs, sq_abs]

/-- `‖S‖² = ‖λ₊‖² + ‖λ₋‖²`. -/
lemma frobSq_eq_norm_sq_add_norm_sq (S : Matrix (Fin N) (Fin N) ℝ) (hS : S.IsHermitian) :
    frobSq S = ‖lamPlusVec S hS‖ ^ 2 + ‖lamMinusVec S hS‖ ^ 2 := by
  rw [lamPlusVec, lamMinusVec, norm_sq_euclidean, norm_sq_euclidean,
    sum_alpha_sq_add_beta_sq hS, frobSq_of_isHermitian hS]

/-- The paper's argument for two vectors `a = λ₊ ≠ 0`, `b = λ₋` with
`F = ‖S‖² = ‖a‖² + ‖b‖²`. -/
lemma pairing_bound_aux (a b : EuclideanSpace ℝ (Fin N)) (F : ℝ)
    (hF2 : F = ‖a‖ ^ 2 + ‖b‖ ^ 2) (hplus : a ≠ 0) :
    1 - ‖a - b‖ / Real.sqrt F = 1 - Real.sqrt (1 - 2 * inner ℝ a b / F) ∧
      1 - ‖a - b‖ / Real.sqrt F
        ≤ 1 - |1 - ‖b‖ / ‖a‖| / Real.sqrt (1 + (‖b‖ / ‖a‖) ^ 2) ∧
      (1 - ‖a - b‖ / Real.sqrt F
          = 1 - |1 - ‖b‖ / ‖a‖| / Real.sqrt (1 + (‖b‖ / ‖a‖) ^ 2) ↔
        b = (‖b‖ / ‖a‖) • a) := by
  have ha : 0 < ‖a‖ := norm_pos_iff.mpr hplus
  have hF : 0 < F := by rw [hF2]; positivity
  have hρ : ‖b‖ = ‖b‖ / ‖a‖ * ‖a‖ := (div_mul_cancel₀ _ ha.ne').symm
  -- `‖S‖ = √(1 + ρ²) ‖λ₊‖`
  have hnormS : Real.sqrt F = Real.sqrt (1 + (‖b‖ / ‖a‖) ^ 2) * ‖a‖ := by
    rw [hF2, show ‖a‖ ^ 2 + ‖b‖ ^ 2 = (1 + (‖b‖ / ‖a‖) ^ 2) * ‖a‖ ^ 2 by
      field_simp, Real.sqrt_mul (by positivity), Real.sqrt_sq ha.le]
  -- the equality: `‖λ₊ - λ₋‖² = ‖S‖² - 2⟨λ₊, λ₋⟩`
  have hsub : ‖a - b‖ ^ 2 = F - 2 * inner ℝ a b := by
    rw [norm_sub_sq_real, hF2]; ring
  have heq : ‖a - b‖ / Real.sqrt F = Real.sqrt (1 - 2 * inner ℝ a b / F) := by
    rw [← Real.sqrt_sq (norm_nonneg (a - b)), ← Real.sqrt_div' _ hF.le, hsub]
    congr 1
    field_simp
  -- the triangle inequality: `‖λ₊ - λ₋‖ ≥ |‖λ₊‖ - ‖λ₋‖| = |1 - ρ| ‖λ₊‖`
  have htri : |1 - ‖b‖ / ‖a‖| * ‖a‖ = |‖a‖ - ‖b‖| := by
    conv_rhs => rw [hρ]
    rw [← one_sub_mul, abs_mul, abs_of_pos ha]
  have hsqrt : 0 < Real.sqrt F := Real.sqrt_pos.mpr hF
  have hbound : |1 - ‖b‖ / ‖a‖| / Real.sqrt (1 + (‖b‖ / ‖a‖) ^ 2)
      = |‖a‖ - ‖b‖| / Real.sqrt F := by
    rw [hnormS, ← htri, mul_div_mul_right _ _ ha.ne']
  refine ⟨by rw [heq], ?_, ?_⟩
  · rw [hbound]
    gcongr
    exact abs_norm_sub_norm_le a b
  · -- equality iff one of `λ±` is a nonnegative multiple of the other
    rw [hbound, sub_right_inj, div_left_inj' hsqrt.ne', ← sameRay_iff_norm_sub]
    constructor
    · intro h
      obtain ⟨r, hr, hrb⟩ := h.exists_nonneg_left hplus
      have hr' : r = ‖b‖ / ‖a‖ := by
        rw [← hrb, norm_smul, Real.norm_eq_abs, abs_of_nonneg hr]
        field_simp
      rw [← hr', hrb]
    · intro h
      rw [h]
      exact SameRay.sameRay_nonneg_smul_right a (by positivity)

/-- **Lemma (pairing bound).**  For a real symmetric matrix `S` with `λ₊ ≠ 0`
and `ρ = ‖λ₋‖/‖λ₊‖`,
`𝒫 = 1 - √(1 - 2⟨λ₊, λ₋⟩/‖S‖²) ≤ 1 - |1 - ρ|/√(1 + ρ²)`,
with equality if and only if `λ₋ = ρ λ₊`. -/
theorem pairing_bound (S : Matrix (Fin N) (Fin N) ℝ) (hS : S.IsHermitian)
    (hplus : lamPlusVec S hS ≠ 0) :
    let ρ := ‖lamMinusVec S hS‖ / ‖lamPlusVec S hS‖
    pairingScoreSym S hS
        = 1 - Real.sqrt (1 - 2 * inner ℝ (lamPlusVec S hS) (lamMinusVec S hS) / frobSq S) ∧
      pairingScoreSym S hS ≤ 1 - |1 - ρ| / Real.sqrt (1 + ρ ^ 2) ∧
      (pairingScoreSym S hS = 1 - |1 - ρ| / Real.sqrt (1 + ρ ^ 2) ↔
        lamMinusVec S hS = ρ • lamPlusVec S hS) :=
  pairing_bound_aux _ _ _ (frobSq_eq_norm_sq_add_norm_sq S hS) hplus

end Appendices.MainText
