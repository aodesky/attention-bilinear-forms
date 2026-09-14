/-
Appendix "Random matrices",
Lemma (Symmetric and skew energy identities).

Let `L ∈ M_N(ℝ)`, and write `S = ½(L + Lᵀ)`, `T = ½(L - Lᵀ)`. Then

    ‖L‖² = ‖S‖² + ‖T‖²,

and

    ‖S‖² = ½‖L‖² + ½ tr(L²),    ‖T‖² = ½‖L‖² - ½ tr(L²).

Consequently,

    ‖S‖²/‖L‖² = ½ + tr(L²)/(2‖L‖²),
    ‖T‖²/‖L‖² = ½ - tr(L²)/(2‖L‖²).

All norms are Frobenius norms; we state the identities for the squared
norm `frobSq`.  The consequent ratio identities require `L ≠ 0`.
-/
import Appendices.Common

namespace Appendices

open Matrix BigOperators

variable {N : ℕ} (L : Matrix (Fin N) (Fin N) ℝ)

private lemma trace_mul_self_eq_sum :
    Matrix.trace (L * L) = ∑ i, ∑ j, L i j * L j i := by
  simp [Matrix.trace, Matrix.diag, Matrix.mul_apply]

/-- `‖S‖² = ½‖L‖² + ½ tr(L²)`. -/
theorem frobSq_symPart :
    frobSq (symPart L) = frobSq L / 2 + Matrix.trace (L * L) / 2 := by
  rw [frobSq_eq_sum_sq, frobSq_eq_sum_sq, trace_mul_self_eq_sum]
  calc ∑ i, ∑ j, symPart L i j ^ 2
      = ∑ i, ∑ j, (L i j ^ 2 / 4 + L i j * L j i / 2 + L j i ^ 2 / 4) := by
        refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
        simp only [symPart, Matrix.smul_apply, Matrix.add_apply,
          Matrix.transpose_apply, smul_eq_mul]
        ring
    _ = (∑ i, ∑ j, L i j ^ 2) / 4 + (∑ i, ∑ j, L i j * L j i) / 2
          + (∑ i, ∑ j, L j i ^ 2) / 4 := by
        simp [Finset.sum_add_distrib, Finset.sum_div]
    _ = (∑ i, ∑ j, L i j ^ 2) / 2 + (∑ i, ∑ j, L i j * L j i) / 2 := by
        rw [Finset.sum_comm (f := fun i j => L j i ^ 2)]
        ring

/-- `‖T‖² = ½‖L‖² - ½ tr(L²)`. -/
theorem frobSq_skewPart :
    frobSq (skewPart L) = frobSq L / 2 - Matrix.trace (L * L) / 2 := by
  rw [frobSq_eq_sum_sq, frobSq_eq_sum_sq, trace_mul_self_eq_sum]
  calc ∑ i, ∑ j, skewPart L i j ^ 2
      = ∑ i, ∑ j, (L i j ^ 2 / 4 - L i j * L j i / 2 + L j i ^ 2 / 4) := by
        refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
        simp only [skewPart, Matrix.smul_apply, Matrix.sub_apply,
          Matrix.transpose_apply, smul_eq_mul]
        ring
    _ = (∑ i, ∑ j, L i j ^ 2) / 4 - (∑ i, ∑ j, L i j * L j i) / 2
          + (∑ i, ∑ j, L j i ^ 2) / 4 := by
        simp [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_div]
    _ = (∑ i, ∑ j, L i j ^ 2) / 2 - (∑ i, ∑ j, L i j * L j i) / 2 := by
        rw [Finset.sum_comm (f := fun i j => L j i ^ 2)]
        ring

/-- `‖L‖² = ‖S‖² + ‖T‖²`: the symmetric and antisymmetric parts are
orthogonal in the Frobenius inner product. -/
theorem frobSq_eq_frobSq_symPart_add_frobSq_skewPart :
    frobSq L = frobSq (symPart L) + frobSq (skewPart L) := by
  rw [frobSq_symPart, frobSq_skewPart]; ring

/-- `‖S‖²/‖L‖² = ½ + tr(L²)/(2‖L‖²)` for `L ≠ 0`. -/
theorem frobSq_symPart_div (hL : L ≠ 0) :
    frobSq (symPart L) / frobSq L
      = 1 / 2 + Matrix.trace (L * L) / (2 * frobSq L) := by
  have h : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  rw [frobSq_symPart]
  field_simp

/-- `‖T‖²/‖L‖² = ½ - tr(L²)/(2‖L‖²)` for `L ≠ 0`. -/
theorem frobSq_skewPart_div (hL : L ≠ 0) :
    frobSq (skewPart L) / frobSq L
      = 1 / 2 - Matrix.trace (L * L) / (2 * frobSq L) := by
  have h : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  rw [frobSq_skewPart]
  field_simp

end Appendices
