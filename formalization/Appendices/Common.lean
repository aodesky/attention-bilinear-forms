/-
Common definitions used throughout the appendices.

All matrix norms in the paper are Frobenius norms, `‖L‖ = √(tr (Lᵀ L))`.
We work with the *squared* Frobenius norm `frobSq`, which avoids square
roots; every statement in the paper involving `‖·‖` is squared.

For `L ∈ M_N(ℝ)` the paper writes `S = ½(L + Lᵀ)` and `T = ½(L - Lᵀ)`
for the symmetric and antisymmetric parts of `L`; these are `symPart L`
and `skewPart L` below.
-/
import Mathlib

namespace Appendices

open Matrix BigOperators

variable {N : ℕ}

/-- The squared Frobenius norm `‖L‖² = tr (Lᵀ L)` of a real square matrix. -/
def frobSq (L : Matrix (Fin N) (Fin N) ℝ) : ℝ :=
  Matrix.trace (Lᵀ * L)

/-- The symmetric part `S = ½(L + Lᵀ)`. -/
noncomputable def symPart (L : Matrix (Fin N) (Fin N) ℝ) : Matrix (Fin N) (Fin N) ℝ :=
  (1 / 2 : ℝ) • (L + Lᵀ)

/-- The antisymmetric part `T = ½(L - Lᵀ)`. -/
noncomputable def skewPart (L : Matrix (Fin N) (Fin N) ℝ) : Matrix (Fin N) (Fin N) ℝ :=
  (1 / 2 : ℝ) • (L - Lᵀ)

/-- The squared Frobenius norm is the sum of the squared entries. -/
lemma frobSq_eq_sum_sq (L : Matrix (Fin N) (Fin N) ℝ) :
    frobSq L = ∑ i, ∑ j, L i j ^ 2 := by
  simp only [frobSq, Matrix.trace, Matrix.diag, Matrix.mul_apply,
    Matrix.transpose_apply, sq]
  exact Finset.sum_comm

lemma frobSq_nonneg (L : Matrix (Fin N) (Fin N) ℝ) : 0 ≤ frobSq L := by
  rw [frobSq_eq_sum_sq]
  exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg _

lemma frobSq_eq_zero_iff (L : Matrix (Fin N) (Fin N) ℝ) :
    frobSq L = 0 ↔ L = 0 := by
  rw [frobSq_eq_sum_sq]
  constructor
  · intro h
    ext i j
    have hi := (Finset.sum_eq_zero_iff_of_nonneg
      (fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg (L i j))).mp h i
      (Finset.mem_univ i)
    have hij := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j _ => sq_nonneg (L i j))).mp hi j (Finset.mem_univ j)
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hij
  · rintro rfl; simp

lemma symPart_add_skewPart (L : Matrix (Fin N) (Fin N) ℝ) :
    symPart L + skewPart L = L := by
  simp only [symPart, skewPart, smul_add, smul_sub]
  abel_nf
  rw [← smul_assoc]
  norm_num

lemma symPart_isSymm (L : Matrix (Fin N) (Fin N) ℝ) : (symPart L).IsSymm := by
  simp [symPart, Matrix.IsSymm, Matrix.transpose_smul, Matrix.transpose_add,
    add_comm]

lemma skewPart_transpose (L : Matrix (Fin N) (Fin N) ℝ) :
    (skewPart L)ᵀ = -skewPart L := by
  simp only [skewPart, Matrix.transpose_smul, Matrix.transpose_sub,
    Matrix.transpose_transpose, smul_neg, ← neg_sub L Lᵀ, smul_neg]

end Appendices
