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

The proof follows the paper's.  With the Frobenius inner product
`⟨X, Y⟩ = tr(X Yᵀ)` (`frobInner`), symmetric and antisymmetric
matrices are orthogonal, by the paper's chain of trace identities
(`frobInner_eq_zero_of_isSymm_of_transpose_eq_neg`); since `L = S + T`
this gives `‖L‖² = ‖S‖² + ‖T‖²`.  Expanding
`‖S‖² = ¼(⟨L,L⟩ + 2⟨L,Lᵀ⟩ + ⟨Lᵀ,Lᵀ⟩)` with `⟨L,L⟩ = ⟨Lᵀ,Lᵀ⟩ = ‖L‖²`
and `⟨L,Lᵀ⟩ = tr(L²)` gives the formula for `‖S‖²`, and the formula
for `‖T‖²` follows by subtraction.
-/
import Appendices.Common

namespace Appendices

open Matrix BigOperators

variable {N : ℕ}

/-- The Frobenius inner product `⟨X, Y⟩ = tr(X Yᵀ)`. -/
def frobInner (X Y : Matrix (Fin N) (Fin N) ℝ) : ℝ :=
  Matrix.trace (X * Yᵀ)

lemma frobInner_comm (X Y : Matrix (Fin N) (Fin N) ℝ) : frobInner X Y = frobInner Y X := by
  rw [frobInner, frobInner, ← Matrix.trace_transpose, Matrix.transpose_mul,
    Matrix.transpose_transpose]

lemma frobInner_add_left (X Y Z : Matrix (Fin N) (Fin N) ℝ) :
    frobInner (X + Y) Z = frobInner X Z + frobInner Y Z := by
  simp [frobInner, Matrix.add_mul, Matrix.trace_add]

lemma frobInner_add_right (X Y Z : Matrix (Fin N) (Fin N) ℝ) :
    frobInner X (Y + Z) = frobInner X Y + frobInner X Z := by
  simp [frobInner, Matrix.mul_add, Matrix.trace_add]

lemma frobInner_smul_left (c : ℝ) (X Y : Matrix (Fin N) (Fin N) ℝ) :
    frobInner (c • X) Y = c * frobInner X Y := by
  simp [frobInner, Matrix.trace_smul]

lemma frobInner_smul_right (c : ℝ) (X Y : Matrix (Fin N) (Fin N) ℝ) :
    frobInner X (c • Y) = c * frobInner X Y := by
  simp [frobInner, Matrix.trace_smul]

/-- `‖X‖² = ⟨X, X⟩`. -/
lemma frobSq_eq_frobInner (X : Matrix (Fin N) (Fin N) ℝ) : frobSq X = frobInner X X := by
  rw [frobSq, frobInner, Matrix.trace_mul_comm]

/-- Symmetric and antisymmetric matrices are orthogonal for the Frobenius inner
product: `⟨S,T⟩ = tr(S Tᵀ) = -tr(ST) = -tr((ST)ᵀ) = -tr(TᵀSᵀ) = tr(TS) = tr(ST)`,
so `tr(ST) = 0` and `⟨S,T⟩ = 0`. -/
theorem frobInner_eq_zero_of_isSymm_of_transpose_eq_neg {S T : Matrix (Fin N) (Fin N) ℝ}
    (hS : S.IsSymm) (hT : Tᵀ = -T) : frobInner S T = 0 := by
  have h1 : frobInner S T = -Matrix.trace (S * T) := by
    rw [frobInner, hT, Matrix.mul_neg, Matrix.trace_neg]
  have h2 : Matrix.trace (S * T) = -Matrix.trace (S * T) :=
    calc Matrix.trace (S * T) = Matrix.trace ((S * T)ᵀ) := (Matrix.trace_transpose _).symm
      _ = Matrix.trace (Tᵀ * Sᵀ) := by rw [Matrix.transpose_mul]
      _ = -Matrix.trace (T * S) := by
          rw [hT, hS.eq, Matrix.neg_mul, Matrix.trace_neg]
      _ = -Matrix.trace (S * T) := by rw [Matrix.trace_mul_comm]
  rw [h1]
  linarith

variable (L : Matrix (Fin N) (Fin N) ℝ)

/-- `‖L‖² = ‖S‖² + ‖T‖²`: the symmetric and antisymmetric parts are
orthogonal in the Frobenius inner product, and `L = S + T`. -/
theorem frobSq_eq_frobSq_symPart_add_frobSq_skewPart :
    frobSq L = frobSq (symPart L) + frobSq (skewPart L) := by
  have h0 : frobInner (symPart L) (skewPart L) = 0 :=
    frobInner_eq_zero_of_isSymm_of_transpose_eq_neg (symPart_isSymm L) (skewPart_transpose L)
  conv_lhs => rw [← symPart_add_skewPart L]
  rw [frobSq_eq_frobInner, frobSq_eq_frobInner, frobSq_eq_frobInner, frobInner_add_left,
    frobInner_add_right, frobInner_add_right, frobInner_comm (skewPart L) (symPart L), h0]
  ring

/-- `‖S‖² = ½‖L‖² + ½ tr(L²)`, by expanding
`‖S‖² = ¼(⟨L,L⟩ + 2⟨L,Lᵀ⟩ + ⟨Lᵀ,Lᵀ⟩)`. -/
theorem frobSq_symPart :
    frobSq (symPart L) = frobSq L / 2 + Matrix.trace (L * L) / 2 := by
  have hLT : frobInner L Lᵀ = Matrix.trace (L * L) := by
    rw [frobInner, Matrix.transpose_transpose]
  have hTT : frobInner Lᵀ Lᵀ = frobSq L := by
    rw [frobInner, Matrix.transpose_transpose, frobSq]
  have hexp : frobSq (symPart L)
      = (frobInner L L + 2 * frobInner L Lᵀ + frobInner Lᵀ Lᵀ) / 4 := by
    rw [frobSq_eq_frobInner, symPart, frobInner_smul_left, frobInner_smul_right,
      frobInner_add_left, frobInner_add_right, frobInner_add_right,
      frobInner_comm Lᵀ L]
    ring
  rw [hexp, ← frobSq_eq_frobInner, hLT, hTT]
  ring

/-- `‖T‖² = ½‖L‖² - ½ tr(L²)`, by subtraction from `‖L‖²`. -/
theorem frobSq_skewPart :
    frobSq (skewPart L) = frobSq L / 2 - Matrix.trace (L * L) / 2 := by
  have h := frobSq_eq_frobSq_symPart_add_frobSq_skewPart L
  rw [frobSq_symPart] at h
  linarith

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
