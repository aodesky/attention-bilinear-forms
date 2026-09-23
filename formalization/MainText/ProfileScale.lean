/-
Main text, §"The profile of a real bilinear form": Lemma `lem:profile-scale`.

    Let `L ∈ M_N(ℝ)` with symmetric part `S ≠ 0`, and write
    `π(L) = (a, b, c, d)` and `π(S) = (0, b_S, c_S, d_S)`.  Then
    `(b, c, d) = (1 - a)(b_S, c_S, d_S)`.

The paper's proof: `b, c, d` are computed from the eigenvalues of `S` and
divided by `‖L‖²`, whereas `b_S, c_S, d_S` are the same quantities divided by
`‖S‖²`; and `‖S‖²/‖L‖² = 1 - ‖T‖²/‖L‖² = 1 - a`.

We follow it: `profile_symPart_numerators` records that the numerators agree
(the symmetric part of `S` is `S` itself), `one_sub_profileA` is the identity
`1 - a = ‖S‖²/‖L‖²`, and the three equalities follow by dividing.  The first
coordinate of `π(S)` is `0`, as the statement writes it (`profileA_symPart`).
-/
import Appendices.Common
import Appendices.Profile
import Appendices.RandomMatrices.EnergyIdentities

namespace Appendices.MainText

open Matrix

variable {N : ℕ}

/-- The symmetric part of a symmetric part is itself. -/
lemma symPart_symPart (L : Matrix (Fin N) (Fin N) ℝ) :
    symPart (symPart L) = symPart L := by
  have h : (symPart L)ᵀ = symPart L := symPart_isSymm L
  rw [symPart, h, ← two_smul ℝ (symPart L), smul_smul]
  norm_num

/-- The antisymmetric part of a symmetric part is zero. -/
lemma skewPart_symPart (L : Matrix (Fin N) (Fin N) ℝ) :
    skewPart (symPart L) = 0 := by
  have h : (symPart L)ᵀ = symPart L := symPart_isSymm L
  rw [skewPart, h, sub_self, smul_zero]

/-- The lists `α`, `β` depend only on the matrix, not on the proof that it is
symmetric. -/
lemma alphaList_congr {S S' : Matrix (Fin N) (Fin N) ℝ} (h : S = S')
    (hS : S.IsHermitian) (hS' : S'.IsHermitian) :
    alphaList S hS = alphaList S' hS' := by
  subst h; rfl

lemma betaList_congr {S S' : Matrix (Fin N) (Fin N) ℝ} (h : S = S')
    (hS : S.IsHermitian) (hS' : S'.IsHermitian) :
    betaList S hS = betaList S' hS' := by
  subst h; rfl

/-- `π(S) = (0, b_S, c_S, d_S)`: the first coordinate of the profile of a
symmetric part is zero. -/
theorem profileA_symPart (L : Matrix (Fin N) (Fin N) ℝ) :
    profileA (symPart L) = 0 := by
  rw [profileA, skewPart_symPart]
  simp [frobSq]

/-- `‖S‖²/‖L‖² = 1 - ‖T‖²/‖L‖² = 1 - a`. -/
theorem one_sub_profileA (L : Matrix (Fin N) (Fin N) ℝ) (hL : L ≠ 0) :
    1 - profileA L = frobSq (symPart L) / frobSq L := by
  have hfs : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  rw [profileA, eq_div_iff hfs, sub_mul, div_mul_cancel₀ _ hfs, one_mul,
    frobSq_eq_frobSq_symPart_add_frobSq_skewPart L]
  ring

/-- **Lemma (profile scale).**  If the symmetric part `S` of `L` is nonzero,
then `(b, c, d) = (1 - a)(b_S, c_S, d_S)`, where `π(L) = (a, b, c, d)` and
`π(S) = (0, b_S, c_S, d_S)`. -/
theorem profile_scale (L : Matrix (Fin N) (Fin N) ℝ) (hS : symPart L ≠ 0) :
    profileA (symPart L) = 0 ∧
    profileB L = (1 - profileA L) * profileB (symPart L) ∧
    profileC L = (1 - profileA L) * profileC (symPart L) ∧
    profileD L = (1 - profileA L) * profileD (symPart L) := by
  have hL : L ≠ 0 := by
    rintro rfl
    exact hS (by simp [symPart])
  have hfL : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  have hfS : frobSq (symPart L) ≠ 0 :=
    fun h => hS ((frobSq_eq_zero_iff _).mp h)
  -- `b, c, d` and `b_S, c_S, d_S` are the same eigenvalue expressions of `S`
  have ha := alphaList_congr (symPart_symPart L)
    (symPart_isHermitian (symPart L)) (symPart_isHermitian L)
  have hb := betaList_congr (symPart_symPart L)
    (symPart_isHermitian (symPart L)) (symPart_isHermitian L)
  refine ⟨profileA_symPart L, ?_, ?_, ?_⟩ <;>
  · rw [one_sub_profileA L hL]
    simp only [profileB, profileC, profileD]
    rw [ha, hb]
    field_simp

end Appendices.MainText
