/-
Main text, §"The shape of a real bilinear form", Definition (defn:shapeMap).

Let `L ∈ M_N(ℝ)` with `L = S + T`, `S` symmetric, `T` antisymmetric.
Let `α` (resp. `β`) list the positive eigenvalues (resp. the absolute
values of the negative eigenvalues) of `S` in decreasing order,
appending zeros so that the two lists have equal length.  The
*shape* of `L` (for `‖L‖ = 1`) is `s(L) = (a, b, c, d)`, where

    a = ‖T‖²,  b = 2⟨α, β⟩,  c = ‖(α-β)₊‖²,  d = ‖(α-β)₋‖²,

where `x₊ = max(x,0)` and `x₋ = max(-x,0)` entrywise.  The shape
is extended to all `L ≠ 0` invariantly under positive scaling; we build the normalization `‖L‖² = 1` into the definitions by
dividing by `‖L‖²`.

Encoding: both lists are padded with zeros to the common length `N`;
`α` is realized as the decreasing sort of the vector `(max(λ_i, 0))_i`
of positive parts of the eigenvalues (sorting appends the padding
zeros automatically), and `β` as the decreasing sort of
`(max(-λ_i, 0))_i`.  The shape does not depend on the amount of zero
padding.
-/
import Appendices.Common

namespace Appendices

open BigOperators

/-! ### Decreasing rearrangement -/

/-- The decreasing (nonincreasing) rearrangement of a tuple. -/
noncomputable def sortDesc {n : ℕ} (v : Fin n → ℝ) : Fin n → ℝ :=
  fun i => (v ∘ Tuple.sort v) (Fin.rev i)

lemma sortDesc_antitone {n : ℕ} (v : Fin n → ℝ) : Antitone (sortDesc v) :=
  fun _ _ h => Tuple.monotone_sort v (Fin.rev_le_rev.mpr h)

/-- `sortDesc v` is `v` composed with a permutation. -/
lemma sortDesc_eq_comp_perm {n : ℕ} (v : Fin n → ℝ) :
    sortDesc v = v ∘ (Fin.revPerm.trans (Tuple.sort v)) := rfl

/-- Sums of a function of the entries are invariant under `sortDesc`. -/
lemma sum_comp_sortDesc {n : ℕ} (v : Fin n → ℝ) (g : ℝ → ℝ) :
    ∑ i, g (sortDesc v i) = ∑ i, g (v i) := by
  rw [sortDesc_eq_comp_perm]
  exact Fintype.sum_equiv (Fin.revPerm.trans (Tuple.sort v))
    (fun i => g (v ((Fin.revPerm.trans (Tuple.sort v)) i))) (fun i => g (v i))
    (fun i => rfl) |>.symm ▸ rfl

/-! ### The lists `α` and `β` and the shape -/

variable {N : ℕ}

/-- The symmetric part of a real matrix is Hermitian. -/
lemma symPart_isHermitian (L : Matrix (Fin N) (Fin N) ℝ) :
    (symPart L).IsHermitian := by
  have h := symPart_isSymm L
  simpa [Matrix.IsHermitian, Matrix.conjTranspose, Matrix.IsSymm] using h

/-- `α`: the positive eigenvalues of a symmetric matrix `S` in
decreasing order, padded with zeros to length `N`. -/
noncomputable def alphaList (S : Matrix (Fin N) (Fin N) ℝ)
    (hS : S.IsHermitian) : Fin N → ℝ :=
  sortDesc (fun i => max (hS.eigenvalues i) 0)

/-- `β`: the absolute values of the negative eigenvalues of a
symmetric matrix `S` in decreasing order, padded with zeros to
length `N`. -/
noncomputable def betaList (S : Matrix (Fin N) (Fin N) ℝ)
    (hS : S.IsHermitian) : Fin N → ℝ :=
  sortDesc (fun i => max (-hS.eigenvalues i) 0)

/-- The coordinate `a = ‖T‖²/‖L‖²` of the shape. -/
noncomputable def shapeA (L : Matrix (Fin N) (Fin N) ℝ) : ℝ :=
  frobSq (skewPart L) / frobSq L

/-- The coordinate `b = 2⟨α, β⟩/‖L‖²` of the shape, where `α`, `β` are the lists of the
symmetric part of `L`. -/
noncomputable def shapeB (L : Matrix (Fin N) (Fin N) ℝ) : ℝ :=
  2 * (∑ j, alphaList (symPart L) (symPart_isHermitian L) j *
    betaList (symPart L) (symPart_isHermitian L) j) / frobSq L

/-- The coordinate `c = ‖(α-β)₊‖²/‖L‖²` of the shape. -/
noncomputable def shapeC (L : Matrix (Fin N) (Fin N) ℝ) : ℝ :=
  (∑ j, max (alphaList (symPart L) (symPart_isHermitian L) j -
    betaList (symPart L) (symPart_isHermitian L) j) 0 ^ 2) / frobSq L

/-- The coordinate `d = ‖(α-β)₋‖²/‖L‖²` of the shape. -/
noncomputable def shapeD (L : Matrix (Fin N) (Fin N) ℝ) : ℝ :=
  (∑ j, max (betaList (symPart L) (symPart_isHermitian L) j -
    alphaList (symPart L) (symPart_isHermitian L) j) 0 ^ 2) / frobSq L

end Appendices
