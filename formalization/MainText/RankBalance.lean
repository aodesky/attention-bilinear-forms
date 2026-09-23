/-
Main text, §"Why heads are balanced": Lemma `lem:rank-balance`.

    Let `L ∈ M_N(ℝ)` with `rank L ≤ n`, and let `(n₊, n₋, n₀)` be the
    signature of its symmetric part `S = ½(L + Lᵀ)`.  Then `n₊ ≤ n`
    and `n₋ ≤ n`.

The paper's proof is subspace-dimension counting.  The quadratic form of
`S` vanishes on `K = ker L`, because `2 xᵀSx = xᵀLx + xᵀLᵀx = 0` there.
A subspace `V₊` on which the form is positive definite therefore meets
`K` trivially, so `dim V₊ + dim K ≤ N`, and `dim K ≥ N - n` gives
`dim V₊ ≤ n`.

We follow that argument verbatim.  `finrank_le_of_posDefOn` is its core,
phrased for subspaces: `posDefOn S V` says the quadratic form of `S` is
positive on the nonzero vectors of `V`, and the conclusion bounds
`finrank V`.  `rank_balance` is the lemma as stated, with `n₊` and `n₋` the
numbers of positive and negative eigenvalues of `S` (with multiplicity): `V₊`
is the span of the eigenvectors with positive eigenvalues (`eigSpan`, of
dimension `n₊`), on which the form is positive definite; the negative
eigenvectors span a subspace on which the form of `-S`, the symmetric part of
`-L`, is positive definite.
-/
import Appendices.BalancedBimodalSpectra.SimplexFaces

namespace Appendices.MainText

open Matrix Module

variable {N : ℕ}

/-- The quadratic form `x ↦ xᵀ S x` of a real square matrix. -/
def quadForm (S : Matrix (Fin N) (Fin N) ℝ) (x : Fin N → ℝ) : ℝ :=
  x ⬝ᵥ (S *ᵥ x)

/-- `S` is positive definite on the subspace `V`. -/
def posDefOn (S : Matrix (Fin N) (Fin N) ℝ) (V : Submodule ℝ (Fin N → ℝ)) : Prop :=
  ∀ x ∈ V, x ≠ 0 → 0 < quadForm S x

/-- The quadratic form of the symmetric part vanishes on `ker L`:
this is the paper's `2 xᵀSx = xᵀLx + xᵀLᵀx = 0`. -/
lemma quadForm_symPart_eq_zero_of_mulVec_eq_zero
    (L : Matrix (Fin N) (Fin N) ℝ) {x : Fin N → ℝ} (hx : L *ᵥ x = 0) :
    quadForm (symPart L) x = 0 := by
  have hL : x ⬝ᵥ (L *ᵥ x) = 0 := by rw [hx, dotProduct_zero]
  have hLT : x ⬝ᵥ (Lᵀ *ᵥ x) = 0 := by
    rw [dotProduct_mulVec, vecMul_transpose, dotProduct_comm]
    exact hL
  simp only [quadForm, symPart, Matrix.smul_mulVec, dotProduct_smul, add_mulVec,
    dotProduct_add, hL, hLT, add_zero, smul_eq_mul, mul_zero]

/-- A subspace on which `S` is positive definite meets the kernel of `L` trivially,
when the quadratic form of `S` vanishes on that kernel. -/
lemma disjoint_of_posDefOn (L : Matrix (Fin N) (Fin N) ℝ)
    {V : Submodule ℝ (Fin N → ℝ)} (hV : posDefOn (symPart L) V) :
    Disjoint V (LinearMap.ker (Matrix.mulVecLin L)) := by
  rw [Submodule.disjoint_def]
  intro x hxV hxK
  by_contra hne
  have hzero : quadForm (symPart L) x = 0 :=
    quadForm_symPart_eq_zero_of_mulVec_eq_zero L (by simpa using hxK)
  exact absurd hzero (ne_of_gt (hV x hxV hne))

/-- **Lemma (rank balance).**  If `rank L ≤ n` and the symmetric part of `L` is
positive definite on a subspace `V`, then `dim V ≤ n`.

Applied to the span of the positive eigenvectors of `S` this is the paper's
`n₊ ≤ n`; applied to `-S`, it is `n₋ ≤ n`. -/
theorem finrank_le_of_posDefOn (L : Matrix (Fin N) (Fin N) ℝ) (n : ℕ)
    (hrank : L.rank ≤ n) {V : Submodule ℝ (Fin N → ℝ)} (hV : posDefOn (symPart L) V) :
    finrank ℝ V ≤ n := by
  classical
  set f := Matrix.mulVecLin L with hf
  have hdisj := disjoint_of_posDefOn L hV
  -- `dim V + dim (ker f) ≤ N` since the two subspaces are disjoint
  have hsum : finrank ℝ V + finrank ℝ (LinearMap.ker f) ≤ N := by
    have := Submodule.finrank_add_finrank_le_of_disjoint hdisj
    simpa using this
  -- rank-nullity: `dim (ker f) = N - rank L`
  have hrn : finrank ℝ (LinearMap.ker f) + finrank ℝ (LinearMap.range f) = N := by
    have := LinearMap.finrank_range_add_finrank_ker f
    simpa [add_comm] using this
  have hrange : finrank ℝ (LinearMap.range f) = L.rank := rfl
  omega
/-- **Lemma (rank balance).**  If `rank L ≤ n`, the symmetric part
`S = ½(L + Lᵀ)` has at most `n` positive and at most `n` negative eigenvalues,
counted with multiplicity. -/
theorem rank_balance (L : Matrix (Fin N) (Fin N) ℝ) (n : ℕ) (hrank : L.rank ≤ n) :
    (Finset.univ.filter fun i => 0 < (symPart_isHermitian L).eigenvalues i).card ≤ n ∧
      (Finset.univ.filter fun i => (symPart_isHermitian L).eigenvalues i < 0).card ≤ n := by
  set hS := symPart_isHermitian L
  constructor
  · -- `V₊`, the span of the eigenvectors with positive eigenvalues
    set s := Finset.univ.filter fun i => 0 < hS.eigenvalues i
    rw [← finrank_eigSpan hS s]
    refine finrank_le_of_posDefOn L n hrank fun x hx hx0 => ?_
    have := quadform_gt_on_eigSpan hS s 0 (fun i hi => (Finset.mem_filter.mp hi).2) x hx hx0
    simpa [quadForm] using this
  · -- the span of the eigenvectors with negative eigenvalues, for `-L`
    set s := Finset.univ.filter fun i => hS.eigenvalues i < 0
    rw [← finrank_eigSpan hS s]
    have hrneg : (-L).rank = L.rank := by
      unfold Matrix.rank
      have : (-L).mulVecLin = -L.mulVecLin := by
        ext x; simp
      rw [this, LinearMap.range_neg]
    refine finrank_le_of_posDefOn (-L) n (by rwa [hrneg]) fun x hx hx0 => ?_
    have := quadform_lt_on_eigSpan hS s 0 (fun i hi => (Finset.mem_filter.mp hi).2) x hx hx0
    have hneg : symPart (-L) = -symPart L := by
      simp only [symPart, Matrix.transpose_neg, ← neg_add, smul_neg]
    rw [quadForm, hneg, Matrix.neg_mulVec, dotProduct_neg]
    simpa using this

end Appendices.MainText
