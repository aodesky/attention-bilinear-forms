/-
Supporting facts about the profile `π(L) = (a, b, c, d)` of
`Appendices.Profile`, used by the main-text results on profiles.

* `sortDesc` depends only on the multiset of entries
  (`sortDesc_eq_of_map_eq`), so the lists `α`, `β` of a symmetric matrix
  depend only on its multiset of eigenvalues, that is, on the roots of its
  characteristic polynomial (`alphaList_eq_of_charpoly_eq`).
* The eigenvalues of a diagonal matrix are its diagonal entries
  (`charpoly_roots_diagonal`).
* The profile is invariant under orthogonal changes of coordinates
  `L ↦ Uᵀ L U` (`profile_conj`).  This is what licenses the paper's
  "by an orthogonal change of coordinates" in the proof of
  `prop:rankOneLocus`.
-/
import Appendices.Common
import Appendices.Profile

namespace Appendices.MainText

open Matrix Polynomial

variable {N : ℕ}

/-! ### Sorting depends only on the multiset of entries -/

/-- Two antitone tuples with the same multiset of entries are equal. -/
lemma antitone_eq_of_map_eq {n : ℕ} {x y : Fin n → ℝ}
    (hx : Antitone x) (hy : Antitone y)
    (h : Multiset.map x Finset.univ.val = Multiset.map y Finset.univ.val) :
    x = y := by
  have hx' : Monotone (x ∘ Fin.rev) := fun i j hij => hx (Fin.rev_le_rev.mpr hij)
  have hy' : Monotone (y ∘ Fin.rev) := fun i j hij => hy (Fin.rev_le_rev.mpr hij)
  have hrev : ∀ z : Fin n → ℝ,
      Multiset.map (z ∘ Fin.rev) Finset.univ.val = Multiset.map z Finset.univ.val := by
    intro z
    rw [← Multiset.map_map]
    congr 1
    exact Multiset.map_univ_val_equiv Fin.revPerm
  have h' : Multiset.map (x ∘ Fin.rev) Finset.univ.val
      = Multiset.map (y ∘ Fin.rev) Finset.univ.val := by
    rw [hrev, hrev, h]
  rw [Fin.univ_val_map, Fin.univ_val_map, Multiset.coe_eq_coe] at h'
  have := List.ofFn_injective (h'.eq_of_sortedLE hx'.sortedLE_ofFn hy'.sortedLE_ofFn)
  funext i
  simpa using congrFun this (Fin.rev i)

/-- The multiset of entries of `sortDesc v` is that of `v`. -/
lemma map_sortDesc {n : ℕ} (v : Fin n → ℝ) :
    Multiset.map (sortDesc v) Finset.univ.val = Multiset.map v Finset.univ.val := by
  rw [sortDesc_eq_comp_perm, ← Multiset.map_map]
  congr 1
  exact Multiset.map_univ_val_equiv _

/-- `sortDesc` depends only on the multiset of entries. -/
lemma sortDesc_eq_of_map_eq {n : ℕ} {v w : Fin n → ℝ}
    (h : Multiset.map v Finset.univ.val = Multiset.map w Finset.univ.val) :
    sortDesc v = sortDesc w :=
  antitone_eq_of_map_eq (sortDesc_antitone v) (sortDesc_antitone w)
    (by rw [map_sortDesc, map_sortDesc, h])

/-- An antitone tuple is its own decreasing rearrangement. -/
lemma sortDesc_eq_self {n : ℕ} {v : Fin n → ℝ} (hv : Antitone v) : sortDesc v = v :=
  antitone_eq_of_map_eq (sortDesc_antitone v) hv (map_sortDesc v)

/-! ### Eigenvalues through the characteristic polynomial -/

/-- The multiset of eigenvalues of a real symmetric matrix is the multiset of
roots of its characteristic polynomial. -/
lemma map_eigenvalues_eq_roots {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian) :
    Multiset.map hS.eigenvalues Finset.univ.val = S.charpoly.roots := by
  rw [hS.roots_charpoly_eq_eigenvalues]
  rfl

/-- The roots of the characteristic polynomial of a diagonal matrix are its
diagonal entries. -/
lemma charpoly_roots_diagonal (d : Fin N → ℝ) :
    (diagonal d).charpoly.roots = Multiset.map d Finset.univ.val := by
  rw [Matrix.charpoly_diagonal]
  have : (∏ i, (X - C (d i))) = ((Finset.univ.val.map d).map (fun a => X - C a)).prod := by
    rw [Multiset.map_map]; rfl
  rw [this, Polynomial.roots_multiset_prod_X_sub_C]

/-- The lists `α` of two symmetric matrices with the same characteristic
polynomial agree. -/
lemma alphaList_eq_of_charpoly_eq {S S' : Matrix (Fin N) (Fin N) ℝ}
    (hS : S.IsHermitian) (hS' : S'.IsHermitian) (h : S.charpoly = S'.charpoly) :
    alphaList S hS = alphaList S' hS' := by
  refine sortDesc_eq_of_map_eq ?_
  change Multiset.map ((fun x => max x 0) ∘ hS.eigenvalues) _
    = Multiset.map ((fun x => max x 0) ∘ hS'.eigenvalues) _
  rw [← Multiset.map_map, ← Multiset.map_map,
    map_eigenvalues_eq_roots, map_eigenvalues_eq_roots, h]

/-- The lists `β` of two symmetric matrices with the same characteristic
polynomial agree. -/
lemma betaList_eq_of_charpoly_eq {S S' : Matrix (Fin N) (Fin N) ℝ}
    (hS : S.IsHermitian) (hS' : S'.IsHermitian) (h : S.charpoly = S'.charpoly) :
    betaList S hS = betaList S' hS' := by
  refine sortDesc_eq_of_map_eq ?_
  change Multiset.map ((fun x => max (-x) 0) ∘ hS.eigenvalues) _
    = Multiset.map ((fun x => max (-x) 0) ∘ hS'.eigenvalues) _
  rw [← Multiset.map_map, ← Multiset.map_map,
    map_eigenvalues_eq_roots, map_eigenvalues_eq_roots, h]

/-- For a diagonal symmetric part, `α` is the decreasing rearrangement of the
positive parts of the diagonal. -/
lemma alphaList_diagonal {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian)
    {d : Fin N → ℝ} (hd : S = diagonal d) :
    alphaList S hS = sortDesc (fun i => max (d i) 0) := by
  refine sortDesc_eq_of_map_eq ?_
  change Multiset.map ((fun x => max x 0) ∘ hS.eigenvalues) _
    = Multiset.map ((fun x => max x 0) ∘ d) _
  rw [← Multiset.map_map, ← Multiset.map_map,
    map_eigenvalues_eq_roots, hd, charpoly_roots_diagonal]

/-- For a diagonal symmetric part, `β` is the decreasing rearrangement of the
negative parts of the diagonal. -/
lemma betaList_diagonal {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian)
    {d : Fin N → ℝ} (hd : S = diagonal d) :
    betaList S hS = sortDesc (fun i => max (-d i) 0) := by
  refine sortDesc_eq_of_map_eq ?_
  change Multiset.map ((fun x => max (-x) 0) ∘ hS.eigenvalues) _
    = Multiset.map ((fun x => max (-x) 0) ∘ d) _
  rw [← Multiset.map_map, ← Multiset.map_map,
    map_eigenvalues_eq_roots, hd, charpoly_roots_diagonal]

/-! ### Orthogonal changes of coordinates -/

variable {U : Matrix (Fin N) (Fin N) ℝ}

lemma mul_transpose_eq_one_of (hU : Uᵀ * U = 1) : U * Uᵀ = 1 :=
  mul_eq_one_comm.mp hU

/-- The characteristic polynomial is invariant under `A ↦ Uᵀ A U`. -/
lemma charpoly_conj (hU : Uᵀ * U = 1) (A : Matrix (Fin N) (Fin N) ℝ) :
    (Uᵀ * A * U).charpoly = A.charpoly := by
  rw [Matrix.charpoly_mul_comm, ← Matrix.mul_assoc, mul_transpose_eq_one_of hU,
    Matrix.one_mul]

/-- The squared Frobenius norm is invariant under `A ↦ Uᵀ A U`. -/
lemma frobSq_conj (hU : Uᵀ * U = 1) (A : Matrix (Fin N) (Fin N) ℝ) :
    frobSq (Uᵀ * A * U) = frobSq A := by
  have hU' := mul_transpose_eq_one_of hU
  unfold frobSq
  rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose]
  have : Uᵀ * (Aᵀ * U) * (Uᵀ * A * U) = Uᵀ * (Aᵀ * A * U) := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc U, hU', Matrix.one_mul]
  rw [this, Matrix.trace_mul_comm, Matrix.mul_assoc, hU', Matrix.mul_one]

lemma symPart_conj (A : Matrix (Fin N) (Fin N) ℝ) :
    symPart (Uᵀ * A * U) = Uᵀ * symPart A * U := by
  simp only [symPart, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc]

lemma skewPart_conj (A : Matrix (Fin N) (Fin N) ℝ) :
    skewPart (Uᵀ * A * U) = Uᵀ * skewPart A * U := by
  simp only [skewPart, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_assoc]

/-- **The profile is invariant under orthogonal changes of coordinates.** -/
theorem profile_conj (hU : Uᵀ * U = 1) (L : Matrix (Fin N) (Fin N) ℝ) :
    profileA (Uᵀ * L * U) = profileA L ∧ profileB (Uᵀ * L * U) = profileB L ∧
      profileC (Uᵀ * L * U) = profileC L ∧ profileD (Uᵀ * L * U) = profileD L := by
  have hcp : (symPart (Uᵀ * L * U)).charpoly = (symPart L).charpoly := by
    rw [symPart_conj, charpoly_conj hU]
  have ha := alphaList_eq_of_charpoly_eq (symPart_isHermitian (Uᵀ * L * U))
    (symPart_isHermitian L) hcp
  have hb := betaList_eq_of_charpoly_eq (symPart_isHermitian (Uᵀ * L * U))
    (symPart_isHermitian L) hcp
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [profileA, profileA, skewPart_conj, frobSq_conj hU, frobSq_conj hU]
  · rw [profileB, profileB, ha, hb, frobSq_conj hU]
  · rw [profileC, profileC, ha, hb, frobSq_conj hU]
  · rw [profileD, profileD, ha, hb, frobSq_conj hU]

end Appendices.MainText
