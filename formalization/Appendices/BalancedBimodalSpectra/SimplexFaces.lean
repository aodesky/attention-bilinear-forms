/-
Theorem (thm:simplex-faces) of the main text, proved in Appendix
"Balanced bimodal spectra".

For `0 ≠ L ∈ M_N(ℝ)` write `L = S + T` with `S` symmetric and `T`
antisymmetric, let `α` (resp. `β`) list the positive eigenvalues
(resp. the absolute values of the negative eigenvalues) of `S` in
decreasing order, padded with zeros to a common length, and let

    a = ‖T‖²/‖L‖²,          b = 2⟨α, β⟩/‖L‖²,
    c = ‖(α-β)₊‖²/‖L‖²,     d = ‖(α-β)₋‖²/‖L‖²

be the weights of `L` (`weightA`, …, `weightD` of
`Appendices.Weights`).  Then `a + b + c + d = 1`, each weight is
nonnegative, and:

1. `a = 0` iff `L` is symmetric, and `a = 1` iff `L` is antisymmetric;
2. `b = 0` iff `S` is semidefinite, and `b = 1` iff `L` is symmetric
   with `α = β`;
3. `d = 0` iff `S = H + P` for symmetric matrices `H` and `P` such
   that the spectrum of `H` is symmetric about zero and `P` is
   positive semidefinite; equivalently `α_j ≥ β_j` for every `j`.
   In that case `H` and `P` may be chosen to commute with `S` and
   with each other, with `‖P‖² = c‖L‖²`.  Moreover `d = 1` iff `L` is
   symmetric negative semidefinite;
4. as (3) with `S = H - P`, `β_j ≥ α_j`, `‖P‖² = d‖L‖²`, and `c = 1`
   iff `L` is symmetric positive semidefinite.

"The spectrum of `H` is symmetric about zero" is formalized as: the
multiset of eigenvalues of `H` (with multiplicity) is invariant under
negation.  Over `ℝ`, `Matrix.IsHermitian` means symmetric, and
`Matrix.PosSemidef` includes symmetry.
-/
import Appendices.Common
import Appendices.Weights
import Appendices.RandomMatrices.EnergyIdentities

namespace Appendices

open Matrix BigOperators Finset

variable {N : ℕ}

/-- The multiset of eigenvalues, with multiplicity, of a real
Hermitian (= symmetric) matrix. -/
noncomputable def eigenvalueMultiset {A : Matrix (Fin N) (Fin N) ℝ}
    (hA : A.IsHermitian) : Multiset ℝ :=
  Finset.univ.val.map hA.eigenvalues

/-- The spectrum of `H`, with multiplicity, is symmetric about zero. -/
def SpectrumSymmAboutZero {H : Matrix (Fin N) (Fin N) ℝ}
    (hH : H.IsHermitian) : Prop :=
  (eigenvalueMultiset hH).map (fun x => -x) = eigenvalueMultiset hH

/-! ### Preliminaries on `sortDesc` and the sorted eigenvalue list -/

private lemma card_filter_comp_equiv {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (σ : α ≃ β) (p : β → Prop)
    [DecidablePred p] :
    (Finset.univ.filter fun a => p (σ a)).card
      = (Finset.univ.filter p).card := by
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact Fintype.card_congr (σ.subtypeEquiv fun a => Iff.rfl)

/-- A rearrangement of `v` that is antitone must be `sortDesc v`. -/
private lemma sortDesc_eq_of_antitone {n : ℕ} {v w : Fin n → ℝ}
    (hw : Antitone w) (π : Equiv.Perm (Fin n)) (hvw : v = w ∘ π) :
    sortDesc v = w := by
  have h1 : sortDesc v = w ∘ ((Fin.revPerm.trans (Tuple.sort v)).trans π) := by
    rw [sortDesc_eq_comp_perm, hvw]
    rfl
  have h2 : Antitone (w ∘ ((Fin.revPerm.trans (Tuple.sort v)).trans π)) := by
    rw [← h1]; exact sortDesc_antitone v
  have h3 := Tuple.unique_antitone (f := w)
    (σ := (Fin.revPerm.trans (Tuple.sort v)).trans π) (τ := Equiv.refl _)
    h2 (by simpa using hw)
  rw [h1, h3]
  funext i
  rfl

section EvDesc

variable {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian)

/-- The eigenvalues of `S` in decreasing order, reindexed by `Fin N`. -/
private noncomputable def evDesc : Fin N → ℝ :=
  hS.eigenvalues₀ ∘ finCongr (Fintype.card_fin N).symm

private lemma evDesc_antitone : Antitone (evDesc hS) := by
  intro i j hij
  exact hS.eigenvalues₀_antitone hij

/-- The permutation carrying the descending list to `hS.eigenvalues`. -/
private noncomputable def evPerm : Equiv.Perm (Fin N) :=
  ((Fintype.equivOfCardEq (Fintype.card_fin _)).symm.trans
    (finCongr (Fintype.card_fin N)))

private lemma eigenvalues_eq_evDesc_comp :
    hS.eigenvalues = evDesc hS ∘ evPerm (N := N) := by
  funext i
  show hS.eigenvalues₀ _ = hS.eigenvalues₀ _
  congr 1

/-- `α` is `max(λ_j, 0)` down the decreasing eigenvalue list. -/
private lemma alphaList_eq : alphaList S hS = fun j => max (evDesc hS j) 0 := by
  apply sortDesc_eq_of_antitone
  · intro i j hij
    exact max_le_max (evDesc_antitone hS hij) le_rfl
  · funext i
    simp only [Function.comp_apply]
    rw [eigenvalues_eq_evDesc_comp hS]
    rfl

/-- `β` is `max(-λ_j, 0)` up the increasing end of the eigenvalue
list: `β_j = max(-λ_{rev j}, 0)`. -/
private lemma betaList_eq :
    betaList S hS = fun j => max (-(evDesc hS (Fin.rev j))) 0 := by
  apply sortDesc_eq_of_antitone (π := (evPerm (N := N)).trans Fin.revPerm)
  · intro i j hij
    have : Fin.rev j ≤ Fin.rev i := Fin.rev_le_rev.mpr hij
    exact max_le_max (neg_le_neg (evDesc_antitone hS this)) le_rfl
  · funext i
    simp only [Function.comp_apply, Equiv.trans_apply, Fin.revPerm_apply]
    rw [eigenvalues_eq_evDesc_comp hS]
    simp [Fin.rev_rev]

private lemma alphaList_nonneg (j : Fin N) : 0 ≤ alphaList S hS j := by
  rw [alphaList_eq]; exact le_max_right _ _

private lemma betaList_nonneg (j : Fin N) : 0 ≤ betaList S hS j := by
  rw [betaList_eq]; exact le_max_right _ _

end EvDesc

/-! ### Traces and Frobenius norms via eigenvalues -/

section Spectral

variable {A : Matrix (Fin N) (Fin N) ℝ} (hA : A.IsHermitian)

private lemma star_mul_self_unitary :
    star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
      * (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) = 1 := by
  simp


/-- The spectral theorem over `ℝ`, in matrix-product form. -/
private lemma spectral_real :
    A = (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
        * diagonal hA.eigenvalues
        * star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) := by
  conv_lhs => rw [hA.spectral_theorem]
  rw [Unitary.conjStarAlgAut_apply]
  congr 2

private lemma trace_conj_unitary (M : Matrix (Fin N) (Fin N) ℝ) :
    Matrix.trace ((hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * M
      * star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)) = Matrix.trace M := by
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, star_mul_self_unitary hA,
    Matrix.one_mul]

/-- For a real symmetric matrix, `‖A‖² = Σ λ_i²`. -/
private lemma frobSq_of_isHermitian :
    frobSq A = ∑ i, hA.eigenvalues i ^ 2 := by
  have ht : Aᵀ = A := by
    rw [← Matrix.conjTranspose_eq_transpose_of_trivial, hA.eq]
  have hAA : A * A
      = (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
        * (diagonal hA.eigenvalues * diagonal hA.eigenvalues)
        * star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) := by
    conv_lhs => rw [spectral_real hA]
    calc ((hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
            * diagonal hA.eigenvalues
            * star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
          * ((hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
            * diagonal hA.eigenvalues
            * star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
        = (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
          * diagonal hA.eigenvalues
          * ((star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
              * (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
            * (diagonal hA.eigenvalues
              * star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))) := by
          simp only [Matrix.mul_assoc]
      _ = _ := by
          rw [star_mul_self_unitary hA]
          simp only [Matrix.one_mul, Matrix.mul_assoc]
  rw [frobSq, ht, hAA, trace_conj_unitary hA, Matrix.diagonal_mul_diagonal,
    Matrix.trace_diagonal]
  exact Finset.sum_congr rfl fun i _ => (sq _).symm

/-- Under the closed formulas for `α` and `β`,
`α_j² + β_j²` summed over `j` recovers `Σ λ_i²`. -/
private lemma sum_alpha_sq_add_beta_sq {S : Matrix (Fin N) (Fin N) ℝ}
    (hS : S.IsHermitian) :
    (∑ j, alphaList S hS j ^ 2) + ∑ j, betaList S hS j ^ 2
      = ∑ i, hS.eigenvalues i ^ 2 := by
  have ha : ∑ j, alphaList S hS j ^ 2
      = ∑ j, max (hS.eigenvalues j) 0 ^ 2 :=
    sum_comp_sortDesc _ (fun x => x ^ 2)
  have hb : ∑ j, betaList S hS j ^ 2
      = ∑ j, max (-(hS.eigenvalues j)) 0 ^ 2 :=
    sum_comp_sortDesc _ (fun x => x ^ 2)
  rw [ha, hb, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rcases le_total 0 (hS.eigenvalues i) with h | h
  · rw [max_eq_left h, max_eq_right (by linarith)]
    ring
  · rw [max_eq_right h, max_eq_left (by linarith)]
    ring

end Spectral

/-! ### The partition of the energy and the weight `a` -/

variable (L : Matrix (Fin N) (Fin N) ℝ)

/-- The weights are nonnegative. -/
theorem weightA_nonneg : 0 ≤ weightA L :=
  div_nonneg (frobSq_nonneg _) (frobSq_nonneg _)

theorem weightB_nonneg : 0 ≤ weightB L := by
  unfold weightB
  refine div_nonneg ?_ (frobSq_nonneg L)
  refine mul_nonneg (by norm_num) (Finset.sum_nonneg fun j _ => ?_)
  exact mul_nonneg (alphaList_nonneg (symPart_isHermitian L) j)
    (betaList_nonneg (symPart_isHermitian L) j)

theorem weightC_nonneg : 0 ≤ weightC L :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (frobSq_nonneg _)

theorem weightD_nonneg : 0 ≤ weightD L :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (frobSq_nonneg _)

/-- The partition of the energy: `a + b + c + d = 1` for `L ≠ 0`. -/
theorem weight_sum_eq_one (hL : L ≠ 0) :
    weightA L + weightB L + weightC L + weightD L = 1 := by
  have hfs : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  set hS := symPart_isHermitian L
  have key : frobSq (skewPart L)
      + (2 * ∑ j, alphaList (symPart L) hS j * betaList (symPart L) hS j
        + ∑ j, max (alphaList (symPart L) hS j - betaList (symPart L) hS j) 0 ^ 2
        + ∑ j, max (betaList (symPart L) hS j - alphaList (symPart L) hS j) 0 ^ 2)
      = frobSq L := by
    have hsplit : ∀ j,
        2 * (alphaList (symPart L) hS j * betaList (symPart L) hS j)
          + max (alphaList (symPart L) hS j - betaList (symPart L) hS j) 0 ^ 2
          + max (betaList (symPart L) hS j - alphaList (symPart L) hS j) 0 ^ 2
        = alphaList (symPart L) hS j ^ 2 + betaList (symPart L) hS j ^ 2 := by
      intro j
      rcases le_total (alphaList (symPart L) hS j) (betaList (symPart L) hS j)
        with h | h
      · rw [max_eq_right (by linarith), max_eq_left (by linarith)]
        ring
      · rw [max_eq_left (by linarith), max_eq_right (by linarith)]
        ring
    have h1 : 2 * ∑ j, alphaList (symPart L) hS j * betaList (symPart L) hS j
          + ∑ j, max (alphaList (symPart L) hS j - betaList (symPart L) hS j) 0 ^ 2
          + ∑ j, max (betaList (symPart L) hS j - alphaList (symPart L) hS j) 0 ^ 2
        = (∑ j, alphaList (symPart L) hS j ^ 2)
          + ∑ j, betaList (symPart L) hS j ^ 2 := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib,
        ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => hsplit j
    rw [h1, sum_alpha_sq_add_beta_sq hS, ← frobSq_of_isHermitian hS,
      frobSq_eq_frobSq_symPart_add_frobSq_skewPart L]
    ring
  unfold weightA weightB weightC weightD
  field_simp
  linarith [key]

/-- Theorem (1), first half: `a = 0` iff `L` is symmetric. -/
theorem weightA_eq_zero_iff (hL : L ≠ 0) : weightA L = 0 ↔ L.IsSymm := by
  have hfs : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  rw [weightA, div_eq_zero_iff, or_iff_left hfs, frobSq_eq_zero_iff,
    Matrix.IsSymm]
  constructor
  · intro h
    have h2 : L - Lᵀ = 0 := by
      rcases smul_eq_zero.mp h with h' | h'
      · norm_num at h'
      · exact h'
    exact (sub_eq_zero.mp h2).symm
  · intro h
    simp [skewPart, h]

/-- Theorem (1), second half: `a = 1` iff `L` is antisymmetric. -/
theorem weightA_eq_one_iff (hL : L ≠ 0) : weightA L = 1 ↔ Lᵀ = -L := by
  have hfs : frobSq L ≠ 0 := fun h => hL ((frobSq_eq_zero_iff L).mp h)
  rw [weightA, div_eq_one_iff_eq hfs]
  rw [frobSq_eq_frobSq_symPart_add_frobSq_skewPart L]
  constructor
  · intro h
    have h0 : frobSq (symPart L) = 0 := by linarith
    have h1 : symPart L = 0 := (frobSq_eq_zero_iff _).mp h0
    have h2 : L + Lᵀ = 0 := by
      rcases smul_eq_zero.mp h1 with h' | h'
      · norm_num at h'
      · exact h'
    exact eq_neg_of_add_eq_zero_right h2
  · intro h
    have h1 : symPart L = 0 := by
      simp [symPart, h]
    rw [h1, (frobSq_eq_zero_iff (0 : Matrix (Fin N) (Fin N) ℝ)).mpr rfl]
    ring

/-! ### Eigenvalue multisets -/

private lemma multiset_map_perm (σ : Equiv.Perm (Fin N)) (f : Fin N → ℝ) :
    Finset.univ.val.map (f ∘ ⇑σ) = Finset.univ.val.map f := by
  have h2 : Multiset.map (⇑σ) Finset.univ.val = Finset.univ.val := by
    simp
  calc Finset.univ.val.map (f ∘ ⇑σ) = (Finset.univ.val.map ⇑σ).map f := by
        rw [Multiset.map_map]
    _ = Finset.univ.val.map f := by rw [h2]

private lemma card_filter_eq_countP (f : Fin N → ℝ) (p : ℝ → Prop)
    [DecidablePred p] :
    (Finset.univ.filter fun i => p (f i)).card
      = Multiset.countP p (Finset.univ.val.map f) := by
  rw [Multiset.countP_map]
  rfl

/-- If the spectrum of `H` is symmetric about zero then, for every `t`,
`H` has as many eigenvalues below `-t` as above `t`. -/
private lemma card_lt_neg_eq_card_gt {H : Matrix (Fin N) (Fin N) ℝ}
    (hH : H.IsHermitian) (hsym : SpectrumSymmAboutZero hH) (t : ℝ) :
    (Finset.univ.filter fun i => hH.eigenvalues i < -t).card
      = (Finset.univ.filter fun i => t < hH.eigenvalues i).card := by
  rw [card_filter_eq_countP hH.eigenvalues (fun x => x < -t),
    card_filter_eq_countP hH.eigenvalues (fun x => t < x),
    Multiset.countP_eq_card_filter, Multiset.countP_eq_card_filter]
  have h1 : Multiset.filter (fun x => x < -t) (Finset.univ.val.map hH.eigenvalues)
      = Multiset.map (fun x => -x)
        (Multiset.filter (fun x => t < x) (Finset.univ.val.map hH.eigenvalues)) := by
    have hsym' : Multiset.map (fun x => -x)
        (Multiset.map hH.eigenvalues Finset.univ.val)
        = Multiset.map hH.eigenvalues Finset.univ.val := hsym
    conv_lhs => rw [← hsym']
    rw [Multiset.filter_map]
    congr 1
    apply Multiset.filter_congr
    intro x _
    constructor
    · intro hx; simp only [Function.comp_apply] at hx ⊢; linarith
    · intro hx; simp only [Function.comp_apply] at hx ⊢; linarith
  rw [h1, Multiset.card_map]

/-! ### Orthonormal eigenvectors as plain vectors -/

section Counting

variable {A : Matrix (Fin N) (Fin N) ℝ} (hA : A.IsHermitian)

/-- The orthonormal eigenvectors of `A`, as plain vectors in `ℝ^N`. -/
private noncomputable def evec (i : Fin N) : Fin N → ℝ :=
  ⇑(hA.eigenvectorBasis i)

private lemma mulVec_evec (j : Fin N) :
    A *ᵥ evec hA j = hA.eigenvalues j • evec hA j :=
  hA.mulVec_eigenvectorBasis j

private lemma evec_dot (i j : Fin N) :
    evec hA i ⬝ᵥ evec hA j = if i = j then 1 else 0 := by
  have h := star_mul_self_unitary hA
  have h2 : (star (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
      * (hA.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)) i j
      = (1 : Matrix (Fin N) (Fin N) ℝ) i j := by rw [h]
  rw [Matrix.mul_apply, Matrix.one_apply] at h2
  rw [← h2]
  simp only [dotProduct]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Matrix.star_apply, star_trivial]
  simp [evec]

private lemma sum_smul_evec_dot (s : Finset (Fin N)) (c : {i // i ∈ s} → ℝ)
    (j : {i // i ∈ s}) :
    (∑ i : {i // i ∈ s}, c i • evec hA i) ⬝ᵥ evec hA j = c j := by
  rw [sum_dotProduct]
  rw [Finset.sum_eq_single j]
  · rw [smul_dotProduct, evec_dot, if_pos rfl, smul_eq_mul, mul_one]
  · intro b _ hb
    rw [smul_dotProduct, evec_dot, if_neg (fun h => hb (Subtype.ext h)),
      smul_eq_mul, mul_zero]
  · intro h
    exact absurd (Finset.mem_univ j) h

private lemma evec_linearIndependent (s : Finset (Fin N)) :
    LinearIndependent ℝ (fun i : {i // i ∈ s} => evec hA i) := by
  rw [linearIndependent_iff']
  intro t g hg j hj
  have h := congrArg (fun v => v ⬝ᵥ evec hA j) hg
  simp only [zero_dotProduct] at h
  rw [sum_dotProduct] at h
  rw [Finset.sum_eq_single j] at h
  · rwa [smul_dotProduct, evec_dot, if_pos rfl, smul_eq_mul, mul_one] at h
  · intro b _ hb
    rw [smul_dotProduct, evec_dot, if_neg (fun hbj => hb (Subtype.ext hbj)),
      smul_eq_mul, mul_zero]
  · intro h'
    exact absurd hj h'

/-- The span of the eigenvectors indexed by `s`. -/
private noncomputable def eigSpan (s : Finset (Fin N)) :
    Submodule ℝ (Fin N → ℝ) :=
  Submodule.span ℝ (Set.range fun i : {i // i ∈ s} => evec hA i)

private lemma finrank_eigSpan (s : Finset (Fin N)) :
    Module.finrank ℝ (eigSpan hA s) = s.card := by
  rw [eigSpan, finrank_span_eq_card (evec_linearIndependent hA s)]
  exact Fintype.card_coe s

private lemma mem_eigSpan_iff (s : Finset (Fin N)) (x : Fin N → ℝ) :
    x ∈ eigSpan hA s
      ↔ ∃ c : {i // i ∈ s} → ℝ, x = ∑ i : {i // i ∈ s}, c i • evec hA i := by
  rw [eigSpan, Submodule.mem_span_range_iff_exists_fun]
  constructor
  · rintro ⟨c, hc⟩; exact ⟨c, hc.symm⟩
  · rintro ⟨c, hc⟩; exact ⟨c, hc.symm⟩

private lemma dot_self_expansion (s : Finset (Fin N)) (c : {i // i ∈ s} → ℝ) :
    (∑ i : {i // i ∈ s}, c i • evec hA i) ⬝ᵥ (∑ i : {i // i ∈ s}, c i • evec hA i)
      = ∑ i : {i // i ∈ s}, c i ^ 2 := by
  rw [dotProduct_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [dotProduct_smul, sum_smul_evec_dot hA s c j, smul_eq_mul, sq]

private lemma quadform_expansion (s : Finset (Fin N)) (c : {i // i ∈ s} → ℝ) :
    (∑ i : {i // i ∈ s}, c i • evec hA i)
        ⬝ᵥ (A *ᵥ ∑ i : {i // i ∈ s}, c i • evec hA i)
      = ∑ i : {i // i ∈ s}, hA.eigenvalues i * c i ^ 2 := by
  have hAx : A *ᵥ (∑ i : {i // i ∈ s}, c i • evec hA i)
      = ∑ i : {i // i ∈ s}, (c i * hA.eigenvalues i) • evec hA i := by
    rw [show A *ᵥ (∑ i : {i // i ∈ s}, c i • evec hA i)
        = A.mulVecLin (∑ i : {i // i ∈ s}, c i • evec hA i) from rfl, map_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [map_smul, Matrix.mulVecLin_apply, mulVec_evec hA, smul_smul, mul_comm]
  rw [hAx, dotProduct_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [dotProduct_smul, sum_smul_evec_dot hA s c j, smul_eq_mul]
  ring

end Counting

/-! ### The dimension bound and the counting inequality -/

section Counting2

variable {A : Matrix (Fin N) (Fin N) ℝ} (hA : A.IsHermitian)

private lemma quadform_ge_on_eigSpan (s : Finset (Fin N)) (r : ℝ)
    (hs : ∀ i ∈ s, r ≤ hA.eigenvalues i) :
    ∀ x ∈ eigSpan hA s, r * (x ⬝ᵥ x) ≤ x ⬝ᵥ (A *ᵥ x) := by
  intro x hx
  obtain ⟨c, rfl⟩ := (mem_eigSpan_iff hA s _).mp hx
  rw [dot_self_expansion hA s c, quadform_expansion hA s c, Finset.mul_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  have h := hs j.1 j.2
  nlinarith [sq_nonneg (c j)]

private lemma quadform_le_on_eigSpan (s : Finset (Fin N)) (r : ℝ)
    (hs : ∀ i ∈ s, hA.eigenvalues i ≤ r) :
    ∀ x ∈ eigSpan hA s, x ⬝ᵥ (A *ᵥ x) ≤ r * (x ⬝ᵥ x) := by
  intro x hx
  obtain ⟨c, rfl⟩ := (mem_eigSpan_iff hA s _).mp hx
  rw [dot_self_expansion hA s c, quadform_expansion hA s c, Finset.mul_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  have h := hs j.1 j.2
  nlinarith [sq_nonneg (c j)]

private lemma quadform_lt_on_eigSpan (s : Finset (Fin N)) (r : ℝ)
    (hs : ∀ i ∈ s, hA.eigenvalues i < r) :
    ∀ x ∈ eigSpan hA s, x ≠ 0 → x ⬝ᵥ (A *ᵥ x) < r * (x ⬝ᵥ x) := by
  intro x hx hx0
  obtain ⟨c, rfl⟩ := (mem_eigSpan_iff hA s _).mp hx
  rw [dot_self_expansion hA s c, quadform_expansion hA s c, Finset.mul_sum]
  have hc : c ≠ 0 := by
    rintro rfl
    simp at hx0
  obtain ⟨j, hj⟩ := Function.ne_iff.mp hc
  refine Finset.sum_lt_sum (fun i _ => ?_) ⟨j, Finset.mem_univ j, ?_⟩
  · have h := (hs i.1 i.2).le
    nlinarith [sq_nonneg (c i)]
  · have h1 := hs j.1 j.2
    have h2 : 0 < c j ^ 2 := (sq_nonneg _).lt_of_ne (Ne.symm (pow_ne_zero 2 hj))
    nlinarith

private lemma quadform_gt_on_eigSpan (s : Finset (Fin N)) (r : ℝ)
    (hs : ∀ i ∈ s, r < hA.eigenvalues i) :
    ∀ x ∈ eigSpan hA s, x ≠ 0 → r * (x ⬝ᵥ x) < x ⬝ᵥ (A *ᵥ x) := by
  intro x hx hx0
  obtain ⟨c, rfl⟩ := (mem_eigSpan_iff hA s _).mp hx
  rw [dot_self_expansion hA s c, quadform_expansion hA s c, Finset.mul_sum]
  have hc : c ≠ 0 := by
    rintro rfl
    simp at hx0
  obtain ⟨j, hj⟩ := Function.ne_iff.mp hc
  refine Finset.sum_lt_sum (fun i _ => ?_) ⟨j, Finset.mem_univ j, ?_⟩
  · have h := (hs i.1 i.2).le
    nlinarith [sq_nonneg (c i)]
  · have h1 := hs j.1 j.2
    have h2 : 0 < c j ^ 2 := (sq_nonneg _).lt_of_ne (Ne.symm (pow_ne_zero 2 hj))
    nlinarith

/-- If the quadratic form of `A` is `< r‖x‖²` on a subspace `W ∖ {0}`,
then `dim W` is at most the number of eigenvalues of `A` below `r`. -/
private lemma finrank_le_card_of_quadform_lt (r : ℝ)
    (W : Submodule ℝ (Fin N → ℝ))
    (hW : ∀ x ∈ W, x ≠ 0 → x ⬝ᵥ (A *ᵥ x) < r * (x ⬝ᵥ x)) :
    Module.finrank ℝ W
      ≤ (Finset.univ.filter fun i => hA.eigenvalues i < r).card := by
  classical
  set s := Finset.univ.filter (fun i => ¬ hA.eigenvalues i < r) with hs_def
  have hdisj : Disjoint W (eigSpan hA s) := by
    rw [Submodule.disjoint_def]
    intro x hxW hxS
    by_contra hx0
    have h1 := hW x hxW hx0
    have h2 := quadform_ge_on_eigSpan hA s r
      (fun i hi => not_lt.mp (Finset.mem_filter.mp hi).2) x hxS
    linarith
  have hle := Submodule.finrank_add_finrank_le_of_disjoint hdisj
  rw [finrank_eigSpan hA s, Module.finrank_fintype_fun_eq_card,
    Fintype.card_fin] at hle
  have hcard : (Finset.univ.filter fun i => hA.eigenvalues i < r).card
      + s.card = N := by
    rw [hs_def]
    simpa using Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Fin N)))
      (fun i => hA.eigenvalues i < r)
  omega

/-- Mirror form: quadratic form `> r‖x‖²` bounds `dim W` by the number
of eigenvalues above `r`. -/
private lemma finrank_le_card_of_quadform_gt (r : ℝ)
    (W : Submodule ℝ (Fin N → ℝ))
    (hW : ∀ x ∈ W, x ≠ 0 → r * (x ⬝ᵥ x) < x ⬝ᵥ (A *ᵥ x)) :
    Module.finrank ℝ W
      ≤ (Finset.univ.filter fun i => r < hA.eigenvalues i).card := by
  classical
  set s := Finset.univ.filter (fun i => ¬ r < hA.eigenvalues i) with hs_def
  have hdisj : Disjoint W (eigSpan hA s) := by
    rw [Submodule.disjoint_def]
    intro x hxW hxS
    by_contra hx0
    have h1 := hW x hxW hx0
    have h2 := quadform_le_on_eigSpan hA s r
      (fun i hi => not_lt.mp (Finset.mem_filter.mp hi).2) x hxS
    linarith
  have hle := Submodule.finrank_add_finrank_le_of_disjoint hdisj
  rw [finrank_eigSpan hA s, Module.finrank_fintype_fun_eq_card,
    Fintype.card_fin] at hle
  have hcard : (Finset.univ.filter fun i => r < hA.eigenvalues i).card
      + s.card = N := by
    rw [hs_def]
    simpa using Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Fin N)))
      (fun i => r < hA.eigenvalues i)
  omega

end Counting2

/-- The counting inequality: if `S = H + P` with the spectrum of `H`
symmetric about zero and `P` positive semidefinite then, for every
`t`, `S` has at most as many eigenvalues below `-t` as above `t`
(display (eqn:counting-dominance) of the paper, stated there for
`t > 0`). -/
private lemma counting_dominance {S H P : Matrix (Fin N) (Fin N) ℝ}
    (hS : S.IsHermitian) (hH : H.IsHermitian) (hP : P.PosSemidef)
    (hSHP : S = H + P) (hsym : SpectrumSymmAboutZero hH) {t : ℝ} :
    (Finset.univ.filter fun i => hS.eigenvalues i < -t).card
      ≤ (Finset.univ.filter fun i => t < hS.eigenvalues i).card := by
  have hPq : ∀ x : Fin N → ℝ, 0 ≤ x ⬝ᵥ (P *ᵥ x) := by
    intro x
    have h := hP.dotProduct_mulVec_nonneg x
    simpa using h
  have h1 : (Finset.univ.filter fun i => hS.eigenvalues i < -t).card
      ≤ (Finset.univ.filter fun i => hH.eigenvalues i < -t).card := by
    rw [← finrank_eigSpan hS (Finset.univ.filter fun i => hS.eigenvalues i < -t)]
    apply finrank_le_card_of_quadform_lt hH
    intro x hx hx0
    have hq := quadform_lt_on_eigSpan hS _ (-t)
      (fun i hi => (Finset.mem_filter.mp hi).2) x hx hx0
    have hHx : x ⬝ᵥ (H *ᵥ x) = x ⬝ᵥ (S *ᵥ x) - x ⬝ᵥ (P *ᵥ x) := by
      rw [hSHP, Matrix.add_mulVec, dotProduct_add]
      ring
    have h := hPq x
    linarith
  have h2 := card_lt_neg_eq_card_gt hH hsym t
  have h3 : (Finset.univ.filter fun i => t < hH.eigenvalues i).card
      ≤ (Finset.univ.filter fun i => t < hS.eigenvalues i).card := by
    rw [← finrank_eigSpan hH (Finset.univ.filter fun i => t < hH.eigenvalues i)]
    apply finrank_le_card_of_quadform_gt hS
    intro x hx hx0
    have hq := quadform_gt_on_eigSpan hH _ t
      (fun i hi => (Finset.mem_filter.mp hi).2) x hx hx0
    have hSx : x ⬝ᵥ (S *ᵥ x) = x ⬝ᵥ (H *ᵥ x) + x ⬝ᵥ (P *ᵥ x) := by
      rw [hSHP, Matrix.add_mulVec, dotProduct_add]
    have h := hPq x
    linarith
  omega

/-! ### Conjugation by the eigenvector unitary -/

section Conj

variable {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian)

private lemma mul_star_self_unitary :
    (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
      * star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) = 1 := by
  simp

/-- `U * diagonal v * star U`, the symmetric matrix with eigenvalue `v i`
on the `i`-th eigenvector of `S`. -/
private noncomputable def conjDiag (v : Fin N → ℝ) : Matrix (Fin N) (Fin N) ℝ :=
  (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * diagonal v
    * star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)

private lemma conjDiag_isHermitian (v : Fin N → ℝ) :
    (conjDiag hS v).IsHermitian := by
  unfold conjDiag
  have hv : (diagonal v)ᴴ = diagonal v := by
    rw [Matrix.diagonal_conjTranspose, star_trivial]
  rw [Matrix.IsHermitian, Matrix.conjTranspose_mul, Matrix.conjTranspose_mul, hv,
    ← Matrix.star_eq_conjTranspose, ← Matrix.star_eq_conjTranspose, star_star,
    Matrix.mul_assoc]

private lemma conjDiag_add (v w : Fin N → ℝ) :
    conjDiag hS v + conjDiag hS w = conjDiag hS (v + w) := by
  unfold conjDiag
  have hd : diagonal (v + w) = diagonal v + diagonal w := by
    rw [Matrix.diagonal_add]; rfl
  rw [hd, Matrix.mul_add, Matrix.add_mul]

private lemma conjDiag_mul (v w : Fin N → ℝ) :
    conjDiag hS v * conjDiag hS w = conjDiag hS (v * w) := by
  unfold conjDiag
  have hd : diagonal (v * w) = diagonal v * diagonal w := by
    rw [Matrix.diagonal_mul_diagonal]; rfl
  rw [hd]
  calc ((hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * diagonal v
        * star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
      * ((hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * diagonal w
        * star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
      = (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * diagonal v
        * ((star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
            * (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
          * (diagonal w
            * star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))) := by
        simp only [Matrix.mul_assoc]
    _ = _ := by
        rw [star_mul_self_unitary hS]
        simp only [Matrix.one_mul, Matrix.mul_assoc]

private lemma conjDiag_commute (v w : Fin N → ℝ) :
    conjDiag hS v * conjDiag hS w = conjDiag hS w * conjDiag hS v := by
  rw [conjDiag_mul, conjDiag_mul, mul_comm]

private lemma spectral_conjDiag : S = conjDiag hS hS.eigenvalues :=
  spectral_real hS

private lemma conjDiag_posSemidef_iff (v : Fin N → ℝ) :
    (conjDiag hS v).PosSemidef ↔ ∀ i, 0 ≤ v i := by
  constructor
  · intro h
    have h2 := h.mul_mul_conjTranspose_same
      (star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
    have h3 : star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
        * conjDiag hS v
        * (star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))ᴴ
        = diagonal v := by
      rw [← Matrix.star_eq_conjTranspose, star_star]
      unfold conjDiag
      calc star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
          * ((hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * diagonal v
            * star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
          * (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
          = (star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
              * (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ))
            * diagonal v
            * (star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
              * (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)) := by
            simp only [Matrix.mul_assoc]
        _ = diagonal v := by
            rw [star_mul_self_unitary hS]
            simp
    rw [h3] at h2
    exact posSemidef_diagonal_iff.mp h2
  · intro h
    have h2 := (Matrix.posSemidef_diagonal_iff.mpr h).mul_mul_conjTranspose_same
      (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)
    rwa [← Matrix.star_eq_conjTranspose] at h2

/-- The eigenvalue multiset of `U * diagonal v * star U` is the values
of `v`. -/
private lemma eigenvalueMultiset_conjDiag (v : Fin N → ℝ) :
    eigenvalueMultiset (conjDiag_isHermitian hS v)
      = Finset.univ.val.map v := by
  have hcp : (conjDiag hS v).charpoly = (diagonal v).charpoly := by
    unfold conjDiag
    rw [Matrix.charpoly_mul_comm
      ((hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) * diagonal v)
      (star (hS.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ)),
      ← Matrix.mul_assoc, star_mul_self_unitary hS, Matrix.one_mul]
  have hroots1 : (conjDiag hS v).charpoly.roots
      = eigenvalueMultiset (conjDiag_isHermitian hS v) := by
    rw [(conjDiag_isHermitian hS v).roots_charpoly_eq_eigenvalues]
    rfl
  have hroots2 : (diagonal v).charpoly.roots = Finset.univ.val.map v := by
    rw [Matrix.charpoly_diagonal]
    have : (∏ i, (Polynomial.X - Polynomial.C (v i)))
        = ((Finset.univ.val.map v).map
            (fun a => Polynomial.X - Polynomial.C a)).prod := by
      rw [Multiset.map_map]
      rfl
    rw [this, Polynomial.roots_multiset_prod_X_sub_C]
  rw [← hroots1, hcp, hroots2]

end Conj

/-! ### Counts of the lists `α` and `β` vs. counts of eigenvalues -/

section CountBridges

variable {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian)

private lemma card_alpha_gt (t : ℝ) (ht : 0 < t) :
    (Finset.univ.filter fun j => t < alphaList S hS j).card
      = (Finset.univ.filter fun i => t < hS.eigenvalues i).card := by
  have h1 : (Finset.univ.filter fun j => t < alphaList S hS j)
      = Finset.univ.filter fun j => t < evDesc hS j := by
    apply Finset.filter_congr
    intro j _
    rw [alphaList_eq]
    show t < max (evDesc hS j) 0 ↔ t < evDesc hS j
    constructor
    · intro h
      rcases max_cases (evDesc hS j) 0 with ⟨he, _⟩ | ⟨he, _⟩ <;> rw [he] at h
      · exact h
      · linarith
    · intro h
      exact h.trans_le (le_max_left _ _)
  have h2 : (Finset.univ.filter fun i => t < hS.eigenvalues i)
      = Finset.univ.filter fun i => t < evDesc hS (evPerm (N := N) i) := by
    apply Finset.filter_congr
    intro i _
    rw [eigenvalues_eq_evDesc_comp hS]
    rfl
  rw [h1, h2, card_filter_comp_equiv (evPerm (N := N)) (fun i => t < evDesc hS i)]

private lemma card_beta_gt (t : ℝ) (ht : 0 < t) :
    (Finset.univ.filter fun j => t < betaList S hS j).card
      = (Finset.univ.filter fun i => hS.eigenvalues i < -t).card := by
  have h1 : (Finset.univ.filter fun j => t < betaList S hS j)
      = Finset.univ.filter fun j => evDesc hS (Fin.rev j) < -t := by
    apply Finset.filter_congr
    intro j _
    rw [betaList_eq]
    show t < max (-(evDesc hS (Fin.rev j))) 0 ↔ evDesc hS (Fin.rev j) < -t
    constructor
    · intro h
      rcases max_cases (-(evDesc hS (Fin.rev j))) 0 with ⟨he, _⟩ | ⟨he, _⟩ <;>
        rw [he] at h
      · linarith
      · linarith
    · intro h
      have : t < -(evDesc hS (Fin.rev j)) := by linarith
      exact this.trans_le (le_max_left _ _)
  have h2 : (Finset.univ.filter fun i => hS.eigenvalues i < -t)
      = Finset.univ.filter fun i => evDesc hS (evPerm (N := N) i) < -t := by
    apply Finset.filter_congr
    intro i _
    rw [eigenvalues_eq_evDesc_comp hS]
    rfl
  rw [h1, h2, card_filter_comp_equiv (evPerm (N := N)) (fun i => evDesc hS i < -t)]
  exact card_filter_comp_equiv Fin.revPerm (fun j => evDesc hS j < -t)

/-- From the counting inequalities at every positive threshold to the
entrywise comparison of two nonincreasing lists `u`, `v` with `u ≥ 0`. -/
private lemma forall_le_of_card_le_of_antitone {u v : Fin N → ℝ}
    (hu : Antitone u) (hv : Antitone v) (hu0 : ∀ j, 0 ≤ u j)
    (h : ∀ t : ℝ, 0 < t →
      (Finset.univ.filter fun j => t < v j).card
        ≤ (Finset.univ.filter fun j => t < u j).card) :
    ∀ j, v j ≤ u j := by
  intro j
  by_contra hlt
  push_neg at hlt
  set t := (u j + v j) / 2 with ht_def
  have ht : 0 < t := by have := hu0 j; rw [ht_def]; linarith
  have h1 : u j < t := by rw [ht_def]; linarith
  have h2 : t < v j := by rw [ht_def]; linarith
  have hcardu : (Finset.univ.filter fun i => t < u i).card ≤ (j : ℕ) := by
    calc (Finset.univ.filter fun i => t < u i).card
        ≤ (Finset.Iio j).card := by
          apply Finset.card_le_card
          intro i hi
          rw [Finset.mem_filter] at hi
          rw [Finset.mem_Iio]
          by_contra hij
          push_neg at hij
          have := hu hij
          linarith [hi.2]
      _ = (j : ℕ) := Fin.card_Iio j
  have hcardv : (j : ℕ) + 1
      ≤ (Finset.univ.filter fun i => t < v i).card := by
    calc (j : ℕ) + 1 = (Finset.Iic j).card := (Fin.card_Iic j).symm
      _ ≤ (Finset.univ.filter fun i => t < v i).card := by
          apply Finset.card_le_card
          intro i hi
          rw [Finset.mem_Iic] at hi
          rw [Finset.mem_filter]
          refine ⟨Finset.mem_univ i, ?_⟩
          have := hv hi
          linarith
  have := h t ht
  omega

private lemma alphaList_antitone : Antitone (alphaList S hS) :=
  sortDesc_antitone _

private lemma betaList_antitone : Antitone (betaList S hS) :=
  sortDesc_antitone _

/-- From the counting inequalities at every positive threshold to
`β_j ≤ α_j` for every `j`. -/
private lemma forall_le_of_card_le
    (h : ∀ t : ℝ, 0 < t →
      (Finset.univ.filter fun j => t < betaList S hS j).card
        ≤ (Finset.univ.filter fun j => t < alphaList S hS j).card) :
    ∀ j, betaList S hS j ≤ alphaList S hS j :=
  forall_le_of_card_le_of_antitone (alphaList_antitone hS) (betaList_antitone hS)
    (alphaList_nonneg hS) h

/-- The same with `α` and `β` exchanged. -/
private lemma forall_le_of_card_le'
    (h : ∀ t : ℝ, 0 < t →
      (Finset.univ.filter fun j => t < alphaList S hS j).card
        ≤ (Finset.univ.filter fun j => t < betaList S hS j).card) :
    ∀ j, alphaList S hS j ≤ betaList S hS j :=
  forall_le_of_card_le_of_antitone (betaList_antitone hS) (alphaList_antitone hS)
    (betaList_nonneg hS) h

end CountBridges

/-! ### Semidefiniteness via eigenvalues -/

section Semidef

variable {A : Matrix (Fin N) (Fin N) ℝ} (hA : A.IsHermitian)

private lemma neg_posSemidef_iff_eigenvalues_nonpos :
    (-A).PosSemidef ↔ ∀ i, hA.eigenvalues i ≤ 0 := by
  have h : -A = conjDiag hA (fun i => -(hA.eigenvalues i)) := by
    have hd : diagonal (fun i => -(hA.eigenvalues i)) = -diagonal hA.eigenvalues := by
      rw [Matrix.diagonal_neg]
    conv_lhs => rw [spectral_conjDiag hA]
    unfold conjDiag
    rw [hd, Matrix.mul_neg, Matrix.neg_mul]
  rw [h, conjDiag_posSemidef_iff]
  constructor
  · intro hh i
    have := hh i
    linarith
  · intro hh i
    have := hh i
    linarith

private lemma posSemidef_iff_evDesc_nonneg :
    A.PosSemidef ↔ ∀ i, 0 ≤ evDesc hA i := by
  rw [hA.posSemidef_iff_eigenvalues_nonneg]
  constructor
  · intro h i
    have := h
    rw [eigenvalues_eq_evDesc_comp hA] at this
    have h2 := this (evPerm.symm i)
    simpa using h2
  · intro h i
    rw [eigenvalues_eq_evDesc_comp hA]
    exact h _

private lemma neg_posSemidef_iff_evDesc_nonpos :
    (-A).PosSemidef ↔ ∀ i, evDesc hA i ≤ 0 := by
  rw [neg_posSemidef_iff_eigenvalues_nonpos hA]
  constructor
  · intro h i
    have := h (evPerm.symm i)
    rw [eigenvalues_eq_evDesc_comp hA] at this
    simpa using this
  · intro h i
    rw [eigenvalues_eq_evDesc_comp hA]
    exact h _

end Semidef

/-! ### The counting inequality with `S = H - P` -/

/-- The counting inequality with `S = H - P`: for every `t`, `S` has at
most as many eigenvalues above `t` as below `-t`. -/
private lemma counting_dominance_sub {S H P : Matrix (Fin N) (Fin N) ℝ}
    (hS : S.IsHermitian) (hH : H.IsHermitian) (hP : P.PosSemidef)
    (hSHP : S = H - P) (hsym : SpectrumSymmAboutZero hH) {t : ℝ} :
    (Finset.univ.filter fun i => t < hS.eigenvalues i).card
      ≤ (Finset.univ.filter fun i => hS.eigenvalues i < -t).card := by
  have hPq : ∀ x : Fin N → ℝ, 0 ≤ x ⬝ᵥ (P *ᵥ x) := by
    intro x
    have h := hP.dotProduct_mulVec_nonneg x
    simpa using h
  have h1 : (Finset.univ.filter fun i => t < hS.eigenvalues i).card
      ≤ (Finset.univ.filter fun i => t < hH.eigenvalues i).card := by
    rw [← finrank_eigSpan hS (Finset.univ.filter fun i => t < hS.eigenvalues i)]
    apply finrank_le_card_of_quadform_gt hH
    intro x hx hx0
    have hq := quadform_gt_on_eigSpan hS _ t
      (fun i hi => (Finset.mem_filter.mp hi).2) x hx hx0
    have hHx : x ⬝ᵥ (H *ᵥ x) = x ⬝ᵥ (S *ᵥ x) + x ⬝ᵥ (P *ᵥ x) := by
      rw [hSHP, Matrix.sub_mulVec, dotProduct_sub]
      ring
    have h := hPq x
    linarith
  have h2 := card_lt_neg_eq_card_gt hH hsym t
  have h3 : (Finset.univ.filter fun i => hH.eigenvalues i < -t).card
      ≤ (Finset.univ.filter fun i => hS.eigenvalues i < -t).card := by
    rw [← finrank_eigSpan hH (Finset.univ.filter fun i => hH.eigenvalues i < -t)]
    apply finrank_le_card_of_quadform_lt hS
    intro x hx hx0
    have hq := quadform_lt_on_eigSpan hH _ (-t)
      (fun i hi => (Finset.mem_filter.mp hi).2) x hx hx0
    have hSx : x ⬝ᵥ (S *ᵥ x) = x ⬝ᵥ (H *ᵥ x) - x ⬝ᵥ (P *ᵥ x) := by
      rw [hSHP, Matrix.sub_mulVec, dotProduct_sub]
    have h := hPq x
    linarith
  omega

/-! ### The decomposition `S = H ± P` of Theorem (3) and (4) -/

section Decomposition

variable {S : Matrix (Fin N) (Fin N) ℝ} (hS : S.IsHermitian)

private lemma alphaList_apply (j : Fin N) :
    alphaList S hS j = max (evDesc hS j) 0 := by
  rw [alphaList_eq]

private lemma betaList_apply (j : Fin N) :
    betaList S hS j = max (-(evDesc hS (Fin.rev j))) 0 := by
  rw [betaList_eq]

private lemma conjDiag_neg (v : Fin N → ℝ) :
    -conjDiag hS v = conjDiag hS (-v) := by
  unfold conjDiag
  have hd : diagonal (-v) = -diagonal v := by
    rw [Matrix.diagonal_neg]; rfl
  rw [hd, Matrix.mul_neg, Matrix.neg_mul]

private lemma conjDiag_sub (v w : Fin N → ℝ) :
    conjDiag hS v - conjDiag hS w = conjDiag hS (v - w) := by
  rw [sub_eq_add_neg, conjDiag_neg, conjDiag_add, ← sub_eq_add_neg]

private lemma conjDiag_commute_self (v : Fin N → ℝ) :
    Commute (conjDiag hS v) S := by
  have h := conjDiag_commute hS v hS.eigenvalues
  rw [← spectral_conjDiag hS] at h
  exact h

/-- `‖U * diagonal v * star U‖² = Σ v_i²`. -/
private lemma frobSq_conjDiag (v : Fin N → ℝ) :
    frobSq (conjDiag hS v) = ∑ i, v i ^ 2 := by
  have ht : (conjDiag hS v)ᵀ = conjDiag hS v := by
    rw [← Matrix.conjTranspose_eq_transpose_of_trivial,
      (conjDiag_isHermitian hS v).eq]
  rw [frobSq, ht, conjDiag_mul]
  unfold conjDiag
  rw [trace_conj_unitary hS, Matrix.trace_diagonal]
  exact Finset.sum_congr rfl fun i _ => (sq _).symm

/-- `S` is `conjDiag` of its decreasing eigenvalue list, reindexed. -/
private lemma conjDiag_evDesc :
    conjDiag hS (evDesc hS ∘ evPerm (N := N)) = S := by
  rw [← eigenvalues_eq_evDesc_comp hS]
  exact (spectral_conjDiag hS).symm

/-- The eigenvalue of `H` on the `j`-th eigenvector of `S` (in
decreasing order of the eigenvalues of `S`): `λ_j - α_j + β_j`, which
equals `β_j` if `λ_j > 0` and `-β_{rev j} = λ_j` otherwise. -/
private noncomputable def hvec (j : Fin N) : ℝ :=
  evDesc hS j - alphaList S hS j + betaList S hS j

private lemma sub_max_eq (x : ℝ) : x - max x 0 = -max (-x) 0 := by
  rcases le_total 0 x with h | h
  · rw [max_eq_left h, max_eq_right (by linarith)]; ring
  · rw [max_eq_right h, max_eq_left (by linarith)]; ring

/-- The pairs `±β_j`: the value of `hvec` at `rev j` is minus its
value at `j`. -/
private lemma hvec_rev (j : Fin N) : hvec hS (Fin.rev j) = -hvec hS j := by
  unfold hvec
  rw [alphaList_apply, alphaList_apply, betaList_apply, betaList_apply,
    Fin.rev_rev, sub_max_eq, sub_max_eq]
  ring

/-- The spectrum of `H` is symmetric about zero. -/
private lemma spectrumSymm_H :
    SpectrumSymmAboutZero
      (conjDiag_isHermitian hS (hvec hS ∘ evPerm (N := N))) := by
  unfold SpectrumSymmAboutZero
  rw [eigenvalueMultiset_conjDiag, multiset_map_perm, Multiset.map_map]
  have h : ((fun x : ℝ => -x) ∘ hvec hS) = hvec hS ∘ ⇑(Fin.revPerm (n := N)) := by
    funext j
    simp only [Function.comp_apply, Fin.revPerm_apply, hvec_rev]
  rw [h, multiset_map_perm]

/-- `S = H + P` with `P` having eigenvalue `α_j - β_j`. -/
private lemma decomp_add :
    S = conjDiag hS (hvec hS ∘ evPerm (N := N))
      + conjDiag hS ((fun j => alphaList S hS j - betaList S hS j)
          ∘ evPerm (N := N)) := by
  rw [conjDiag_add]
  refine (conjDiag_evDesc hS).symm.trans ?_
  congr 1
  funext i
  simp only [Pi.add_apply, Function.comp_apply, hvec]
  ring

/-- `S = H - P` with `P` having eigenvalue `β_j - α_j`. -/
private lemma decomp_sub :
    S = conjDiag hS (hvec hS ∘ evPerm (N := N))
      - conjDiag hS ((fun j => betaList S hS j - alphaList S hS j)
          ∘ evPerm (N := N)) := by
  rw [conjDiag_sub]
  refine (conjDiag_evDesc hS).symm.trans ?_
  congr 1
  funext i
  simp only [Pi.sub_apply, Function.comp_apply, hvec]
  ring

end Decomposition

/-! ### Theorems (2)–(4) -/

private lemma frobSq_ne_zero (hL : L ≠ 0) : frobSq L ≠ 0 :=
  fun h => hL ((frobSq_eq_zero_iff L).mp h)

/-- A symmetric matrix is its own symmetric part. -/
private lemma symPart_eq_of_isSymm (hsymm : L.IsSymm) : symPart L = L := by
  unfold symPart
  rw [Matrix.IsSymm] at hsymm
  rw [hsymm, ← two_smul ℝ L, smul_smul]
  norm_num

/-- Theorem (3), first half: `d = 0` iff `α_j ≥ β_j` for every `j`. -/
theorem weightD_eq_zero_iff_forall (hL : L ≠ 0) :
    weightD L = 0
      ↔ ∀ j, betaList (symPart L) (symPart_isHermitian L) j
          ≤ alphaList (symPart L) (symPart_isHermitian L) j := by
  rw [weightD, div_eq_zero_iff, or_iff_left (frobSq_ne_zero L hL),
    Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg _)]
  constructor
  · intro h j
    have h1 := h j (Finset.mem_univ j)
    rw [pow_eq_zero_iff two_ne_zero, max_eq_right_iff] at h1
    linarith
  · intro h j _
    rw [max_eq_right (by linarith [h j])]
    norm_num

/-- Theorem (4), first half: `c = 0` iff `β_j ≥ α_j` for every `j`. -/
theorem weightC_eq_zero_iff_forall (hL : L ≠ 0) :
    weightC L = 0
      ↔ ∀ j, alphaList (symPart L) (symPart_isHermitian L) j
          ≤ betaList (symPart L) (symPart_isHermitian L) j := by
  rw [weightC, div_eq_zero_iff, or_iff_left (frobSq_ne_zero L hL),
    Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg _)]
  constructor
  · intro h j
    have h1 := h j (Finset.mem_univ j)
    rw [pow_eq_zero_iff two_ne_zero, max_eq_right_iff] at h1
    linarith
  · intro h j _
    rw [max_eq_right (by linarith [h j])]
    norm_num

/-- Theorem (2), first half: `b = 0` iff `S` is semidefinite. -/
theorem weightB_eq_zero_iff (hL : L ≠ 0) :
    weightB L = 0 ↔ (symPart L).PosSemidef ∨ (-(symPart L)).PosSemidef := by
  have hsum : weightB L = 0
      ↔ ∀ j, alphaList (symPart L) (symPart_isHermitian L) j
          * betaList (symPart L) (symPart_isHermitian L) j = 0 := by
    rw [weightB, div_eq_zero_iff, or_iff_left (frobSq_ne_zero L hL), mul_eq_zero,
      or_iff_right two_ne_zero,
      Finset.sum_eq_zero_iff_of_nonneg (fun j _ =>
        mul_nonneg (alphaList_nonneg (symPart_isHermitian L) j)
          (betaList_nonneg (symPart_isHermitian L) j))]
    simp only [Finset.mem_univ, true_implies]
  rw [hsum, posSemidef_iff_evDesc_nonneg (symPart_isHermitian L),
    neg_posSemidef_iff_evDesc_nonpos (symPart_isHermitian L)]
  constructor
  · intro h
    by_contra hcon
    push_neg at hcon
    obtain ⟨⟨i, hi⟩, ⟨i', hi'⟩⟩ := hcon
    have hN : 0 < N := i.pos
    have hj0 : ∀ k : Fin N, (⟨0, hN⟩ : Fin N) ≤ k :=
      fun k => Fin.le_iff_val_le_val.mpr (Nat.zero_le _)
    have hα : 0 < alphaList (symPart L) (symPart_isHermitian L) ⟨0, hN⟩ := by
      rw [alphaList_apply]
      exact lt_max_of_lt_left
        (hi'.trans_le (evDesc_antitone (symPart_isHermitian L) (hj0 i')))
    have hβ : 0 < betaList (symPart L) (symPart_isHermitian L) ⟨0, hN⟩ := by
      rw [betaList_apply]
      have hle : Fin.rev (Fin.rev i) ≤ Fin.rev ⟨0, hN⟩ :=
        Fin.rev_le_rev.mpr (hj0 _)
      rw [Fin.rev_rev] at hle
      have := evDesc_antitone (symPart_isHermitian L) hle
      exact lt_max_of_lt_left (by linarith)
    exact (mul_pos hα hβ).ne' (h ⟨0, hN⟩)
  · rintro (h | h) <;> intro j
    · rw [betaList_apply, max_eq_right (by linarith [h (Fin.rev j)]), mul_zero]
    · rw [alphaList_apply, max_eq_right (h j), zero_mul]

/-- Theorem (2), second half: `b = 1` iff `L` is symmetric with `α = β`. -/
theorem weightB_eq_one_iff (hL : L ≠ 0) :
    weightB L = 1
      ↔ L.IsSymm ∧ alphaList (symPart L) (symPart_isHermitian L)
          = betaList (symPart L) (symPart_isHermitian L) := by
  have hsum := weight_sum_eq_one L hL
  have ha := weightA_nonneg L
  have hc := weightC_nonneg L
  have hd := weightD_nonneg L
  constructor
  · intro h
    have ha0 : weightA L = 0 := by linarith
    have hc0 : weightC L = 0 := by linarith
    have hd0 : weightD L = 0 := by linarith
    refine ⟨(weightA_eq_zero_iff L hL).mp ha0, funext fun j => le_antisymm ?_ ?_⟩
    · exact (weightC_eq_zero_iff_forall L hL).mp hc0 j
    · exact (weightD_eq_zero_iff_forall L hL).mp hd0 j
  · rintro ⟨hsymm, hαβ⟩
    have ha0 : weightA L = 0 := (weightA_eq_zero_iff L hL).mpr hsymm
    have hc0 : weightC L = 0 :=
      (weightC_eq_zero_iff_forall L hL).mpr fun j => by rw [hαβ]
    have hd0 : weightD L = 0 :=
      (weightD_eq_zero_iff_forall L hL).mpr fun j => by rw [hαβ]
    linarith

/-- Theorem (3), the decomposition: if `d = 0` then `S = H + P` for
symmetric `H` and `P` with the spectrum of `H` symmetric about zero and
`P` positive semidefinite, where `H` and `P` commute with `S` and with
each other and `‖P‖² = c‖L‖²`. -/
theorem weightD_eq_zero_decomp_commute (hL : L ≠ 0) (h : weightD L = 0) :
    ∃ H P : Matrix (Fin N) (Fin N) ℝ, ∃ hH : H.IsHermitian,
      SpectrumSymmAboutZero hH ∧ P.PosSemidef ∧ symPart L = H + P
      ∧ Commute H (symPart L) ∧ Commute P (symPart L) ∧ Commute H P
      ∧ frobSq P = weightC L * frobSq L := by
  have hβα := (weightD_eq_zero_iff_forall L hL).mp h
  refine ⟨conjDiag (symPart_isHermitian L)
      (hvec (symPart_isHermitian L) ∘ evPerm (N := N)),
    conjDiag (symPart_isHermitian L)
      ((fun j => alphaList (symPart L) (symPart_isHermitian L) j
        - betaList (symPart L) (symPart_isHermitian L) j) ∘ evPerm (N := N)),
    conjDiag_isHermitian _ _, spectrumSymm_H _, ?_, decomp_add _,
    conjDiag_commute_self _ _, conjDiag_commute_self _ _, conjDiag_commute _ _ _, ?_⟩
  · rw [conjDiag_posSemidef_iff]
    intro i
    simp only [Function.comp_apply]
    linarith [hβα (evPerm i)]
  · rw [frobSq_conjDiag, weightC, div_mul_cancel₀ _ (frobSq_ne_zero L hL)]
    refine (Fintype.sum_equiv (evPerm (N := N)) _
      (fun j => (alphaList (symPart L) (symPart_isHermitian L) j
        - betaList (symPart L) (symPart_isHermitian L) j) ^ 2)
      (fun i => rfl)).trans ?_
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [max_eq_left (sub_nonneg.mpr (hβα j))]

/-- Theorem (3), second half: `d = 0` iff `S = H + P` for symmetric `H`
and `P` such that the spectrum of `H` is symmetric about zero and `P`
is positive semidefinite. -/
theorem weightD_eq_zero_iff_decomp (hL : L ≠ 0) :
    weightD L = 0
      ↔ ∃ H P : Matrix (Fin N) (Fin N) ℝ, ∃ hH : H.IsHermitian,
          SpectrumSymmAboutZero hH ∧ P.PosSemidef ∧ symPart L = H + P := by
  constructor
  · intro h
    obtain ⟨H, P, hH, hsym, hP, hSHP, -⟩ := weightD_eq_zero_decomp_commute L hL h
    exact ⟨H, P, hH, hsym, hP, hSHP⟩
  · rintro ⟨H, P, hH, hsym, hP, hSHP⟩
    rw [weightD_eq_zero_iff_forall L hL]
    apply forall_le_of_card_le
    intro t ht
    rw [card_alpha_gt (symPart_isHermitian L) t ht,
      card_beta_gt (symPart_isHermitian L) t ht]
    exact counting_dominance (symPart_isHermitian L) hH hP hSHP hsym

/-- Theorem (4), the decomposition: if `c = 0` then `S = H - P` with
`H` as in (3) and `P` positive semidefinite, where `H` and `P` commute
with `S` and with each other and `‖P‖² = d‖L‖²`. -/
theorem weightC_eq_zero_decomp_commute (hL : L ≠ 0) (h : weightC L = 0) :
    ∃ H P : Matrix (Fin N) (Fin N) ℝ, ∃ hH : H.IsHermitian,
      SpectrumSymmAboutZero hH ∧ P.PosSemidef ∧ symPart L = H - P
      ∧ Commute H (symPart L) ∧ Commute P (symPart L) ∧ Commute H P
      ∧ frobSq P = weightD L * frobSq L := by
  have hαβ := (weightC_eq_zero_iff_forall L hL).mp h
  refine ⟨conjDiag (symPart_isHermitian L)
      (hvec (symPart_isHermitian L) ∘ evPerm (N := N)),
    conjDiag (symPart_isHermitian L)
      ((fun j => betaList (symPart L) (symPart_isHermitian L) j
        - alphaList (symPart L) (symPart_isHermitian L) j) ∘ evPerm (N := N)),
    conjDiag_isHermitian _ _, spectrumSymm_H _, ?_, decomp_sub _,
    conjDiag_commute_self _ _, conjDiag_commute_self _ _, conjDiag_commute _ _ _, ?_⟩
  · rw [conjDiag_posSemidef_iff]
    intro i
    simp only [Function.comp_apply]
    linarith [hαβ (evPerm i)]
  · rw [frobSq_conjDiag, weightD, div_mul_cancel₀ _ (frobSq_ne_zero L hL)]
    refine (Fintype.sum_equiv (evPerm (N := N)) _
      (fun j => (betaList (symPart L) (symPart_isHermitian L) j
        - alphaList (symPart L) (symPart_isHermitian L) j) ^ 2)
      (fun i => rfl)).trans ?_
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [max_eq_left (sub_nonneg.mpr (hαβ j))]

/-- Theorem (4), second half: `c = 0` iff `S = H - P` with `H` as in
(3) and `P` positive semidefinite. -/
theorem weightC_eq_zero_iff_decomp (hL : L ≠ 0) :
    weightC L = 0
      ↔ ∃ H P : Matrix (Fin N) (Fin N) ℝ, ∃ hH : H.IsHermitian,
          SpectrumSymmAboutZero hH ∧ P.PosSemidef ∧ symPart L = H - P := by
  constructor
  · intro h
    obtain ⟨H, P, hH, hsym, hP, hSHP, -⟩ := weightC_eq_zero_decomp_commute L hL h
    exact ⟨H, P, hH, hsym, hP, hSHP⟩
  · rintro ⟨H, P, hH, hsym, hP, hSHP⟩
    rw [weightC_eq_zero_iff_forall L hL]
    apply forall_le_of_card_le'
    intro t ht
    rw [card_alpha_gt (symPart_isHermitian L) t ht,
      card_beta_gt (symPart_isHermitian L) t ht]
    exact counting_dominance_sub (symPart_isHermitian L) hH hP hSHP hsym

/-- Theorem (3), last part: `d = 1` iff `L` is symmetric negative
semidefinite. -/
theorem weightD_eq_one_iff (hL : L ≠ 0) :
    weightD L = 1 ↔ L.IsSymm ∧ (-L).PosSemidef := by
  have hsum := weight_sum_eq_one L hL
  have ha := weightA_nonneg L
  have hb := weightB_nonneg L
  have hc := weightC_nonneg L
  constructor
  · intro h
    have ha0 : weightA L = 0 := by linarith
    have hb0 : weightB L = 0 := by linarith
    have hc0 : weightC L = 0 := by linarith
    have hsymm := (weightA_eq_zero_iff L hL).mp ha0
    refine ⟨hsymm, ?_⟩
    rw [← symPart_eq_of_isSymm L hsymm,
      neg_posSemidef_iff_evDesc_nonpos (symPart_isHermitian L)]
    have hαβ := (weightC_eq_zero_iff_forall L hL).mp hc0
    rcases (weightB_eq_zero_iff L hL).mp hb0 with hpos | hneg
    · rw [posSemidef_iff_evDesc_nonneg (symPart_isHermitian L)] at hpos
      intro i
      have h1 := hαβ i
      rw [alphaList_apply, betaList_apply,
        max_eq_right (show -(evDesc (symPart_isHermitian L) (Fin.rev i)) ≤ (0 : ℝ) by
          linarith [hpos (Fin.rev i)])] at h1
      exact (le_max_left _ _).trans h1
    · exact (neg_posSemidef_iff_evDesc_nonpos (symPart_isHermitian L)).mp hneg
  · rintro ⟨hsymm, hneg⟩
    have hS : symPart L = L := symPart_eq_of_isSymm L hsymm
    have ha0 : weightA L = 0 := (weightA_eq_zero_iff L hL).mpr hsymm
    have hneg' : (-(symPart L)).PosSemidef := by rw [hS]; exact hneg
    have hb0 : weightB L = 0 := (weightB_eq_zero_iff L hL).mpr (Or.inr hneg')
    have hc0 : weightC L = 0 := by
      rw [weightC_eq_zero_iff_forall L hL]
      intro j
      rw [alphaList_apply, max_eq_right
        ((neg_posSemidef_iff_evDesc_nonpos (symPart_isHermitian L)).mp hneg' j)]
      exact betaList_nonneg _ j
    linarith

/-- Theorem (4), last part: `c = 1` iff `L` is symmetric positive
semidefinite. -/
theorem weightC_eq_one_iff (hL : L ≠ 0) :
    weightC L = 1 ↔ L.IsSymm ∧ L.PosSemidef := by
  have hsum := weight_sum_eq_one L hL
  have ha := weightA_nonneg L
  have hb := weightB_nonneg L
  have hd := weightD_nonneg L
  constructor
  · intro h
    have ha0 : weightA L = 0 := by linarith
    have hb0 : weightB L = 0 := by linarith
    have hd0 : weightD L = 0 := by linarith
    have hsymm := (weightA_eq_zero_iff L hL).mp ha0
    refine ⟨hsymm, ?_⟩
    rw [← symPart_eq_of_isSymm L hsymm,
      posSemidef_iff_evDesc_nonneg (symPart_isHermitian L)]
    have hβα := (weightD_eq_zero_iff_forall L hL).mp hd0
    rcases (weightB_eq_zero_iff L hL).mp hb0 with hpos | hneg
    · exact (posSemidef_iff_evDesc_nonneg (symPart_isHermitian L)).mp hpos
    · rw [neg_posSemidef_iff_evDesc_nonpos (symPart_isHermitian L)] at hneg
      intro i
      have h1 := hβα (Fin.rev i)
      rw [alphaList_apply, betaList_apply, Fin.rev_rev,
        max_eq_right (hneg (Fin.rev i))] at h1
      linarith [le_max_left (-(evDesc (symPart_isHermitian L) i)) 0]
  · rintro ⟨hsymm, hpos⟩
    have hS : symPart L = L := symPart_eq_of_isSymm L hsymm
    have ha0 : weightA L = 0 := (weightA_eq_zero_iff L hL).mpr hsymm
    have hpos' : (symPart L).PosSemidef := by rw [hS]; exact hpos
    have hb0 : weightB L = 0 := (weightB_eq_zero_iff L hL).mpr (Or.inl hpos')
    have hd0 : weightD L = 0 := by
      rw [weightD_eq_zero_iff_forall L hL]
      intro j
      rw [betaList_apply, max_eq_right (by
        linarith [(posSemidef_iff_evDesc_nonneg (symPart_isHermitian L)).mp hpos'
          (Fin.rev j)])]
      exact alphaList_nonneg _ j
    linarith

end Appendices
