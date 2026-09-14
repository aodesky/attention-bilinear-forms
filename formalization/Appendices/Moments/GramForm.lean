/-
Appendix "Moments", Lemma (lem:gram-form),
"Computation from the key and query matrices".

Let `W_K, W_Q ∈ M_{n×N}(ℝ)`, let `S = ½(W_KᵀW_Q + W_QᵀW_K)`, and let

    M = (W_K; W_Q) : ℝ^N → ℝ^{2n},   G = MMᵀ,   J = (0 Iₙ; Iₙ 0).

Then `S = ½ MᵀJM` and

    tr(S^k) = tr((½JG)^k)   (k ≥ 1);

in particular `‖S‖² = ¼ tr((JG)²)`.  If moreover `M` is surjective and
`n ≥ 1`, then the nonzero spectrum of `S`, with multiplicity, is the
spectrum of `½JG`; the sorted normalized nonzero spectrum `λ` of `S`
lies in `Λ_n`, and with `H = JG/(2‖S‖)`,

    O(λ) = tr(H(I-H²)⁻¹),  E(λ) = tr(H²(I-H²)⁻¹),  R(λ) = tr((I-H)⁻¹).

Here `S = symPart (W_Kᵀ * W_Q)`, and the vertical block matrix `M` is
`Matrix.fromRows W_K W_Q`, indexed by `Fin n ⊕ Fin n`.

Encodings:
* "`M` is surjective" is `Function.Surjective (Matrix.mulVecLin (M WK WQ))`.
* `‖S‖` is `frobNorm S = √(frobSq S)`, the Frobenius norm.
* "the spectrum of a real matrix, with multiplicity" is the multiset
  of roots of its characteristic polynomial; this is faithful when the
  characteristic polynomial has only real roots, which is proved for
  both `S` (`card_charpoly_roots_symPart`: `N` roots) and `½JG`
  (`card_charpoly_roots_half_JG`: `2n` roots).  "The nonzero spectrum
  of `S`, with multiplicity, is the spectrum of `½JG`" is then
  `charpoly_roots_filter_ne_zero`: the multiset of nonzero roots of the
  characteristic polynomial of `S` equals the multiset of roots of the
  characteristic polynomial of `½JG`.
* "the sorted normalized nonzero spectrum `λ` of `S`" is a vector
  `λ ∈ ℝ^{2n}` satisfying `IsSortedNormalizedNonzeroSpectrum S λ`:
  `λ` is nondecreasing and the multiset of its entries is the multiset
  of nonzero roots of the characteristic polynomial of `S`, each divided
  by `‖S‖`.  Such a `λ` exists
  (`exists_isSortedNormalizedNonzeroSpectrum`) and is unique
  (`isSortedNormalizedNonzeroSpectrum_unique`); it lies in `Λ_n`
  (`memLambdaSet_of_isSortedNormalizedNonzeroSpectrum`) and satisfies
  the three trace identities (`statO_eq_trace`, `statE_eq_trace`,
  `statR_eq_trace`) with `H WK WQ = JG/(2‖S‖)`.

The trace identities are proved directly from the diagonalization
`H = W diag(e) W'` (`W W' = W' W = 1`, `e` real), which the proof in
the paper obtains from the similarity
`JG = G^{-1/2} (G^{1/2} J G^{1/2}) G^{1/2}` and the spectral theorem;
the entries of `e` are the entries of `λ`, whose absolute values are
less than one by Lemma OE-equivalences
(`MemLambdaSet.abs_entry_lt_one`).
-/
import Appendices.Common
import Appendices.Profile
import Appendices.Moments.Defs
import Appendices.Moments.OEEquivalences
import Appendices.BalancedBimodalSpectra.Defs

namespace Appendices

namespace GramForm

open Matrix BigOperators

variable {n N : ℕ}

/-- The vertical block matrix `M = (W_K; W_Q) : ℝ^N → ℝ^{2n}`. -/
def M (WK WQ : Matrix (Fin n) (Fin N) ℝ) : Matrix (Fin n ⊕ Fin n) (Fin N) ℝ :=
  Matrix.fromRows WK WQ

/-- The matrix `J = (0 Iₙ; Iₙ 0)` of the hyperbolic form. -/
def J (n : ℕ) : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  Matrix.fromBlocks 0 1 1 0

/-- The matrix `G = MMᵀ`, whose blocks are the `n × n` Gram matrices
`W_KW_Kᵀ`, `W_KW_Qᵀ`, `W_QW_Kᵀ`, `W_QW_Qᵀ`. -/
def G (WK WQ : Matrix (Fin n) (Fin N) ℝ) :
    Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  M WK WQ * (M WK WQ)ᵀ

variable (WK WQ : Matrix (Fin n) (Fin N) ℝ)

/-- The paper's `S = ½(W_KᵀW_Q + W_QᵀW_K)` is the symmetric part of
`W_KᵀW_Q`. -/
lemma symPart_transpose_mul :
    symPart (WKᵀ * WQ) = (1 / 2 : ℝ) • (WKᵀ * WQ + WQᵀ * WK) := by
  simp [symPart, Matrix.transpose_mul]

/-- `S = ½ MᵀJM`. -/
theorem symPart_eq_half_conj :
    symPart (WKᵀ * WQ) = (1 / 2 : ℝ) • ((M WK WQ)ᵀ * J n * M WK WQ) := by
  rw [symPart_transpose_mul]
  congr 1
  rw [M, J, Matrix.transpose_fromRows, Matrix.mul_assoc,
    Matrix.fromBlocks_mul_fromRows]
  simp [Matrix.fromCols_mul_fromRows]

private lemma trace_pow_mul_comm {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    (A : Matrix α β ℝ) (B : Matrix β α ℝ) (k : ℕ) (hk : 1 ≤ k) :
    Matrix.trace ((A * B) ^ k) = Matrix.trace ((B * A) ^ k) := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  have h : ∀ j : ℕ, (A * B) ^ (j + 1) = A * ((B * A) ^ j * B) := by
    intro j
    induction j with
    | zero => simp [pow_succ]
    | succ i ih => simp only [pow_succ, ih, Matrix.mul_assoc]
  rw [h j, Matrix.trace_mul_comm, Matrix.mul_assoc, ← pow_succ]

/-- `tr(S^k) = tr((½JG)^k)` for `k ≥ 1`. -/
theorem trace_pow_symPart (k : ℕ) (hk : 1 ≤ k) :
    Matrix.trace (symPart (WKᵀ * WQ) ^ k)
      = Matrix.trace (((1 / 2 : ℝ) • (J n * G WK WQ)) ^ k) := by
  rw [symPart_eq_half_conj, smul_pow, smul_pow, Matrix.trace_smul,
    Matrix.trace_smul]
  congr 1
  have h : (M WK WQ)ᵀ * J n * M WK WQ = (M WK WQ)ᵀ * (J n * M WK WQ) :=
    Matrix.mul_assoc _ _ _
  rw [h, trace_pow_mul_comm _ _ k hk, Matrix.mul_assoc, G]

/-- `‖S‖² = ¼ tr((JG)²)`. -/
theorem frobSq_symPart_eq_quarter_trace :
    frobSq (symPart (WKᵀ * WQ))
      = (1 / 4 : ℝ) * Matrix.trace ((J n * G WK WQ) ^ 2) := by
  have hsym : (symPart (WKᵀ * WQ))ᵀ = symPart (WKᵀ * WQ) :=
    symPart_isSymm (WKᵀ * WQ)
  have h1 : frobSq (symPart (WKᵀ * WQ))
      = Matrix.trace (symPart (WKᵀ * WQ) ^ 2) := by
    rw [show frobSq (symPart (WKᵀ * WQ))
        = Matrix.trace ((symPart (WKᵀ * WQ))ᵀ * symPart (WKᵀ * WQ)) from rfl,
      hsym, ← pow_two]
  rw [h1, trace_pow_symPart WK WQ 2 (by norm_num), smul_pow,
    Matrix.trace_smul, smul_eq_mul]
  norm_num

/-! ### Private machinery: conjugated-diagonal normal forms

Algebra of matrices of the form `W * diagonal d * W'` where
`W * W' = W' * W = 1`.  The matrix `½JG` in part (b) is of this form
with real diagonal, which yields all its trace, inverse, and
characteristic-polynomial identities. -/

section ConjDiag

open Polynomial

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {W W' : Matrix α α ℝ}

private lemma conj_mul (hWr : W' * W = 1) (a b : α → ℝ) :
    (W * Matrix.diagonal a * W') * (W * Matrix.diagonal b * W')
      = W * Matrix.diagonal (a * b) * W' := by
  simp only [Matrix.mul_assoc]
  rw [← Matrix.mul_assoc W' W, hWr, Matrix.one_mul,
    ← Matrix.mul_assoc (Matrix.diagonal a), Matrix.diagonal_mul_diagonal]
  rfl

private lemma conj_one (hWl : W * W' = 1) :
    (1 : Matrix α α ℝ) = W * Matrix.diagonal 1 * W' := by
  have h1 : Matrix.diagonal (1 : α → ℝ) = 1 := Matrix.diagonal_one
  rw [h1, Matrix.mul_one, hWl]

private lemma conj_sub (a b : α → ℝ) :
    W * Matrix.diagonal a * W' - W * Matrix.diagonal b * W'
      = W * Matrix.diagonal (a - b) * W' := by
  rw [← sub_mul, ← mul_sub, Matrix.diagonal_sub]
  rfl

private lemma conj_smul (s : ℝ) (a : α → ℝ) :
    s • (W * Matrix.diagonal a * W') = W * Matrix.diagonal (s • a) * W' := by
  simp only [Matrix.diagonal_smul, mul_smul_comm, smul_mul_assoc]

private lemma conj_inv (hWl : W * W' = 1) (hWr : W' * W = 1)
    {a : α → ℝ} (ha : ∀ i, a i ≠ 0) :
    (W * Matrix.diagonal a * W')⁻¹ = W * Matrix.diagonal a⁻¹ * W' := by
  refine Matrix.inv_eq_right_inv ?_
  rw [conj_mul hWr]
  have : a * a⁻¹ = 1 := by
    funext i
    exact mul_inv_cancel₀ (ha i)
  rw [this, ← conj_one hWl]

private lemma conj_trace (hWr : W' * W = 1) (a : α → ℝ) :
    Matrix.trace (W * Matrix.diagonal a * W') = ∑ i, a i := by
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, hWr, Matrix.one_mul,
    Matrix.trace_diagonal]

private lemma conj_charpoly (hWl : W * W' = 1) (hWr : W' * W = 1)
    (a : α → ℝ) :
    (W * Matrix.diagonal a * W').charpoly = ∏ i, (X - C (a i)) := by
  have h := Matrix.charpoly_units_conj ⟨W, W', hWl, hWr⟩ (Matrix.diagonal a)
  rw [show (⟨W, W', hWl, hWr⟩ : (Matrix α α ℝ)ˣ).val = W from rfl] at h
  rw [show ((⟨W, W', hWl, hWr⟩ : (Matrix α α ℝ)ˣ)⁻¹).val = W' from rfl] at h
  rw [h, Matrix.charpoly_diagonal]

private lemma conj_roots (hWl : W * W' = 1) (hWr : W' * W = 1)
    (a : α → ℝ) :
    (W * Matrix.diagonal a * W').charpoly.roots
      = Multiset.map a Finset.univ.val := by
  rw [conj_charpoly hWl hWr]
  rw [show (∏ i, (X - C (a i)))
      = ((Finset.univ.val.map a).map fun r => X - C r).prod by
    rw [Multiset.map_map]; rfl]
  exact roots_multiset_prod_X_sub_C _

end ConjDiag

/-- Real form of the spectral theorem: a real symmetric matrix is
`U * diagonal (eigenvalues) * star U` with `U` its eigenvector unitary. -/
private lemma spectral_real {α : Type*} [Fintype α] [DecidableEq α]
    {A : Matrix α α ℝ} (hA : A.IsHermitian) :
    A = (hA.eigenvectorUnitary : Matrix α α ℝ)
        * Matrix.diagonal hA.eigenvalues
        * star (hA.eigenvectorUnitary : Matrix α α ℝ) := by
  conv_lhs => rw [hA.spectral_theorem]
  rw [Unitary.conjStarAlgAut_apply]
  congr 2


/-! ### Private machinery: positive definiteness of `G` and its square root -/

private lemma transpose_mulVec_ne_zero
    (hsurj : Function.Surjective (Matrix.mulVecLin (M WK WQ)))
    {x : Fin n ⊕ Fin n → ℝ} (hx : x ≠ 0) : (M WK WQ)ᵀ *ᵥ x ≠ 0 := by
  intro h0
  obtain ⟨v, hv⟩ := hsurj x
  have hv' : M WK WQ *ᵥ v = x := hv
  have hxx : x ⬝ᵥ x = 0 := by
    calc x ⬝ᵥ x = x ⬝ᵥ (M WK WQ *ᵥ v) := by rw [hv']
    _ = (x ᵥ* M WK WQ) ⬝ᵥ v := Matrix.dotProduct_mulVec _ _ _
    _ = 0 := by rw [← Matrix.mulVec_transpose, h0]; simp
  exact hx (dotProduct_self_eq_zero.mp hxx)

private lemma posDef_G
    (hsurj : Function.Surjective (Matrix.mulVecLin (M WK WQ))) :
    (G WK WQ).PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · rw [show G WK WQ = M WK WQ * (M WK WQ)ᴴ by
      rw [G, Matrix.conjTranspose_eq_transpose_of_trivial]]
    exact Matrix.isHermitian_mul_conjTranspose_self _
  · intro x hx
    have hy : (M WK WQ)ᵀ *ᵥ x ≠ 0 := transpose_mulVec_ne_zero WK WQ hsurj hx
    have hstar : star x = x := by
      funext i; exact star_trivial _
    have hcalc : star x ⬝ᵥ (G WK WQ *ᵥ x)
        = ((M WK WQ)ᵀ *ᵥ x) ⬝ᵥ ((M WK WQ)ᵀ *ᵥ x) := by
      rw [hstar, G, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec,
        ← Matrix.mulVec_transpose]
    rw [hcalc]
    rcases lt_or_eq_of_le (Fintype.sum_nonneg fun i => mul_self_nonneg _ :
        (0:ℝ) ≤ ((M WK WQ)ᵀ *ᵥ x) ⬝ᵥ ((M WK WQ)ᵀ *ᵥ x)) with h | h
    · exact h
    · exact absurd (dotProduct_self_eq_zero.mp h.symm) hy

/-- Any real positive definite matrix has a symmetric square root that
is invertible with symmetric inverse. -/
private lemma exists_sqrt {α : Type*} [Fintype α] [DecidableEq α]
    {P : Matrix α α ℝ} (hP : P.PosDef) :
    ∃ Q Q' : Matrix α α ℝ, Qᵀ = Q ∧ Q'ᵀ = Q' ∧ Q * Q = P ∧
      Q * Q' = 1 ∧ Q' * Q = 1 := by
  classical
  set U : Matrix α α ℝ := ↑(hP.1.eigenvectorUnitary) with hU
  have hUl : U * star U = 1 := hP.1.eigenvectorUnitary.2.2
  have hUr : star U * U = 1 := hP.1.eigenvectorUnitary.2.1
  have hd : ∀ i, 0 < hP.1.eigenvalues i := fun i => hP.eigenvalues_pos i
  have hsymm : ∀ e : α → ℝ, (U * Matrix.diagonal e * star U)ᵀ
      = U * Matrix.diagonal e * star U := by
    intro e
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.diagonal_transpose]
    rw [show star U = Uᵀ by
      rw [Matrix.star_eq_conjTranspose,
        Matrix.conjTranspose_eq_transpose_of_trivial]]
    rw [Matrix.transpose_transpose, Matrix.mul_assoc]
  refine ⟨U * Matrix.diagonal (fun i => Real.sqrt (hP.1.eigenvalues i)) * star U,
    U * Matrix.diagonal (fun i => (Real.sqrt (hP.1.eigenvalues i))⁻¹) * star U,
    hsymm _, hsymm _, ?_, ?_, ?_⟩
  · rw [conj_mul hUr]
    rw [show (fun i => Real.sqrt (hP.1.eigenvalues i))
        * (fun i => Real.sqrt (hP.1.eigenvalues i)) = hP.1.eigenvalues by
      funext i
      exact Real.mul_self_sqrt (hd i).le]
    exact (spectral_real hP.1).symm
  · rw [conj_mul hUr]
    rw [show (fun i => Real.sqrt (hP.1.eigenvalues i))
        * (fun i => (Real.sqrt (hP.1.eigenvalues i))⁻¹) = 1 by
      funext i
      exact mul_inv_cancel₀ (Real.sqrt_ne_zero'.mpr (hd i))]
    exact (conj_one hUl).symm
  · rw [conj_mul hUr]
    rw [show (fun i => (Real.sqrt (hP.1.eigenvalues i))⁻¹)
        * (fun i => Real.sqrt (hP.1.eigenvalues i)) = 1 by
      funext i
      exact inv_mul_cancel₀ (Real.sqrt_ne_zero'.mpr (hd i))]
    exact (conj_one hUl).symm


/-! ### Private machinery: eigenvalue signs of `QJQ` (Sylvester counting)

The symmetric matrix `Q (sJ) Q` (`Q` symmetric invertible, `s > 0`) is
congruent to the hyperbolic form and has exactly `n` positive and `n`
negative eigenvalues.  We prove the counting by the standard subspace
argument: a family on which the form is positive definite together
with a linearly independent family on which it is `≤ 0` is linearly
independent. -/

private lemma card_add_card_le_of_forms
    {α ι κ : Type*} [Fintype α] [DecidableEq α] [Fintype ι] [Fintype κ]
    (P : Matrix α α ℝ) (v : ι → α → ℝ) (w : κ → α → ℝ)
    (hv : ∀ a : ι → ℝ, a ≠ 0 →
      0 < (∑ i, a i • v i) ⬝ᵥ (P *ᵥ ∑ i, a i • v i))
    (hw₁ : ∀ b : κ → ℝ,
      (∑ j, b j • w j) ⬝ᵥ (P *ᵥ ∑ j, b j • w j) ≤ 0)
    (hw₂ : ∀ b : κ → ℝ, ∑ j, b j • w j = 0 → b = 0) :
    Fintype.card ι + Fintype.card κ ≤ Fintype.card α := by
  classical
  have hli : LinearIndependent ℝ (Sum.elim v w) := by
    rw [Fintype.linearIndependent_iff]
    intro g hg
    rw [Fintype.sum_sum_type] at hg
    simp only [Sum.elim_inl, Sum.elim_inr] at hg
    have ha : (fun i => g (Sum.inl i)) = 0 := by
      by_contra hne
      have hx := hv _ hne
      have hxw : (∑ i, g (Sum.inl i) • v i)
          = ∑ j, (-g (Sum.inr j)) • w j := by
        have := eq_neg_of_add_eq_zero_left hg
        rw [this, ← Finset.sum_neg_distrib]
        exact Finset.sum_congr rfl fun j _ => (neg_smul _ _).symm
      rw [hxw] at hx
      exact absurd hx (not_lt.mpr (hw₁ _))
    have hb : (fun j => g (Sum.inr j)) = 0 := by
      apply hw₂
      have h0 : (∑ i, g (Sum.inl i) • v i) = 0 := by
        rw [show (∑ i, g (Sum.inl i) • v i)
            = ∑ i, (fun i => g (Sum.inl i)) i • v i from rfl]
        rw [ha]
        simp
      rw [h0, zero_add] at hg
      exact hg
    intro s
    cases s with
    | inl i => exact congrFun ha i
    | inr j => exact congrFun hb j
  have hcard := hli.fintype_card_le_finrank
  rwa [Fintype.card_sum, Module.finrank_pi] at hcard

private lemma eigenbasis_dotProduct {α : Type*} [Fintype α] [DecidableEq α]
    {B : Matrix α α ℝ} (hB : B.IsHermitian) (i j : α) :
    ⇑(hB.eigenvectorBasis i) ⬝ᵥ ⇑(hB.eigenvectorBasis j)
      = if i = j then 1 else 0 := by
  have h := orthonormal_iff_ite.mp hB.eigenvectorBasis.orthonormal i j
  rw [EuclideanSpace.inner_eq_star_dotProduct] at h
  rw [← h, dotProduct_comm, star_trivial]

private lemma form_on_eigencombo {α : Type*} [Fintype α] [DecidableEq α]
    {B : Matrix α α ℝ} (hB : B.IsHermitian) {ι : Type*} [Fintype ι]
    [DecidableEq ι] (f : ι → α) (hf : Function.Injective f) (a : ι → ℝ) :
    (∑ s, a s • ⇑(hB.eigenvectorBasis (f s)))
        ⬝ᵥ (B *ᵥ ∑ s, a s • ⇑(hB.eigenvectorBasis (f s)))
      = ∑ s, hB.eigenvalues (f s) * a s ^ 2 := by
  have hmv : B *ᵥ (∑ s, a s • ⇑(hB.eigenvectorBasis (f s)))
      = ∑ s, (hB.eigenvalues (f s) * a s) • ⇑(hB.eigenvectorBasis (f s)) := by
    rw [← Matrix.mulVecLin_apply, map_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [LinearMap.map_smul, Matrix.mulVecLin_apply,
      hB.mulVec_eigenvectorBasis, smul_smul, mul_comm]
  rw [hmv, dotProduct_sum]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [dotProduct_smul, sum_dotProduct]
  have hsum : (∑ s, (a s • ⇑(hB.eigenvectorBasis (f s)))
      ⬝ᵥ ⇑(hB.eigenvectorBasis (f t))) = a t := by
    rw [Finset.sum_eq_single t]
    · rw [smul_dotProduct, eigenbasis_dotProduct hB, if_pos rfl,
        smul_eq_mul, mul_one]
    · intro s _ hst
      rw [smul_dotProduct, eigenbasis_dotProduct hB,
        if_neg (fun h => hst (hf h)), smul_eq_mul, mul_zero]
    · intro h
      exact absurd (Finset.mem_univ t) h
  rw [hsum, smul_eq_mul]
  ring

private lemma J_mulVec_elim (b c : Fin n → ℝ) :
    J n *ᵥ Sum.elim b c = Sum.elim c b := by
  rw [J, Matrix.fromBlocks_mulVec]
  funext i
  cases i <;> simp

private lemma elim_dotProduct_elim (b c b' c' : Fin n → ℝ) :
    Sum.elim b c ⬝ᵥ Sum.elim b' c' = b ⬝ᵥ b' + c ⬝ᵥ c' := by
  simp [dotProduct, Fintype.sum_sum_type]

private lemma sum_smul_single_elim (b : Fin n → ℝ) (ε : ℝ) :
    (∑ j, b j • (Sum.elim (Pi.single j 1) (Pi.single j ε)
        : Fin n ⊕ Fin n → ℝ))
      = Sum.elim b (ε • b) := by
  funext x
  cases x with
  | inl i => simp [Pi.single_apply]
  | inr i => simp [Pi.single_apply, mul_comm]

private lemma inertia {n : ℕ} {Q Q' : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ}
    (hQ : Qᵀ = Q) (hQQ' : Q * Q' = 1)
    (hB : (Q * J n * Q).IsHermitian) :
    (Finset.univ.filter fun i => 0 < hB.eigenvalues i).card = n
    ∧ (Finset.univ.filter fun i => hB.eigenvalues i < 0).card = n
    ∧ ∀ i, hB.eigenvalues i ≠ 0 := by
  classical
  set B := Q * J n * Q with hBdef
  -- the quadratic form of `B` on vectors `Q' *ᵥ z` is the hyperbolic form at `z`
  have hform : ∀ z : Fin n ⊕ Fin n → ℝ,
      (Q' *ᵥ z) ⬝ᵥ (B *ᵥ (Q' *ᵥ z)) = z ⬝ᵥ (J n *ᵥ z) := by
    intro z
    rw [Matrix.mulVec_mulVec,
      show B * Q' = Q * J n by rw [hBdef, Matrix.mul_assoc, hQQ', Matrix.mul_one],
      ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec]
    congr 1
    rw [← Matrix.mulVec_transpose, hQ, Matrix.mulVec_mulVec, hQQ',
      Matrix.one_mulVec]
  -- no zero eigenvalues
  have hJJ : J n * J n = 1 := by
    rw [J]
    simp [Matrix.fromBlocks_multiply, ← Matrix.fromBlocks_one]
  have hUnit : IsUnit B := by
    refine IsUnit.of_mul_eq_one (b := Q' * J n * Q') ?_
    calc B * (Q' * J n * Q')
        = Q * J n * (Q * Q') * J n * Q' := by
          rw [hBdef]; noncomm_ring
      _ = Q * (J n * J n) * Q' := by rw [hQQ']; noncomm_ring
      _ = 1 := by rw [hJJ, Matrix.mul_one, hQQ']
  have hne : ∀ i, hB.eigenvalues i ≠ 0 := by
    have hdet : B.det ≠ 0 := by
      intro h
      exact (Matrix.isUnit_iff_isUnit_det B |>.mp hUnit).ne_zero h
    intro i hi
    apply hdet
    rw [hB.det_eq_prod_eigenvalues]
    exact Finset.prod_eq_zero (Finset.mem_univ i) (by exact_mod_cast hi)
  -- injectivity of `Q' *ᵥ ·`
  have hQ'inj : ∀ z : Fin n ⊕ Fin n → ℝ, Q' *ᵥ z = 0 → z = 0 := by
    intro z h
    have := congrArg (Q *ᵥ ·) h
    simpa [Matrix.mulVec_mulVec, hQQ'] using this
  -- the two applications of the abstract counting lemma
  have hposcount : Fintype.card {i // 0 < hB.eigenvalues i} + n ≤ n + n := by
    have h := card_add_card_le_of_forms B
      (fun s : {i // 0 < hB.eigenvalues i} => ⇑(hB.eigenvectorBasis s.1))
      (fun j : Fin n => Q' *ᵥ (Sum.elim (Pi.single j 1) (Pi.single j (-1)) : Fin n ⊕ Fin n → ℝ))
      ?_ ?_ ?_
    · simpa [Fintype.card_sum] using h
    · intro a ha
      rw [form_on_eigencombo hB Subtype.val Subtype.val_injective a]
      obtain ⟨s₀, hs₀⟩ := Function.ne_iff.mp ha
      refine Finset.sum_pos' (fun s _ => ?_) ⟨s₀, Finset.mem_univ _, ?_⟩
      · exact mul_nonneg s.2.le (sq_nonneg _)
      · exact mul_pos s₀.2 (sq_pos_iff.mpr hs₀)
    · intro b
      rw [show (∑ j, b j • (Q' *ᵥ (Sum.elim (Pi.single j 1) (Pi.single j (-1)) : Fin n ⊕ Fin n → ℝ)))
          = Q' *ᵥ Sum.elim b ((-1 : ℝ) • b) by
        rw [← sum_smul_single_elim b (-1 : ℝ)]
        rw [← Matrix.mulVecLin_apply, map_sum]
        exact Finset.sum_congr rfl fun j _ => by
          rw [LinearMap.map_smul, Matrix.mulVecLin_apply]]
      rw [hform, J_mulVec_elim, elim_dotProduct_elim, dotProduct_smul,
        smul_dotProduct]
      have hb : (0:ℝ) ≤ b ⬝ᵥ b := Fintype.sum_nonneg fun i => mul_self_nonneg _
      simp only [smul_eq_mul]
      nlinarith [hb]
    · intro b hb0
      rw [show (∑ j, b j • (Q' *ᵥ (Sum.elim (Pi.single j 1) (Pi.single j (-1)) : Fin n ⊕ Fin n → ℝ)))
          = Q' *ᵥ Sum.elim b ((-1 : ℝ) • b) by
        rw [← sum_smul_single_elim b (-1 : ℝ)]
        rw [← Matrix.mulVecLin_apply, map_sum]
        exact Finset.sum_congr rfl fun j _ => by
          rw [LinearMap.map_smul, Matrix.mulVecLin_apply]] at hb0
      have hz := hQ'inj _ hb0
      funext i
      exact congrFun (congrArg (fun f => f ∘ Sum.inl) hz) i
  have hnegcount : Fintype.card {i // hB.eigenvalues i < 0} + n ≤ n + n := by
    have h := card_add_card_le_of_forms (-B)
      (fun s : {i // hB.eigenvalues i < 0} => ⇑(hB.eigenvectorBasis s.1))
      (fun j : Fin n => Q' *ᵥ (Sum.elim (Pi.single j 1) (Pi.single j 1) : Fin n ⊕ Fin n → ℝ))
      ?_ ?_ ?_
    · simpa [Fintype.card_sum] using h
    · intro a ha
      rw [Matrix.neg_mulVec, dotProduct_neg,
        form_on_eigencombo hB Subtype.val Subtype.val_injective a]
      rw [← Finset.sum_neg_distrib]
      obtain ⟨s₀, hs₀⟩ := Function.ne_iff.mp ha
      refine Finset.sum_pos' (fun s _ => ?_) ⟨s₀, Finset.mem_univ _, ?_⟩
      · rw [← neg_mul]
        exact mul_nonneg (neg_nonneg.mpr s.2.le) (sq_nonneg _)
      · rw [← neg_mul]
        exact mul_pos (neg_pos.mpr s₀.2) (sq_pos_iff.mpr hs₀)
    · intro b
      rw [show (∑ j, b j • (Q' *ᵥ (Sum.elim (Pi.single j 1) (Pi.single j 1) : Fin n ⊕ Fin n → ℝ)))
          = Q' *ᵥ Sum.elim b ((1 : ℝ) • b) by
        rw [← sum_smul_single_elim b (1 : ℝ)]
        rw [← Matrix.mulVecLin_apply, map_sum]
        exact Finset.sum_congr rfl fun j _ => by
          rw [LinearMap.map_smul, Matrix.mulVecLin_apply]]
      rw [Matrix.neg_mulVec, dotProduct_neg, hform, J_mulVec_elim,
        elim_dotProduct_elim, dotProduct_smul, smul_dotProduct]
      have hb : (0:ℝ) ≤ b ⬝ᵥ b := Fintype.sum_nonneg fun i => mul_self_nonneg _
      simp only [smul_eq_mul]
      nlinarith [hb]
    · intro b hb0
      rw [show (∑ j, b j • (Q' *ᵥ (Sum.elim (Pi.single j 1) (Pi.single j 1) : Fin n ⊕ Fin n → ℝ)))
          = Q' *ᵥ Sum.elim b ((1 : ℝ) • b) by
        rw [← sum_smul_single_elim b (1 : ℝ)]
        rw [← Matrix.mulVecLin_apply, map_sum]
        exact Finset.sum_congr rfl fun j _ => by
          rw [LinearMap.map_smul, Matrix.mulVecLin_apply]] at hb0
      have hz := hQ'inj _ hb0
      funext i
      exact congrFun (congrArg (fun f => f ∘ Sum.inl) hz) i
  -- assemble the counts
  have hsub₁ : Fintype.card {i // 0 < hB.eigenvalues i}
      = (Finset.univ.filter fun i => 0 < hB.eigenvalues i).card :=
    Fintype.card_subtype _
  have hsub₂ : Fintype.card {i // hB.eigenvalues i < 0}
      = (Finset.univ.filter fun i => hB.eigenvalues i < 0).card :=
    Fintype.card_subtype _
  have hsplit : (Finset.univ.filter fun i => 0 < hB.eigenvalues i).card
      + (Finset.univ.filter fun i => hB.eigenvalues i < 0).card = n + n := by
    have hcongr : Finset.univ.filter (fun i => 0 < hB.eigenvalues i)
        = Finset.univ.filter fun i => ¬ hB.eigenvalues i < 0 := by
      apply Finset.filter_congr
      intro i _
      constructor
      · intro h; exact not_lt.mpr h.le
      · intro h
        exact lt_of_le_of_ne (not_lt.mp h) (Ne.symm (hne i))
    rw [hcongr, add_comm, Finset.card_filter_add_card_filter_not]
    simp [Fintype.card_sum]
  rw [hsub₁] at hposcount
  rw [hsub₂] at hnegcount
  exact ⟨by omega, by omega, hne⟩


/-! ### Private machinery: the diagonalization `JG = W diag(d) W'`

If `M` is surjective then `G` is positive definite.  With `Q = G^{1/2}`
the matrix `B = QJQ` is symmetric and congruent to `J`, so by
Sylvester's law of inertia it has exactly `n` positive and `n` negative
eigenvalues; and `JG = Q⁻¹ B Q` is similar to `B`, hence
diagonalizable with the same real spectrum. -/

private lemma J_transpose : (J n)ᵀ = J n := by
  rw [J, Matrix.fromBlocks_transpose]
  simp

private lemma exists_conjDiag
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) :
    ∃ (W W' : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ)
      (d : Fin n ⊕ Fin n → ℝ),
      W * W' = 1 ∧ W' * W = 1 ∧
      J n * G WK WQ = W * Matrix.diagonal d * W' ∧
      (Finset.univ.filter fun i => 0 < d i).card = n ∧
      (Finset.univ.filter fun i => d i < 0).card = n ∧
      ∀ i, d i ≠ 0 := by
  obtain ⟨Q, Q', hQ, -, hQQ, hQQ', hQ'Q⟩ := exists_sqrt (posDef_G WK WQ hM)
  have hB : (Q * J n * Q).IsHermitian := by
    show (Q * J n * Q)ᴴ = Q * J n * Q
    rw [Matrix.conjTranspose_eq_transpose_of_trivial, Matrix.transpose_mul,
      Matrix.transpose_mul, hQ, J_transpose, Matrix.mul_assoc]
  obtain ⟨hpos, hneg, hne⟩ := inertia hQ hQQ' hB
  set U : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
    ↑hB.eigenvectorUnitary with hUdef
  have hUl : U * star U = 1 := Unitary.coe_mul_star_self _
  have hUr : star U * U = 1 := mul_eq_one_comm.mp hUl
  refine ⟨Q' * U, star U * Q, hB.eigenvalues, ?_, ?_, ?_, hpos, hneg, hne⟩
  · calc Q' * U * (star U * Q) = Q' * (U * star U) * Q := by
          simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hUl, Matrix.mul_one, hQ'Q]
  · calc star U * Q * (Q' * U) = star U * (Q * Q') * U := by
          simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hQQ', Matrix.mul_one, hUr]
  · calc J n * G WK WQ = (Q' * Q) * J n * (Q * Q) := by
          rw [hQ'Q, Matrix.one_mul, hQQ]
      _ = Q' * (Q * J n * Q) * Q := by simp only [Matrix.mul_assoc]
      _ = Q' * (U * Matrix.diagonal hB.eigenvalues * star U) * Q := by
          rw [← spectral_real hB]
      _ = Q' * U * Matrix.diagonal hB.eigenvalues * (star U * Q) := by
          simp only [Matrix.mul_assoc]

/-- The roots of the characteristic polynomial of `½JG` are the `½ d_i`. -/
private lemma roots_half_JG {W W' : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ}
    {d : Fin n ⊕ Fin n → ℝ} (hWl : W * W' = 1) (hWr : W' * W = 1)
    (hJG : J n * G WK WQ = W * Matrix.diagonal d * W') :
    ((1 / 2 : ℝ) • (J n * G WK WQ)).charpoly.roots
      = Multiset.map (fun i => (1 / 2 : ℝ) * d i) Finset.univ.val := by
  rw [hJG, conj_smul, conj_roots hWl hWr]
  rfl

/-! ### Private machinery: multisets of entries

Two tuples (possibly on different index types) with the same multiset
of entries have the same sums of any function of the entries and the
same number of entries satisfying any predicate; a nondecreasing tuple
is determined by its multiset of entries. -/

private lemma sum_congr_of_map_eq {ι κ : Type*} [Fintype ι] [Fintype κ]
    {f : ι → ℝ} {g : κ → ℝ}
    (h : Multiset.map f Finset.univ.val = Multiset.map g Finset.univ.val)
    (φ : ℝ → ℝ) : ∑ i, φ (f i) = ∑ j, φ (g j) := by
  rw [Finset.sum_eq_multiset_sum, Finset.sum_eq_multiset_sum]
  have e1 : Multiset.map (fun i => φ (f i)) Finset.univ.val
      = Multiset.map φ (Multiset.map f Finset.univ.val) :=
    (Multiset.map_map φ f _).symm
  have e2 : Multiset.map (fun j => φ (g j)) Finset.univ.val
      = Multiset.map φ (Multiset.map g Finset.univ.val) :=
    (Multiset.map_map φ g _).symm
  rw [e1, e2, h]

private lemma card_filter_congr_of_map_eq {ι κ : Type*} [Fintype ι] [Fintype κ]
    {f : ι → ℝ} {g : κ → ℝ}
    (h : Multiset.map f Finset.univ.val = Multiset.map g Finset.univ.val)
    (p : ℝ → Prop) [DecidablePred p] :
    (Finset.univ.filter fun i => p (f i)).card
      = (Finset.univ.filter fun j => p (g j)).card := by
  have := congrArg (Multiset.countP p) h
  rw [Multiset.countP_map, Multiset.countP_map] at this
  simpa [Finset.card_def, Finset.filter_val] using this

private lemma monotone_eq_of_map_eq {k : ℕ} {x y : Fin k → ℝ}
    (hx : Monotone x) (hy : Monotone y)
    (h : Multiset.map x Finset.univ.val = Multiset.map y Finset.univ.val) :
    x = y := by
  rw [Fin.univ_val_map, Fin.univ_val_map, Multiset.coe_eq_coe] at h
  exact List.ofFn_injective (h.eq_of_sortedLE hx.sortedLE_ofFn hy.sortedLE_ofFn)

/-- A nondecreasing tuple in `ℝ^{2m}` with exactly `m` negative and `m`
positive entries has `x_m < 0 < x_{m+1}` (1-indexed). -/
private lemma monotone_middle {m : ℕ} {x : Fin (2 * m) → ℝ} (hx : Monotone x)
    (hm : 0 < m)
    (hneg : (Finset.univ.filter fun i => x i < 0).card = m)
    (hpos : (Finset.univ.filter fun i => 0 < x i).card = m) :
    x ⟨m - 1, by omega⟩ < 0 ∧ 0 < x ⟨m, by omega⟩ := by
  constructor
  · by_contra hc
    push_neg at hc
    have hsub : (Finset.univ.filter fun i => x i < 0)
        ⊆ Finset.Iio ⟨m - 1, by omega⟩ := by
      intro i hi
      rw [Finset.mem_filter] at hi
      rw [Finset.mem_Iio]
      by_contra hle
      push_neg at hle
      exact absurd (lt_of_le_of_lt (hx hle) hi.2) (not_lt.mpr hc)
    have := Finset.card_le_card hsub
    rw [hneg, Fin.card_Iio] at this
    simp at this
    omega
  · by_contra hc
    push_neg at hc
    have hsub : (Finset.univ.filter fun i => 0 < x i)
        ⊆ Finset.Ioi ⟨m, by omega⟩ := by
      intro i hi
      rw [Finset.mem_filter] at hi
      rw [Finset.mem_Ioi]
      by_contra hle
      push_neg at hle
      exact absurd (lt_of_lt_of_le hi.2 (hx hle)) (not_lt.mpr hc)
    have := Finset.card_le_card hsub
    rw [hpos, Fin.card_Ioi] at this
    simp at this
    omega

private lemma filter_ne_zero_nsmul_zero (k : ℕ) :
    Multiset.filter (fun r : ℝ => r ≠ 0) (k • ({0} : Multiset ℝ)) = 0 := by
  rw [Multiset.filter_eq_nil]
  intro a ha
  rw [Multiset.mem_nsmul, Multiset.mem_singleton] at ha
  simpa using ha.2

private lemma mul_neg_iff_of_pos {c t : ℝ} (hc : 0 < c) : c * t < 0 ↔ t < 0 := by
  constructor
  · intro h
    by_contra h'
    push_neg at h'
    nlinarith
  · intro h
    nlinarith

/-! ### The nonzero spectrum of `S` -/

section Spectrum

open Polynomial

/-- "The nonzero spectrum of `S`, with multiplicity, is the spectrum of
`½JG`": the multiset of nonzero roots of the characteristic polynomial
of `S` equals the multiset of roots of the characteristic polynomial of
`½JG` (all of which are nonzero).  This follows from the fact, quoted in
the paper, that for `A ∈ M_{N×2n}` and `B ∈ M_{2n×N}` the nonzero
eigenvalues of `AB` and `BA` coincide with multiplicity
(`Matrix.charpoly_mul_comm'`), applied to `A = Mᵀ`, `B = ½JM`. -/
theorem charpoly_roots_filter_ne_zero
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) :
    (symPart (WKᵀ * WQ)).charpoly.roots.filter (fun r => r ≠ 0)
      = ((1 / 2 : ℝ) • (J n * G WK WQ)).charpoly.roots := by
  obtain ⟨W, W', d, hWl, hWr, hJG, -, -, hne⟩ := exists_conjDiag WK WQ hM
  have hS : symPart (WKᵀ * WQ)
      = (M WK WQ)ᵀ * ((1 / 2 : ℝ) • (J n * M WK WQ)) := by
    rw [symPart_eq_half_conj, Matrix.mul_smul, Matrix.mul_assoc]
  have hJG' : ((1 / 2 : ℝ) • (J n * M WK WQ)) * (M WK WQ)ᵀ
      = (1 / 2 : ℝ) • (J n * G WK WQ) := by
    rw [Matrix.smul_mul, Matrix.mul_assoc, G]
  have key := Matrix.charpoly_mul_comm' (M WK WQ)ᵀ
    ((1 / 2 : ℝ) • (J n * M WK WQ))
  rw [← hS, hJG'] at key
  have hne1 : (X ^ Fintype.card (Fin n ⊕ Fin n)
      * (symPart (WKᵀ * WQ)).charpoly) ≠ 0 :=
    mul_ne_zero (pow_ne_zero _ X_ne_zero) (Matrix.charpoly_monic _).ne_zero
  have hne2 : (X ^ Fintype.card (Fin N)
      * ((1 / 2 : ℝ) • (J n * G WK WQ)).charpoly) ≠ 0 :=
    mul_ne_zero (pow_ne_zero _ X_ne_zero) (Matrix.charpoly_monic _).ne_zero
  have hroots := congrArg Polynomial.roots key
  rw [Polynomial.roots_mul hne1, Polynomial.roots_mul hne2,
    Polynomial.roots_X_pow, Polynomial.roots_X_pow] at hroots
  have hfilt := congrArg (Multiset.filter fun r : ℝ => r ≠ 0) hroots
  rw [Multiset.filter_add, Multiset.filter_add, filter_ne_zero_nsmul_zero,
    filter_ne_zero_nsmul_zero, zero_add, zero_add] at hfilt
  rw [hfilt, Multiset.filter_eq_self.mpr]
  intro r hr
  rw [roots_half_JG WK WQ hWl hWr hJG, Multiset.mem_map] at hr
  obtain ⟨i, -, rfl⟩ := hr
  exact mul_ne_zero (by norm_num) (hne i)

/-- The characteristic polynomial of the real symmetric matrix `S`
(of degree `N`) has `N` real roots, so its root multiset is the
spectrum of `S` with multiplicity. -/
theorem card_charpoly_roots_symPart :
    Multiset.card (symPart (WKᵀ * WQ)).charpoly.roots = N := by
  rw [(symPart_isHermitian _).roots_charpoly_eq_eigenvalues,
    Multiset.card_map, Finset.card_val, Finset.card_univ, Fintype.card_fin]

/-- The characteristic polynomial of `½JG` (of degree `2n`) has `2n`
real roots: the spectrum of `½JG` is real, so its root multiset is its
spectrum with multiplicity. -/
theorem card_charpoly_roots_half_JG
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) :
    Multiset.card ((1 / 2 : ℝ) • (J n * G WK WQ)).charpoly.roots = 2 * n := by
  obtain ⟨W, W', d, hWl, hWr, hJG, -, -, -⟩ := exists_conjDiag WK WQ hM
  rw [roots_half_JG WK WQ hWl hWr hJG, Multiset.card_map, Finset.card_val,
    Finset.card_univ, Fintype.card_sum, Fintype.card_fin]
  ring

end Spectrum

/-! ### The sorted normalized nonzero spectrum and `O`, `E`, `R` -/

/-- The Frobenius norm `‖L‖ = √(tr(LᵀL))`; `frobSq` is its square. -/
noncomputable def frobNorm {k : ℕ} (L : Matrix (Fin k) (Fin k) ℝ) : ℝ :=
  Real.sqrt (frobSq L)

/-- `H = JG/(2‖S‖)`. -/
noncomputable def H (WK WQ : Matrix (Fin n) (Fin N) ℝ) :
    Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  (1 / (2 * frobNorm (symPart (WKᵀ * WQ)))) • (J n * G WK WQ)

/-- `λ` is "the sorted normalized nonzero spectrum of `S`": `λ` is
nondecreasing, and its entries, with multiplicity, are the nonzero roots
of the characteristic polynomial of `S` divided by `‖S‖`.  (The
multiset of roots of a real symmetric matrix's characteristic polynomial
is its spectrum with multiplicity.)  Such a `λ` is unique
(`isSortedNormalizedNonzeroSpectrum_unique`). -/
def IsSortedNormalizedNonzeroSpectrum {k m : ℕ} (S : Matrix (Fin k) (Fin k) ℝ)
    (lam : Fin m → ℝ) : Prop :=
  Monotone lam ∧
    Multiset.map lam Finset.univ.val
      = (S.charpoly.roots.filter fun r => r ≠ 0).map fun r => r / frobNorm S

/-- A nondecreasing tuple is determined by its multiset of entries, so
the sorted normalized nonzero spectrum is unique. -/
theorem isSortedNormalizedNonzeroSpectrum_unique {k m : ℕ}
    {S : Matrix (Fin k) (Fin k) ℝ} {lam lam' : Fin m → ℝ}
    (h : IsSortedNormalizedNonzeroSpectrum S lam)
    (h' : IsSortedNormalizedNonzeroSpectrum S lam') : lam = lam' :=
  monotone_eq_of_map_eq h.1 h'.1 (h.2.trans h'.2.symm)

/-- If `M` is surjective, `S` has a sorted normalized nonzero spectrum in
`ℝ^{2n}`: the `2n` roots of the characteristic polynomial of `½JG`,
divided by `‖S‖` and sorted nondecreasingly. -/
theorem exists_isSortedNormalizedNonzeroSpectrum
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) :
    ∃ lam : Fin (2 * n) → ℝ,
      IsSortedNormalizedNonzeroSpectrum (symPart (WKᵀ * WQ)) lam := by
  obtain ⟨W, W', d, hWl, hWr, hJG, -, -, -⟩ := exists_conjDiag WK WQ hM
  let ι : Fin (2 * n) ≃ Fin n ⊕ Fin n :=
    (finCongr (by ring)).trans finSumFinEquiv.symm
  let v : Fin (2 * n) → ℝ :=
    fun i => (1 / 2 : ℝ) * d (ι i) / frobNorm (symPart (WKᵀ * WQ))
  refine ⟨v ∘ Tuple.sort v, Tuple.monotone_sort v, ?_⟩
  rw [charpoly_roots_filter_ne_zero WK WQ hM, roots_half_JG WK WQ hWl hWr hJG,
    Multiset.map_map, ← Multiset.map_map v, Multiset.map_univ_val_equiv,
    ← Multiset.map_univ_val_equiv ι, Multiset.map_map]
  rfl

/-- The sorted normalized nonzero spectrum `λ` of `S`, written through
the diagonalization `H = W diag(e) W'` of `H = JG/(2‖S‖)`: the entries
of `λ` are, with multiplicity, the `e_i`, and `λ ∈ Λ_n`. -/
private lemma sorted_package
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) (hn : 1 ≤ n)
    {lam : Fin (2 * n) → ℝ}
    (h : IsSortedNormalizedNonzeroSpectrum (symPart (WKᵀ * WQ)) lam) :
    ∃ (W W' : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ)
      (e : Fin n ⊕ Fin n → ℝ),
      W * W' = 1 ∧ W' * W = 1 ∧ H WK WQ = W * Matrix.diagonal e * W' ∧
      Multiset.map lam Finset.univ.val = Multiset.map e Finset.univ.val ∧
      MemLambdaSet n lam := by
  obtain ⟨W, W', d, hWl, hWr, hJG, hpos, hneg, hne⟩ := exists_conjDiag WK WQ hM
  have hfrob : frobSq (symPart (WKᵀ * WQ)) = (1 / 4 : ℝ) * ∑ i, d i ^ 2 := by
    rw [frobSq_symPart_eq_quarter_trace, hJG, pow_two, conj_mul hWr,
      conj_trace hWr]
    simp [Pi.mul_apply, sq]
  have hT : 0 < ∑ i, d i ^ 2 :=
    Finset.sum_pos' (fun i _ => sq_nonneg _)
      ⟨Sum.inl ⟨0, by omega⟩, Finset.mem_univ _, sq_pos_iff.mpr (hne _)⟩
  set s := frobNorm (symPart (WKᵀ * WQ)) with hsdef
  have hs : 0 < s := by
    rw [hsdef, frobNorm]
    exact Real.sqrt_pos.mpr (by rw [hfrob]; exact mul_pos (by norm_num) hT)
  have hs2 : s ^ 2 = frobSq (symPart (WKᵀ * WQ)) := by
    rw [hsdef, frobNorm]
    exact Real.sq_sqrt (frobSq_nonneg _)
  set c := 1 / (2 * s) with hc
  have hcpos : 0 < c := by rw [hc]; positivity
  set e : Fin n ⊕ Fin n → ℝ := fun i => c * d i with he
  have hmap : Multiset.map lam Finset.univ.val
      = Multiset.map e Finset.univ.val := by
    rw [h.2, charpoly_roots_filter_ne_zero WK WQ hM,
      roots_half_JG WK WQ hWl hWr hJG, Multiset.map_map]
    refine Multiset.map_congr rfl fun i _ => ?_
    show (1 / 2 : ℝ) * d i / s = c * d i
    rw [hc]
    ring
  have hH : H WK WQ = W * Matrix.diagonal e * W' := by
    rw [H, hJG, conj_smul]
    rfl
  have hunit : ∑ i, lam i ^ 2 = 1 := by
    have h1 : ∑ i, lam i ^ 2 = ∑ j, e j ^ 2 :=
      sum_congr_of_map_eq hmap (fun t => t ^ 2)
    rw [h1]
    show ∑ j, (c * d j) ^ 2 = 1
    rw [show ∑ j, (c * d j) ^ 2 = c ^ 2 * ∑ j, d j ^ 2 by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring]
    have h4 : ∑ j, d j ^ 2 = 4 * s ^ 2 := by rw [hs2, hfrob]; ring
    rw [h4, hc]
    field_simp
    norm_num
  have hnegc : (Finset.univ.filter fun i => lam i < 0).card = n := by
    have h1 : (Finset.univ.filter fun i => lam i < 0).card
        = (Finset.univ.filter fun j => e j < 0).card :=
      card_filter_congr_of_map_eq hmap (fun t => t < 0)
    rw [h1]
    exact (congrArg Finset.card
      (Finset.filter_congr fun j _ => mul_neg_iff_of_pos hcpos)).trans hneg
  have hposc : (Finset.univ.filter fun i => 0 < lam i).card = n := by
    have h1 : (Finset.univ.filter fun i => 0 < lam i).card
        = (Finset.univ.filter fun j => 0 < e j).card :=
      card_filter_congr_of_map_eq hmap (fun t => 0 < t)
    rw [h1]
    exact (congrArg Finset.card
      (Finset.filter_congr fun j _ => mul_pos_iff_of_pos_left hcpos)).trans hpos
  obtain ⟨hmid1, hmid2⟩ := monotone_middle h.1 (by omega) hnegc hposc
  exact ⟨W, W', e, hWl, hWr, hH, hmap, h.1, hunit, fun _ => hmid1, fun _ => hmid2⟩

/-- "The sorted normalized nonzero spectrum `λ` of `S` lies in `Λ_n`":
the nonzero spectrum of `S` consists of `n` positive and `n` negative
eigenvalues (Sylvester's law of inertia), and `‖S‖² = ¼ tr((JG)²)` is
the sum of their squares. -/
theorem memLambdaSet_of_isSortedNormalizedNonzeroSpectrum
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) (hn : 1 ≤ n)
    {lam : Fin (2 * n) → ℝ}
    (h : IsSortedNormalizedNonzeroSpectrum (symPart (WKᵀ * WQ)) lam) :
    MemLambdaSet n lam :=
  (sorted_package WK WQ hM hn h).choose_spec.choose_spec.choose_spec.2.2.2.2

/-- Every entry of the diagonalization of `H` is an entry of `λ`, hence
has absolute value less than one (Lemma OE-equivalences). -/
private lemma abs_diag_lt_one {lam : Fin (2 * n) → ℝ} (hlam : MemLambdaSet n lam)
    {e : Fin n ⊕ Fin n → ℝ}
    (hmap : Multiset.map lam Finset.univ.val = Multiset.map e Finset.univ.val)
    (i : Fin n ⊕ Fin n) : |e i| < 1 := by
  have hi : e i ∈ Multiset.map lam Finset.univ.val := by
    rw [hmap]
    exact Multiset.mem_map_of_mem e (Finset.mem_univ i)
  obtain ⟨j, -, hj⟩ := Multiset.mem_map.mp hi
  rw [← hj]
  exact hlam.abs_entry_lt_one j

/-- `O(λ) = tr(H(I-H²)⁻¹)`. -/
theorem statO_eq_trace
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) (hn : 1 ≤ n)
    {lam : Fin (2 * n) → ℝ}
    (h : IsSortedNormalizedNonzeroSpectrum (symPart (WKᵀ * WQ)) lam) :
    statO lam = Matrix.trace (H WK WQ * (1 - H WK WQ ^ 2)⁻¹) := by
  obtain ⟨W, W', e, hWl, hWr, hH, hmap, hlam⟩ := sorted_package WK WQ hM hn h
  have habs := abs_diag_lt_one hlam hmap
  have hne : ∀ i, (1 - e * e) i ≠ 0 := fun i => by
    have := (sq_lt_one_iff_abs_lt_one (e i)).mpr (habs i)
    rw [sq] at this
    exact ne_of_gt (by simp only [Pi.sub_apply, Pi.one_apply, Pi.mul_apply]; linarith)
  have hH2 : H WK WQ ^ 2 = W * Matrix.diagonal (e * e) * W' := by
    rw [pow_two, hH, conj_mul hWr]
  have hsub : 1 - H WK WQ ^ 2 = W * Matrix.diagonal (1 - e * e) * W' := by
    rw [hH2, ← conj_sub, ← conj_one hWl]
  have hinv : (1 - H WK WQ ^ 2)⁻¹ = W * Matrix.diagonal (1 - e * e)⁻¹ * W' := by
    rw [hsub]
    exact conj_inv hWl hWr hne
  rw [hinv, hH, conj_mul hWr, conj_trace hWr]
  have hsum : ∑ i, lam i / (1 - lam i ^ 2) = ∑ j, e j / (1 - e j ^ 2) :=
    sum_congr_of_map_eq hmap (fun t => t / (1 - t ^ 2))
  rw [statO, hsum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Pi.mul_apply, Pi.sub_apply, Pi.one_apply, Pi.inv_apply, sq,
    div_eq_mul_inv]

/-- `E(λ) = tr(H²(I-H²)⁻¹)`. -/
theorem statE_eq_trace
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) (hn : 1 ≤ n)
    {lam : Fin (2 * n) → ℝ}
    (h : IsSortedNormalizedNonzeroSpectrum (symPart (WKᵀ * WQ)) lam) :
    statE lam = Matrix.trace (H WK WQ ^ 2 * (1 - H WK WQ ^ 2)⁻¹) := by
  obtain ⟨W, W', e, hWl, hWr, hH, hmap, hlam⟩ := sorted_package WK WQ hM hn h
  have habs := abs_diag_lt_one hlam hmap
  have hne : ∀ i, (1 - e * e) i ≠ 0 := fun i => by
    have := (sq_lt_one_iff_abs_lt_one (e i)).mpr (habs i)
    rw [sq] at this
    exact ne_of_gt (by simp only [Pi.sub_apply, Pi.one_apply, Pi.mul_apply]; linarith)
  have hH2 : H WK WQ ^ 2 = W * Matrix.diagonal (e * e) * W' := by
    rw [pow_two, hH, conj_mul hWr]
  have hsub : 1 - H WK WQ ^ 2 = W * Matrix.diagonal (1 - e * e) * W' := by
    rw [hH2, ← conj_sub, ← conj_one hWl]
  have hinv : (1 - H WK WQ ^ 2)⁻¹ = W * Matrix.diagonal (1 - e * e)⁻¹ * W' := by
    rw [hsub]
    exact conj_inv hWl hWr hne
  rw [hinv, hH2, conj_mul hWr, conj_trace hWr]
  have hsum : ∑ i, lam i ^ 2 / (1 - lam i ^ 2) = ∑ j, e j ^ 2 / (1 - e j ^ 2) :=
    sum_congr_of_map_eq hmap (fun t => t ^ 2 / (1 - t ^ 2))
  rw [statE, hsum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Pi.mul_apply, Pi.sub_apply, Pi.one_apply, Pi.inv_apply, sq,
    div_eq_mul_inv]

/-- `R(λ) = tr((I-H)⁻¹)`. -/
theorem statR_eq_trace
    (hM : Function.Surjective (Matrix.mulVecLin (M WK WQ))) (hn : 1 ≤ n)
    {lam : Fin (2 * n) → ℝ}
    (h : IsSortedNormalizedNonzeroSpectrum (symPart (WKᵀ * WQ)) lam) :
    statR lam = Matrix.trace ((1 - H WK WQ)⁻¹) := by
  obtain ⟨W, W', e, hWl, hWr, hH, hmap, hlam⟩ := sorted_package WK WQ hM hn h
  have habs := abs_diag_lt_one hlam hmap
  have hne : ∀ i, (1 - e) i ≠ 0 := fun i => by
    have := (abs_lt.mp (habs i)).2
    exact ne_of_gt (by simp only [Pi.sub_apply, Pi.one_apply]; linarith)
  have hsub : 1 - H WK WQ = W * Matrix.diagonal (1 - e) * W' := by
    rw [hH, ← conj_sub, ← conj_one hWl]
  have hinv : (1 - H WK WQ)⁻¹ = W * Matrix.diagonal (1 - e)⁻¹ * W' := by
    rw [hsub]
    exact conj_inv hWl hWr hne
  rw [hinv, conj_trace hWr]
  have hsum : ∑ i, 1 / (1 - lam i) = ∑ j, 1 / (1 - e j) :=
    sum_congr_of_map_eq hmap (fun t => 1 / (1 - t))
  rw [statR, hsum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Pi.sub_apply, Pi.one_apply, Pi.inv_apply, one_div]

end GramForm

end Appendices
