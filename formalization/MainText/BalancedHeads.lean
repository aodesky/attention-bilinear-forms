/-
Main text, §"Why heads are balanced": Theorem
`thm:heads-have-balanced-attention`.

    Let `A` be a head with query and key maps `Q, K : C → H`, `dim H = n`.
    If `A` is nondegenerate --- that is, the stacked map
    `J = (Q,K) : C → H ⊕ H` is surjective --- then the symmetric form
    `S(x,y) = ½(⟨Kx,Qy⟩ + ⟨Qx,Ky⟩)` has signature `(n, n, N - 2n)`.

The head `A` enters the statement only through its query and key maps and
through the nondegeneracy condition, which the paper defines directly as
surjectivity of `J`.  Nothing about attention functions, weights or
potentials is used, so the theorem is stated here for `Q` and `K`
themselves.

The paper's proof has two halves:

  * the *radical* of `S` is `ker J`, of dimension `N - 2n`, so `S` has
    rank `2n` (`finrank_radical`, via `radical_eq_ker`);
  * `S` is the symmetric part of a matrix of rank `≤ n`, so by
    `lem:rank-balance` neither `n₊` nor `n₋` exceeds `n`
    (`MainText/RankBalance.lean`).

Together with `n₊ + n₋ = rank S = 2n` these force `n₊ = n₋ = n`
(`balanced_of_rank_eq`); the assembled statement is
`heads_have_balanced_attention`.

Encoding of the signature.  Mathlib has no signature API.  The signature
`(n₊, n₋, n₀)` of `S` is taken to be the numbers of positive, negative and
zero eigenvalues, with multiplicity, of the Gram matrix `G_{ij} = S(bᵢ, bⱼ)`
in a basis `b` of `C` (`gram`, `signature`); by Sylvester's law of inertia
these do not depend on `b`, and the theorem is stated for every basis.  The
paper's "Gram matrix of `2S` is `KᵀQ + QᵀK`" is `gram_eq_symPart`, where `KᵀQ`
is `⟨K bᵢ, Q bⱼ⟩`, factored through an orthonormal basis of `H`; the
identification of `ker G` with the radical of `S` is `ker_gram_eq_map`.
-/
import Mathlib
import MainText.RankBalance

namespace Appendices.MainText

open Module LinearMap

variable {C H : Type*}
variable [NormedAddCommGroup C] [InnerProductSpace ℝ C] [FiniteDimensional ℝ C]
variable [NormedAddCommGroup H] [InnerProductSpace ℝ H] [FiniteDimensional ℝ H]

/-- The stacked map `J = (Q,K) : C → H ⊕ H`, `x ↦ (Qx, Kx)`. -/
noncomputable def stacked (Q K : C →ₗ[ℝ] H) : C →ₗ[ℝ] (H × H) := Q.prod K

/-- A head is *nondegenerate* when its stacked map is surjective. -/
def Nondegenerate (Q K : C →ₗ[ℝ] H) : Prop := Function.Surjective (stacked Q K)

/-- The symmetric bilinear form `S(x,y) = ½(⟨Kx,Qy⟩ + ⟨Qx,Ky⟩)` of the head. -/
noncomputable def headForm (Q K : C →ₗ[ℝ] H) (x y : C) : ℝ :=
  (1 / 2 : ℝ) * (inner ℝ (K x) (Q y) + inner ℝ (Q x) (K y))

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
lemma headForm_symm (Q K : C →ₗ[ℝ] H) (x y : C) :
    headForm Q K x y = headForm Q K y x := by
  simp only [headForm, real_inner_comm]
  ring

/-- The nondegenerate symmetric form `B((a,b),(a',b')) = ⟨a,b'⟩ + ⟨b,a'⟩` on `H ⊕ H`,
which the paper introduces to identify the radical of `S`. -/
noncomputable def pairForm (p q : H × H) : ℝ :=
  inner ℝ p.1 q.2 + inner ℝ p.2 q.1

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
/-- `B(Jx, Jy) = 2 S(x,y)`: the head form is the pullback of `pairForm` along `J`. -/
lemma pairForm_stacked (Q K : C →ₗ[ℝ] H) (x y : C) :
    pairForm (stacked Q K x) (stacked Q K y) = 2 * headForm Q K x y := by
  simp only [pairForm, stacked, headForm, LinearMap.prod_apply, Pi.prod]
  ring

omit [FiniteDimensional ℝ H] in
/-- `pairForm` is nondegenerate: if `B(p, ·) = 0` then `p = 0`. -/
lemma pairForm_nondegenerate {p : H × H} (hp : ∀ q, pairForm p q = 0) : p = 0 := by
  have h1 : p.1 = 0 := by
    have := hp (0, p.1)
    simpa [pairForm, real_inner_self_eq_norm_sq] using this
  have h2 : p.2 = 0 := by
    have := hp (p.2, 0)
    simpa [pairForm, real_inner_self_eq_norm_sq] using this
  exact Prod.ext h1 h2

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
/-- **The radical of `S` is `ker J`.**  This is the heart of the paper's proof:
`Jx = 0` gives `S(x,·) = 0`; conversely if `S(x,·) = 0` then `B(Jx,·)` vanishes
on `J(C) = H ⊕ H` by surjectivity, so `Jx = 0` by nondegeneracy of `B`. -/
theorem radical_eq_ker (Q K : C →ₗ[ℝ] H) (hnd : Nondegenerate Q K) :
    {x : C | ∀ y, headForm Q K x y = 0} = (LinearMap.ker (stacked Q K) : Set C) := by
  ext x
  simp only [Set.mem_setOf_eq, SetLike.mem_coe, LinearMap.mem_ker]
  constructor
  · intro hx
    -- `B(Jx, q) = 0` for every `q`, using surjectivity of `J`
    refine pairForm_nondegenerate ?_
    intro q
    obtain ⟨y, rfl⟩ := hnd q
    rw [pairForm_stacked, hx y, mul_zero]
  · intro hx y
    have : pairForm (stacked Q K x) (stacked Q K y) = 0 := by rw [hx]; simp [pairForm]
    rw [pairForm_stacked] at this
    linarith

/-- **The radical of `S` has dimension `N - 2n`.**  By `radical_eq_ker` the radical
is `ker J`, and `J : C → H ⊕ H` is surjective, so rank-nullity gives
`dim (ker J) + 2n = N`.  This is the paper's "`n₀ = N - 2n`", and equivalently
says that `S` has rank `2n`. -/
theorem finrank_radical (Q K : C →ₗ[ℝ] H) (hnd : Nondegenerate Q K) :
    finrank ℝ (LinearMap.ker (stacked Q K)) + 2 * finrank ℝ H = finrank ℝ C := by
  have hrange : LinearMap.range (stacked Q K) = ⊤ := LinearMap.range_eq_top.mpr hnd
  have hrn := LinearMap.finrank_range_add_finrank_ker (stacked Q K)
  rw [hrange] at hrn
  simp [finrank_top, Module.finrank_prod, two_mul] at hrn ⊢
  omega

/-- The arithmetic step closing the paper's proof: `n₊ ≤ n` and `n₋ ≤ n` from
`lem:rank-balance`, together with `n₊ + n₋ = rank S = 2n`, force `n₊ = n₋ = n`. -/
theorem balanced_of_rank_eq {n nplus nminus : ℕ}
    (hp : nplus ≤ n) (hm : nminus ≤ n) (hsum : nplus + nminus = 2 * n) :
    nplus = n ∧ nminus = n := ⟨by omega, by omega⟩

/-! ### The signature of `S` -/

section Signature

variable {N : ℕ}

/-- The Gram matrix `G_{ij} = S(bᵢ, bⱼ)` of the head form in a basis `b` of `C`. -/
noncomputable def gram (b : Basis (Fin N) ℝ C) (Q K : C →ₗ[ℝ] H) :
    Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i j => headForm Q K (b i) (b j)

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
lemma gram_isHermitian (b : Basis (Fin N) ℝ C) (Q K : C →ₗ[ℝ] H) :
    (gram b Q K).IsHermitian := by
  ext i j
  simp [gram, headForm_symm]

/-- The signature `(n₊, n₋, n₀)` of a real symmetric matrix: the numbers of
positive, negative and zero eigenvalues, counted with multiplicity. -/
noncomputable def signature {A : Matrix (Fin N) (Fin N) ℝ} (hA : A.IsHermitian) :
    ℕ × ℕ × ℕ :=
  ((Finset.univ.filter fun i => 0 < hA.eigenvalues i).card,
    (Finset.univ.filter fun i => hA.eigenvalues i < 0).card,
    (Finset.univ.filter fun i => hA.eigenvalues i = 0).card)

lemma eigenvalues_congr {A B : Matrix (Fin N) (Fin N) ℝ} (h : A = B)
    (hA : A.IsHermitian) (hB : B.IsHermitian) : hA.eigenvalues = hB.eigenvalues := by
  subst h; rfl

omit [FiniteDimensional ℝ C] in
/-- "The Gram matrix of `2S` is `KᵀQ + QᵀK`, the symmetric part of `2KᵀQ`; since
`KᵀQ` factors through `H`, it has rank at most `n`."  Here `KᵀQ` is the matrix
`⟨K bᵢ, Q bⱼ⟩`, factored through an orthonormal basis of `H`. -/
lemma gram_eq_symPart (b : Basis (Fin N) ℝ C) (Q K : C →ₗ[ℝ] H) :
    ∃ L : Matrix (Fin N) (Fin N) ℝ, L.rank ≤ finrank ℝ H ∧ gram b Q K = symPart L := by
  let ob := stdOrthonormalBasis ℝ H
  let P : Matrix (Fin N) (Fin (finrank ℝ H)) ℝ := Matrix.of fun i k => inner ℝ (K (b i)) (ob k)
  let R : Matrix (Fin (finrank ℝ H)) (Fin N) ℝ := Matrix.of fun k j => inner ℝ (ob k) (Q (b j))
  refine ⟨P * R, (Matrix.rank_mul_le_left P R).trans (Matrix.rank_le_width P), ?_⟩
  have hPR : ∀ i j, (P * R) i j = inner ℝ (K (b i)) (Q (b j)) := fun i j => by
    simp only [Matrix.mul_apply, P, R, Matrix.of_apply]
    exact ob.sum_inner_mul_inner _ _
  ext i j
  simp only [gram, Matrix.of_apply, headForm, symPart, Matrix.smul_apply, Matrix.add_apply,
    Matrix.transpose_apply, hPR, smul_eq_mul, real_inner_comm (Q (b i)) (K (b j))]

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
/-- `S(x, Σ cⱼ bⱼ) = Σ cⱼ S(x, bⱼ)`. -/
lemma headForm_sum (Q K : C →ₗ[ℝ] H) (x : C) {ι : Type*} [Fintype ι] (c : ι → ℝ)
    (v : ι → C) :
    headForm Q K x (∑ j, c j • v j) = ∑ j, c j * headForm Q K x (v j) := by
  simp only [headForm, map_sum, map_smul, inner_sum, inner_smul_right, Finset.mul_sum,
    ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => by ring

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
/-- The kernel of the Gram matrix corresponds, under the coordinates of `b`, to the
radical of `S`, which is `ker J` (`radical_eq_ker`). -/
lemma ker_gram_eq_map (b : Basis (Fin N) ℝ C) (Q K : C →ₗ[ℝ] H) (hnd : Nondegenerate Q K) :
    LinearMap.ker (gram b Q K).mulVecLin
      = (LinearMap.ker (stacked Q K)).map b.equivFun.toLinearMap := by
  ext c
  rw [Submodule.mem_map_equiv, LinearMap.mem_ker, ← SetLike.mem_coe,
    ← radical_eq_ker Q K hnd, Set.mem_setOf_eq, Basis.equivFun_symm_apply]
  rw [Matrix.mulVecLin_apply]
  constructor
  · intro hc y
    rw [← b.sum_repr y, headForm_sum]
    refine Finset.sum_eq_zero fun i _ => ?_
    have := congrFun hc i
    simp only [Matrix.mulVec, dotProduct, gram, Matrix.of_apply, Pi.zero_apply] at this
    rw [headForm_symm, headForm_sum]
    have h' : ∑ j, c j * headForm Q K (b i) (b j) = 0 := by
      rw [← this]; exact Finset.sum_congr rfl fun j _ => mul_comm _ _
    rw [h', mul_zero]
  · intro hx
    funext i
    have := hx (b i)
    rw [headForm_symm, headForm_sum] at this
    simp only [Matrix.mulVec, dotProduct, gram, Matrix.of_apply, Pi.zero_apply]
    rw [← this]
    exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- **Theorem (heads have balanced attention).**  If the head is nondegenerate,
then the symmetric bilinear form `S(x,y) = ½(⟨Kx,Qy⟩ + ⟨Qx,Ky⟩)` has signature
`(n₊, n₋, n₀) = (n, n, N - 2n)`, where `n = dim H` and `N = dim C`; the signature
is that of the Gram matrix of `S` in any basis `b` of `C`. -/
theorem heads_have_balanced_attention (Q K : C →ₗ[ℝ] H) (hnd : Nondegenerate Q K)
    (b : Basis (Fin N) ℝ C) :
    signature (gram_isHermitian b Q K) = (finrank ℝ H, finrank ℝ H, N - 2 * finrank ℝ H) := by
  set hG := gram_isHermitian b Q K
  set n := finrank ℝ H
  -- `n₊ ≤ n` and `n₋ ≤ n` by `lem:rank-balance`
  obtain ⟨L, hLrank, hGL⟩ := gram_eq_symPart b Q K
  have hev := eigenvalues_congr hGL hG (symPart_isHermitian L)
  obtain ⟨hplus, hminus⟩ := rank_balance L n hLrank
  simp only [← hev] at hplus hminus
  -- `S` has rank `2n`: its radical is `ker J`, of dimension `N - 2n`
  have hker : finrank ℝ (LinearMap.ker (gram b Q K).mulVecLin) + 2 * n = N := by
    rw [ker_gram_eq_map b Q K hnd, LinearEquiv.finrank_map_eq, finrank_radical Q K hnd,
      Module.finrank_eq_card_basis b, Fintype.card_fin]
  have hrn := LinearMap.finrank_range_add_finrank_ker (gram b Q K).mulVecLin
  have hrank : (gram b Q K).rank + finrank ℝ (LinearMap.ker (gram b Q K).mulVecLin) = N := by
    rw [Module.finrank_fin_fun] at hrn
    exact hrn
  have hnz : (gram b Q K).rank
      = (Finset.univ.filter fun i => 0 < hG.eigenvalues i).card
        + (Finset.univ.filter fun i => hG.eigenvalues i < 0).card := by
    rw [hG.rank_eq_card_non_zero_eigs, Fintype.card_subtype, ← Finset.card_union_of_disjoint
      (Finset.disjoint_filter.mpr fun i _ h1 h2 => by linarith)]
    congr 1
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]
    exact ⟨fun h => (lt_or_gt_of_ne h).symm, fun h => h.elim ne_of_gt ne_of_lt⟩
  have hzero : (Finset.univ.filter fun i => hG.eigenvalues i = 0).card
      + (gram b Q K).rank = N := by
    rw [hG.rank_eq_card_non_zero_eigs, Fintype.card_subtype]
    simpa using Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin N)))
      (fun i => hG.eigenvalues i = 0)
  -- `n₊ + n₋ = 2n` with `n₊, n₋ ≤ n` forces `n₊ = n₋ = n`
  obtain ⟨h1, h2⟩ := balanced_of_rank_eq hplus hminus (by omega)
  simp only [signature, Prod.mk.injEq]
  exact ⟨h1, h2, by omega⟩

end Signature

end Appendices.MainText
