/-
Main text, §"Bilinear attention": the definitions of attention functions,
bilinear attention functions, attention heads and transformers, and
Proposition `prop:qkv-homogenizes-to-bilinear`.

    Proposition.  Let `Ĉ = C ⊕ ℝe` and `ι : C → Ĉ`, `x ↦ x + e`.  For every
    single-head QKV transformer `T : C^ℓ → C^ℓ` there is a single-head
    bilinear transformer `T̂ : Ĉ^ℓ → Ĉ^ℓ` with the same potential such that
    `T̂(ιx₁, …, ιx_ℓ) = ι^ℓ(T(x₁, …, x_ℓ))`.

Definitions, as in the paper.
  * (Attention function.)  A smooth `a : C^k → C` together with a smooth
    `ω : C^k → Δ_{k-1}`, a smooth convex `φ : ℝ^k → ℝ` and a polynomial map
    `σ : C^k → ℝ^k` such that `ω = (∇φ) ∘ σ` and `a(x) = Σ_j ω_j(x) x_j`
    (`IsAttentionFunction`; the data `ω, φ, σ` are carried along, as the
    paper does "with an abuse of terminology").
  * (Bilinear.)  The scoring function is
    `σ(x₁, …, x_k) = (P_{k-1}(x₁, x_k), …, P₀(x_k, x_k))` with each `P_d` a
    bilinear form (`IsBilinearScore`).
  * (Attention head.)  `A(x₁, …, x_ℓ) = (x₁, a₂(x₁, x₂), …, a_ℓ(x₁, …, x_ℓ))`
    (`head`).
  * (Transformer, single head.)  `T = T_M ∘ T_A` with `T_A = id + (V ⊗ id) ∘ A`
    and `T_M = id + M ⊗ id` (`transformer`).
  * (QKV attention, §"QKV attention".)  The score of `x_k` on `x_j` is
    `⟨Q(x_k + p_k), K(x_j + p_j)⟩/√n` with `n = dim H` (`qkvScore`).  A
    single-head QKV transformer is a single-head transformer whose attention
    functions `a_k` (`k = 2, …, ℓ`) are attention functions with these
    scoring functions.

Encoding.
  * Lengths are shifted by one: `a m` is the attention function of length
    `k = m + 1`, so the query is `x (Fin.last m)` and the head uses `a 1, …,
    a (ℓ-1)`.  Position vectors are `p : ℕ → C`; only `p 0, …, p (ℓ-1)` enter.
  * `Ĉ = C × ℝ`, `e = (0, 1)`, `ι x = (x, 1)`.
  * `ℝ^k` is `EuclideanSpace ℝ (Fin k)`, so that `∇φ` is mathlib's `gradient`.
    "Smooth" is `ContDiff ℝ ∞`.  A polynomial map on `C^k` is one whose
    components are polynomials in the coordinates with respect to a basis
    of `C` (`IsPolyFun`, using `Module.finBasis`; polynomiality does not
    depend on the basis).
  * The softmax potential of §"QKV attention" is not assumed: the statement
    is proved for QKV transformers with any potentials `φ_k`, which includes
    softmax.  The paper's standing assumption that `Q`, `K` are independent
    plays no role and is not assumed.
  * The neural network `M : C → C` is "piecewise smooth" in the paper, a
    notion it does not define; no regularity is assumed, and the conclusion
    records `M̂ = M ∘ pr_C` (and `V̂ = V ∘ pr_C`) followed by the inclusion
    `C → Ĉ`, so any regularity of `M` stable under composition with linear
    maps passes to `M̂`.

The proof follows the paper's.  The homogenized forms are
`B̂(u + αe, v + βe) = ⟨Q(v + βp_k), K(u + αp_j)⟩/√n`, whose expansion is the
paper's four-term formula; they agree with the QKV scores on the chart `C + e`,
so the weights agree there, and `Σ_j ω_j (x_j + e) = a_k(x) + e`.  The value map
and the neural network are extended by ignoring the new coordinate.  One point
the paper leaves implicit is checked: the homogenized weights `(∇φ) ∘ σ̂` are
simplex-valued on all of `Ĉ^k`, because `σ̂(u + αe, …) = σ(x)` with
`x_j = u_j + (α_j - 1)p_j` (`homScore_eq`); this also gives their smoothness.
-/
import Mathlib

namespace Appendices.MainText

open Module
open scoped ContDiff

/-! ### Definitions -/

section Defs

variable {C : Type*} [NormedAddCommGroup C] [NormedSpace ℝ C] [FiniteDimensional ℝ C]

/-- A polynomial function on `C^k`: a polynomial in the coordinates of the `x_i`
with respect to a basis of `C`. -/
def IsPolyFun {k : ℕ} (f : (Fin k → C) → ℝ) : Prop :=
  ∃ p : MvPolynomial (Fin k × Fin (finrank ℝ C)) ℝ,
    ∀ x, f x = MvPolynomial.eval (fun il => (Module.finBasis ℝ C).repr (x il.1) il.2) p

/-- **Definition (attention function).**  `w` is the paper's `ω` (the attention
weights). -/
structure IsAttentionFunction {k : ℕ} (a : (Fin k → C) → C)
    (w : (Fin k → C) → EuclideanSpace ℝ (Fin k)) (φ : EuclideanSpace ℝ (Fin k) → ℝ)
    (σ : (Fin k → C) → EuclideanSpace ℝ (Fin k)) : Prop where
  contDiff_a : ContDiff ℝ ∞ a
  contDiff_w : ContDiff ℝ ∞ w
  mem_stdSimplex : ∀ x, WithLp.ofLp (w x) ∈ stdSimplex ℝ (Fin k)
  contDiff_φ : ContDiff ℝ ∞ φ
  convexOn_φ : ConvexOn ℝ Set.univ φ
  poly_σ : ∀ j, IsPolyFun fun x => σ x j
  grad : ∀ x, w x = gradient φ (σ x)
  output : ∀ x, a x = ∑ j, w x j • x j

/-- **Definition (bilinear).**  The scoring function of an attention function of
length `m + 1` is `σ(x)_j = P_{m-j}(x_j, x_{m})` for bilinear forms `P_d`
(written here `B j = P_{m-j}`). -/
def IsBilinearScore {m : ℕ} (σ : (Fin (m + 1) → C) → EuclideanSpace ℝ (Fin (m + 1))) :
    Prop :=
  ∃ B : Fin (m + 1) → C →ₗ[ℝ] C →ₗ[ℝ] ℝ, ∀ x j, σ x j = B j (x j) (x (Fin.last m))

/-- **Definition (attention head).**  `A(x₁, …, x_ℓ) = (x₁, a₂(x₁, x₂), …)`;
here `a m` has length `m + 1`. -/
def head (ℓ : ℕ) (a : (m : ℕ) → (Fin (m + 1) → C) → C) (x : Fin ℓ → C) (i : Fin ℓ) : C :=
  if (i : ℕ) = 0 then x i else a i (fun j : Fin ((i : ℕ) + 1) => x (Fin.castLE i.2 j))

/-- **Definition (transformer), single head.**  `T = T_M ∘ T_A` with
`T_A(x)_i = x_i + V(A(x)_i)` and `T_M(y)_i = y_i + M(y_i)`. -/
def transformer {ℓ : ℕ} (A : (Fin ℓ → C) → Fin ℓ → C) (V : C →ₗ[ℝ] C) (M : C → C)
    (x : Fin ℓ → C) (i : Fin ℓ) : C :=
  (x i + V (A x i)) + M (x i + V (A x i))

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [FiniteDimensional ℝ H]

/-- The QKV scoring function of length `m + 1`:
`σ_j(x) = ⟨Q(x_k + p_k), K(x_j + p_j)⟩/√n`, `k = m + 1`, `n = dim H`. -/
noncomputable def qkvScore (Q K : C →ₗ[ℝ] H) (p : ℕ → C) (m : ℕ) (x : Fin (m + 1) → C) :
    EuclideanSpace ℝ (Fin (m + 1)) :=
  WithLp.toLp 2 fun j =>
    inner ℝ (Q (x (Fin.last m) + p m)) (K (x j + p j)) / Real.sqrt (finrank ℝ H)

end Defs

/-! ### Bilinear forms are polynomial -/

section Poly

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] [FiniteDimensional ℝ V]

/-- `y ↦ B(y_j, y_l)` is a polynomial function on `V^k` for a bilinear form `B`. -/
lemma isPolyFun_bilinear {k : ℕ} (B : V →ₗ[ℝ] V →ₗ[ℝ] ℝ) (j l : Fin k) :
    IsPolyFun fun y : Fin k → V => B (y j) (y l) := by
  set bb := Module.finBasis ℝ V
  refine ⟨∑ r, ∑ s, MvPolynomial.C (B (bb r) (bb s)) * MvPolynomial.X (j, r)
    * MvPolynomial.X (l, s), fun y => ?_⟩
  simp only [map_sum, map_mul, MvPolynomial.eval_C, MvPolynomial.eval_X]
  conv_lhs => rw [← bb.sum_repr (y j), ← bb.sum_repr (y l)]
  simp only [map_sum, map_smul, LinearMap.sum_apply, LinearMap.smul_apply, smul_eq_mul,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun s _ => ?_
  ring

end Poly

/-! ### The homogenization -/

section Homog

variable {C : Type*} [NormedAddCommGroup C] [NormedSpace ℝ C] [FiniteDimensional ℝ C]
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [FiniteDimensional ℝ H]

/-- `u + αe ↦ K(u + α p)`, a linear map `Ĉ → H`. -/
noncomputable def homMap (K : C →ₗ[ℝ] H) (q : C) : C × ℝ →ₗ[ℝ] H :=
  K.comp (LinearMap.fst ℝ C ℝ) + (LinearMap.snd ℝ C ℝ).smulRight (K q)

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
lemma homMap_apply (K : C →ₗ[ℝ] H) (q : C) (y : C × ℝ) :
    homMap K q y = K y.1 + y.2 • K q := rfl

/-- The homogenized bilinear form of the key at position `j` and the query at
position `k = m + 1`: `B̂(u + αe, v + βe) = ⟨Q(v + βp_k), K(u + αp_j)⟩/√n`. -/
noncomputable def homForm (Q K : C →ₗ[ℝ] H) (p : ℕ → C) (m j : ℕ) :
    C × ℝ →ₗ[ℝ] C × ℝ →ₗ[ℝ] ℝ :=
  (1 / Real.sqrt (finrank ℝ H)) •
    (innerₗ H).compl₁₂ (homMap K (p j)) (homMap Q (p m))

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
/-- The paper's four-term formula for `B̂`. -/
lemma homForm_apply (Q K : C →ₗ[ℝ] H) (p : ℕ → C) (m j : ℕ) (u v : C) (α β : ℝ) :
    homForm Q K p m j (u, α) (v, β)
      = (inner ℝ (Q v) (K u) + α * inner ℝ (Q v) (K (p j)) + β * inner ℝ (Q (p m)) (K u)
          + α * β * inner ℝ (Q (p m)) (K (p j))) / Real.sqrt (finrank ℝ H) := by
  simp only [homForm, LinearMap.smul_apply, LinearMap.compl₁₂_apply, innerₗ_apply_apply,
    homMap_apply, smul_eq_mul, inner_add_left, inner_add_right, real_inner_smul_left,
    real_inner_smul_right]
  rw [real_inner_comm (K u), real_inner_comm (K u), real_inner_comm (K (p j)),
    real_inner_comm (K (p j))]
  ring

/-- The homogenized (bilinear) scoring function of length `m + 1`. -/
noncomputable def homScore (Q K : C →ₗ[ℝ] H) (p : ℕ → C) (m : ℕ)
    (y : Fin (m + 1) → C × ℝ) : EuclideanSpace ℝ (Fin (m + 1)) :=
  WithLp.toLp 2 fun j => homForm Q K p m j (y j) (y (Fin.last m))

/-- `u + αe ↦ u + (α - 1)p`: the point of `C` whose QKV score equals the
homogenized score. -/
def dehom (p : ℕ → C) (m : ℕ) (y : Fin (m + 1) → C × ℝ) : Fin (m + 1) → C :=
  fun i => (y i).1 + ((y i).2 - 1) • p i

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
/-- `σ̂(u + αe, …) = σ(x)` with `x_j = u_j + (α_j - 1)p_j`. -/
lemma homScore_eq (Q K : C →ₗ[ℝ] H) (p : ℕ → C) (m : ℕ) (y : Fin (m + 1) → C × ℝ) :
    homScore Q K p m y = qkvScore Q K p m (dehom p m y) := by
  have hK : ∀ i : Fin (m + 1), K (dehom p m y i + p i) = homMap K (p i) (y i) := by
    intro i
    simp only [dehom, homMap_apply, map_add, map_smul]
    rw [sub_smul, one_smul]; abel
  have hQ : Q (dehom p m y (Fin.last m) + p m) = homMap Q (p m) (y (Fin.last m)) := by
    simp only [dehom, homMap_apply, map_add, map_smul, Fin.val_last]
    rw [sub_smul, one_smul]; abel
  ext j
  simp only [homScore, qkvScore, PiLp.toLp_apply, homForm, LinearMap.smul_apply,
    LinearMap.compl₁₂_apply, innerₗ_apply_apply, smul_eq_mul]
  rw [hK j, hQ, real_inner_comm]
  ring

omit [FiniteDimensional ℝ C] in
/-- On the chart `C + e`, `dehom` recovers the point. -/
lemma dehom_iota (p : ℕ → C) (m : ℕ) (x : Fin (m + 1) → C) :
    dehom p m (fun i => (x i, (1 : ℝ))) = x := by
  funext i; simp [dehom]

omit [FiniteDimensional ℝ C] [FiniteDimensional ℝ H] in
lemma contDiff_dehom (p : ℕ → C) (m : ℕ) : ContDiff ℝ ∞ (dehom p m) := by
  refine contDiff_pi.mpr fun i => ?_
  have h1 : ContDiff ℝ ∞ fun y : Fin (m + 1) → C × ℝ => y i := contDiff_apply ℝ (C × ℝ) i
  exact (contDiff_fst.comp h1).add (((contDiff_snd.comp h1).sub contDiff_const).smul
    contDiff_const)

omit [FiniteDimensional ℝ H] in
/-- **Proposition (bilinear attention recovers QKV attention).**  For every
single-head QKV transformer `T` on `C` (query and key maps `Q, K`, positions
`p`, value map `V`, neural network `M`, and attention functions `a_k` with QKV
scores and potentials `φ_k`), there is a single-head bilinear transformer `T̂`
on `Ĉ = C × ℝ` with the same potentials such that
`T̂(ιx₁, …, ιx_ℓ) = ι^ℓ(T(x₁, …, x_ℓ))`, `ι x = (x, 1)`. -/
theorem qkv_homogenizes_to_bilinear {ℓ : ℕ} (Q K : C →ₗ[ℝ] H) (p : ℕ → C)
    (V : C →ₗ[ℝ] C) (M : C → C)
    (a : (m : ℕ) → (Fin (m + 1) → C) → C)
    (w : (m : ℕ) → (Fin (m + 1) → C) → EuclideanSpace ℝ (Fin (m + 1)))
    (φ : (m : ℕ) → EuclideanSpace ℝ (Fin (m + 1)) → ℝ)
    (ha : ∀ m, 1 ≤ m → m < ℓ → IsAttentionFunction (a m) (w m) (φ m) (qkvScore Q K p m)) :
    ∃ (ah : (m : ℕ) → (Fin (m + 1) → C × ℝ) → C × ℝ)
      (wh : (m : ℕ) → (Fin (m + 1) → C × ℝ) → EuclideanSpace ℝ (Fin (m + 1)))
      (sh : (m : ℕ) → (Fin (m + 1) → C × ℝ) → EuclideanSpace ℝ (Fin (m + 1)))
      (Vh : C × ℝ →ₗ[ℝ] C × ℝ) (Mh : C × ℝ → C × ℝ),
      (∀ m, 1 ≤ m → m < ℓ →
        IsAttentionFunction (ah m) (wh m) (φ m) (sh m) ∧ IsBilinearScore (sh m)) ∧
      (∀ y, Vh y = (V y.1, 0)) ∧ (∀ y, Mh y = (M y.1, 0)) ∧
      ∀ x : Fin ℓ → C,
        transformer (head ℓ ah) Vh Mh (fun i => (x i, 1))
          = fun i => (transformer (head ℓ a) V M x i, 1) := by
  -- the homogenized weights and attention functions
  let wh : (m : ℕ) → (Fin (m + 1) → C × ℝ) → EuclideanSpace ℝ (Fin (m + 1)) :=
    fun m y => w m (dehom p m y)
  let ah : (m : ℕ) → (Fin (m + 1) → C × ℝ) → C × ℝ := fun m y => ∑ j, wh m y j • y j
  refine ⟨ah, wh, homScore Q K p, (LinearMap.inl ℝ C ℝ).comp (V.comp (LinearMap.fst ℝ C ℝ)),
    fun y => (M y.1, 0), fun m hm hmℓ => ⟨?_, ?_⟩, fun y => rfl, fun y => rfl, ?_⟩
  · obtain ⟨hca, hcw, hsimp, hcφ, hconv, -, hgrad, -⟩ := ha m hm hmℓ
    have hcwh : ContDiff ℝ ∞ (wh m) := hcw.comp (contDiff_dehom p m)
    refine ⟨?_, hcwh, fun y => hsimp _, hcφ, hconv,
      fun j => isPolyFun_bilinear (homForm Q K p m j) j (Fin.last m), fun y => ?_,
      fun y => rfl⟩
    · refine ContDiff.sum fun j _ => ?_
      exact (((EuclideanSpace.proj j : EuclideanSpace ℝ (Fin (m + 1)) →L[ℝ] ℝ).contDiff).comp
        hcwh).smul (contDiff_apply ℝ (C × ℝ) j)
    · -- `wh = (∇φ) ∘ sh`, since `sh(y) = σ(x)`
      show w m (dehom p m y) = gradient (φ m) (homScore Q K p m y)
      rw [hgrad, homScore_eq]
  · exact ⟨fun j => homForm Q K p m j, fun y j => rfl⟩
  · -- `T̂ ∘ ι^ℓ = ι^ℓ ∘ T`
    intro x
    have hhead : ∀ i : Fin ℓ, head ℓ ah (fun i => (x i, 1)) i = (head ℓ a x i, 1) := by
      intro i
      unfold head
      by_cases hi : (i : ℕ) = 0
      · simp [hi]
      · simp only [hi, if_false]
        obtain ⟨-, -, hsimp, -, -, -, -, hout⟩ := ha i (Nat.one_le_iff_ne_zero.mpr hi) i.2
        set xi : Fin ((i : ℕ) + 1) → C := fun j => x (Fin.castLE i.2 j)
        have hwh : wh i (fun j => (xi j, (1 : ℝ))) = w i xi := by
          show w i (dehom p i _) = _
          rw [dehom_iota]
        have hsum : ∑ j, w i xi j = 1 := (hsimp xi).2
        show ∑ j, wh i (fun j => (xi j, (1 : ℝ))) j • (xi j, (1 : ℝ)) = _
        rw [hwh, hout xi]
        refine Prod.ext ?_ ?_
        · simp [Prod.fst_sum]
        · simp [Prod.snd_sum, hsum]
    funext i
    simp only [transformer, hhead, LinearMap.comp_apply, LinearMap.fst_apply,
      LinearMap.inl_apply, Prod.mk_add_mk, add_zero]

end Homog

end Appendices.MainText
