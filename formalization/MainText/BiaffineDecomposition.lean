/-
Main text, §"Biaffine attention": Lemma `lem:biaffine-decomposition`.

    Every polynomial `P : C × C → ℝ` of bidegree at most `(1,1)` admits a
    unique decomposition

        P(u, v) = B(u, v) + L(u) + R(v) + D,

    where `B` is bilinear, `L` and `R` are linear, and `D ∈ ℝ`.  The
    components are given by `D = P(0,0)`, `L(u) = P(u,0) - P(0,0)`,
    `R(v) = P(0,v) - P(0,0)` and `B(u,v) = P(u,v) - P(u,0) - P(0,v) + P(0,0)`.

Encoding.  `C` is a finite-dimensional real vector space with a basis `b`
indexed by `ι`.  A polynomial function on `C × C` is the evaluation of a
polynomial `p ∈ ℝ[x_i, y_i : i ∈ ι]` at the coordinates `x = b.repr u`,
`y = b.repr v`; it has bidegree at most `(1,1)` when every monomial of `p` has
degree at most `1` in the `x`-variables and at most `1` in the `y`-variables
(`IsPolyBidegreeLeOneOne`).  Uniqueness is stated with `∃!` over the quadruple
`(B, L, R, D)`, and the formulas for the components are stated for every
decomposition.

The paper gives no proof ("elementary").  The proof here: each monomial of
bidegree at most `(1,1)` is a product `f(u) g(v)` of two factors, each either
the constant `1` or a coordinate function, so `P(u,v) = Σ_m c_m f_m(u) g_m(v)`
with every `f_m`, `g_m` affine.  Expanding shows that the four expressions in
the formulas are bilinear, linear, linear and constant, which gives existence;
evaluating a decomposition at `(0,0)`, `(u,0)` and `(0,v)` gives the formulas,
and hence uniqueness.
-/
import Mathlib

namespace Appendices.MainText

open MvPolynomial

variable {ι C : Type*} [Fintype ι] [DecidableEq ι]
variable [AddCommGroup C] [Module ℝ C]

/-- `P : C × C → ℝ` is a polynomial of bidegree at most `(1,1)`: it is the
evaluation at the coordinates of `u` and `v` of a polynomial every monomial of
which has degree at most `1` in the coordinates of `u` and at most `1` in the
coordinates of `v`. -/
def IsPolyBidegreeLeOneOne (b : Module.Basis ι ℝ C) (P : C × C → ℝ) : Prop :=
  ∃ p : MvPolynomial (ι ⊕ ι) ℝ,
    (∀ m ∈ p.support, (∑ i, m (Sum.inl i)) ≤ 1 ∧ (∑ i, m (Sum.inr i)) ≤ 1) ∧
    ∀ u v, P (u, v) = eval (Sum.elim (b.repr u) (b.repr v)) p

/-- A monomial of degree at most one is affine: `∏ᵢ yᵢ^{eᵢ} = α + λ(y)`. -/
lemma prod_pow_eq_affine (e : ι → ℕ) (he : ∑ i, e i ≤ 1) :
    ∃ (α : ℝ) (lam : (ι → ℝ) →ₗ[ℝ] ℝ), ∀ y : ι → ℝ, ∏ i, y i ^ e i = α + lam y := by
  by_cases h0 : ∀ i, e i = 0
  · exact ⟨1, 0, fun y => by simp [h0]⟩
  · push_neg at h0
    obtain ⟨i₀, hi₀⟩ := h0
    have hsingle : e i₀ ≤ ∑ i, e i :=
      Finset.single_le_sum (fun i _ => Nat.zero_le (e i)) (Finset.mem_univ i₀)
    have hrest : ∀ j ≠ i₀, e j = 0 := by
      intro j hj
      have h2 : e i₀ + e j ≤ ∑ i, e i := by
        rw [← Finset.sum_pair hj.symm]
        exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
          (fun i _ _ => Nat.zero_le (e i))
      omega
    have hi₀1 : e i₀ = 1 := by omega
    refine ⟨0, LinearMap.proj i₀, fun y => ?_⟩
    rw [Finset.prod_eq_single i₀ (fun j _ hj => by rw [hrest j hj, pow_zero])
      (by simp), hi₀1, pow_one]
    simp

/-- The evaluation of a polynomial of bidegree at most `(1,1)` as
`Σ_m c_m (α_m + λ_m u)(β_m + μ_m v)` with `λ_m`, `μ_m` linear. -/
lemma exists_affine_expansion (b : Module.Basis ι ℝ C) (P : C × C → ℝ)
    (hP : IsPolyBidegreeLeOneOne b P) :
    ∃ (s : Finset ((ι ⊕ ι) →₀ ℕ)) (c α β : ((ι ⊕ ι) →₀ ℕ) → ℝ)
      (lam mu : ((ι ⊕ ι) →₀ ℕ) → C →ₗ[ℝ] ℝ),
      ∀ u v, P (u, v) = ∑ m ∈ s, c m * ((α m + lam m u) * (β m + mu m v)) := by
  obtain ⟨p, hdeg, hP⟩ := hP
  have hex : ∀ m : (ι ⊕ ι) →₀ ℕ, ∃ (a : ℝ) (l : (ι → ℝ) →ₗ[ℝ] ℝ) (a' : ℝ)
      (l' : (ι → ℝ) →ₗ[ℝ] ℝ), m ∈ p.support →
        (∀ y, ∏ i, y i ^ m (Sum.inl i) = a + l y) ∧
        (∀ y, ∏ i, y i ^ m (Sum.inr i) = a' + l' y) := by
    intro m
    by_cases hm : m ∈ p.support
    · obtain ⟨a, l, hl⟩ := prod_pow_eq_affine _ (hdeg m hm).1
      obtain ⟨a', l', hl'⟩ := prod_pow_eq_affine _ (hdeg m hm).2
      exact ⟨a, l, a', l', fun _ => ⟨hl, hl'⟩⟩
    · exact ⟨0, 0, 0, 0, fun h => absurd h hm⟩
  choose a l a' l' hal using hex
  refine ⟨p.support, fun m => coeff m p, a, a',
    fun m => l m ∘ₗ Finsupp.lcoeFun ∘ₗ b.repr.toLinearMap,
    fun m => l' m ∘ₗ Finsupp.lcoeFun ∘ₗ b.repr.toLinearMap, ?_⟩
  intro u v
  rw [hP, eval_eq']
  refine Finset.sum_congr rfl fun m hm => ?_
  rw [Fintype.prod_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr]
  rw [(hal m hm).1 (fun i => b.repr u i), (hal m hm).2 (fun i => b.repr v i)]
  rfl

/-- The formulas for the components, valid for every decomposition
`P(u,v) = B(u,v) + L(u) + R(v) + D`: evaluate at `(0,0)`, `(u,0)`, `(0,v)`. -/
theorem biaffine_components (P : C × C → ℝ) (B : C →ₗ[ℝ] C →ₗ[ℝ] ℝ)
    (L R : C →ₗ[ℝ] ℝ) (D : ℝ) (h : ∀ u v, P (u, v) = B u v + L u + R v + D) :
    D = P (0, 0) ∧ (∀ u, L u = P (u, 0) - P (0, 0)) ∧
      (∀ v, R v = P (0, v) - P (0, 0)) ∧
      ∀ u v, B u v = P (u, v) - P (u, 0) - P (0, v) + P (0, 0) := by
  refine ⟨?_, fun u => ?_, fun v => ?_, fun u v => ?_⟩ <;>
    simp only [h, map_zero, LinearMap.zero_apply] <;> ring

/-- **Lemma (biaffine decomposition).**  A polynomial `P : C × C → ℝ` of
bidegree at most `(1,1)` has a unique decomposition
`P(u,v) = B(u,v) + L(u) + R(v) + D` with `B` bilinear, `L`, `R` linear and
`D ∈ ℝ`; the components are `D = P(0,0)`, `L(u) = P(u,0) - P(0,0)`,
`R(v) = P(0,v) - P(0,0)`, `B(u,v) = P(u,v) - P(u,0) - P(0,v) + P(0,0)`. -/
theorem biaffine_decomposition (b : Module.Basis ι ℝ C) (P : C × C → ℝ)
    (hP : IsPolyBidegreeLeOneOne b P) :
    (∃! BLRD : (C →ₗ[ℝ] C →ₗ[ℝ] ℝ) × (C →ₗ[ℝ] ℝ) × (C →ₗ[ℝ] ℝ) × ℝ,
        ∀ u v, P (u, v) = BLRD.1 u v + BLRD.2.1 u + BLRD.2.2.1 v + BLRD.2.2.2) ∧
      ∀ (B : C →ₗ[ℝ] C →ₗ[ℝ] ℝ) (L R : C →ₗ[ℝ] ℝ) (D : ℝ),
        (∀ u v, P (u, v) = B u v + L u + R v + D) →
        D = P (0, 0) ∧ (∀ u, L u = P (u, 0) - P (0, 0)) ∧
          (∀ v, R v = P (0, v) - P (0, 0)) ∧
          ∀ u v, B u v = P (u, v) - P (u, 0) - P (0, v) + P (0, 0) := by
  refine ⟨?_, biaffine_components P⟩
  obtain ⟨s, c, α, β, lam, mu, hexp⟩ := exists_affine_expansion b P hP
  -- existence: expand `(α + λu)(β + μv) = λu μv + β λu + α μv + αβ`
  have hexist : ∀ u v, P (u, v) = (∑ m ∈ s, c m • (lam m).smulRight (mu m)) u v
      + (∑ m ∈ s, (c m * β m) • lam m) u + (∑ m ∈ s, (c m * α m) • mu m) v
      + ∑ m ∈ s, c m * α m * β m := by
    intro u v
    simp only [hexp, LinearMap.coe_sum, Finset.sum_apply, LinearMap.smul_apply,
      LinearMap.smulRight_apply, smul_eq_mul, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun m _ => by ring
  refine ⟨(∑ m ∈ s, c m • (lam m).smulRight (mu m),
      ∑ m ∈ s, (c m * β m) • lam m, ∑ m ∈ s, (c m * α m) • mu m,
      ∑ m ∈ s, c m * α m * β m), hexist, ?_⟩
  -- uniqueness: the components of any decomposition are given by the formulas
  rintro ⟨B', L', R', D'⟩ h'
  obtain ⟨hD, hL, hR, hB⟩ := biaffine_components P B' L' R' D' h'
  obtain ⟨hD₀, hL₀, hR₀, hB₀⟩ := biaffine_components P _ _ _ _ hexist
  simp only [Prod.mk.injEq]
  refine ⟨LinearMap.ext₂ fun u v => ?_, LinearMap.ext fun u => ?_,
    LinearMap.ext fun v => ?_, ?_⟩
  · rw [hB, hB₀]
  · rw [hL, hL₀]
  · rw [hR, hR₀]
  · rw [hD, hD₀]

end Appendices.MainText
