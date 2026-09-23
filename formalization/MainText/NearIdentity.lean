/-
Main text, §"Transformers are unipotent": Proposition `prop:near-identity`.

    ‖T(x) - x‖_∞ ≤ (L + V + LV) ‖x‖_∞ .

The paper states this for the transformer `T = T_M ∘ T_A` associated with
an attention block `A = ∑ᵢ (Vᵢ ⊗ id) ∘ Aᵢ` and a neural network component
`M`.  Unwinding the paper's definitions, the proof uses exactly three
properties, and no others:

  * each attention head `Aᵢ` returns, in each component, a *convex
    combination* of the inputs `x₁, …, x_ℓ` --- this is all that the
    paper's `ω : C^k → Δ_{k-1}` contributes to the estimate;
  * `M` is `L`-bounded: `‖M y‖ ≤ L ‖y‖`;
  * the value maps `Vᵢ` are linear, with operator norm `‖Vᵢ‖`.

Smoothness of the attention function, convexity of the potential `φ`, the
scoring map `σ` and the relation `ω = (∇φ) ∘ σ` play no role in the
bound.  We therefore take the convex-combination property as the
hypothesis `IsAttentionOutput` below, rather than reconstructing the
paper's definitional tower.  Any attention head in the paper's sense
satisfies it, so the proposition proved here implies the paper's.

`C` is any real normed space; the paper fixes a norm on `C` and uses
`‖x‖_∞ = max_k ‖x_k‖`, which for `Fin ℓ → C` is the `Pi` sup-norm.
-/
import Mathlib

namespace Appendices.MainText

open Finset BigOperators

variable {C : Type*} [NormedAddCommGroup C] [NormedSpace ℝ C]
variable {ℓ : ℕ}

/-- `y` is a convex combination of the entries of `x`: the defining property of
the output of an attention function, whose weights lie in the simplex `Δ_{ℓ-1}`. -/
def IsConvexCombOf (x : Fin ℓ → C) (y : C) : Prop :=
  ∃ w : Fin ℓ → ℝ, (∀ j, 0 ≤ w j) ∧ ∑ j, w j = 1 ∧ y = ∑ j, w j • x j

/-- An attention head, for the purposes of the estimate: every component of the
output is a convex combination of the inputs. -/
def IsAttentionOutput (a : (Fin ℓ → C) → Fin ℓ → C) : Prop :=
  ∀ x k, IsConvexCombOf x (a x k)

/-- The sup-norm `‖x‖_∞ = max_k ‖x_k‖` used in the paper. -/
noncomputable def supNorm (x : Fin ℓ → C) : ℝ := ‖x‖

omit [NormedSpace ℝ C] in
lemma norm_le_supNorm (x : Fin ℓ → C) (k : Fin ℓ) : ‖x k‖ ≤ supNorm x :=
  norm_le_pi_norm x k

omit [NormedSpace ℝ C] in
lemma supNorm_nonneg (x : Fin ℓ → C) : 0 ≤ supNorm x := norm_nonneg x

/-- A convex combination of the entries of `x` has norm at most `‖x‖_∞`.
This is the paper's `‖∑ ωⱼ xⱼ‖ ≤ ∑ ωⱼ ‖xⱼ‖ ≤ max_j ‖xⱼ‖`. -/
lemma norm_le_of_isConvexCombOf {x : Fin ℓ → C} {y : C} (h : IsConvexCombOf x y) :
    ‖y‖ ≤ supNorm x := by
  obtain ⟨w, hw, hsum, rfl⟩ := h
  calc ‖∑ j, w j • x j‖ ≤ ∑ j, ‖w j • x j‖ := norm_sum_le _ _
    _ = ∑ j, w j * ‖x j‖ := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw j)]
    _ ≤ ∑ j, w j * supNorm x := by
        refine Finset.sum_le_sum fun j _ => ?_
        exact mul_le_mul_of_nonneg_left (norm_le_supNorm x j) (hw j)
    _ = supNorm x := by rw [← Finset.sum_mul, hsum, one_mul]

variable {h : ℕ}

/-- The attention block `A = ∑ᵢ (Vᵢ ⊗ id) ∘ Aᵢ`, evaluated at `x` in component `k`. -/
noncomputable def attentionBlock (V : Fin h → (C →L[ℝ] C))
    (A : Fin h → (Fin ℓ → C) → Fin ℓ → C) (x : Fin ℓ → C) (k : Fin ℓ) : C :=
  ∑ i, V i (A i x k)

/-- Each component of the attention block is bounded by `V ‖x‖_∞`, where
`V = ∑ᵢ ‖Vᵢ‖`.  This is the paper's second display. -/
lemma norm_attentionBlock_le (V : Fin h → (C →L[ℝ] C))
    (A : Fin h → (Fin ℓ → C) → Fin ℓ → C) (hA : ∀ i, IsAttentionOutput (A i))
    (x : Fin ℓ → C) (k : Fin ℓ) :
    ‖attentionBlock V A x k‖ ≤ (∑ i, ‖V i‖) * supNorm x := by
  calc ‖∑ i, V i (A i x k)‖ ≤ ∑ i, ‖V i (A i x k)‖ := norm_sum_le _ _
    _ ≤ ∑ i, ‖V i‖ * supNorm x := by
        refine Finset.sum_le_sum fun i _ => ?_
        calc ‖V i (A i x k)‖ ≤ ‖V i‖ * ‖A i x k‖ := (V i).le_opNorm _
          _ ≤ ‖V i‖ * supNorm x :=
              mul_le_mul_of_nonneg_left (norm_le_of_isConvexCombOf (hA i x k)) (norm_nonneg _)
    _ = (∑ i, ‖V i‖) * supNorm x := by rw [← Finset.sum_mul]

/-- **Proposition (transformers are near-identity).**
`T = T_M ∘ T_A` with `T_A = id + A` and `T_M = id + M ⊗ id`; the `k`-th
component of `T x - x` is `A(x)_k + M(x_k + A(x)_k)`.  With `‖M y‖ ≤ L ‖y‖`
and `V = ∑ᵢ ‖Vᵢ‖`,

    ‖T x - x‖_∞ ≤ (L + V + L·V) ‖x‖_∞ . -/
theorem near_identity [Nontrivial C] (V : Fin h → (C →L[ℝ] C))
    (A : Fin h → (Fin ℓ → C) → Fin ℓ → C) (hA : ∀ i, IsAttentionOutput (A i))
    (M : C → C) (L : ℝ) (hM : ∀ y, ‖M y‖ ≤ L * ‖y‖)
    (x : Fin ℓ → C) (k : Fin ℓ) :
    ‖(attentionBlock V A x k + M (x k + attentionBlock V A x k))‖
      ≤ (L + (∑ i, ‖V i‖) + L * (∑ i, ‖V i‖)) * supNorm x := by
  set Vsum := ∑ i, ‖V i‖ with hV
  have hVnn : 0 ≤ Vsum := Finset.sum_nonneg fun i _ => norm_nonneg _
  have hAk : ‖attentionBlock V A x k‖ ≤ Vsum * supNorm x :=
    norm_attentionBlock_le V A hA x k
  -- `L` is nonnegative: `‖M y‖ ≤ L‖y‖` at any nonzero `y` forces it.
  have hLnn : 0 ≤ L := by
    obtain ⟨y, hy⟩ := exists_ne (0 : C)
    have hypos : 0 < ‖y‖ := norm_pos_iff.mpr hy
    nlinarith [hM y, norm_nonneg (M y)]
  -- bound the inner argument `x k + A(x)_k`
  have hinner : ‖x k + attentionBlock V A x k‖ ≤ (1 + Vsum) * supNorm x := by
    calc ‖x k + attentionBlock V A x k‖ ≤ ‖x k‖ + ‖attentionBlock V A x k‖ := norm_add_le _ _
      _ ≤ supNorm x + Vsum * supNorm x := add_le_add (norm_le_supNorm x k) hAk
      _ = (1 + Vsum) * supNorm x := by ring
  have hMbound : ‖M (x k + attentionBlock V A x k)‖ ≤ L * ((1 + Vsum) * supNorm x) :=
    le_trans (hM _) (mul_le_mul_of_nonneg_left hinner hLnn)
  calc ‖attentionBlock V A x k + M (x k + attentionBlock V A x k)‖
      ≤ ‖attentionBlock V A x k‖ + ‖M (x k + attentionBlock V A x k)‖ := norm_add_le _ _
    _ ≤ Vsum * supNorm x + L * ((1 + Vsum) * supNorm x) := add_le_add hAk hMbound
    _ = (L + Vsum + L * Vsum) * supNorm x := by ring

end Appendices.MainText
