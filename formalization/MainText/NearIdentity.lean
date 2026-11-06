/-
Main text, §"Transformers are unipotent": Proposition `prop:near-identity`.

    ‖T(x) - x‖_∞ ≤ (L + V + LV) ‖x‖_∞ .

`T` is the transformer associated with an attention block
`A = ∑ᵢ (Vᵢ ⊗ id) ∘ Aᵢ` and a neural network component `M`, with
`‖M y‖ ≤ L ‖y‖`, and `V = ∑ᵢ ‖Vᵢ‖` (operator norms).  The definitions
of attention functions, heads, attention blocks and transformers are
those of `MainText/Homogenization.lean`.

Encoding.  `C` is a finite-dimensional real normed space (the paper
fixes a norm on `C`).  For `x : Fin ℓ → C` the norm `‖x‖` is the `Pi`
sup-norm, which is the paper's `‖x‖_∞ = max_k ‖x_k‖`.  The value maps
are continuous linear maps, so that `‖Vᵢ‖` is the operator norm; on the
finite-dimensional space `C` every linear map is continuous.

The proof follows the paper's: an attention function returns a convex
combination of its inputs, so its norm is at most the largest input
norm (`norm_le_of_isAttentionFunction`); hence every component of a head
has norm at most `‖x‖_∞`, the first component being `x₁` itself
(`norm_head_le`); hence each component of the attention block has norm
at most `V ‖x‖_∞` (`norm_attentionBlock_le`).  The `k`-th component of
`T(x) - x` is `A(x)_k + M(x_k + A(x)_k)`, whose norm is at most
`V‖x‖_∞ + L(‖x‖_∞ + V‖x‖_∞)`.
-/
import MainText.Homogenization

namespace Appendices.MainText

open Finset
open scoped ContDiff

variable {C : Type*} [NormedAddCommGroup C] [NormedSpace ℝ C] [FiniteDimensional ℝ C]

/-- Since the attention weights lie in the simplex, an attention function
satisfies `‖a(x₁, …, x_k)‖ = ‖∑ⱼ ωⱼ xⱼ‖ ≤ ∑ⱼ ωⱼ ‖xⱼ‖ ≤ maxⱼ ‖xⱼ‖`. -/
lemma norm_le_of_isAttentionFunction {k : ℕ} {a : (Fin k → C) → C}
    {w : (Fin k → C) → EuclideanSpace ℝ (Fin k)} {φ : EuclideanSpace ℝ (Fin k) → ℝ}
    {σ : (Fin k → C) → EuclideanSpace ℝ (Fin k)} (ha : IsAttentionFunction a w φ σ)
    (x : Fin k → C) : ‖a x‖ ≤ ‖x‖ := by
  obtain ⟨hw0, hw1⟩ := ha.mem_stdSimplex x
  rw [ha.output x]
  calc ‖∑ j, w x j • x j‖ ≤ ∑ j, ‖w x j • x j‖ := norm_sum_le _ _
    _ = ∑ j, w x j * ‖x j‖ := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw0 j)]
    _ ≤ ∑ j, w x j * ‖x‖ :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (norm_le_pi_norm x j) (hw0 j)
    _ = ‖x‖ := by rw [← Finset.sum_mul]; simp [hw1]

/-- Every component of an attention head has norm at most `‖x‖_∞`; the first
component is `x₁` itself. -/
lemma norm_head_le {ℓ : ℕ} {a : (m : ℕ) → (Fin (m + 1) → C) → C}
    (ha : ∀ m, 1 ≤ m → m < ℓ → ∃ w φ σ, IsAttentionFunction (a m) w φ σ)
    (x : Fin ℓ → C) (k : Fin ℓ) : ‖head ℓ a x k‖ ≤ ‖x‖ := by
  unfold head
  split_ifs with hk
  · exact norm_le_pi_norm x k
  · obtain ⟨w, φ, σ, h⟩ := ha k (Nat.one_le_iff_ne_zero.mpr hk) k.2
    refine (norm_le_of_isAttentionFunction h _).trans ?_
    exact (pi_norm_le_iff_of_nonneg (norm_nonneg x)).mpr fun j => norm_le_pi_norm x _

/-- Each component of the attention block has norm at most `V ‖x‖_∞`, where
`V = ∑ᵢ ‖Vᵢ‖`. -/
lemma norm_attentionBlock_le {ℓ h : ℕ} {a : Fin h → (m : ℕ) → (Fin (m + 1) → C) → C}
    (ha : ∀ i m, 1 ≤ m → m < ℓ → ∃ w φ σ, IsAttentionFunction (a i m) w φ σ)
    (V : Fin h → (C →L[ℝ] C)) (x : Fin ℓ → C) (k : Fin ℓ) :
    ‖attentionBlock (fun i => head ℓ (a i)) (fun i => (V i : C →ₗ[ℝ] C)) x k‖
      ≤ (∑ i, ‖V i‖) * ‖x‖ := by
  calc ‖∑ i, V i (head ℓ (a i) x k)‖ ≤ ∑ i, ‖V i (head ℓ (a i) x k)‖ := norm_sum_le _ _
    _ ≤ ∑ i, ‖V i‖ * ‖x‖ := by
        refine Finset.sum_le_sum fun i _ => ?_
        exact ((V i).le_opNorm _).trans
          (mul_le_mul_of_nonneg_left (norm_head_le (ha i) x k) (norm_nonneg _))
    _ = (∑ i, ‖V i‖) * ‖x‖ := by rw [← Finset.sum_mul]

/-- **Proposition (transformers are near-identity).**  Let `T` be the
transformer associated to the attention block `A = ∑ᵢ (Vᵢ ⊗ id) ∘ Aᵢ` (the
`Aᵢ` attention heads, built from attention functions `a i`) and a neural
network component `M` with `‖M y‖ ≤ L ‖y‖`.  With `V = ∑ᵢ ‖Vᵢ‖`,

    ‖T x - x‖_∞ ≤ (L + V + L·V) ‖x‖_∞ . -/
theorem near_identity {ℓ h : ℕ} (a : Fin h → (m : ℕ) → (Fin (m + 1) → C) → C)
    (ha : ∀ i m, 1 ≤ m → m < ℓ → ∃ w φ σ, IsAttentionFunction (a i m) w φ σ)
    (V : Fin h → (C →L[ℝ] C)) (M : C → C) (L : ℝ) (hM : ∀ y, ‖M y‖ ≤ L * ‖y‖)
    (x : Fin ℓ → C) :
    ‖transformer (attentionBlock (fun i => head ℓ (a i)) (fun i => (V i : C →ₗ[ℝ] C))) M x - x‖
      ≤ (L + (∑ i, ‖V i‖) + L * (∑ i, ‖V i‖)) * ‖x‖ := by
  set A := attentionBlock (fun i => head ℓ (a i)) (fun i => (V i : C →ₗ[ℝ] C)) with hA
  set Vsum := ∑ i, ‖V i‖ with hV
  have hVnn : 0 ≤ Vsum := Finset.sum_nonneg fun i _ => norm_nonneg _
  -- If `C = 0` both sides vanish; otherwise `C` has a nonzero vector.
  rcases subsingleton_or_nontrivial C with hC | hC
  · rw [Subsingleton.elim (transformer A M x - x) 0, norm_zero, Subsingleton.elim x 0,
      norm_zero, mul_zero]
  -- `L` is nonnegative: `‖M y‖ ≤ L‖y‖` at any nonzero `y` forces it.
  have hLnn : 0 ≤ L := by
    obtain ⟨y, hy⟩ := exists_ne (0 : C)
    have hypos : 0 < ‖y‖ := norm_pos_iff.mpr hy
    nlinarith [hM y, norm_nonneg (M y)]
  refine (pi_norm_le_iff_of_nonneg (mul_nonneg (by positivity) (norm_nonneg x))).mpr
    fun k => ?_
  have hAk : ‖A x k‖ ≤ Vsum * ‖x‖ := norm_attentionBlock_le ha V x k
  -- the `k`-th component of `T(x) - x` is `A(x)_k + M(x_k + A(x)_k)`
  have hcomp : (transformer A M x - x) k = A x k + M (x k + A x k) := by
    rw [Pi.sub_apply, transformer_apply]
    abel
  have hinner : ‖x k + A x k‖ ≤ (1 + Vsum) * ‖x‖ := by
    calc ‖x k + A x k‖ ≤ ‖x k‖ + ‖A x k‖ := norm_add_le _ _
      _ ≤ ‖x‖ + Vsum * ‖x‖ := add_le_add (norm_le_pi_norm x k) hAk
      _ = (1 + Vsum) * ‖x‖ := by ring
  have hMbound : ‖M (x k + A x k)‖ ≤ L * ((1 + Vsum) * ‖x‖) :=
    (hM _).trans (mul_le_mul_of_nonneg_left hinner hLnn)
  rw [hcomp]
  calc ‖A x k + M (x k + A x k)‖ ≤ ‖A x k‖ + ‖M (x k + A x k)‖ := norm_add_le _ _
    _ ≤ Vsum * ‖x‖ + L * ((1 + Vsum) * ‖x‖) := add_le_add hAk hMbound
    _ = (L + Vsum + L * Vsum) * ‖x‖ := by ring

end Appendices.MainText
