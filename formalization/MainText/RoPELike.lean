/-
Main text, §"RoPE-like attention functions": the definition of RoPE-like
bilinear attention functions and the proposition classifying them.

    Let `a : H^k → H` be a bilinear attention function with scoring function
    `σ(u₁, …, u_k) = (B_{k-1}(u₁, u_k), …, B_0(u_k, u_k))` for bilinear forms
    `B_{k-1}, …, B_0` on `H × H`.  Each `B_j` determines `L_j : H → H^∨`,
    `L_j(v)(u) = B_j(u, v)`.  Assume `B_0` is nondegenerate, and define
    `T_d = L_0⁻¹ L_d ∈ End(H)`.

    Definition.  If `T_{d+e} = T_d T_e` for all integers `d, e ≥ 0` with
    `d + e < k`, then `a` is called RoPE-like.

    Proposition.  A bilinear attention function `a` as above with
    nondegenerate `B_0` is RoPE-like if and only if there exists a linear map
    `S ∈ End(H)` such that `T_d = S^d` for all `0 ≤ d < k`.

    Note: when `S` exists it is unique and given by `S = T_1 = L_0⁻¹ L_1`.

Encoding.  Being RoPE-like is a condition on the forms `B_0, …, B_{k-1}` alone:
the attention weights, the potential and the rest of the attention function
do not enter the definition, the proposition or its proof.  So both are stated
for the family of forms.  The family is indexed by `ℕ`; only the forms `B_d`
with `d < k` enter any statement.  The inner product on `H` plays no role and
is not assumed: `H` is a finite-dimensional real vector space.  Mathlib's
`BilinForm.Nondegenerate` (`B(m, ·) = 0 ⇒ m = 0`) is equivalent in finite
dimension to nondegeneracy in the usual sense.

`L_0` is inverted as the linear equivalence `B_0.flip.toDual : H ≃ H^∨`, whose
underlying map is `L_0`.

The proof follows the paper's: RoPE-like gives `L_{d+e} = L_d L_0⁻¹ L_e`, in
particular `L_{d+1} = L_d S` with `S = T_1`; by induction `L_d = L_0 S^d`, hence
`T_d = S^d`.  The converse is `S^{d+e} = S^d S^e`.
-/
import Mathlib

namespace Appendices.MainText

open LinearMap

variable {H : Type*} [AddCommGroup H] [Module ℝ H] [FiniteDimensional ℝ H]

/-- `L_j : H → H^∨`, `L_j(v)(u) = B_j(u, v)`. -/
def ropeL (B : ℕ → LinearMap.BilinForm ℝ H) (j : ℕ) : H →ₗ[ℝ] Module.Dual ℝ H :=
  (B j).flip

/-- `L_0` as a linear equivalence `H ≃ H^∨`, for nondegenerate `B_0`. -/
noncomputable def ropeL0Equiv (B : ℕ → LinearMap.BilinForm ℝ H) (h0 : (B 0).Nondegenerate) :
    H ≃ₗ[ℝ] Module.Dual ℝ H :=
  (B 0).flip.toDual h0.flip

lemma ropeL0Equiv_coe (B : ℕ → LinearMap.BilinForm ℝ H) (h0 : (B 0).Nondegenerate) :
    (ropeL0Equiv B h0 : H →ₗ[ℝ] Module.Dual ℝ H) = ropeL B 0 := rfl

/-- `T_d = L_0⁻¹ L_d ∈ End(H)`. -/
noncomputable def ropeT (B : ℕ → LinearMap.BilinForm ℝ H) (h0 : (B 0).Nondegenerate) (d : ℕ) :
    Module.End ℝ H :=
  (ropeL0Equiv B h0).symm.toLinearMap ∘ₗ ropeL B d

/-- **Definition (RoPE-like).**  `T_{d+e} = T_d T_e` for all `d, e ≥ 0` with
`d + e < k`. -/
def IsRoPELike (k : ℕ) (B : ℕ → LinearMap.BilinForm ℝ H) (h0 : (B 0).Nondegenerate) : Prop :=
  ∀ d e : ℕ, d + e < k → ropeT B h0 (d + e) = ropeT B h0 d * ropeT B h0 e

/-- `L_0 T_d = L_d`. -/
lemma ropeL0_comp_ropeT (B : ℕ → LinearMap.BilinForm ℝ H) (h0 : (B 0).Nondegenerate) (d : ℕ) :
    ropeL B 0 ∘ₗ ropeT B h0 d = ropeL B d := by
  rw [← ropeL0Equiv_coe B h0, ropeT, ← LinearMap.comp_assoc]
  ext v
  simp

/-- **Proposition (classification of RoPE-like attention functions).**  With
`B_0` nondegenerate, the forms are RoPE-like if and only if there is
`S ∈ End(H)` with `T_d = S^d` for all `0 ≤ d < k`. -/
theorem isRoPELike_iff (k : ℕ) (B : ℕ → LinearMap.BilinForm ℝ H) (h0 : (B 0).Nondegenerate) :
    IsRoPELike k B h0 ↔ ∃ S : Module.End ℝ H, ∀ d < k, ropeT B h0 d = S ^ d := by
  set L0 := ropeL0Equiv B h0
  constructor
  · intro hrope
    -- `L_{d+e} = L_d L_0⁻¹ L_e` for `d + e < k`
    have hL : ∀ d e : ℕ, d + e < k →
        ropeL B (d + e) = ropeL B d ∘ₗ L0.symm.toLinearMap ∘ₗ ropeL B e := by
      intro d e hde
      rw [← ropeL0_comp_ropeT B h0 (d + e), hrope d e hde, Module.End.mul_eq_comp,
        ← LinearMap.comp_assoc, ropeL0_comp_ropeT B h0 d]
      rfl
    refine ⟨ropeT B h0 1, ?_⟩
    -- `L_{d+1} = L_d S`, so by induction `L_d = L_0 S^d`
    have hLd : ∀ d < k, ropeL B d = ropeL B 0 ∘ₗ ropeT B h0 1 ^ d := by
      intro d
      induction d with
      | zero => intro _; rw [pow_zero, Module.End.one_eq_id, LinearMap.comp_id]
      | succ d ih =>
        intro hd
        rw [hL d 1 hd, ih (by omega), pow_succ, Module.End.mul_eq_comp,
          LinearMap.comp_assoc]
        rfl
    -- hence `T_d = L_0⁻¹ L_d = S^d`
    intro d hd
    rw [ropeT, hLd d hd, ← ropeL0Equiv_coe B h0, ← LinearMap.comp_assoc]
    ext v
    simp
  · rintro ⟨S, hS⟩ d e hde
    rw [hS (d + e) hde, hS d (by omega), hS e (by omega), pow_add]

/-- **Note.**  When `S` exists it is unique and given by `S = T_1`
(for `k ≥ 2`, so that `T_1` is among the `T_d`, `d < k`). -/
theorem eq_ropeT_one_of_forall (k : ℕ) (hk : 2 ≤ k) (B : ℕ → LinearMap.BilinForm ℝ H)
    (h0 : (B 0).Nondegenerate) {S : Module.End ℝ H}
    (hS : ∀ d < k, ropeT B h0 d = S ^ d) : S = ropeT B h0 1 := by
  rw [hS 1 (by omega), pow_one]

end Appendices.MainText
