/-
Appendix "Moments", Lemma (OE-equivalences), `lem:OE-equivalences`.

Every `λ ∈ Λ_m` satisfies `|λ_i| < 1` for all `i`.  Define

    O(λ) = Σ_i λ_i/(1-λ_i²),  E(λ) = Σ_i λ_i²/(1-λ_i²),
    R(λ) = Σ_i 1/(1-λ_i).

Then:
1. `O(λ) = Σ_{k ≥ 1 odd} Σ_i λ_i^k` and `E(λ) = Σ_{k ≥ 2 even} Σ_i λ_i^k`,
   both series converging absolutely;
2. `O = ½(R - R∘ι)` and `n + E = ½(R + R∘ι)`; in particular
   `O∘ι = -O` and `E∘ι = E`;
3. if `S` is a symmetric matrix whose spectrum, with multiplicity, is
   the entries of `λ`, then `O(λ) = tr(S(I-S²)⁻¹)`,
   `E(λ) = tr(S²(I-S²)⁻¹)`, `R(λ) = tr((I-S)⁻¹)`.

Encodings:
* odd exponents `k ≥ 1` are enumerated as `2k+1`, even exponents
  `k ≥ 2` as `2(k+1)`, for `k : ℕ`;
* "spectrum, with multiplicity, is the entries of `λ`" is encoded as
  `hS.eigenvalues ∘ σ = x` for a permutation `σ` of the indices, where
  `hS.eigenvalues` is the eigenvalue listing of the spectral theorem.
-/
import Appendices.Moments.Defs
import Appendices.BalancedBimodalSpectra.Defs

namespace Appendices

open BigOperators Matrix

variable {m : ℕ} {x : Fin (2 * m) → ℝ}

/-- "Every `λ ∈ Λ_m` satisfies `|λ_i| < 1` for all `i`": if `|λ_i| = 1`
then, since `‖λ‖ = 1`, all other entries vanish, contradicting the
presence of both a negative and a positive entry. -/
theorem MemLambdaSet.abs_entry_lt_one (h : MemLambdaSet m x) (i : Fin (2 * m)) :
    |x i| < 1 := by
  have hm := h.m_pos
  have hneg : x ⟨m - 1, by omega⟩ < 0 := h.middle_neg hm
  have hpos : (0 : ℝ) < x ⟨m, by omega⟩ := h.middle_pos hm
  by_contra hcon
  push_neg at hcon
  have hsq : 1 ≤ x i ^ 2 := by nlinarith [sq_abs (x i), abs_nonneg (x i)]
  obtain ⟨j, hji, hj0⟩ : ∃ j, j ≠ i ∧ x j ≠ 0 := by
    by_cases hin : (⟨m - 1, by omega⟩ : Fin (2 * m)) = i
    · refine ⟨⟨m, by omega⟩, fun hpi => ?_, ne_of_gt hpos⟩
      rw [← hpi] at hin
      have := congrArg Fin.val hin
      simp only at this
      omega
    · exact ⟨_, hin, ne_of_lt hneg⟩
  have hpair : x i ^ 2 + x j ^ 2 ≤ ∑ k, x k ^ 2 := by
    have hs := Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.subset_univ ({i, j} : Finset (Fin (2 * m))))
      (fun k _ _ => sq_nonneg (x k))
    rwa [Finset.sum_pair (Ne.symm hji)] at hs
  rw [h.unit] at hpair
  have hj2 : 0 < x j ^ 2 := by
    rcases lt_or_gt_of_ne hj0 with h' | h' <;> nlinarith
  linarith

/-! ### Part (1): geometric series per entry -/

private lemma abs_sq_lt_one {t : ℝ} (ht : |t| < 1) : |t ^ 2| < 1 := by
  rw [abs_of_nonneg (sq_nonneg t)]
  exact (sq_lt_one_iff_abs_lt_one t).mpr ht

private lemma tsum_odd_geom {t : ℝ} (ht : |t| < 1) :
    ∑' k : ℕ, t ^ (2 * k + 1) = t / (1 - t ^ 2) := by
  calc ∑' k : ℕ, t ^ (2 * k + 1)
      = ∑' k : ℕ, t * (t ^ 2) ^ k := tsum_congr fun k => by ring
    _ = t * ∑' k : ℕ, (t ^ 2) ^ k := tsum_mul_left
    _ = t * (1 - t ^ 2)⁻¹ := by rw [tsum_geometric_of_abs_lt_one (abs_sq_lt_one ht)]
    _ = t / (1 - t ^ 2) := (div_eq_mul_inv _ _).symm

private lemma tsum_even_geom {t : ℝ} (ht : |t| < 1) :
    ∑' k : ℕ, t ^ (2 * (k + 1)) = t ^ 2 / (1 - t ^ 2) := by
  calc ∑' k : ℕ, t ^ (2 * (k + 1))
      = ∑' k : ℕ, t ^ 2 * (t ^ 2) ^ k := tsum_congr fun k => by ring
    _ = t ^ 2 * ∑' k : ℕ, (t ^ 2) ^ k := tsum_mul_left
    _ = t ^ 2 * (1 - t ^ 2)⁻¹ := by rw [tsum_geometric_of_abs_lt_one (abs_sq_lt_one ht)]
    _ = t ^ 2 / (1 - t ^ 2) := (div_eq_mul_inv _ _).symm

private lemma summable_odd_geom {t : ℝ} (ht : |t| < 1) :
    Summable fun k : ℕ => t ^ (2 * k + 1) :=
  ((summable_geometric_of_abs_lt_one (abs_sq_lt_one ht)).mul_left t).congr
    fun k => by ring

private lemma summable_even_geom {t : ℝ} (ht : |t| < 1) :
    Summable fun k : ℕ => t ^ (2 * (k + 1)) :=
  ((summable_geometric_of_abs_lt_one (abs_sq_lt_one ht)).mul_left (t ^ 2)).congr
    fun k => by ring

/-- Part (1), odd half: `O(λ) = Σ_{k ≥ 1 odd} Σ_i λ_i^k`
(odd exponents enumerated as `2k+1`). -/
theorem statO_eq_tsum_odd_moments (h : MemLambdaSet m x) :
    ∑' k : ℕ, ∑ i, x i ^ (2 * k + 1) = statO x := by
  have hlt := h.abs_entry_lt_one
  calc ∑' k : ℕ, ∑ i, x i ^ (2 * k + 1)
      = ∑ i, ∑' k : ℕ, x i ^ (2 * k + 1) :=
        Summable.tsum_finsetSum fun i _ => summable_odd_geom (hlt i)
    _ = ∑ i, x i / (1 - x i ^ 2) :=
        Finset.sum_congr rfl fun i _ => tsum_odd_geom (hlt i)
    _ = statO x := rfl

/-- Part (1), even half: `E(λ) = Σ_{k ≥ 2 even} Σ_i λ_i^k`
(even exponents `≥ 2` enumerated as `2(k+1)`). -/
theorem statE_eq_tsum_even_moments (h : MemLambdaSet m x) :
    ∑' k : ℕ, ∑ i, x i ^ (2 * (k + 1)) = statE x := by
  have hlt := h.abs_entry_lt_one
  calc ∑' k : ℕ, ∑ i, x i ^ (2 * (k + 1))
      = ∑ i, ∑' k : ℕ, x i ^ (2 * (k + 1)) :=
        Summable.tsum_finsetSum fun i _ => summable_even_geom (hlt i)
    _ = ∑ i, x i ^ 2 / (1 - x i ^ 2) :=
        Finset.sum_congr rfl fun i _ => tsum_even_geom (hlt i)
    _ = statE x := rfl

private lemma summable_finset_sum {ι : Type*} (s : Finset ι) (f : ι → ℕ → ℝ)
    (hf : ∀ i ∈ s, Summable (f i)) : Summable fun k => ∑ i ∈ s, f i k := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using summable_zero
  | insert a t ha ih =>
    have h1 : Summable (f a) := hf a (Finset.mem_insert_self a t)
    have h2 := ih fun i hi => hf i (Finset.mem_insert_of_mem hi)
    simpa [Finset.sum_insert ha] using h1.add h2

/-- Part (1): the odd-moment series converges absolutely. -/
theorem summable_abs_odd_moments (h : MemLambdaSet m x) :
    Summable fun k : ℕ => ∑ i, |x i| ^ (2 * k + 1) := by
  refine summable_finset_sum _ _ fun i _ => ?_
  exact summable_odd_geom (by rw [abs_abs]; exact h.abs_entry_lt_one i)

/-- Part (1): the even-moment series converges absolutely. -/
theorem summable_abs_even_moments (h : MemLambdaSet m x) :
    Summable fun k : ℕ => ∑ i, |x i| ^ (2 * (k + 1)) := by
  refine summable_finset_sum _ _ fun i _ => ?_
  exact summable_even_geom (by rw [abs_abs]; exact h.abs_entry_lt_one i)

/-! ### Part (2): partial fractions -/

private lemma statR_iotaMap_eq_sum (x : Fin (2 * m) → ℝ) :
    statR (iotaMap x) = ∑ i, 1 / (1 + x i) := by
  unfold statR
  refine Fintype.sum_equiv Fin.revPerm _ _ fun i => ?_
  simp [iotaMap, sub_neg_eq_add]

/-- Part (2): `O = ½(R - R∘ι)`. -/
theorem statO_eq_half_statR_sub (h : MemLambdaSet m x) :
    statO x = (statR x - statR (iotaMap x)) / 2 := by
  have hlt := h.abs_entry_lt_one
  rw [statR_iotaMap_eq_sum]
  unfold statO statR
  rw [← Finset.sum_sub_distrib, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  obtain ⟨ha, hb⟩ := abs_lt.mp (hlt i)
  have h1 : (1 : ℝ) - x i ≠ 0 := ne_of_gt (by linarith)
  have h2 : (1 : ℝ) + x i ≠ 0 := ne_of_gt (by linarith)
  have h3 : (1 : ℝ) - x i ^ 2 ≠ 0 := ne_of_gt (by nlinarith)
  field_simp
  ring

/-- Part (2): `n + E = ½(R + R∘ι)`, where `n = 2m`. -/
theorem statE_add_card_eq_half_statR_add (h : MemLambdaSet m x) :
    (2 * m : ℝ) + statE x = (statR x + statR (iotaMap x)) / 2 := by
  have hlt := h.abs_entry_lt_one
  rw [statR_iotaMap_eq_sum]
  unfold statE statR
  have hcard : (2 * m : ℝ) = ∑ _i : Fin (2 * m), (1 : ℝ) := by
    simp [Finset.sum_const, Fintype.card_fin]
  rw [hcard, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  obtain ⟨ha, hb⟩ := abs_lt.mp (hlt i)
  have h1 : (1 : ℝ) - x i ≠ 0 := ne_of_gt (by linarith)
  have h2 : (1 : ℝ) + x i ≠ 0 := ne_of_gt (by linarith)
  have h3 : (1 : ℝ) - x i ^ 2 ≠ 0 := ne_of_gt (by nlinarith)
  field_simp
  ring

/-- Part (2), "in particular": `O∘ι = -O`.  (The identity holds for
every vector; the paper states it for `λ ∈ Λ_m`.) -/
theorem statO_iotaMap (_h : MemLambdaSet m x) :
    statO (iotaMap x) = -statO x := by
  unfold statO
  calc ∑ i, iotaMap x i / (1 - iotaMap x i ^ 2)
      = ∑ i, -(x (Fin.rev i) / (1 - x (Fin.rev i) ^ 2)) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        simp [iotaMap, neg_div]
    _ = -∑ i, x (Fin.rev i) / (1 - x (Fin.rev i) ^ 2) := by
        rw [Finset.sum_neg_distrib]
    _ = -∑ i, x i / (1 - x i ^ 2) := by
        congr 1
        exact Equiv.sum_comp Fin.revPerm fun j => x j / (1 - x j ^ 2)

/-- Part (2), "in particular": `E∘ι = E`.  (The identity holds for
every vector; the paper states it for `λ ∈ Λ_m`.) -/
theorem statE_iotaMap (_h : MemLambdaSet m x) :
    statE (iotaMap x) = statE x := by
  unfold statE
  calc ∑ i, iotaMap x i ^ 2 / (1 - iotaMap x i ^ 2)
      = ∑ i, x (Fin.rev i) ^ 2 / (1 - x (Fin.rev i) ^ 2) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        simp [iotaMap]
    _ = ∑ i, x i ^ 2 / (1 - x i ^ 2) :=
        Equiv.sum_comp Fin.revPerm fun j => x j ^ 2 / (1 - x j ^ 2)

/-! ### Part (3): traces

The spectral theorem writes `S = U D U*` with `D` diagonal; the three
trace identities follow from the private conjugation lemmas below. -/

section Trace

variable {S : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ}

private lemma conj_mul {U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ}
    (hU' : star U * U = 1) (d e : Fin (2 * m) → ℝ) :
    (U * diagonal d * star U) * (U * diagonal e * star U)
      = U * diagonal (fun i => d i * e i) * star U := by
  calc (U * diagonal d * star U) * (U * diagonal e * star U)
      = U * diagonal d * (star U * U) * diagonal e * star U := by
        simp only [Matrix.mul_assoc]
    _ = U * diagonal d * diagonal e * star U := by rw [hU', Matrix.mul_one]
    _ = U * diagonal (fun i => d i * e i) * star U := by
        rw [Matrix.mul_assoc U, Matrix.diagonal_mul_diagonal]

private lemma conj_sub (U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ)
    (d e : Fin (2 * m) → ℝ) :
    U * diagonal d * star U - U * diagonal e * star U
      = U * diagonal (fun i => d i - e i) * star U := by
  rw [show diagonal (fun i => d i - e i) = diagonal d - diagonal e from
    (Matrix.diagonal_sub d e).symm, Matrix.mul_sub, Matrix.sub_mul]

private lemma conj_one {U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ}
    (hU : U * star U = 1) :
    U * diagonal (fun _ => (1 : ℝ)) * star U = 1 := by
  rw [Matrix.diagonal_one, Matrix.mul_one, hU]

private lemma inv_conj {U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ}
    (hU : U * star U = 1) (e : Fin (2 * m) → ℝ) (he : ∀ i, e i ≠ 0) :
    (U * diagonal e * star U)⁻¹ = U * diagonal (fun i => (e i)⁻¹) * star U := by
  have hU' : star U * U = 1 := mul_eq_one_comm.mp hU
  apply Matrix.inv_eq_right_inv
  rw [conj_mul hU']
  have he1 : (fun i => e i * (e i)⁻¹) = fun _ => (1 : ℝ) :=
    funext fun i => mul_inv_cancel₀ (he i)
  rw [he1, conj_one hU]

private lemma trace_conj {U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ}
    (hU : U * star U = 1) (d : Fin (2 * m) → ℝ) :
    Matrix.trace (U * diagonal d * star U) = ∑ i, d i := by
  have hU' : star U * U = 1 := mul_eq_one_comm.mp hU
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, hU', Matrix.one_mul,
    Matrix.trace_diagonal]

private lemma spectral_decomp (hS : S.IsHermitian) :
    S = (hS.eigenvectorUnitary : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ) *
        diagonal hS.eigenvalues *
        star (hS.eigenvectorUnitary : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ) := by
  have h1 := hS.spectral_theorem
  rw [Unitary.conjStarAlgAut_apply] at h1
  simpa [RCLike.ofReal_real_eq_id, Function.id_comp, Unitary.coe_star] using h1

private lemma eigenvalues_abs_lt_one (hS : S.IsHermitian) (h : MemLambdaSet m x)
    (σ : Equiv.Perm (Fin (2 * m))) (hσ : hS.eigenvalues ∘ σ = x)
    (i : Fin (2 * m)) : |hS.eigenvalues i| < 1 := by
  have hx := h.abs_entry_lt_one (σ.symm i)
  rw [← congrFun hσ (σ.symm i)] at hx
  simpa using hx

private lemma sum_eigenvalues_comp (hS : S.IsHermitian)
    (σ : Equiv.Perm (Fin (2 * m))) (hσ : hS.eigenvalues ∘ σ = x) (g : ℝ → ℝ) :
    ∑ i, g (hS.eigenvalues i) = ∑ i, g (x i) := by
  refine (Fintype.sum_equiv σ _ _ fun i => ?_).symm
  rw [← congrFun hσ i]
  rfl

variable (hS : S.IsHermitian)

/-- Part (3): `O(λ) = tr(S(I-S²)⁻¹)` for a symmetric matrix `S` whose
spectrum, with multiplicity, is the entries of `λ`. -/
theorem trace_eq_statO (h : MemLambdaSet m x) (σ : Equiv.Perm (Fin (2 * m)))
    (hσ : hS.eigenvalues ∘ σ = x) :
    Matrix.trace (S * (1 - S ^ 2)⁻¹) = statO x := by
  set U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ := ↑hS.eigenvectorUnitary with hUdef
  have hU : U * star U = 1 := Unitary.coe_mul_star_self _
  have hU' : star U * U = 1 := mul_eq_one_comm.mp hU
  set d := hS.eigenvalues with hd
  have hdlt : ∀ i, |d i| < 1 := eigenvalues_abs_lt_one hS h σ hσ
  have hne : ∀ i, (1 : ℝ) - d i ^ 2 ≠ 0 := fun i =>
    ne_of_gt (by have := (sq_lt_one_iff_abs_lt_one (d i)).mpr (hdlt i); linarith)
  have hSdec : S = U * diagonal d * star U := spectral_decomp hS
  have hS2 : S ^ 2 = U * diagonal (fun i => d i ^ 2) * star U := by
    have e : (fun i => d i * d i) = fun i => d i ^ 2 :=
      funext fun i => (pow_two (d i)).symm
    rw [pow_two, hSdec, conj_mul hU', e]
  have hsub : 1 - S ^ 2 = U * diagonal (fun i => 1 - d i ^ 2) * star U := by
    calc 1 - S ^ 2
        = U * diagonal (fun _ => (1 : ℝ)) * star U -
          U * diagonal (fun i => d i ^ 2) * star U := by rw [← hS2, conj_one hU]
      _ = U * diagonal (fun i => 1 - d i ^ 2) * star U := conj_sub _ _ _
  have hinv : (1 - S ^ 2)⁻¹ = U * diagonal (fun i => (1 - d i ^ 2)⁻¹) * star U := by
    rw [hsub]; exact inv_conj hU _ hne
  calc Matrix.trace (S * (1 - S ^ 2)⁻¹)
      = Matrix.trace (U * diagonal (fun i => d i * (1 - d i ^ 2)⁻¹) * star U) := by
        rw [hinv, hSdec, conj_mul hU']
    _ = ∑ i, d i * (1 - d i ^ 2)⁻¹ := trace_conj hU _
    _ = ∑ i, d i / (1 - d i ^ 2) := by simp_rw [div_eq_mul_inv]
    _ = ∑ i, x i / (1 - x i ^ 2) := by
        simpa using sum_eigenvalues_comp hS σ hσ (fun t => t / (1 - t ^ 2))
    _ = statO x := rfl

/-- Part (3): `E(λ) = tr(S²(I-S²)⁻¹)`. -/
theorem trace_eq_statE (h : MemLambdaSet m x) (σ : Equiv.Perm (Fin (2 * m)))
    (hσ : hS.eigenvalues ∘ σ = x) :
    Matrix.trace (S ^ 2 * (1 - S ^ 2)⁻¹) = statE x := by
  set U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ := ↑hS.eigenvectorUnitary with hUdef
  have hU : U * star U = 1 := Unitary.coe_mul_star_self _
  have hU' : star U * U = 1 := mul_eq_one_comm.mp hU
  set d := hS.eigenvalues with hd
  have hdlt : ∀ i, |d i| < 1 := eigenvalues_abs_lt_one hS h σ hσ
  have hne : ∀ i, (1 : ℝ) - d i ^ 2 ≠ 0 := fun i =>
    ne_of_gt (by have := (sq_lt_one_iff_abs_lt_one (d i)).mpr (hdlt i); linarith)
  have hSdec : S = U * diagonal d * star U := spectral_decomp hS
  have hS2 : S ^ 2 = U * diagonal (fun i => d i ^ 2) * star U := by
    have e : (fun i => d i * d i) = fun i => d i ^ 2 :=
      funext fun i => (pow_two (d i)).symm
    rw [pow_two, hSdec, conj_mul hU', e]
  have hsub : 1 - S ^ 2 = U * diagonal (fun i => 1 - d i ^ 2) * star U := by
    calc 1 - S ^ 2
        = U * diagonal (fun _ => (1 : ℝ)) * star U -
          U * diagonal (fun i => d i ^ 2) * star U := by rw [← hS2, conj_one hU]
      _ = U * diagonal (fun i => 1 - d i ^ 2) * star U := conj_sub _ _ _
  have hinv : (1 - S ^ 2)⁻¹ = U * diagonal (fun i => (1 - d i ^ 2)⁻¹) * star U := by
    rw [hsub]; exact inv_conj hU _ hne
  calc Matrix.trace (S ^ 2 * (1 - S ^ 2)⁻¹)
      = Matrix.trace (U * diagonal (fun i => d i ^ 2 * (1 - d i ^ 2)⁻¹) * star U) := by
        rw [hinv, hS2, conj_mul hU']
    _ = ∑ i, d i ^ 2 * (1 - d i ^ 2)⁻¹ := trace_conj hU _
    _ = ∑ i, d i ^ 2 / (1 - d i ^ 2) := by simp_rw [div_eq_mul_inv]
    _ = ∑ i, x i ^ 2 / (1 - x i ^ 2) := by
        simpa using sum_eigenvalues_comp hS σ hσ (fun t => t ^ 2 / (1 - t ^ 2))
    _ = statE x := rfl

/-- Part (3): `R(λ) = tr((I-S)⁻¹)`. -/
theorem trace_eq_statR (h : MemLambdaSet m x) (σ : Equiv.Perm (Fin (2 * m)))
    (hσ : hS.eigenvalues ∘ σ = x) :
    Matrix.trace ((1 - S)⁻¹) = statR x := by
  set U : Matrix (Fin (2 * m)) (Fin (2 * m)) ℝ := ↑hS.eigenvectorUnitary with hUdef
  have hU : U * star U = 1 := Unitary.coe_mul_star_self _
  have hU' : star U * U = 1 := mul_eq_one_comm.mp hU
  set d := hS.eigenvalues with hd
  have hdlt : ∀ i, |d i| < 1 := eigenvalues_abs_lt_one hS h σ hσ
  have hne : ∀ i, (1 : ℝ) - d i ≠ 0 := fun i =>
    ne_of_gt (by have := abs_lt.mp (hdlt i); linarith [this.2])
  have hSdec : S = U * diagonal d * star U := spectral_decomp hS
  have hsub : 1 - S = U * diagonal (fun i => 1 - d i) * star U := by
    calc 1 - S
        = U * diagonal (fun _ => (1 : ℝ)) * star U -
          U * diagonal d * star U := by rw [← hSdec, conj_one hU]
      _ = U * diagonal (fun i => 1 - d i) * star U := conj_sub _ _ _
  have hinv : (1 - S)⁻¹ = U * diagonal (fun i => (1 - d i)⁻¹) * star U := by
    rw [hsub]; exact inv_conj hU _ hne
  calc Matrix.trace ((1 - S)⁻¹)
      = ∑ i, (1 - d i)⁻¹ := by rw [hinv]; exact trace_conj hU _
    _ = ∑ i, 1 / (1 - d i) := by simp_rw [one_div]
    _ = ∑ i, 1 / (1 - x i) := by
        simpa using sum_eigenvalues_comp hS σ hσ (fun t => 1 / (1 - t))
    _ = statR x := rfl

end Trace

end Appendices

