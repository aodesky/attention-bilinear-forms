/-
Appendix "Random matrices",
Corollary (Random QK-product baseline).

Let `W_K, W_Q ∈ M_{n×N}(ℝ)` be independent random matrices with
independent, mean-zero entries of variances `σ_K²` and `σ_Q²`.
Set `L = W_Kᵀ W_Q` and `S, T` as before.  Then

    E‖L‖² = nN²σ_K²σ_Q²,
    E‖S‖² = nN(N+1)/2 · σ_K²σ_Q²,
    E‖T‖² = nN(N-1)/2 · σ_K²σ_Q²,

so

    E‖S‖²/E‖L‖² = (N+1)/(2N),    E‖T‖²/E‖L‖² = (N-1)/(2N).

Modelling: "independent random matrices with independent entries" is
joint independence (`iIndepFun`) of the family of all `2nN` entry maps,
indexed by `(Fin n × Fin N) ⊕ (Fin n × Fin N)` (left summand: entries
of `W_K`; right summand: entries of `W_Q`).  Mean zero and variance are
integral conditions together with square-integrability of each entry.
The ratio statements divide by `E‖L‖² = nN²σ_K²σ_Q²`, so they require
`σ_K ≠ 0`, `σ_Q ≠ 0`, `0 < n`, `0 < N` (the paper implicitly assumes
this nondegeneracy).
-/
import Appendices.Common
import Appendices.RandomMatrices.EnergyIdentities

-- Some section hypotheses are not needed by every lemma in the
-- section; silence the corresponding linter.
set_option linter.unusedSectionVars false

namespace Appendices

open Matrix MeasureTheory ProbabilityTheory BigOperators

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}
  [IsProbabilityMeasure μ] {n N : ℕ}
  {WK WQ : Ω → Matrix (Fin n) (Fin N) ℝ} {σK σQ : ℝ}

section QKProduct

/-- The family of all `2nN` entries of the pair `(W_K, W_Q)`. -/
private def entryFamily (WK WQ : Ω → Matrix (Fin n) (Fin N) ℝ) :
    (Fin n × Fin N) ⊕ (Fin n × Fin N) → Ω → ℝ :=
  Sum.elim (fun q ω => WK ω q.1 q.2) (fun q ω => WQ ω q.1 q.2)

variable (hIndep : iIndepFun (entryFamily WK WQ) μ)
  (hKL2 : ∀ r a, MemLp (fun ω => WK ω r a) 2 μ)
  (hQL2 : ∀ r b, MemLp (fun ω => WQ ω r b) 2 μ)
  (hKMean : ∀ r a, ∫ ω, WK ω r a ∂μ = 0)
  (hQMean : ∀ r b, ∫ ω, WQ ω r b ∂μ = 0)
  (hKVar : ∀ r a, ∫ ω, (WK ω r a) ^ 2 ∂μ = σK ^ 2)
  (hQVar : ∀ r b, ∫ ω, (WQ ω r b) ^ 2 ∂μ = σQ ^ 2)

include hKL2 hQL2 in
private lemma entryFamily_aemeasurable :
    ∀ i, AEMeasurable (entryFamily WK WQ i) μ := by
  rintro (q | q)
  · exact (hKL2 q.1 q.2).aestronglyMeasurable.aemeasurable
  · exact (hQL2 q.1 q.2).aestronglyMeasurable.aemeasurable

include hIndep hKL2 hQL2 in
/-- The product of two `W_K`-entries is independent of the product of
two `W_Q`-entries. -/
private lemma block_indepFun (r s : Fin n) (a b c d : Fin N) :
    IndepFun (fun ω => WK ω r a * WK ω s c)
      (fun ω => WQ ω r b * WQ ω s d) μ := by
  have h := hIndep.indepFun_prodMk_prodMk₀
    (entryFamily_aemeasurable hKL2 hQL2)
    (Sum.inl (r, a)) (Sum.inl (s, c)) (Sum.inr (r, b)) (Sum.inr (s, d))
    (by simp) (by simp) (by simp) (by simp)
  exact h.comp (measurable_fst.mul measurable_snd)
    (measurable_fst.mul measurable_snd)

include hIndep hKL2 hKMean hKVar in
/-- Second moments of the `W_K` entries: `E[K_{ra} K_{sc}]` is `σ_K²`
on the diagonal and `0` off it. -/
private lemma integral_K_mul_K (r s : Fin n) (a c : Fin N) :
    ∫ ω, WK ω r a * WK ω s c ∂μ
      = if (r, a) = (s, c) then σK ^ 2 else 0 := by
  by_cases h : (r, a) = (s, c)
  · obtain ⟨rfl, rfl⟩ := Prod.mk.injEq .. ▸ h
    rw [if_pos rfl, ← hKVar r a]
    exact integral_congr_ae (Filter.Eventually.of_forall fun ω => (sq _).symm)
  · rw [if_neg h]
    have hind : (fun ω => WK ω r a) ⟂ᵢ[μ] (fun ω => WK ω s c) :=
      hIndep.indepFun (i := Sum.inl (r, a)) (j := Sum.inl (s, c)) (by simp [h])
    rw [hind.integral_fun_mul_eq_mul_integral
      (hKL2 r a).aestronglyMeasurable (hKL2 s c).aestronglyMeasurable,
      hKMean, zero_mul]

include hIndep hQL2 hQMean hQVar in
private lemma integral_Q_mul_Q (r s : Fin n) (b d : Fin N) :
    ∫ ω, WQ ω r b * WQ ω s d ∂μ
      = if (r, b) = (s, d) then σQ ^ 2 else 0 := by
  by_cases h : (r, b) = (s, d)
  · obtain ⟨rfl, rfl⟩ := Prod.mk.injEq .. ▸ h
    rw [if_pos rfl, ← hQVar r b]
    exact integral_congr_ae (Filter.Eventually.of_forall fun ω => (sq _).symm)
  · rw [if_neg h]
    have hind : (fun ω => WQ ω r b) ⟂ᵢ[μ] (fun ω => WQ ω s d) :=
      hIndep.indepFun (i := Sum.inr (r, b)) (j := Sum.inr (s, d)) (by simp [h])
    rw [hind.integral_fun_mul_eq_mul_integral
      (hQL2 r b).aestronglyMeasurable (hQL2 s d).aestronglyMeasurable,
      hQMean, zero_mul]

include hIndep hKL2 hQL2 in
private lemma integrable_fourProduct (r s : Fin n) (a b c d : Fin N) :
    Integrable (fun ω => (WK ω r a * WK ω s c) * (WQ ω r b * WQ ω s d)) μ := by
  have hK : Integrable (fun ω => WK ω r a * WK ω s c) μ := by
    by_cases h : (r, a) = (s, c)
    · obtain ⟨rfl, rfl⟩ := Prod.mk.injEq .. ▸ h
      simpa [sq] using (hKL2 r a).integrable_sq
    · exact ((hIndep.indepFun (i := Sum.inl (r, a)) (j := Sum.inl (s, c))
        (by simp [h]))).integrable_mul
        ((hKL2 r a).integrable one_le_two) ((hKL2 s c).integrable one_le_two)
  have hQ : Integrable (fun ω => WQ ω r b * WQ ω s d) μ := by
    by_cases h : (r, b) = (s, d)
    · obtain ⟨rfl, rfl⟩ := Prod.mk.injEq .. ▸ h
      simpa [sq] using (hQL2 r b).integrable_sq
    · exact ((hIndep.indepFun (i := Sum.inr (r, b)) (j := Sum.inr (s, d))
        (by simp [h]))).integrable_mul
        ((hQL2 r b).integrable one_le_two) ((hQL2 s d).integrable one_le_two)
  exact (block_indepFun hIndep hKL2 hQL2 r s a b c d).integrable_mul hK hQ

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- The fundamental second-moment computation:
`E[L_{ab} L_{cd}] = (a = c ∧ b = d) · n σ_K² σ_Q²` for `L = W_Kᵀ W_Q`. -/
private lemma integral_L_mul_L (a b c d : Fin N) :
    ∫ ω, ((WK ω)ᵀ * WQ ω) a b * ((WK ω)ᵀ * WQ ω) c d ∂μ
      = if a = c ∧ b = d then n * σK ^ 2 * σQ ^ 2 else 0 := by
  have hentry : ∀ ω, ((WK ω)ᵀ * WQ ω) a b * ((WK ω)ᵀ * WQ ω) c d
      = ∑ r, ∑ s, (WK ω r a * WK ω s c) * (WQ ω r b * WQ ω s d) := by
    intro ω
    rw [Matrix.mul_apply, Matrix.mul_apply, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun s _ => ?_
    simp only [Matrix.transpose_apply]
    ring
  have hint : ∀ (r s : Fin n),
      Integrable (fun ω => (WK ω r a * WK ω s c) * (WQ ω r b * WQ ω s d)) μ :=
    fun r s => integrable_fourProduct hIndep hKL2 hQL2 r s a b c d
  calc ∫ ω, ((WK ω)ᵀ * WQ ω) a b * ((WK ω)ᵀ * WQ ω) c d ∂μ
      = ∫ ω, ∑ r, ∑ s, (WK ω r a * WK ω s c) * (WQ ω r b * WQ ω s d) ∂μ :=
        integral_congr_ae (Filter.Eventually.of_forall fun ω => hentry ω)
    _ = ∑ r, ∑ s, ∫ ω, (WK ω r a * WK ω s c) * (WQ ω r b * WQ ω s d) ∂μ := by
        rw [integral_finset_sum _ fun r _ => integrable_finset_sum _ fun s _ =>
          hint r s]
        exact Finset.sum_congr rfl fun r _ =>
          integral_finset_sum _ fun s _ => hint r s
    _ = ∑ r, ∑ s, (if (r, a) = (s, c) then σK ^ 2 else 0)
          * (if (r, b) = (s, d) then σQ ^ 2 else 0) := by
        refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun s _ => ?_
        rw [(block_indepFun hIndep hKL2 hQL2 r s a b c d).integral_fun_mul_eq_mul_integral
          ((hKL2 r a).aestronglyMeasurable.mul (hKL2 s c).aestronglyMeasurable)
          ((hQL2 r b).aestronglyMeasurable.mul (hQL2 s d).aestronglyMeasurable),
          integral_K_mul_K hIndep hKL2 hKMean hKVar,
          integral_Q_mul_Q hIndep hQL2 hQMean hQVar]
    _ = if a = c ∧ b = d then n * σK ^ 2 * σQ ^ 2 else 0 := by
        by_cases hac : a = c
        · by_cases hbd : b = d
          · subst hac; subst hbd
            rw [if_pos ⟨rfl, rfl⟩]
            have hterm : ∀ r s : Fin n,
                (if ((r, a) : Fin n × Fin N) = (s, a) then σK ^ 2 else 0)
                  * (if ((r, b) : Fin n × Fin N) = (s, b) then σQ ^ 2 else 0)
                = if r = s then σK ^ 2 * σQ ^ 2 else 0 := by
              intro r s
              by_cases h : r = s
              · subst h
                simp
              · rw [if_neg (by simp [h]), if_neg h, zero_mul]
            simp_rw [hterm]
            rw [Finset.sum_comm]
            simp [mul_assoc]
          · rw [if_neg (by tauto)]
            refine Finset.sum_eq_zero fun r _ => Finset.sum_eq_zero fun s _ => ?_
            have hq : ¬(((r, b) : Fin n × Fin N) = (s, d)) := by
              simp only [Prod.mk.injEq, not_and]
              exact fun _ => hbd
            rw [if_neg hq, mul_zero]
        · rw [if_neg (by tauto)]
          refine Finset.sum_eq_zero fun r _ => Finset.sum_eq_zero fun s _ => ?_
          have hk : ¬(((r, a) : Fin n × Fin N) = (s, c)) := by
            simp only [Prod.mk.injEq, not_and]
            exact fun _ => hac
          rw [if_neg hk, zero_mul]

/-! ### The statements of the corollary -/

private lemma frobSq_eq_sum_prod' (A : Matrix (Fin N) (Fin N) ℝ) :
    frobSq A = ∑ p : Fin N × Fin N, A p.1 p.2 ^ 2 := by
  rw [frobSq_eq_sum_sq]
  exact (Fintype.sum_prod_type (f := fun p : Fin N × Fin N => A p.1 p.2 ^ 2)).symm

private lemma trace_mul_self_eq_sum_prod' (A : Matrix (Fin N) (Fin N) ℝ) :
    Matrix.trace (A * A) = ∑ p : Fin N × Fin N, A p.1 p.2 * A p.2 p.1 := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply]
  exact (Fintype.sum_prod_type
    (f := fun p : Fin N × Fin N => A p.1 p.2 * A p.2 p.1)).symm

include hIndep hKL2 hQL2 in
private lemma integrable_L_entry_mul (a b c d : Fin N) :
    Integrable (fun ω => ((WK ω)ᵀ * WQ ω) a b * ((WK ω)ᵀ * WQ ω) c d) μ := by
  have h : (fun ω => ((WK ω)ᵀ * WQ ω) a b * ((WK ω)ᵀ * WQ ω) c d)
      = fun ω => ∑ r, ∑ s, (WK ω r a * WK ω s c) * (WQ ω r b * WQ ω s d) := by
    funext ω
    rw [Matrix.mul_apply, Matrix.mul_apply, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun s _ => ?_
    simp only [Matrix.transpose_apply]
    ring
  rw [h]
  exact integrable_finset_sum _ fun r _ => integrable_finset_sum _ fun s _ =>
    integrable_fourProduct hIndep hKL2 hQL2 r s a b c d

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- `E‖L‖² = nN²σ_K²σ_Q²` for `L = W_Kᵀ W_Q`. -/
theorem integral_frobSq_qk :
    ∫ ω, frobSq ((WK ω)ᵀ * WQ ω) ∂μ = n * N ^ 2 * σK ^ 2 * σQ ^ 2 := by
  simp_rw [frobSq_eq_sum_prod']
  rw [integral_finset_sum _ fun p _ => by
    simpa [sq] using integrable_L_entry_mul hIndep hKL2 hQL2 p.1 p.2 p.1 p.2]
  have hterm : ∀ p : Fin N × Fin N,
      (∫ ω, ((WK ω)ᵀ * WQ ω) p.1 p.2 ^ 2 ∂μ) = n * σK ^ 2 * σQ ^ 2 := by
    intro p
    have h := integral_L_mul_L hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar
      p.1 p.2 p.1 p.2
    rw [if_pos ⟨rfl, rfl⟩] at h
    rw [← h]
    exact integral_congr_ae (Filter.Eventually.of_forall fun ω => sq _)
  simp_rw [hterm]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
    nsmul_eq_mul]
  push_cast
  ring

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- `E tr(L²) = nNσ_K²σ_Q²`. -/
theorem integral_trace_mul_self_qk :
    ∫ ω, Matrix.trace ((WK ω)ᵀ * WQ ω * ((WK ω)ᵀ * WQ ω)) ∂μ
      = n * N * σK ^ 2 * σQ ^ 2 := by
  simp_rw [trace_mul_self_eq_sum_prod']
  rw [integral_finset_sum _ fun p _ =>
    integrable_L_entry_mul hIndep hKL2 hQL2 p.1 p.2 p.2 p.1]
  have hterm : ∀ p : Fin N × Fin N,
      (∫ ω, ((WK ω)ᵀ * WQ ω) p.1 p.2 * ((WK ω)ᵀ * WQ ω) p.2 p.1 ∂μ)
        = if p.1 = p.2 then n * σK ^ 2 * σQ ^ 2 else 0 := by
    intro p
    have h := integral_L_mul_L hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar
      p.1 p.2 p.2 p.1
    by_cases hp : p.1 = p.2
    · rw [if_pos hp, h, if_pos ⟨hp, hp.symm⟩]
    · rw [if_neg hp, h, if_neg (by tauto)]
  simp_rw [hterm]
  rw [Fintype.sum_prod_type]
  have hinner : ∀ i : Fin N,
      (∑ j : Fin N, if i = j then n * σK ^ 2 * σQ ^ 2 else (0 : ℝ))
        = n * σK ^ 2 * σQ ^ 2 := by
    intro i
    simp
  simp_rw [hinner]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- `E‖S‖² = nN(N+1)/2 · σ_K²σ_Q²` for `S` the symmetric part of
`L = W_Kᵀ W_Q`. -/
theorem integral_frobSq_symPart_qk :
    ∫ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω)) ∂μ
      = n * N * (N + 1) / 2 * σK ^ 2 * σQ ^ 2 := by
  have hpt : ∀ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω))
      = frobSq ((WK ω)ᵀ * WQ ω) / 2
        + Matrix.trace ((WK ω)ᵀ * WQ ω * ((WK ω)ᵀ * WQ ω)) / 2 :=
    fun ω => frobSq_symPart _
  simp_rw [hpt]
  have hint1 : Integrable (fun ω => frobSq ((WK ω)ᵀ * WQ ω)) μ := by
    simp_rw [frobSq_eq_sum_prod']
    exact integrable_finset_sum _ fun p _ => by
      simpa [sq] using integrable_L_entry_mul hIndep hKL2 hQL2 p.1 p.2 p.1 p.2
  have hint2 : Integrable
      (fun ω => Matrix.trace ((WK ω)ᵀ * WQ ω * ((WK ω)ᵀ * WQ ω))) μ := by
    simp_rw [trace_mul_self_eq_sum_prod']
    exact integrable_finset_sum _ fun p _ =>
      integrable_L_entry_mul hIndep hKL2 hQL2 p.1 p.2 p.2 p.1
  rw [integral_add (hint1.div_const 2) (hint2.div_const 2), integral_div,
    integral_div, integral_frobSq_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar,
    integral_trace_mul_self_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar]
  ring

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- `E‖T‖² = nN(N-1)/2 · σ_K²σ_Q²` for `T` the antisymmetric part of
`L = W_Kᵀ W_Q`. -/
theorem integral_frobSq_skewPart_qk :
    ∫ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω)) ∂μ
      = n * N * (N - 1) / 2 * σK ^ 2 * σQ ^ 2 := by
  have hpt : ∀ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω))
      = frobSq ((WK ω)ᵀ * WQ ω) / 2
        - Matrix.trace ((WK ω)ᵀ * WQ ω * ((WK ω)ᵀ * WQ ω)) / 2 :=
    fun ω => frobSq_skewPart _
  simp_rw [hpt]
  have hint1 : Integrable (fun ω => frobSq ((WK ω)ᵀ * WQ ω)) μ := by
    simp_rw [frobSq_eq_sum_prod']
    exact integrable_finset_sum _ fun p _ => by
      simpa [sq] using integrable_L_entry_mul hIndep hKL2 hQL2 p.1 p.2 p.1 p.2
  have hint2 : Integrable
      (fun ω => Matrix.trace ((WK ω)ᵀ * WQ ω * ((WK ω)ᵀ * WQ ω))) μ := by
    simp_rw [trace_mul_self_eq_sum_prod']
    exact integrable_finset_sum _ fun p _ =>
      integrable_L_entry_mul hIndep hKL2 hQL2 p.1 p.2 p.2 p.1
  rw [integral_sub (hint1.div_const 2) (hint2.div_const 2), integral_div,
    integral_div, integral_frobSq_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar,
    integral_trace_mul_self_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar]
  ring

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- `E‖S‖² / E‖L‖² = (N+1)/(2N)`.  (The division requires the
nondegeneracy `σ_K, σ_Q ≠ 0`, `0 < n`, `0 < N`, which the paper leaves
implicit.) -/
theorem integral_frobSq_symPart_div_integral_frobSq_qk
    (hσK : σK ≠ 0) (hσQ : σQ ≠ 0) (hn : 0 < n) (hN : 0 < N) :
    (∫ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω)) ∂μ)
        / (∫ ω, frobSq ((WK ω)ᵀ * WQ ω) ∂μ)
      = (N + 1) / (2 * N) := by
  rw [integral_frobSq_symPart_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar,
    integral_frobSq_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar]
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  field_simp

include hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar in
/-- `E‖T‖² / E‖L‖² = (N-1)/(2N)`. -/
theorem integral_frobSq_skewPart_div_integral_frobSq_qk
    (hσK : σK ≠ 0) (hσQ : σQ ≠ 0) (hn : 0 < n) (hN : 0 < N) :
    (∫ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω)) ∂μ)
        / (∫ ω, frobSq ((WK ω)ᵀ * WQ ω) ∂μ)
      = (N - 1) / (2 * N) := by
  rw [integral_frobSq_skewPart_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar,
    integral_frobSq_qk hIndep hKL2 hQL2 hKMean hQMean hKVar hQVar]
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  field_simp

end QKProduct

end Appendices

