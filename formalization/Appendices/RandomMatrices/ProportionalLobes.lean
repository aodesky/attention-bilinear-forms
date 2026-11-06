/-
Appendix "Random matrices", Lemma (lem:proportional-lobes).

  Let `X` be a positive random variable with finite, nonzero second
  moment.  Fix scales `u, v > 0` and set `ρ = u/v`.  For each `n`, let
  `α_n ∈ ℝⁿ` be the decreasing sorting of `vX_1, …, vX_n` and let
  `β_n ∈ ℝⁿ` be the decreasing sorting of `uX'_1, …, uX'_n`, where the
  `X_i` and `X'_i` are independent copies of `X`.  Then

      ‖β_n − ρ α_n‖ / ‖α_n‖ → 0    almost surely as n → ∞.

Encoding choices (documented in the theorem docstring):
* the law of `X` is a probability law `γ` on `ℝ` giving no mass to
  `(-∞, 0]` (`X` is positive) whose second moment
  `∫⁻ y, ofReal (y²) ∂γ` is finite and nonzero;
* the copies are sequences `X X' : ℕ → Ω → ℝ` of random variables,
  each with law `γ`; the hypothesis `iIndepFun (Sum.elim X X') P` says
  that all of `X 0, X 1, …, X' 0, X' 1, …` are independent;
* `α_n` is `sortDesc (fun i : Fin n => v * X i ω)` and `β_n` is
  `sortDesc (fun i : Fin n => u * X' i ω)`: the `n` samples defining
  `α_n` are the first `n` terms of the sequence `X` (this is the
  setting of the strong law of large numbers used in the paper's
  proof);
* norms are written out as `√(∑ …²)`;
* `U` uniform on `[0,1]` is Lebesgue measure on `(0,1)` (`vol01`).

The proof follows the paper's (in the revised wording of its first step,
author-approved 2026-11-05; see `README.md`).
* `F̄(x) = P(X > x)` (`survival`) and the decreasing generalized inverse
  `G(t) = inf {x > 0 : F̄(x) ≤ t}` (`quantile`), with
  `G(t) > x ↔ t < F̄(x)` (`lt_quantile_iff`); `G(U)` has the law of `X`
  (`map_quantile`), so `∫₀¹ G² = m₂` (`lintegral_quantile_sq`).
* The step function `h_n(t) = x_{⌈nt⌉}` of the sorted samples (`stepFn`),
  with `∫₀¹ h_n² = (1/n) ∑ xᵢ²` (`integral_stepFn`) and
  `h_n(t) > x ↔ F̄_n(x) ≥ ⌈nt⌉/n` (`lt_stepFn_iff`).
* First step: almost surely `F̄_n(x) → F̄(x)` for every rational `x > 0`
  (strong law at countably many thresholds); on such an outcome
  `h_n(t) → G(t)` at every continuity point `t` of `G`, by the paper's
  upper bound (via `t' < t` with `G(t') < G(t) + ε` and a rational
  `x ∈ (G(t'), G(t) + ε)`) and lower bound (via a rational
  `x ∈ (G(t) - ε, G(t))` and `⌈nt⌉/n < t + 1/n`)
  (`tendsto_stepFn_of_continuousWithinAt`); hence for almost every `t`
  (`ae_tendsto_stepFn`).  The lower bound takes `x > 0`, which is
  possible because `G(t) > 0`.
* Second step: `∫₀¹ h_n² → m₂` by the strong law (`tendsto_integral_stepFn_sq`).
* Third step: expanding `∫(h_n - G)²` and applying Fatou's lemma to
  `h_n G` gives `∫(h_n - G)² → 0` (`tendsto_integral_stepFn_sub_quantile_sq`).
* Conclusion: `‖α_n - α_n'‖²/n = ∫(g_n - g_n')² ≤ 2∫(g_n - G)² + 2∫(g_n' - G)²`
  and `‖α_n‖²/n → m₂ > 0` (`tendsto_sortDesc_diff_div_of_seq`); the
  reduction to `v = 1` and `β_n = ρ α_n'` is in `proportional_lobes`.
-/
import Appendices.Common
import Appendices.Profile
import Appendices.RandomMatrices.EnergyIdentities

open MeasureTheory ProbabilityTheory Filter Set Finset
open scoped ENNReal NNReal Topology

namespace Appendices

/-! ### The quantile function

For the law `γ` of `X`, `F̄(x) = P(X > x)` is `survival γ x` and the
decreasing generalized inverse `G(t) = inf {x > 0 : F̄(x) ≤ t}` is
`quantile γ t`.  The defining property is `G(t) > x ↔ t < F̄(x)` for
`t ∈ (0,1)` (`lt_quantile_iff`). -/

section Quantile

variable (γ : Measure ℝ)

/-- `F̄(x) = P(X > x)`. -/
noncomputable def survival (x : ℝ) : ℝ := γ.real (Ioi x)

/-- The decreasing generalized inverse `G(t) = inf {x > 0 : F̄(x) ≤ t}`. -/
noncomputable def quantile (t : ℝ) : ℝ := sInf {x | 0 < x ∧ survival γ x ≤ t}

variable {γ} [IsProbabilityMeasure γ]

lemma survival_eq_one_sub_cdf (x : ℝ) : survival γ x = 1 - cdf γ x := by
  rw [survival, cdf_eq_real, ← Set.compl_Iic, probReal_compl_eq_one_sub measurableSet_Iic]

lemma survival_antitone : Antitone (survival γ) := fun _ _ h => by
  simp only [survival_eq_one_sub_cdf]; linarith [monotone_cdf γ h]

lemma survival_of_nonpos (hγ : γ (Iic 0) = 0) {x : ℝ} (hx : x ≤ 0) : survival γ x = 1 := by
  have h0 : survival γ 0 = 1 := by
    rw [survival, ← Set.compl_Iic, probReal_compl_eq_one_sub measurableSet_Iic,
      measureReal_def, hγ]
    simp
  have h1 : survival γ x ≤ 1 := by
    rw [survival_eq_one_sub_cdf]; linarith [cdf_nonneg γ x]
  linarith [survival_antitone (γ := γ) hx]

/-- `G(t) > x` if and only if `t < F̄(x)`, for `t ∈ (0,1)`. -/
lemma lt_quantile_iff (hγ : γ (Iic 0) = 0) {t : ℝ} (ht : t ∈ Ioo (0:ℝ) 1) (x : ℝ) :
    x < quantile γ t ↔ t < survival γ x := by
  set S := {x | 0 < x ∧ survival γ x ≤ t} with hS
  have hbdd : BddBelow S := ⟨0, fun y hy => hy.1.le⟩
  have hne : S.Nonempty := by
    have hlim : Tendsto (survival γ) atTop (𝓝 0) := by
      have := (tendsto_cdf_atTop (μ := γ)).const_sub 1
      simp only [sub_self] at this
      exact this.congr fun y => (survival_eq_one_sub_cdf y).symm
    obtain ⟨y, hy⟩ := ((hlim.eventually (gt_mem_nhds ht.1)).and (eventually_gt_atTop 0)).exists
    exact ⟨y, hy.2, hy.1.le⟩
  constructor
  · intro hx
    by_contra hle
    push_neg at hle
    rcases le_or_gt x 0 with h0 | h0
    · rw [survival_of_nonpos hγ h0] at hle; linarith [ht.2]
    · exact absurd (csInf_le hbdd ⟨h0, hle⟩) (not_le.mpr hx)
  · intro hx
    -- right-continuity of `F̄` at `x`
    have hrc : ∀ᶠ y in 𝓝[>] x, t < survival γ y := by
      have h := ((cdf γ).right_continuous x).mono_left (nhdsWithin_mono _ Set.Ioi_subset_Ici_self)
      have h' : Tendsto (survival γ) (𝓝[>] x) (𝓝 (survival γ x)) := by
        have := h.const_sub 1
        simp only [← survival_eq_one_sub_cdf] at this
        exact this
      exact h'.eventually (lt_mem_nhds hx)
    obtain ⟨y, hyt, hyx⟩ := (hrc.and self_mem_nhdsWithin).exists
    refine lt_of_lt_of_le hyx (le_csInf hne fun s hs => ?_)
    by_contra hsy
    push_neg at hsy
    linarith [hs.2, survival_antitone (γ := γ) hsy.le]

lemma quantile_pos (hγ : γ (Iic 0) = 0) {t : ℝ} (ht : t ∈ Ioo (0:ℝ) 1) : 0 < quantile γ t := by
  rw [lt_quantile_iff hγ ht, survival_of_nonpos hγ le_rfl]; exact ht.2

lemma quantile_antitoneOn (hγ : γ (Iic 0) = 0) : AntitoneOn (quantile γ) (Ioo 0 1) := by
  intro t ht t' ht' htt'
  by_contra hlt
  push_neg at hlt
  obtain ⟨x, hx1, hx2⟩ := exists_between hlt
  have h1 := (lt_quantile_iff hγ ht' x).mp hx2
  have h2 : survival γ x ≤ t := not_lt.mp fun h => (not_lt.mpr hx1.le) ((lt_quantile_iff hγ ht x).mpr h)
  linarith

/-- Lebesgue measure on `(0,1)`, the law of a uniform random variable `U`. -/
noncomputable abbrev vol01 : Measure ℝ := volume.restrict (Ioo (0:ℝ) 1)

instance : IsProbabilityMeasure vol01 :=
  ⟨by simp [Measure.restrict_apply MeasurableSet.univ, Real.volume_Ioo]⟩

lemma aemeasurable_quantile (hγ : γ (Iic 0) = 0) : AEMeasurable (quantile γ) vol01 :=
  aemeasurable_restrict_of_antitoneOn measurableSet_Ioo (quantile_antitoneOn hγ)

/-- `G(U)` has the law of `X`: `P(G(U) > x) = P(U < F̄(x)) = F̄(x)`. -/
lemma map_quantile (hγ : γ (Iic 0) = 0) : vol01.map (quantile γ) = γ := by
  have hm := aemeasurable_quantile (γ := γ) hγ
  haveI : IsProbabilityMeasure (vol01.map (quantile γ)) := Measure.isProbabilityMeasure_map hm
  have hIoi : ∀ x, (vol01.map (quantile γ)).real (Ioi x) = survival γ x := by
    intro x
    have hset : quantile γ ⁻¹' Ioi x ∩ Ioo 0 1 = Ioo 0 (survival γ x) ∩ Ioo 0 1 := by
      ext t
      simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_Ioi, Set.mem_Ioo]
      constructor
      · rintro ⟨h, h0, h1⟩; exact ⟨⟨h0, (lt_quantile_iff hγ ⟨h0, h1⟩ x).mp h⟩, h0, h1⟩
      · rintro ⟨⟨_, h⟩, h0, h1⟩; exact ⟨(lt_quantile_iff hγ ⟨h0, h1⟩ x).mpr h, h0, h1⟩
    have hle : survival γ x ≤ 1 := by
      rw [survival_eq_one_sub_cdf]; linarith [cdf_nonneg γ x]
    rw [measureReal_def, Measure.map_apply_of_aemeasurable hm measurableSet_Ioi,
      Measure.restrict_apply' measurableSet_Ioo, hset, Set.inter_eq_left.mpr
        (Set.Ioo_subset_Ioo_right hle), Real.volume_Ioo, sub_zero,
      ENNReal.toReal_ofReal (by rw [survival]; exact measureReal_nonneg)]
  refine Measure.eq_of_cdf _ _ ?_
  ext x
  have h1 := survival_eq_one_sub_cdf (γ := vol01.map (quantile γ)) x
  have h2 := survival_eq_one_sub_cdf (γ := γ) x
  rw [survival, hIoi] at h1
  linarith

/-- `∫₀¹ G(t)² dt = m₂`. -/
lemma lintegral_quantile_sq (hγ : γ (Iic 0) = 0) :
    ∫⁻ t, ENNReal.ofReal (quantile γ t ^ 2) ∂vol01 = ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ := by
  conv_rhs => rw [← map_quantile hγ]
  rw [lintegral_map' (by fun_prop) (aemeasurable_quantile hγ)]

end Quantile

/-! ### Step functions of sorted samples

For a list `a = (a₁ ≥ ⋯ ≥ a_n)` the step function is `h(t) = a_{⌈nt⌉}` on
`(0,1)` (`stepFn`; Lean indexes from `0`, so this is `a ⟨⌈nt⌉ - 1, _⟩`).
Then `∫₀¹ f(h_a, h_b) = (1/n) ∑ᵢ f(aᵢ, bᵢ)` (`integral_stepFn`), and
`h(t) > x` exactly when `#{i : aᵢ > x}/n ≥ ⌈nt⌉/n` (`lt_stepFn_iff`). -/

section StepFunction

variable {n : ℕ}

/-- The step function `h(t) = a_{⌈nt⌉}` (and `0` where `⌈nt⌉ ∉ [1, n]`). -/
noncomputable def stepFn (a : Fin n → ℝ) (t : ℝ) : ℝ :=
  if h : ⌈(n:ℝ) * t⌉₊ - 1 < n then a ⟨⌈(n:ℝ) * t⌉₊ - 1, h⟩ else 0

lemma measurable_stepFn (a : Fin n → ℝ) : Measurable (stepFn a) := by
  have h : stepFn a = (fun k : ℕ => if h : k - 1 < n then a ⟨k - 1, h⟩ else 0)
      ∘ (fun t : ℝ => ⌈(n:ℝ) * t⌉₊) := rfl
  rw [h]
  exact measurable_from_nat.comp (measurable_const_mul _).nat_ceil

lemma abs_stepFn_le (a : Fin n → ℝ) (t : ℝ) : |stepFn a t| ≤ ∑ i, |a i| := by
  have h0 : 0 ≤ ∑ i, |a i| := Finset.sum_nonneg fun i _ => abs_nonneg _
  unfold stepFn
  split_ifs with h
  · exact Finset.single_le_sum (f := fun i => |a i|) (fun i _ => abs_nonneg _)
      (Finset.mem_univ _)
  · simp only [abs_zero]; exact h0

lemma memLp_stepFn (a : Fin n → ℝ) : MemLp (stepFn a) 2 vol01 :=
  MemLp.of_bound (measurable_stepFn a).aestronglyMeasurable (∑ i, |a i|)
    (ae_of_all _ fun t => by rw [Real.norm_eq_abs]; exact abs_stepFn_le a t)

lemma stepFn_nonneg {a : Fin n → ℝ} (ha : ∀ i, 0 ≤ a i) (t : ℝ) : 0 ≤ stepFn a t := by
  unfold stepFn; split_ifs <;> simp [ha]

/-- On `(i/n, (i+1)/n]` the step function is `aᵢ` (`0`-indexed). -/
lemma stepFn_of_mem (a : Fin n → ℝ) (i : Fin n) {t : ℝ}
    (ht : t ∈ Ioc ((i:ℝ) / n) (((i:ℝ) + 1) / n)) : stepFn a t = a i := by
  have hn : (0:ℝ) < n := Nat.cast_pos.mpr (Nat.zero_lt_of_lt i.2)
  have hceil : ⌈(n:ℝ) * t⌉₊ = (i:ℕ) + 1 := by
    rw [Nat.ceil_eq_iff (Nat.succ_ne_zero _)]
    push_cast
    constructor
    · have := ht.1; rw [div_lt_iff₀ hn] at this; linarith
    · have := ht.2; rw [le_div_iff₀ hn] at this; linarith
  have hi : ⌈(n:ℝ) * t⌉₊ - 1 < n := by rw [hceil]; simp
  simp only [stepFn, dif_pos hi]
  congr 1
  ext
  simp [hceil]

/-- `∫₀¹ f(h_a(t), h_b(t)) dt = (1/n) ∑ᵢ f(aᵢ, bᵢ)`. -/
lemma integral_stepFn (hn : 0 < n) (f : ℝ → ℝ → ℝ) (a b : Fin n → ℝ) :
    ∫ t, f (stepFn a t) (stepFn b t) ∂vol01 = (∑ i, f (a i) (b i)) / n := by
  have hn' : (0:ℝ) < n := Nat.cast_pos.mpr hn
  set c : ℕ → ℝ := fun k => (k:ℝ) / n with hc
  have hpiece : ∀ i : Fin n, ∀ t ∈ Ioc (c i) (c (i + 1)),
      f (stepFn a t) (stepFn b t) = f (a i) (b i) := by
    intro i t ht
    have ht' : t ∈ Ioc ((i:ℝ) / n) (((i:ℝ) + 1) / n) := by simpa [hc] using ht
    rw [stepFn_of_mem a i ht', stepFn_of_mem b i ht']
  have hle : ∀ k, c k ≤ c (k + 1) := fun k => by
    simp only [hc]; gcongr; simp
  have hint : ∀ k < n, IntervalIntegrable (fun t => f (stepFn a t) (stepFn b t)) volume
      (c k) (c (k + 1)) := by
    intro k hk
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le (hle k)]
    refine (integrableOn_const (C := f (a ⟨k, hk⟩) (b ⟨k, hk⟩))
      (by simp [Real.volume_Ioc])).congr_fun (fun t ht => ?_) measurableSet_Ioc
    exact (hpiece ⟨k, hk⟩ t ht).symm
  have hsum := intervalIntegral.sum_integral_adjacent_intervals hint
  have hc0 : c 0 = 0 := by simp [hc]
  have hcn : c n = 1 := by simp [hc, hn'.ne']
  rw [hc0, hcn, intervalIntegral.integral_of_le zero_le_one, integral_Ioc_eq_integral_Ioo]
    at hsum
  rw [← hsum, ← Fin.sum_univ_eq_sum_range (fun k => ∫ t in c k..c (k + 1),
    f (stepFn a t) (stepFn b t)), Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [intervalIntegral.integral_of_le (hle i),
    setIntegral_congr_fun measurableSet_Ioc (hpiece i), setIntegral_const, smul_eq_mul,
    measureReal_def, Real.volume_Ioc, ENNReal.toReal_ofReal (sub_nonneg.mpr (hle i))]
  simp only [hc]
  push_cast
  field_simp
  ring

/-- For antitone tuples the index sets `{i | s < a i}` are initial segments. -/
private lemma mem_iff_val_lt_card_of_initial {S : Finset (Fin n)}
    (hS : ∀ ⦃i j : Fin n⦄, j ≤ i → i ∈ S → j ∈ S) {i : Fin n} :
    i ∈ S ↔ (i : ℕ) < S.card := by
  constructor
  · intro hi
    have h1 : Finset.Iic i ⊆ S := fun j hj => hS (Finset.mem_Iic.mp hj) hi
    have h2 := Finset.card_le_card h1
    rw [Fin.card_Iic] at h2
    omega
  · intro hi
    by_contra hnot
    have h1 : S ⊆ Finset.Iio i := by
      intro j hj
      rw [Finset.mem_Iio]
      rcases lt_or_ge j i with h | h
      · exact h
      · exact absurd (hS h hj) hnot
    have h2 := Finset.card_le_card h1
    rw [Fin.card_Iio] at h2
    omega

/-- `h(t) > x` exactly when `#{i : aᵢ > x}/n ≥ ⌈nt⌉/n`, for antitone `a` and
`t ∈ (0,1)`. -/
lemma lt_stepFn_iff (hn : 0 < n) {a : Fin n → ℝ} (ha : Antitone a) {t : ℝ}
    (ht : t ∈ Ioo (0:ℝ) 1) (x : ℝ) :
    x < stepFn a t ↔
      (⌈(n:ℝ) * t⌉₊ : ℝ) / n ≤ ((univ.filter fun i => x < a i).card : ℝ) / n := by
  have hn' : (0:ℝ) < n := Nat.cast_pos.mpr hn
  have hk1 : 1 ≤ ⌈(n:ℝ) * t⌉₊ := Nat.one_le_iff_ne_zero.mpr
    (Nat.ceil_pos.mpr (mul_pos hn' ht.1)).ne'
  have hkn : ⌈(n:ℝ) * t⌉₊ ≤ n := Nat.ceil_le.mpr (by nlinarith [ht.2])
  have hi : ⌈(n:ℝ) * t⌉₊ - 1 < n := by omega
  have hS : ∀ ⦃i j : Fin n⦄, j ≤ i → i ∈ univ.filter (fun i => x < a i) →
      j ∈ univ.filter (fun i => x < a i) := by
    intro i j hij hi
    simp only [Finset.mem_filter_univ] at hi ⊢
    exact lt_of_lt_of_le hi (ha hij)
  rw [div_le_div_iff_of_pos_right hn', Nat.cast_le, stepFn, dif_pos hi]
  have := mem_iff_val_lt_card_of_initial hS (i := ⟨⌈(n:ℝ) * t⌉₊ - 1, hi⟩)
  simp only [Finset.mem_filter_univ] at this
  rw [this]
  omega

end StepFunction

/-! ### Step 1: pointwise convergence of the step functions

`F̄_n(x) = #{i < n : Xᵢ > x}/n` is the empirical survival function.  By the
strong law of large numbers at each of the countably many rational `x > 0`,
almost surely `F̄_n(x) → F̄(x)` for every rational `x > 0`.  On such an
outcome `h_n(t) → G(t)` at every `t ∈ (0,1)` at which `G` is continuous
(`tendsto_stepFn_of_continuousWithinAt`), hence for almost every `t`
(`ae_tendsto_stepFn`). -/

section Empirical

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- Strong law of large numbers for the indicator variables of one
threshold. -/
private lemma tendsto_empirical_survival {X : ℕ → Ω → ℝ}
    (hmeas : ∀ i, Measurable (X i)) (hindep : iIndepFun X P)
    (hident : ∀ i, Measure.map (X i) P = Measure.map (X 0) P) (s : ℝ) :
    ∀ᵐ ω ∂P, Tendsto
      (fun n => (((Finset.range n).filter (fun i => s < X i ω)).card : ℝ) / n)
      atTop (𝓝 ((Measure.map (X 0) P).real (Set.Ioi s))) := by
  set g : ℝ → ℝ := fun x => if s < x then 1 else 0 with hg_def
  have hg : Measurable g := Measurable.ite measurableSet_Ioi
    measurable_const measurable_const
  set Z : ℕ → Ω → ℝ := fun i ω => g (X i ω) with hZ_def
  have hZind : ∀ i, Z i = (X i ⁻¹' Set.Ioi s).indicator (1 : Ω → ℝ) := by
    intro i
    funext ω
    by_cases h : s < X i ω <;>
      simp [hZ_def, hg_def, h, Set.mem_preimage]
  have hint : Integrable (Z 0) P := by
    rw [hZind 0]
    rw [integrable_indicator_iff ((hmeas 0) measurableSet_Ioi)]
    exact integrableOn_const
  have hpairs : Pairwise (fun i j => IndepFun (Z i) (Z j) P) := by
    intro i j hij
    exact (hindep.indepFun hij).comp hg hg
  have hid : ∀ i, IdentDistrib (Z i) (Z 0) P P := by
    intro i
    have hXi : IdentDistrib (X i) (X 0) P P :=
      ⟨(hmeas i).aemeasurable, (hmeas 0).aemeasurable, hident i⟩
    exact hXi.comp hg
  have hlln := strong_law_ae_real Z hint hpairs hid
  have hmean : P[Z 0] = (Measure.map (X 0) P).real (Set.Ioi s) := by
    rw [hZind 0, integral_indicator_one ((hmeas 0) measurableSet_Ioi)]
    rw [measureReal_def, measureReal_def, Measure.map_apply (hmeas 0) measurableSet_Ioi]
  rw [hmean] at hlln
  filter_upwards [hlln] with ω hω
  convert hω using 2 with n
  rw [Finset.card_filter]
  push_cast
  rfl

end Empirical

section StepOne

private lemma card_filter_comp_equiv {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (σ : α ≃ β) (p : β → Prop)
    [DecidablePred p] :
    (Finset.univ.filter fun a => p (σ a)).card
      = (Finset.univ.filter p).card := by
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact Fintype.card_congr (σ.subtypeEquiv fun a => Iff.rfl)

/-- The number of entries of the sorted list exceeding `s` is the number
of samples exceeding `s`. -/
private lemma card_filter_sortDesc_lt (x : ℕ → ℝ) (n : ℕ) (s : ℝ) :
    (univ.filter fun j : Fin n => s < sortDesc (fun i : Fin n => x i) j).card
      = ((range n).filter fun i => s < x i).card := by
  rw [sortDesc_eq_comp_perm]
  refine (card_filter_comp_equiv (Fin.revPerm.trans (Tuple.sort (fun i : Fin n => x i)))
    (fun j : Fin n => s < x j)).trans ?_
  rw [Finset.card_filter, Finset.card_filter]
  exact Fin.sum_univ_eq_sum_range (fun i => if s < x i then 1 else 0) n

private lemma sortDesc_pos {x : ℕ → ℝ} (hx : ∀ i, 0 < x i) (n : ℕ) (j : Fin n) :
    0 < sortDesc (fun i : Fin n => x i) j := by
  rw [sortDesc_eq_comp_perm]; exact hx _

/-- On a sample path along which `F̄_n(x) → F̄(x)` for every rational `x > 0`,
`h_n(t) → G(t)` at every `t ∈ (0,1)` at which `G` is continuous. -/
lemma tendsto_stepFn_of_continuousWithinAt {γ : Measure ℝ} [IsProbabilityMeasure γ]
    (hγ : γ (Iic 0) = 0) {x : ℕ → ℝ}
    (hemp : ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < x i).card : ℝ) / n)
      atTop (𝓝 (survival γ q)))
    {t : ℝ} (ht : t ∈ Ioo (0:ℝ) 1) (hc : ContinuousWithinAt (quantile γ) (Ioo 0 1) t) :
    Tendsto (fun n => stepFn (sortDesc (fun i : Fin n => x i)) t) atTop
      (𝓝 (quantile γ t)) := by
  -- `h_n(t) > q` exactly when `F̄_n(q) ≥ ⌈nt⌉/n`
  have hiff : ∀ n, 0 < n → ∀ q : ℝ, q < stepFn (sortDesc (fun i : Fin n => x i)) t ↔
      (⌈(n:ℝ) * t⌉₊ : ℝ) / n ≤ (((range n).filter fun i => q < x i).card : ℝ) / n := by
    intro n hn q
    rw [lt_stepFn_iff hn (sortDesc_antitone _) ht, card_filter_sortDesc_lt]
  rw [tendsto_order]
  constructor
  · -- lower bound: a rational `q` with `G(t) - ε < q < G(t)` has `F̄(q) > t`
    intro b hb
    obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn (max_lt hb (quantile_pos hγ ht))
    have hq0 : (0:ℝ) < q := lt_of_le_of_lt (le_max_right _ _) hq1
    have hFq : t < survival γ q := (lt_quantile_iff hγ ht q).mp hq2
    -- `⌈nt⌉/n < t + 1/n`, and eventually `F̄_n(q) ≥ t + 1/n`
    have hlim : Tendsto (fun n : ℕ => t + 1 / (n:ℝ)) atTop (𝓝 t) := by
      have h1 : Tendsto (fun n : ℕ => 1 / (n:ℝ)) atTop (𝓝 0) :=
        tendsto_one_div_atTop_nhds_zero_nat
      simpa using (tendsto_const_nhds (x := t)).add h1
    have hev : ∀ᶠ n : ℕ in atTop, t + 1 / (n:ℝ) <
        (((range n).filter fun i => (q:ℝ) < x i).card : ℝ) / n :=
      (hlim.eventually (gt_mem_nhds (by linarith : t < (t + survival γ q) / 2))).and
        ((hemp q (by exact_mod_cast hq0)).eventually
          (lt_mem_nhds (by linarith : (t + survival γ q) / 2 < survival γ q)))
        |>.mono fun n h => h.1.trans h.2
    filter_upwards [hev, eventually_gt_atTop 0] with n hn hn0
    have hn' : (0:ℝ) < n := Nat.cast_pos.mpr hn0
    have hceil : (⌈(n:ℝ) * t⌉₊ : ℝ) / n < t + 1 / n := by
      rw [div_lt_iff₀ hn', add_mul, one_div_mul_cancel hn'.ne']
      linarith [Nat.ceil_lt_add_one (mul_pos hn' ht.1).le]
    exact lt_of_le_of_lt (le_max_left _ _)
      (lt_trans hq1 ((hiff n hn0 q).mpr (hceil.trans hn).le))
  · -- upper bound: continuity gives `t' < t` with `G(t') < b`; take a rational
    -- `q ∈ (G(t'), b)`, so that `F̄(q) ≤ t' < t`
    intro b hb
    obtain ⟨δ, hδ, hδG⟩ := Metric.continuousWithinAt_iff.mp hc (b - quantile γ t)
      (by linarith)
    set t' := max (t / 2) (t - δ / 2) with ht'
    have ht'lt : t' < t := max_lt (by linarith [ht.1]) (by linarith)
    have ht'mem : t' ∈ Ioo (0:ℝ) 1 :=
      ⟨lt_of_lt_of_le (by linarith [ht.1]) (le_max_left _ _), ht'lt.trans ht.2⟩
    have hdist : dist t' t < δ := by
      rw [Real.dist_eq, abs_sub_lt_iff]
      constructor
      · linarith
      · linarith [le_max_right (t / 2) (t - δ / 2)]
    have hGt' : quantile γ t' < b := by
      have := hδG ht'mem hdist
      rw [Real.dist_eq, abs_sub_lt_iff] at this
      linarith [this.1]
    obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn hGt'
    have hq0 : (0:ℝ) < q := (quantile_pos hγ ht'mem).trans hq1
    have hFq : survival γ q ≤ t' :=
      not_lt.mp fun h => (not_lt.mpr hq1.le) ((lt_quantile_iff hγ ht'mem q).mpr h)
    have hev : ∀ᶠ n : ℕ in atTop,
        (((range n).filter fun i => (q:ℝ) < x i).card : ℝ) / n < t :=
      (hemp q (by exact_mod_cast hq0)).eventually (gt_mem_nhds (by linarith))
    filter_upwards [hev, eventually_gt_atTop 0] with n hn hn0
    have hn' : (0:ℝ) < n := Nat.cast_pos.mpr hn0
    have hceil : t ≤ (⌈(n:ℝ) * t⌉₊ : ℝ) / n := by
      rw [le_div_iff₀ hn', mul_comm]; exact Nat.le_ceil _
    have hle : stepFn (sortDesc (fun i : Fin n => x i)) t ≤ q :=
      not_lt.mp fun h => absurd ((hiff n hn0 q).mp h) (not_le.mpr (hn.trans_le hceil))
    exact hle.trans_lt hq2

/-- On such a sample path, `h_n(t) → G(t)` for almost every `t ∈ (0,1)`: the
discontinuities of the monotone function `G` are countable. -/
lemma ae_tendsto_stepFn {γ : Measure ℝ} [IsProbabilityMeasure γ]
    (hγ : γ (Iic 0) = 0) {x : ℕ → ℝ}
    (hemp : ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < x i).card : ℝ) / n)
      atTop (𝓝 (survival γ q))) :
    ∀ᵐ t ∂vol01, Tendsto (fun n => stepFn (sortDesc (fun i : Fin n => x i)) t) atTop
      (𝓝 (quantile γ t)) := by
  have hcount := (quantile_antitoneOn (γ := γ) hγ).countable_not_continuousWithinAt
  have hnull : ∀ᵐ t ∂(volume : Measure ℝ),
      t ∉ {t | t ∈ Ioo (0:ℝ) 1 ∧ ¬ContinuousWithinAt (quantile γ) (Ioo 0 1) t} :=
    measure_eq_zero_iff_ae_notMem.mp (hcount.measure_zero volume)
  rw [ae_restrict_iff' measurableSet_Ioo]
  filter_upwards [hnull] with t hnot ht
  have hc : ContinuousWithinAt (quantile γ) (Ioo 0 1) t := by
    by_contra h; exact hnot ⟨ht, h⟩
  exact tendsto_stepFn_of_continuousWithinAt hγ hemp ht hc

end StepOne

/-! ### Steps 2 and 3: `h_n → G` in `L²(0,1)`

Second, `∫₀¹ h_n² = (1/n) ∑ᵢ Xᵢ² → m₂ = ∫₀¹ G²` by the strong law of large
numbers.  Third, `∫ (h_n - G)² = ∫ h_n² + ∫ G² - 2 ∫ h_n G`, and Fatou's
lemma applied to the nonnegative functions `h_n G`, which converge to `G²`
almost everywhere by the first step, gives `liminf ∫ h_n G ≥ ∫ G²`; hence
`limsup ∫ (h_n - G)² ≤ m₂ + m₂ - 2m₂ = 0`. -/

section StepThree

variable {γ : Measure ℝ} [IsProbabilityMeasure γ]

lemma memLp_quantile (hγ : γ (Iic 0) = 0) (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤) :
    MemLp (quantile γ) 2 vol01 := by
  have hae := (aemeasurable_quantile hγ).aestronglyMeasurable
  rw [memLp_two_iff_integrable_sq hae]
  refine ⟨hae.pow 2, ?_⟩
  rw [hasFiniteIntegral_iff_ofReal (ae_of_all _ fun t => sq_nonneg _),
    lintegral_quantile_sq hγ]
  exact hM.lt_top

lemma integral_quantile_sq (hγ : γ (Iic 0) = 0) :
    ∫ t, quantile γ t ^ 2 ∂vol01 = (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal := by
  rw [integral_eq_lintegral_of_nonneg_ae (ae_of_all _ fun t => sq_nonneg _)
    ((aemeasurable_quantile hγ).pow_const 2).aestronglyMeasurable, lintegral_quantile_sq hγ]

/-- If `A_n → m`, `A'_n → m`, and `liminf C_n ≥ m` (in `ℝ≥0∞`), then
`A_n + A'_n − 2 C_n → 0` provided it is nonnegative. -/
private lemma tendsto_of_liminf_ofReal {A A' C : ℕ → ℝ} {m : ℝ} (hm : 0 < m)
    (hA : Tendsto A atTop (𝓝 m)) (hA' : Tendsto A' atTop (𝓝 m))
    (hC : ENNReal.ofReal m ≤ atTop.liminf (fun n => ENNReal.ofReal (C n)))
    (hD : ∀ n, 0 ≤ A n + A' n - 2 * C n) :
    Tendsto (fun n => A n + A' n - 2 * C n) atTop (𝓝 0) := by
  rw [tendsto_order]
  constructor
  · intro a ha
    exact Eventually.of_forall fun n => lt_of_lt_of_le ha (hD n)
  · intro ε hε
    have hCe : ∀ᶠ n in atTop, m - ε / 4 < C n := by
      have hlt : ENNReal.ofReal (m - ε / 4)
          < atTop.liminf (fun n => ENNReal.ofReal (C n)) :=
        lt_of_lt_of_le ((ENNReal.ofReal_lt_ofReal_iff hm).mpr (by linarith)) hC
      filter_upwards [eventually_lt_of_lt_liminf hlt] with n hn
      exact (ENNReal.ofReal_lt_ofReal_iff'.mp hn).1
    have hAe := (tendsto_order.mp hA).2 (m + ε / 4) (by linarith)
    have hA'e := (tendsto_order.mp hA').2 (m + ε / 4) (by linarith)
    filter_upwards [hCe, hAe, hA'e] with n h1 h2 h3
    linarith

private lemma tendsto_sqrt_div_sqrt {D A : ℕ → ℝ} {m : ℝ} (hm : 0 < m)
    (hD : Tendsto D atTop (𝓝 0)) (hA : Tendsto A atTop (𝓝 m)) :
    Tendsto (fun n => √(D n) / √(A n)) atTop (𝓝 0) := by
  have h1 : Tendsto (fun n => √(D n)) atTop (𝓝 0) := by
    have := (Real.continuous_sqrt.tendsto 0).comp hD
    simpa using this
  have h2 : Tendsto (fun n => √(A n)) atTop (𝓝 (√m)) :=
    (Real.continuous_sqrt.tendsto m).comp hA
  have := h1.div h2 (Real.sqrt_pos.mpr hm).ne'
  simpa using this

/-- Second step: `∫₀¹ h_n² = (1/n) ∑ᵢ Xᵢ²`, which tends to `m₂`. -/
lemma tendsto_integral_stepFn_sq {x : ℕ → ℝ} {m : ℝ}
    (hsq : Tendsto (fun n => (∑ i ∈ range n, x i ^ 2) / n) atTop (𝓝 m)) :
    Tendsto (fun n => ∫ t, stepFn (sortDesc (fun i : Fin n => x i)) t ^ 2 ∂vol01) atTop
      (𝓝 m) := by
  refine hsq.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with n hn
  rw [integral_stepFn hn (fun a _ => a ^ 2) _ (sortDesc (fun i : Fin n => x i)),
    sum_comp_sortDesc (fun i : Fin n => x i) (fun y => y ^ 2),
    Fin.sum_univ_eq_sum_range (fun i => x i ^ 2) n]

/-- Third step: along a sample path as in the first and second steps,
`∫₀¹ (h_n - G)² → 0`. -/
lemma tendsto_integral_stepFn_sub_quantile_sq (hγ : γ (Iic 0) = 0)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤)
    (hMpos : 0 < ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ)
    {x : ℕ → ℝ} (hx : ∀ i, 0 < x i)
    (hemp : ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < x i).card : ℝ) / n)
      atTop (𝓝 (survival γ q)))
    (hsq : Tendsto (fun n => (∑ i ∈ range n, x i ^ 2) / n) atTop
      (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal)) :
    Tendsto (fun n => ∫ t, (stepFn (sortDesc (fun i : Fin n => x i)) t - quantile γ t) ^ 2
      ∂vol01) atTop (𝓝 0) := by
  set m := (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal with hm
  have hmpos : 0 < m := ENNReal.toReal_pos hMpos.ne' hM
  set h : ℕ → ℝ → ℝ := fun n => stepFn (sortDesc (fun i : Fin n => x i)) with hh
  set G := quantile γ with hGdef
  have hG2 : MemLp G 2 vol01 := memLp_quantile hγ hM
  have hh2 : ∀ n, MemLp (h n) 2 vol01 := fun n => memLp_stepFn _
  have hint : ∀ n, Integrable (fun t => h n t * G t) vol01 := fun n => (hh2 n).integrable_mul hG2
  -- `h_n G ≥ 0` on `(0,1)`
  have hnn : ∀ n, 0 ≤ᵐ[vol01] fun t => h n t * G t := by
    intro n
    filter_upwards [ae_restrict_mem measurableSet_Ioo] with t ht
    exact mul_nonneg (stepFn_nonneg (fun i => (sortDesc_pos hx n i).le) t)
      (quantile_pos hγ ht).le
  -- Fatou: `liminf ∫ h_n G ≥ ∫ G² = m₂`
  have hC : ENNReal.ofReal m ≤ atTop.liminf (fun n => ENNReal.ofReal (∫ t, h n t * G t ∂vol01)) := by
    have hlim := ae_tendsto_stepFn hγ hemp
    calc ENNReal.ofReal m = ∫⁻ t, ENNReal.ofReal (G t ^ 2) ∂vol01 := by
          rw [hm, ENNReal.ofReal_toReal hM, hGdef, lintegral_quantile_sq hγ]
      _ = ∫⁻ t, atTop.liminf (fun n => ENNReal.ofReal (h n t * G t)) ∂vol01 := by
          refine lintegral_congr_ae ?_
          filter_upwards [hlim] with t ht
          have := (ENNReal.tendsto_ofReal (ht.mul_const (G t)))
          rw [this.liminf_eq, sq]
      _ ≤ atTop.liminf (fun n => ∫⁻ t, ENNReal.ofReal (h n t * G t) ∂vol01) :=
          lintegral_liminf_le' fun n =>
            ((measurable_stepFn _).aemeasurable.mul (aemeasurable_quantile hγ)).ennreal_ofReal
      _ = atTop.liminf (fun n => ENNReal.ofReal (∫ t, h n t * G t ∂vol01)) := by
          congr 1
          funext n
          exact (ofReal_integral_eq_lintegral_ofReal (hint n) (hnn n)).symm
  -- expand `∫ (h_n - G)² = ∫ h_n² + ∫ G² - 2 ∫ h_n G`
  have hexp : ∀ n, ∫ t, (h n t - G t) ^ 2 ∂vol01
      = ∫ t, h n t ^ 2 ∂vol01 + m - 2 * ∫ t, h n t * G t ∂vol01 := by
    intro n
    have hpt : ∀ t, (h n t - G t) ^ 2 = h n t ^ 2 + G t ^ 2 - 2 * (h n t * G t) :=
      fun t => by ring
    simp_rw [hpt]
    rw [integral_sub (f := fun t => h n t ^ 2 + G t ^ 2)
        ((hh2 n).integrable_sq.add hG2.integrable_sq) ((hint n).const_mul 2),
      integral_add (hh2 n).integrable_sq hG2.integrable_sq, integral_const_mul, hGdef,
      integral_quantile_sq hγ]
  have hD0 : ∀ n, 0 ≤ ∫ t, h n t ^ 2 ∂vol01 + m - 2 * ∫ t, h n t * G t ∂vol01 := by
    intro n
    rw [← hexp n]
    exact integral_nonneg fun t => sq_nonneg _
  have := tendsto_of_liminf_ofReal hmpos (tendsto_integral_stepFn_sq hsq) tendsto_const_nhds
    hC hD0
  exact this.congr fun n => (hexp n).symm

end StepThree

/-! ### Assembly

With `g_n`, `g_n'` the step functions of the two samples,
`‖α_n - α_n'‖²/n = ∫₀¹ (g_n - g_n')² ≤ 2 ∫₀¹ (g_n - G)² + 2 ∫₀¹ (g_n' - G)² → 0`
and `‖α_n‖²/n = ∫₀¹ g_n² → m₂ > 0`. -/

section Assembly

/-- Deterministic assembly along one sample path. -/
private lemma tendsto_sortDesc_diff_div_of_seq {γ : Measure ℝ} [IsProbabilityMeasure γ]
    (hγ : γ (Iic 0) = 0)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤)
    (hMpos : 0 < ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ)
    {x x' : ℕ → ℝ} (hx : ∀ i, 0 < x i) (hx' : ∀ i, 0 < x' i)
    (hemp : ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < x i).card : ℝ) / n)
      atTop (𝓝 (survival γ q)))
    (hemp' : ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < x' i).card : ℝ) / n)
      atTop (𝓝 (survival γ q)))
    (hsq : Tendsto (fun n => (∑ i ∈ range n, x i ^ 2) / n) atTop
      (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal))
    (hsq' : Tendsto (fun n => (∑ i ∈ range n, x' i ^ 2) / n) atTop
      (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal)) :
    Tendsto (fun n =>
        √(∑ i : Fin n,
            (sortDesc (fun i : Fin n => x' i) i - sortDesc (fun i : Fin n => x i) i) ^ 2)
          / √(∑ i : Fin n, (sortDesc (fun i : Fin n => x i) i) ^ 2)) atTop (𝓝 0) := by
  have hmpos : 0 < (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal :=
    ENNReal.toReal_pos hMpos.ne' hM
  set g : ℕ → ℝ → ℝ := fun n => stepFn (sortDesc (fun i : Fin n => x i)) with hg
  set g' : ℕ → ℝ → ℝ := fun n => stepFn (sortDesc (fun i : Fin n => x' i)) with hg'
  have hL2 := tendsto_integral_stepFn_sub_quantile_sq hγ hM hMpos hx hemp hsq
  have hL2' := tendsto_integral_stepFn_sub_quantile_sq hγ hM hMpos hx' hemp' hsq'
  -- `‖α_n - α_n'‖²/n = ∫ (g_n - g_n')² ≤ 2 ∫ (g_n - G)² + 2 ∫ (g_n' - G)²`
  have hle : ∀ n, 0 < n → (∑ i : Fin n,
        (sortDesc (fun i : Fin n => x' i) i - sortDesc (fun i : Fin n => x i) i) ^ 2) / n
      ≤ 2 * ∫ t, (g n t - quantile γ t) ^ 2 ∂vol01
        + 2 * ∫ t, (g' n t - quantile γ t) ^ 2 ∂vol01 := by
    intro n hn
    have hG2 := memLp_quantile hγ hM
    rw [← integral_stepFn hn (fun a b => (b - a) ^ 2), ← integral_const_mul,
      ← integral_const_mul, ← integral_add]
    · refine integral_mono (((memLp_stepFn _).sub (memLp_stepFn _)).integrable_sq)
        ((((memLp_stepFn _).sub hG2).integrable_sq.const_mul 2).add
          (((memLp_stepFn _).sub hG2).integrable_sq.const_mul 2)) fun t => ?_
      simp only [hg, hg']
      nlinarith [sq_nonneg (stepFn (sortDesc (fun i : Fin n => x i)) t
        + stepFn (sortDesc (fun i : Fin n => x' i)) t - 2 * quantile γ t)]
    · exact ((memLp_stepFn _).sub hG2).integrable_sq.const_mul 2
    · exact ((memLp_stepFn _).sub hG2).integrable_sq.const_mul 2
  have hD : Tendsto (fun n : ℕ => (∑ i : Fin n,
        (sortDesc (fun i : Fin n => x' i) i - sortDesc (fun i : Fin n => x i) i) ^ 2) / n)
      atTop (𝓝 0) := by
    have hup := (hL2.const_mul 2).add (hL2'.const_mul 2)
    simp only [mul_zero, add_zero] at hup
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup
      (Eventually.of_forall fun n => by positivity) ?_
    filter_upwards [eventually_gt_atTop 0] with n hn
    exact hle n hn
  have hA : Tendsto (fun n : ℕ => (∑ i : Fin n, (sortDesc (fun i : Fin n => x i) i) ^ 2) / n)
      atTop (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal) :=
    hsq.congr fun n => by
      rw [sum_comp_sortDesc (fun i : Fin n => x i) (fun y => y ^ 2),
        Fin.sum_univ_eq_sum_range (fun i => x i ^ 2) n]
  refine (tendsto_sqrt_div_sqrt hmpos hD hA).congr' ?_
  filter_upwards [eventually_gt_atTop 0] with n hn
  rw [Real.sqrt_div' _ (Nat.cast_nonneg n), Real.sqrt_div' _ (Nat.cast_nonneg n),
    div_div_div_cancel_right₀ (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hn)).ne']

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-- Strong law of large numbers for the squares of the samples. -/
private lemma tendsto_mean_sq {X : ℕ → Ω → ℝ}
    (hmeas : ∀ i, Measurable (X i)) (hindep : iIndepFun X P)
    {γ : Measure ℝ} (hlaw : ∀ i, Measure.map (X i) P = γ)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤) :
    ∀ᵐ ω ∂P, Tendsto (fun n => (∑ i ∈ range n, (X i ω) ^ 2) / n) atTop
      (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal) := by
  have hsq : Measurable (fun y : ℝ => y ^ 2) := measurable_id.pow_const 2
  have hint : Integrable (fun ω => (X 0 ω) ^ 2) P := by
    have h : Integrable (fun y : ℝ => y ^ 2) γ := by
      refine ⟨hsq.aestronglyMeasurable, ?_⟩
      rw [hasFiniteIntegral_iff_enorm]
      refine lt_of_le_of_lt (le_of_eq (lintegral_congr fun y => ?_)) hM.lt_top
      exact Real.enorm_eq_ofReal (sq_nonneg y)
    rw [← hlaw 0] at h
    exact (integrable_map_measure hsq.aestronglyMeasurable (hmeas 0).aemeasurable).mp h
  have hpairs :
      Pairwise (fun i j => IndepFun (fun ω => (X i ω) ^ 2) (fun ω => (X j ω) ^ 2) P) :=
    fun i j hij => (hindep.indepFun hij).comp hsq hsq
  have hid : ∀ i, IdentDistrib (fun ω => (X i ω) ^ 2) (fun ω => (X 0 ω) ^ 2) P P := fun i =>
    (⟨(hmeas i).aemeasurable, (hmeas 0).aemeasurable, (hlaw i).trans (hlaw 0).symm⟩ :
      IdentDistrib (X i) (X 0) P P).comp hsq
  have hlln := strong_law_ae_real (fun i ω => (X i ω) ^ 2) hint hpairs hid
  have hmean : P[fun ω => (X 0 ω) ^ 2] = (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal := by
    have h1 : ∫ y, y ^ 2 ∂γ = (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal :=
      integral_eq_lintegral_of_nonneg_ae (ae_of_all _ fun y => sq_nonneg y)
        hsq.aestronglyMeasurable
    have h2 : ∫ y, y ^ 2 ∂γ = P[fun ω => (X 0 ω) ^ 2] := by
      rw [← hlaw 0]
      exact integral_map (hmeas 0).aemeasurable hsq.aestronglyMeasurable
    rw [← h2, h1]
  rw [hmean] at hlln
  exact hlln

/-- Samples from a law giving no mass to `(-∞, 0]` are almost surely
positive. -/
private lemma ae_pos_of_map_eq {X : Ω → ℝ} (hX : Measurable X) {γ : Measure ℝ}
    (hlaw : Measure.map X P = γ) (hγ : γ (Set.Iic 0) = 0) : ∀ᵐ ω ∂P, 0 < X ω := by
  have h : P (X ⁻¹' Set.Iic 0) = 0 := by
    rw [← Measure.map_apply hX measurableSet_Iic, hlaw, hγ]
  rw [ae_iff]
  convert h using 2
  ext ω; simp [not_lt]

variable [IsProbabilityMeasure P]

/-- Core statement: for two sequences of independent samples `X`, `X'`
from a common law `γ` on `(0,∞)` with finite positive second moment,
`‖α_n' − α_n‖ / ‖α_n‖ → 0` almost surely, where `α_n`, `α_n'` are the
decreasing sortings of the first `n` samples. -/
private theorem tendsto_sortDesc_diff_div_of_iid {X X' : ℕ → Ω → ℝ}
    (hX : ∀ i, Measurable (X i)) (hX' : ∀ i, Measurable (X' i))
    (hXi : iIndepFun X P) (hX'i : iIndepFun X' P)
    {γ : Measure ℝ} (hXl : ∀ i, Measure.map (X i) P = γ)
    (hX'l : ∀ i, Measure.map (X' i) P = γ) (hγ0 : γ (Set.Iic 0) = 0)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤)
    (hMpos : 0 < ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ) :
    ∀ᵐ ω ∂P, Tendsto (fun n =>
        √(∑ i : Fin n, (sortDesc (fun i => X' i ω) i - sortDesc (fun i => X i ω) i) ^ 2)
          / √(∑ i : Fin n, (sortDesc (fun i => X i ω) i) ^ 2)) atTop (𝓝 0) := by
  haveI : IsProbabilityMeasure γ := by
    rw [← hXl 0]; exact Measure.isProbabilityMeasure_map (hX 0).aemeasurable
  have hpos : ∀ᵐ ω ∂P, ∀ i, 0 < X i ω :=
    ae_all_iff.mpr fun i => ae_pos_of_map_eq (hX i) (hXl i) hγ0
  have hpos' : ∀ᵐ ω ∂P, ∀ i, 0 < X' i ω :=
    ae_all_iff.mpr fun i => ae_pos_of_map_eq (hX' i) (hX'l i) hγ0
  -- the strong law at each of the countably many rational thresholds
  have hemp : ∀ᵐ ω ∂P, ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < X i ω).card : ℝ) / n)
      atTop (𝓝 (survival γ q)) := by
    rw [ae_all_iff]
    intro q
    have h := tendsto_empirical_survival hX hXi (fun i => by rw [hXl i, hXl 0]) (q:ℝ)
    rw [hXl 0] at h
    exact h.mono fun ω hω _ => hω
  have hemp' : ∀ᵐ ω ∂P, ∀ q : ℚ, 0 < q → Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < X' i ω).card : ℝ) / n)
      atTop (𝓝 (survival γ q)) := by
    rw [ae_all_iff]
    intro q
    have h := tendsto_empirical_survival hX' hX'i (fun i => by rw [hX'l i, hX'l 0]) (q:ℝ)
    rw [hX'l 0] at h
    exact h.mono fun ω hω _ => hω
  have hsq := tendsto_mean_sq hX hXi hXl hM
  have hsq' := tendsto_mean_sq hX' hX'i hX'l hM
  filter_upwards [hpos, hpos', hemp, hemp', hsq, hsq'] with ω hpos hpos' hemp hemp' hsq hsq'
  exact tendsto_sortDesc_diff_div_of_seq (x := fun i => X i ω) (x' := fun i => X' i ω)
    hγ0 hM hMpos hpos hpos' hemp hemp' hsq hsq'

end Assembly

/-! ### The lemma -/

section Main

variable {n : ℕ}

/-- A rearrangement of `v` that is antitone must be `sortDesc v`. -/
private lemma sortDesc_eq_of_antitone {v w : Fin n → ℝ}
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

/-- Scaling by a nonnegative constant commutes with `sortDesc`. -/
private lemma sortDesc_const_mul {c : ℝ} (hc : 0 ≤ c) (v : Fin n → ℝ) :
    sortDesc (fun i => c * v i) = fun j => c * sortDesc v j := by
  have hw : Antitone (fun j => c * sortDesc v j) :=
    fun a b hab => mul_le_mul_of_nonneg_left (sortDesc_antitone v hab) hc
  refine sortDesc_eq_of_antitone hw (Fin.revPerm.trans (Tuple.sort v)).symm ?_
  funext i
  simp only [Function.comp, sortDesc_eq_comp_perm, Equiv.apply_symm_apply]

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- Lemma (lem:proportional-lobes).

Let `X` be a positive random variable with finite, nonzero second
moment; fix scales `u, v > 0` and set `ρ = u/v`.  For each `n`, let
`α_n` be the decreasing sorting of `vX_1, …, vX_n` and `β_n` the
decreasing sorting of `uX'_1, …, uX'_n`, where the `X_i` and `X'_i`
are independent copies of `X`.  Then `‖β_n − ρ α_n‖ / ‖α_n‖ → 0`
almost surely.

Encoding: `γ` is the law of `X`, a law on `ℝ` with `γ (Iic 0) = 0`
(`X` is positive) and second moment `∫⁻ y, ofReal (y²) ∂γ` finite and
nonzero; `X i` and `X' i` are the independent copies, each with law
`γ`, and `hindep` says that all of `X 0, X 1, …, X' 0, X' 1, …` are
independent; `α_n = sortDesc (fun i : Fin n => v * X i ω)`,
`β_n = sortDesc (fun i : Fin n => u * X' i ω)`, `ρ = u/v`, and the
norms are written out as `√(∑ …²)`. -/
theorem proportional_lobes {u v : ℝ} (hu : 0 < u) (hv : 0 < v)
    {X X' : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) (hX' : ∀ i, Measurable (X' i))
    (hindep : iIndepFun (Sum.elim X X') P)
    {γ : Measure ℝ} (hXlaw : ∀ i, Measure.map (X i) P = γ)
    (hX'law : ∀ i, Measure.map (X' i) P = γ)
    (hγpos : γ (Set.Iic 0) = 0)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤)
    (hMpos : 0 < ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ) :
    ∀ᵐ ω ∂P, Tendsto (fun n =>
        √(∑ i : Fin n,
            (sortDesc (fun i => u * X' i ω) i
              - (u / v) * sortDesc (fun i => v * X i ω) i) ^ 2)
          / √(∑ i : Fin n, (sortDesc (fun i => v * X i ω) i) ^ 2))
      atTop (𝓝 0) := by
  have hXi : iIndepFun X P :=
    iIndepFun.precomp (f := Sum.elim X X') (g := Sum.inl) Sum.inl_injective hindep
  have hX'i : iIndepFun X' P :=
    iIndepFun.precomp (f := Sum.elim X X') (g := Sum.inr) Sum.inr_injective hindep
  have hmain := tendsto_sortDesc_diff_div_of_iid hX hX' hXi hX'i hXlaw hX'law
    hγpos hM hMpos
  filter_upwards [hmain] with ω hω
  have hkey : ∀ n : ℕ,
      √(∑ i : Fin n,
          (sortDesc (fun i => u * X' i ω) i
            - (u / v) * sortDesc (fun i => v * X i ω) i) ^ 2)
        / √(∑ i : Fin n, (sortDesc (fun i => v * X i ω) i) ^ 2)
      = (u / v) * (√(∑ i : Fin n,
            (sortDesc (fun i => X' i ω) i - sortDesc (fun i => X i ω) i) ^ 2)
          / √(∑ i : Fin n, (sortDesc (fun i => X i ω) i) ^ 2)) := by
    intro n
    have e1 : sortDesc (fun i : Fin n => u * X' i ω)
        = fun j => u * sortDesc (fun i : Fin n => X' i ω) j :=
      sortDesc_const_mul hu.le _
    have e2 : sortDesc (fun i : Fin n => v * X i ω)
        = fun j => v * sortDesc (fun i : Fin n => X i ω) j :=
      sortDesc_const_mul hv.le _
    rw [e1, e2]
    have hnum : ∑ i : Fin n,
          (u * sortDesc (fun i : Fin n => X' i ω) i
            - (u / v) * (v * sortDesc (fun i : Fin n => X i ω) i)) ^ 2
        = u ^ 2 * ∑ i : Fin n,
            (sortDesc (fun i : Fin n => X' i ω) i
              - sortDesc (fun i : Fin n => X i ω) i) ^ 2 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [← mul_assoc (u / v), div_mul_cancel₀ u hv.ne']
      ring
    have hden : ∑ i : Fin n, (v * sortDesc (fun i : Fin n => X i ω) i) ^ 2
        = v ^ 2 * ∑ i : Fin n, (sortDesc (fun i : Fin n => X i ω) i) ^ 2 := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [hnum, hden, Real.sqrt_mul (sq_nonneg u), Real.sqrt_mul (sq_nonneg v),
      Real.sqrt_sq hu.le, Real.sqrt_sq hv.le, mul_div_mul_comm]
  have h := hω.const_mul (u / v)
  rw [mul_zero] at h
  exact h.congr fun n => (hkey n).symm

end Main

end Appendices
