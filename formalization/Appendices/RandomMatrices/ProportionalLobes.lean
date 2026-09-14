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
* norms are written out as `√(∑ …²)`.

The instance used in the proof of Theorem (thm:accumulation) of the
main text — `X` with law `γ_{ν,k}`, `v = 1`, `u = ρ = |μ|/ν` — is
recorded as `proportional_lobes_gamma`.  The paper's distribution
`γ_{ν,k}` (for `ν > 0`, `k > 1`) has density
`Γ(k)⁻¹ ((k−1)/ν)^k x^{k−1} e^{−(k−1)x/ν}` on `x > 0`: it is the Gamma
distribution with shape `k` and rate `(k−1)/ν`, i.e. mathlib's
`ProbabilityTheory.gammaMeasure k ((k−1)/ν)`.

The proof follows the paper's three steps (pointwise convergence of the
sorted lists, the strong law of large numbers for the second moment,
and Fatou's lemma), with the first and third steps combined: a
layer-cake identity writes `⟨α_n, α_n'⟩ / n` as an integral over
`(0,∞)²` of the minimum of two empirical survival counts, whose liminf
Fatou's lemma bounds below by the second moment of `γ_{ν,k}`.
-/
import Appendices.Common
import Appendices.Weights
import Appendices.RandomMatrices.EnergyIdentities

open MeasureTheory ProbabilityTheory Filter Set Finset
open scoped ENNReal NNReal Topology

namespace Appendices

/-! ### Counting above thresholds in an antitone tuple

For antitone tuples the index sets `{i | s < a i}` are initial
segments, so the count of indices where two antitone tuples both
exceed thresholds is the minimum of the two separate counts.  This is
the combinatorial heart of the layer-cake identity for
`∑ i, a i * b i` below. -/

section Counting

variable {n : ℕ}

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

private lemma card_eq_of_val_lt_iff {c d : ℕ} (hc : c ≤ n) (hd : d ≤ n)
    (h : ∀ i : Fin n, (i : ℕ) < c ↔ (i : ℕ) < d) : c = d := by
  by_contra hne
  rcases Nat.lt_or_ge c d with hlt | hge
  · exact absurd ((h ⟨c, lt_of_lt_of_le hlt hd⟩).mpr hlt) (by simp)
  · have hlt : d < c := lt_of_le_of_ne hge (Ne.symm hne)
    exact absurd ((h ⟨d, lt_of_lt_of_le hlt hc⟩).mp hlt) (by simp)

/-- For antitone tuples `a`, `b`, the number of indices where `s < a i`
and `t < b i` simultaneously is the minimum of the two counts. -/
private lemma card_filter_and_eq_min {a b : Fin n → ℝ}
    (ha : Antitone a) (hb : Antitone b) (s t : ℝ) :
    (univ.filter fun i => s < a i ∧ t < b i).card
      = min (univ.filter fun i => s < a i).card
          (univ.filter fun i => t < b i).card := by
  have hA : ∀ ⦃i j : Fin n⦄, j ≤ i → i ∈ univ.filter (fun i => s < a i) →
      j ∈ univ.filter (fun i => s < a i) := by
    intro i j hij hi
    simp only [Finset.mem_filter_univ] at hi ⊢
    exact lt_of_lt_of_le hi (ha hij)
  have hB : ∀ ⦃i j : Fin n⦄, j ≤ i → i ∈ univ.filter (fun i => t < b i) →
      j ∈ univ.filter (fun i => t < b i) := by
    intro i j hij hi
    simp only [Finset.mem_filter_univ] at hi ⊢
    exact lt_of_lt_of_le hi (hb hij)
  have hAB : ∀ ⦃i j : Fin n⦄, j ≤ i →
      i ∈ univ.filter (fun i => s < a i ∧ t < b i) →
      j ∈ univ.filter (fun i => s < a i ∧ t < b i) := by
    intro i j hij hi
    simp only [Finset.mem_filter_univ] at hi ⊢
    exact ⟨lt_of_lt_of_le hi.1 (ha hij), lt_of_lt_of_le hi.2 (hb hij)⟩
  have hle : ∀ (p : Fin n → Prop) (_ : DecidablePred p),
      (univ.filter p).card ≤ n := fun p _ =>
    le_trans (Finset.card_filter_le _ _)
      (le_of_eq (Finset.card_univ.trans (Fintype.card_fin n)))
  refine card_eq_of_val_lt_iff (hle _ _)
    (le_trans (Nat.min_le_left _ _) (hle _ _)) fun i => ?_
  rw [← mem_iff_val_lt_card_of_initial hAB, Nat.lt_min,
    ← mem_iff_val_lt_card_of_initial hA, ← mem_iff_val_lt_card_of_initial hB]
  simp only [Finset.mem_filter_univ]

end Counting

/-! ### A layer-cake identity for `∑ i, a i * b i`

For nonnegative antitone tuples `a`, `b`,

    ∑ i, a i * b i = ∫₀^∞ ∫₀^∞ min #{i | s < a i} #{i | t < b i} dt ds.

This identity, its counterpart for the second moment of a law on
`[0,∞)`, and Fatou's lemma drive the almost-sure convergence below. -/

section LayerCake

variable {n : ℕ}

private lemma lintegral_ite_lt (c : ℝ) :
    ∫⁻ s in Ioi (0:ℝ), (if s < c then (1:ℝ≥0∞) else 0) = ENNReal.ofReal c := by
  have h : ∀ s : ℝ, (if s < c then (1:ℝ≥0∞) else 0)
      = (Iio c).indicator (fun _ => (1:ℝ≥0∞)) s := by
    intro s; rw [Set.indicator_apply]; simp
  simp_rw [h]
  rw [lintegral_indicator measurableSet_Iio, setLIntegral_one,
    Measure.restrict_apply measurableSet_Iio, Set.Iio_inter_Ioi,
    Real.volume_Ioo, sub_zero]

private lemma measurable_ite_lt (c : ℝ) :
    Measurable (fun s : ℝ => if s < c then (1:ℝ≥0∞) else 0) := by
  apply Measurable.ite measurableSet_Iio <;> exact measurable_const

/-- Layer-cake identity for a finite sum of products of nonnegative
entries. -/
private lemma ofReal_sum_mul_eq_lintegral_card {a b : Fin n → ℝ}
    (ha0 : ∀ i, 0 ≤ a i) (hb0 : ∀ i, 0 ≤ b i) :
    ENNReal.ofReal (∑ i, a i * b i)
      = ∫⁻ s in Ioi (0:ℝ), ∫⁻ t in Ioi (0:ℝ),
          ((univ.filter fun i => s < a i ∧ t < b i).card : ℝ≥0∞) := by
  have hcard : ∀ s t : ℝ,
      ((univ.filter fun i => s < a i ∧ t < b i).card : ℝ≥0∞)
        = ∑ i, (if s < a i then (1:ℝ≥0∞) else 0) * (if t < b i then (1:ℝ≥0∞) else 0) := by
    intro s t
    rw [Finset.card_filter]
    push_cast
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h1 : s < a i <;> by_cases h2 : t < b i <;> simp [h1, h2]
  simp_rw [hcard]
  rw [lintegral_congr_ae (ae_of_all _ fun s => lintegral_finset_sum univ
    (fun i _ => ((measurable_ite_lt (b i)).const_mul _)))]
  have hinner : ∀ (i : Fin n) (s : ℝ),
      (∫⁻ t in Ioi (0:ℝ), (if s < a i then (1:ℝ≥0∞) else 0)
          * (if t < b i then (1:ℝ≥0∞) else 0))
        = (if s < a i then (1:ℝ≥0∞) else 0) * ENNReal.ofReal (b i) := by
    intro i s
    rw [lintegral_const_mul _ (measurable_ite_lt (b i)), lintegral_ite_lt]
  simp_rw [hinner]
  rw [lintegral_finset_sum univ (fun i _ => (measurable_ite_lt (a i)).mul_const _)]
  have houter : ∀ i : Fin n,
      (∫⁻ s in Ioi (0:ℝ), (if s < a i then (1:ℝ≥0∞) else 0) * ENNReal.ofReal (b i))
        = ENNReal.ofReal (a i) * ENNReal.ofReal (b i) := by
    intro i
    rw [lintegral_mul_const _ (measurable_ite_lt (a i)), lintegral_ite_lt]
  simp_rw [houter]
  rw [ENNReal.ofReal_sum_of_nonneg (fun i _ => mul_nonneg (ha0 i) (hb0 i))]
  exact Finset.sum_congr rfl fun i _ => ENNReal.ofReal_mul (ha0 i)

end LayerCake

/-! ### Facts about the gamma distribution

`γ_{ν,k}` is `gammaMeasure k ((k-1)/ν)`.  We need: it gives no mass to
`(-∞, 0]`; the scaling relation `γ_{|μ|,k} = (ρ • ·)_* γ_{ν,k}` with
`ρ = |μ|/ν`; and that its second moment is finite and positive. -/

section GammaFacts

open ProbabilityTheory

private lemma gammaMeasure_singleton (a r x : ℝ) : gammaMeasure a r {x} = 0 := by
  rw [gammaMeasure, withDensity_apply _ (measurableSet_singleton x)]
  rw [Measure.restrict_eq_zero.mpr (measure_singleton x), lintegral_zero_measure]

private lemma gammaMeasure_Iic_zero (a r : ℝ) : gammaMeasure a r (Set.Iic 0) = 0 := by
  have h : (Set.Iic (0:ℝ)) = Set.Iio 0 ∪ {0} := by
    ext x; simp [le_iff_lt_or_eq]
  rw [h]
  refine le_antisymm (le_trans (measure_union_le _ _) ?_) (zero_le _)
  rw [gammaMeasure_singleton]
  rw [gammaMeasure, withDensity_apply _ measurableSet_Iio]
  rw [setLIntegral_congr_fun measurableSet_Iio
    (fun x (hx : x < 0) => gammaPDF_of_neg hx), lintegral_zero, zero_add]

private lemma ofReal_inv_mul_gammaPDF_div {a r ρ : ℝ} (ha : 0 < a) (hr : 0 < r)
    (hρ : 0 < ρ) (y : ℝ) :
    ENNReal.ofReal ρ⁻¹ * gammaPDF a r (y / ρ) = gammaPDF a (r / ρ) y := by
  rcases le_or_gt 0 y with hy | hy
  · rw [gammaPDF_of_nonneg (by positivity), gammaPDF_of_nonneg hy,
      ← ENNReal.ofReal_mul (by positivity)]
    congr 1
    have harg : r * (y / ρ) = r / ρ * y := by field_simp
    rw [harg, Real.div_rpow hy hρ.le, Real.div_rpow hr.le hρ.le,
      Real.rpow_sub hρ, Real.rpow_one]
    have hΓ : Real.Gamma a ≠ 0 := (Real.Gamma_pos_of_pos ha).ne'
    have hρa : (0:ℝ) < ρ ^ a := Real.rpow_pos_of_pos hρ a
    field_simp
  · rw [gammaPDF_of_neg hy, gammaPDF_of_neg (div_neg_of_neg_of_pos hy hρ), mul_zero]

/-- Scaling: pushing `gammaMeasure a r` forward along `x ↦ ρx` (`ρ > 0`)
gives `gammaMeasure a (r/ρ)`. -/
private lemma map_mul_left_gammaMeasure {a r ρ : ℝ} (ha : 0 < a) (hr : 0 < r)
    (hρ : 0 < ρ) :
    Measure.map (fun x => ρ * x) (gammaMeasure a r) = gammaMeasure a (r / ρ) := by
  have hmeas : Measurable (fun x : ℝ => ρ * x) := measurable_const_mul ρ
  refine Measure.ext fun A hA => ?_
  rw [Measure.map_apply hmeas hA, gammaMeasure,
    withDensity_apply _ (hA.preimage hmeas), gammaMeasure, withDensity_apply _ hA]
  have hfun : Measurable (A.indicator (fun y => gammaPDF a r (y / ρ))) :=
    Measurable.indicator (Measurable.ennreal_ofReal
      ((measurable_gammaPDFReal a r).comp (measurable_id.div_const ρ))) hA
  have h1 : (∫⁻ x in (fun x => ρ * x) ⁻¹' A, gammaPDF a r x)
      = ∫⁻ x, (A.indicator (fun y => gammaPDF a r (y / ρ))) (ρ * x) := by
    rw [← lintegral_indicator (hA.preimage hmeas)]
    refine lintegral_congr fun x => ?_
    by_cases hx : ρ * x ∈ A
    · rw [Set.indicator_of_mem hx, Set.indicator_of_mem (by exact hx),
        mul_div_cancel_left₀ _ hρ.ne']
    · rw [Set.indicator_of_notMem hx, Set.indicator_of_notMem (by exact hx)]
  have h2 : (∫⁻ x, (A.indicator (fun y => gammaPDF a r (y / ρ))) (ρ * x))
      = ∫⁻ y, (A.indicator (fun y => gammaPDF a r (y / ρ))) y
          ∂(Measure.map (fun x => ρ * x) volume) :=
    (lintegral_map hfun hmeas).symm
  rw [h1, h2, Real.map_volume_mul_left hρ.ne', abs_inv, abs_of_pos hρ,
    lintegral_smul_measure, lintegral_indicator hA, smul_eq_mul,
    ← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  exact setLIntegral_congr_fun hA
    (fun y _ => ofReal_inv_mul_gammaPDF_div ha hr hρ y)

private lemma measurable_ofReal_sq : Measurable (fun x : ℝ => ENNReal.ofReal (x ^ 2)) :=
  Measurable.ennreal_ofReal (measurable_id.pow_const 2)

private lemma lintegral_sq_gammaMeasure_one_lt_top {a : ℝ} (ha : 0 < a) :
    (∫⁻ x, ENNReal.ofReal (x ^ 2) ∂(gammaMeasure a 1)) < ⊤ := by
  have hpdf : Measurable (gammaPDF a 1) :=
    Measurable.ennreal_ofReal (measurable_gammaPDFReal a 1)
  rw [gammaMeasure, lintegral_withDensity_eq_lintegral_mul _ hpdf measurable_ofReal_sq]
  rw [← lintegral_add_compl _ (measurableSet_Ioi (a := (0:ℝ))), compl_Ioi]
  have hIic : (∫⁻ x in Set.Iic 0, (gammaPDF a 1 * fun x => ENNReal.ofReal (x ^ 2)) x) = 0 := by
    rw [setLIntegral_congr_fun measurableSet_Iic (g := fun _ => 0), lintegral_zero]
    intro x hx
    rcases lt_or_eq_of_le (Set.mem_Iic.mp hx) with h | h
    · simp [Pi.mul_apply, gammaPDF_of_neg h]
    · simp [Pi.mul_apply, ← h]
  have hIoi : (∫⁻ x in Set.Ioi 0, (gammaPDF a 1 * fun x => ENNReal.ofReal (x ^ 2)) x)
      = ENNReal.ofReal (1 / Real.Gamma a)
        * ∫⁻ x in Set.Ioi 0, ENNReal.ofReal (Real.exp (-x) * x ^ (a + 2 - 1)) := by
    rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    refine setLIntegral_congr_fun measurableSet_Ioi (fun x hx => ?_)
    have hx : (0:ℝ) < x := hx
    rw [Pi.mul_apply, gammaPDF_of_nonneg hx.le,
      ← ENNReal.ofReal_mul (by positivity),
      ← ENNReal.ofReal_mul (by positivity)]
    congr 1
    rw [Real.one_rpow, one_mul]
    rw [show a + 2 - 1 = (a - 1) + 2 by ring, Real.rpow_add hx, Real.rpow_two]
    ring
  rw [hIic, hIoi, add_zero]
  refine ENNReal.mul_lt_top ENNReal.ofReal_lt_top ?_
  have hint := Real.GammaIntegral_convergent (s := a + 2) (by linarith)
  have h2 := hint.2
  rw [hasFiniteIntegral_iff_enorm] at h2
  refine lt_of_le_of_lt (le_of_eq ?_) h2
  refine setLIntegral_congr_fun measurableSet_Ioi (fun x hx => ?_)
  have hx' : (0:ℝ) < x := hx
  rw [Real.enorm_eq_ofReal (by positivity)]

private lemma lintegral_sq_gammaMeasure_lt_top {a r : ℝ} (ha : 0 < a) (hr : 0 < r) :
    (∫⁻ x, ENNReal.ofReal (x ^ 2) ∂(gammaMeasure a r)) < ⊤ := by
  have hmap : gammaMeasure a r = Measure.map (fun x => r⁻¹ * x) (gammaMeasure a 1) := by
    rw [map_mul_left_gammaMeasure ha one_pos (inv_pos.mpr hr), one_div, inv_inv]
  rw [hmap, lintegral_map measurable_ofReal_sq (measurable_const_mul r⁻¹)]
  have h : ∀ x : ℝ, ENNReal.ofReal ((r⁻¹ * x) ^ 2)
      = ENNReal.ofReal (r⁻¹ ^ 2) * ENNReal.ofReal (x ^ 2) := by
    intro x; rw [mul_pow, ENNReal.ofReal_mul (by positivity)]
  simp_rw [h]
  rw [lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top (lintegral_sq_gammaMeasure_one_lt_top ha)

private lemma lintegral_sq_gammaMeasure_pos {a r : ℝ} (ha : 0 < a) (hr : 0 < r) :
    0 < ∫⁻ x, ENNReal.ofReal (x ^ 2) ∂(gammaMeasure a r) := by
  have hprob := isProbabilityMeasure_gammaMeasure ha hr
  rw [pos_iff_ne_zero]
  intro h0
  rw [lintegral_eq_zero_iff measurable_ofReal_sq] at h0
  have hae : ∀ᵐ x ∂(gammaMeasure a r), x = 0 := by
    filter_upwards [h0] with x hx
    have hx' : ENNReal.ofReal (x ^ 2) = 0 := hx
    rw [ENNReal.ofReal_eq_zero] at hx'
    nlinarith [sq_nonneg x]
  have h1 : gammaMeasure a r ({(0:ℝ)}ᶜ) = 0 := by
    have hset : {x : ℝ | ¬ x = 0} = {(0:ℝ)}ᶜ := by ext x; simp
    exact hset ▸ ae_iff.mp hae
  have h3 := measure_union_le (μ := gammaMeasure a r) ({(0:ℝ)}) ({(0:ℝ)}ᶜ)
  rw [Set.union_compl_self, measure_univ, gammaMeasure_singleton, h1] at h3
  simp at h3

/-- Samples from a gamma law are almost surely positive. -/
private lemma ae_pos_of_map_eq_gammaMeasure {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {X : Ω → ℝ} (hX : Measurable X) {a r : ℝ}
    (hlaw : Measure.map X P = gammaMeasure a r) : ∀ᵐ ω ∂P, 0 < X ω := by
  have h : P (X ⁻¹' Set.Iic 0) = 0 := by
    rw [← Measure.map_apply hX measurableSet_Iic, hlaw, gammaMeasure_Iic_zero]
  rw [ae_iff]
  convert h using 2
  ext ω; simp [not_lt]

end GammaFacts

/-! ### Empirical survival counts

`(range n).filter (s < X · ω) |>.card / n` is the fraction of the
first `n` samples exceeding `s`.  The strong law of large numbers
gives its almost-sure convergence to `P(X₀ > s)` for each fixed `s`;
monotonicity in `s` and density of the rationals upgrade this to the
liminf lower bound at every real threshold simultaneously. -/

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

/-- Deterministic upgrade: if the empirical survival fractions of a
sample path converge at every rational threshold, then at every real
threshold `s` the liminf (in `ℝ≥0∞`) dominates `γ (Ioi s)`. -/
private lemma le_liminf_empirical_of_rat {x : ℕ → ℝ} {γ : Measure ℝ}
    [IsFiniteMeasure γ]
    (h : ∀ q : ℚ, Tendsto
      (fun n => (((Finset.range n).filter (fun i => (q:ℝ) < x i)).card : ℝ) / n)
      atTop (𝓝 (γ.real (Set.Ioi (q:ℝ))))) (s : ℝ) :
    γ (Set.Ioi s) ≤ atTop.liminf
      (fun n => ENNReal.ofReal
        ((((Finset.range n).filter (fun i => s < x i)).card : ℝ) / n)) := by
  -- reduce to rational thresholds above `s`
  have key : ∀ q : ℚ, s < (q:ℝ) → γ (Set.Ioi (q:ℝ)) ≤ atTop.liminf
      (fun n => ENNReal.ofReal
        ((((Finset.range n).filter (fun i => s < x i)).card : ℝ) / n)) := by
    intro q hq
    have hmono : ∀ n, ENNReal.ofReal
        ((((Finset.range n).filter (fun i => (q:ℝ) < x i)).card : ℝ) / n)
        ≤ ENNReal.ofReal
        ((((Finset.range n).filter (fun i => s < x i)).card : ℝ) / n) := by
      intro n
      apply ENNReal.ofReal_le_ofReal
      apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg n)
      exact_mod_cast Finset.card_le_card
        (Finset.monotone_filter_right _ fun i _ hi => lt_trans hq hi)
    have hlim : Tendsto (fun n => ENNReal.ofReal
        ((((Finset.range n).filter (fun i => (q:ℝ) < x i)).card : ℝ) / n))
        atTop (𝓝 (γ (Set.Ioi (q:ℝ)))) := by
      have := ENNReal.tendsto_ofReal (h q)
      rwa [measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)] at this
    rw [← hlim.liminf_eq]
    exact liminf_le_liminf (Eventually.of_forall hmono)
  -- continuity from below along the rational thresholds above `s`
  have hU : Set.Ioi s = ⋃ q : {q : ℚ // s < (q:ℝ)}, Set.Ioi ((q.1 : ℚ) : ℝ) := by
    ext y
    simp only [Set.mem_Ioi, Set.mem_iUnion]
    constructor
    · intro hy
      obtain ⟨q, hq1, hq2⟩ := exists_rat_btwn hy
      exact ⟨⟨q, hq1⟩, hq2⟩
    · rintro ⟨q, hq⟩
      exact lt_trans q.2 hq
  have hdir : Directed (· ⊆ ·)
      (fun q : {q : ℚ // s < (q:ℝ)} => Set.Ioi ((q.1 : ℚ) : ℝ)) := by
    intro q₁ q₂
    refine ⟨⟨min q₁.1 q₂.1, ?_⟩, Set.Ioi_subset_Ioi ?_, Set.Ioi_subset_Ioi ?_⟩
    · rw [Rat.cast_min]; exact lt_min q₁.2 q₂.2
    · rw [Rat.cast_min]; exact min_le_left _ _
    · rw [Rat.cast_min]; exact min_le_right _ _
  rw [hU, hdir.measure_iUnion]
  exact iSup_le fun q => key q.1 q.2

end Empirical

/-! ### Fatou's lemma and the second moment

Write `vol2` for Lebesgue measure on `(0,∞) × (0,∞)`.  The layer-cake
identity expresses `∑ i, a i * b i` as the `vol2`-integral of
`min #{i | s < a i} #{i | t < b i}`; the second moment of a law `γ`
on `(0,∞)` is the `vol2`-integral of `min (γ (Ioi s)) (γ (Ioi t))`.
Fatou's lemma then bounds the liminf of the normalized inner products
`⟨α_n, α_n'⟩ / n` from below by the second moment. -/

section Fatou

variable {n : ℕ}

/-- Lebesgue measure on `(0,∞) × (0,∞)`. -/
private noncomputable abbrev vol2 : Measure (ℝ × ℝ) :=
  (volume.restrict (Set.Ioi (0:ℝ))).prod (volume.restrict (Set.Ioi (0:ℝ)))

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

private lemma measurable_card_filter_lt (a : Fin n → ℝ) :
    Measurable (fun s : ℝ => ((univ.filter fun i => s < a i).card : ℝ≥0∞)) := by
  have h : ∀ s : ℝ, ((univ.filter fun i => s < a i).card : ℝ≥0∞)
      = ∑ i, if s < a i then (1:ℝ≥0∞) else 0 := by
    intro s
    rw [Finset.card_filter]
    push_cast
    rfl
  simp_rw [h]
  exact Finset.measurable_sum _ fun i _ => measurable_ite_lt (a i)

private lemma measurable_ofReal_card_div (x : ℕ → ℝ) (n : ℕ) :
    Measurable (fun s : ℝ =>
      ENNReal.ofReal ((((range n).filter fun i => s < x i).card : ℝ) / n)) := by
  refine Measurable.ennreal_ofReal (Measurable.div_const ?_ _)
  have h : (fun s : ℝ => (((range n).filter fun i => s < x i).card : ℝ))
      = fun s => ∑ i ∈ range n, if s < x i then (1:ℝ) else 0 := by
    funext s
    rw [Finset.card_filter]
    push_cast
    rfl
  rw [h]
  exact Finset.measurable_sum _ fun i _ =>
    Measurable.ite measurableSet_Iio measurable_const measurable_const

/-- Layer-cake identity over `vol2`, for antitone nonnegative tuples. -/
private lemma ofReal_sum_mul_eq_lintegral_min {a b : Fin n → ℝ}
    (ha : Antitone a) (hb : Antitone b) (ha0 : ∀ i, 0 ≤ a i) (hb0 : ∀ i, 0 ≤ b i) :
    ENNReal.ofReal (∑ i, a i * b i)
      = ∫⁻ p, min ((univ.filter fun i => p.1 < a i).card : ℝ≥0∞)
          ((univ.filter fun i => p.2 < b i).card : ℝ≥0∞) ∂vol2 := by
  have hmeas : Measurable (fun p : ℝ × ℝ =>
      ((univ.filter fun i => p.1 < a i ∧ p.2 < b i).card : ℝ≥0∞)) := by
    have h : (fun p : ℝ × ℝ =>
          ((univ.filter fun i => p.1 < a i ∧ p.2 < b i).card : ℝ≥0∞))
        = fun p => min ((univ.filter fun i => p.1 < a i).card : ℝ≥0∞)
            ((univ.filter fun i => p.2 < b i).card : ℝ≥0∞) := by
      funext p
      rw [card_filter_and_eq_min ha hb, (Nat.mono_cast (α := ℝ≥0∞)).map_min]
    rw [h]
    exact ((measurable_card_filter_lt a).comp measurable_fst).min
      ((measurable_card_filter_lt b).comp measurable_snd)
  rw [ofReal_sum_mul_eq_lintegral_card ha0 hb0, ← lintegral_prod _ hmeas.aemeasurable]
  refine lintegral_congr fun p => ?_
  rw [card_filter_and_eq_min ha hb, (Nat.mono_cast (α := ℝ≥0∞)).map_min]

private lemma ofReal_div_natCast_eq (c n : ℕ) :
    ENNReal.ofReal ((c : ℝ) / n) = ENNReal.ofReal (n:ℝ)⁻¹ * c := by
  rw [div_eq_inv_mul, ENNReal.ofReal_mul (inv_nonneg.mpr (Nat.cast_nonneg n)),
    ENNReal.ofReal_natCast]

/-- The normalized inner product of the two sorted lists as a `vol2`-integral
of the normalized survival counts. -/
private lemma lintegral_min_ofReal_div_eq {x x' : ℕ → ℝ}
    (hx : ∀ i, 0 < x i) (hx' : ∀ i, 0 < x' i) (n : ℕ) :
    ∫⁻ p, min (ENNReal.ofReal ((((range n).filter fun i => p.1 < x i).card : ℝ) / n))
        (ENNReal.ofReal ((((range n).filter fun i => p.2 < x' i).card : ℝ) / n)) ∂vol2
      = ENNReal.ofReal ((∑ i : Fin n,
          sortDesc (fun i : Fin n => x i) i * sortDesc (fun i : Fin n => x' i) i) / n) := by
  simp_rw [ofReal_div_natCast_eq, ← card_filter_sortDesc_lt, ← mul_min]
  rw [lintegral_const_mul' _ _ ENNReal.ofReal_ne_top,
    ← ofReal_sum_mul_eq_lintegral_min (sortDesc_antitone _) (sortDesc_antitone _)
      (fun i => (sortDesc_pos hx n i).le) (fun i => (sortDesc_pos hx' n i).le),
    div_eq_inv_mul, ENNReal.ofReal_mul (inv_nonneg.mpr (Nat.cast_nonneg n))]

/-- Fatou's lemma for the `vol2`-integrals of minima. -/
private lemma lintegral_min_le_liminf {γ : Measure ℝ} {u v : ℕ → ℝ → ℝ≥0∞}
    (hu : ∀ n, Measurable (u n)) (hv : ∀ n, Measurable (v n))
    (hlu : ∀ s, γ (Set.Ioi s) ≤ atTop.liminf (fun n => u n s))
    (hlv : ∀ t, γ (Set.Ioi t) ≤ atTop.liminf (fun n => v n t)) :
    ∫⁻ p, min (γ (Set.Ioi p.1)) (γ (Set.Ioi p.2)) ∂vol2
      ≤ atTop.liminf (fun n => ∫⁻ p, min (u n p.1) (v n p.2) ∂vol2) := by
  refine le_trans (lintegral_mono fun p => ?_)
    (lintegral_liminf_le fun n => ((hu n).comp measurable_fst).min ((hv n).comp measurable_snd))
  rw [liminf_min]
  exact min_le_min (hlu p.1) (hlv p.2)

/-- The second moment of a law on `(0,∞)` as a `vol2`-integral. -/
private lemma lintegral_min_measure_Ioi {γ : Measure ℝ} [IsFiniteMeasure γ]
    (hγ : γ (Set.Iic 0) = 0) :
    ∫⁻ p, min (γ (Set.Ioi p.1)) (γ (Set.Ioi p.2)) ∂vol2
      = ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ := by
  have hSm : MeasurableSet {q : (ℝ × ℝ) × ℝ | max q.1.1 q.1.2 < q.2} :=
    measurableSet_lt (measurable_fst.fst.max measurable_fst.snd) measurable_snd
  have h1 : ∀ p : ℝ × ℝ, min (γ (Set.Ioi p.1)) (γ (Set.Ioi p.2))
      = γ (Prod.mk p ⁻¹' {q : (ℝ × ℝ) × ℝ | max q.1.1 q.1.2 < q.2}) := by
    intro p
    have hpre : Prod.mk p ⁻¹' {q : (ℝ × ℝ) × ℝ | max q.1.1 q.1.2 < q.2}
        = Set.Ioi (max p.1 p.2) := by
      ext y; simp
    rw [hpre]
    rcases le_total p.1 p.2 with h | h
    · rw [max_eq_right h, min_eq_right (measure_mono (Set.Ioi_subset_Ioi h))]
    · rw [max_eq_left h, min_eq_left (measure_mono (Set.Ioi_subset_Ioi h))]
  have h2 : ∀ y : ℝ,
      (fun p : ℝ × ℝ => (p, y)) ⁻¹' {q : (ℝ × ℝ) × ℝ | max q.1.1 q.1.2 < q.2}
      = Set.Iio y ×ˢ Set.Iio y := by
    intro y; ext p; simp
  simp_rw [h1]
  rw [← Measure.prod_apply hSm, Measure.prod_apply_symm hSm]
  simp_rw [h2, Measure.prod_prod, Measure.restrict_apply measurableSet_Iio, Set.Iio_inter_Ioi,
    Real.volume_Ioo, sub_zero]
  refine lintegral_congr_ae ?_
  have hpos : ∀ᵐ y ∂γ, 0 < y := by
    rw [ae_iff]
    convert hγ using 2
    ext y; simp [not_lt]
  filter_upwards [hpos] with y hy
  rw [← ENNReal.ofReal_mul hy.le, sq]

/-- Deterministic core of the argument: along a sample path whose
empirical survival fractions dominate `γ (Ioi s)` in the liminf at every
threshold, the liminf of the normalized inner product of the two sorted
lists dominates the second moment of `γ`. -/
private lemma le_liminf_ofReal_sum_mul {γ : Measure ℝ} [IsFiniteMeasure γ]
    (hγ : γ (Set.Iic 0) = 0) {x x' : ℕ → ℝ} (hx : ∀ i, 0 < x i) (hx' : ∀ i, 0 < x' i)
    (hl : ∀ s, γ (Set.Ioi s) ≤ atTop.liminf (fun n => ENNReal.ofReal
      ((((range n).filter fun i => s < x i).card : ℝ) / n)))
    (hl' : ∀ t, γ (Set.Ioi t) ≤ atTop.liminf (fun n => ENNReal.ofReal
      ((((range n).filter fun i => t < x' i).card : ℝ) / n))) :
    ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≤ atTop.liminf (fun n => ENNReal.ofReal
      ((∑ i : Fin n, sortDesc (fun i : Fin n => x i) i * sortDesc (fun i : Fin n => x' i) i)
        / n)) := by
  rw [← lintegral_min_measure_Ioi hγ]
  refine le_trans (lintegral_min_le_liminf (measurable_ofReal_card_div x)
    (measurable_ofReal_card_div x') hl hl') (le_of_eq ?_)
  exact liminf_congr (Eventually.of_forall fun n => lintegral_min_ofReal_div_eq hx hx' n)

end Fatou

/-! ### Assembly -/

section Assembly

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

/-- Deterministic assembly along one sample path. -/
private lemma tendsto_sortDesc_diff_div_of_seq {γ : Measure ℝ} [IsFiniteMeasure γ]
    (hγ : γ (Set.Iic 0) = 0)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤)
    (hMpos : 0 < ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ)
    {x x' : ℕ → ℝ} (hx : ∀ i, 0 < x i) (hx' : ∀ i, 0 < x' i)
    (hl : ∀ s, γ (Set.Ioi s) ≤ atTop.liminf (fun n => ENNReal.ofReal
      ((((range n).filter fun i => s < x i).card : ℝ) / n)))
    (hl' : ∀ t, γ (Set.Ioi t) ≤ atTop.liminf (fun n => ENNReal.ofReal
      ((((range n).filter fun i => t < x' i).card : ℝ) / n)))
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
  have hsum : ∀ n, ∑ i : Fin n, (sortDesc (fun i : Fin n => x i) i) ^ 2
      = ∑ i ∈ range n, x i ^ 2 := fun n =>
    (sum_comp_sortDesc (fun i : Fin n => x i) (fun y => y ^ 2)).trans
      (Fin.sum_univ_eq_sum_range (fun i => x i ^ 2) n)
  have hsum' : ∀ n, ∑ i : Fin n, (sortDesc (fun i : Fin n => x' i) i) ^ 2
      = ∑ i ∈ range n, x' i ^ 2 := fun n =>
    (sum_comp_sortDesc (fun i : Fin n => x' i) (fun y => y ^ 2)).trans
      (Fin.sum_univ_eq_sum_range (fun i => x' i ^ 2) n)
  have hA : Tendsto (fun n => (∑ i : Fin n, (sortDesc (fun i : Fin n => x i) i) ^ 2) / n)
      atTop (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal) :=
    hsq.congr fun n => by rw [hsum n]
  have hA' : Tendsto (fun n => (∑ i : Fin n, (sortDesc (fun i : Fin n => x' i) i) ^ 2) / n)
      atTop (𝓝 (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal) :=
    hsq'.congr fun n => by rw [hsum' n]
  have hC : ENNReal.ofReal (∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ).toReal
      ≤ atTop.liminf (fun n => ENNReal.ofReal ((∑ i : Fin n,
          sortDesc (fun i : Fin n => x i) i * sortDesc (fun i : Fin n => x' i) i) / n)) := by
    rw [ENNReal.ofReal_toReal hM]
    exact le_liminf_ofReal_sum_mul hγ hx hx' hl hl'
  have hexp : ∀ n, (∑ i : Fin n, (sortDesc (fun i : Fin n => x i) i) ^ 2) / n
      + (∑ i : Fin n, (sortDesc (fun i : Fin n => x' i) i) ^ 2) / n
      - 2 * ((∑ i : Fin n,
          sortDesc (fun i : Fin n => x i) i * sortDesc (fun i : Fin n => x' i) i) / n)
      = (∑ i : Fin n,
          (sortDesc (fun i : Fin n => x' i) i - sortDesc (fun i : Fin n => x i) i) ^ 2) / n := by
    intro n
    have h : ∀ i : Fin n,
        (sortDesc (fun i : Fin n => x' i) i - sortDesc (fun i : Fin n => x i) i) ^ 2
        = (sortDesc (fun i : Fin n => x i) i) ^ 2 + (sortDesc (fun i : Fin n => x' i) i) ^ 2
          - 2 * (sortDesc (fun i : Fin n => x i) i * sortDesc (fun i : Fin n => x' i) i) :=
      fun i => by ring
    simp_rw [h]
    rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
    ring
  have hD0 : ∀ n, 0 ≤ (∑ i : Fin n, (sortDesc (fun i : Fin n => x i) i) ^ 2) / n
      + (∑ i : Fin n, (sortDesc (fun i : Fin n => x' i) i) ^ 2) / n
      - 2 * ((∑ i : Fin n,
          sortDesc (fun i : Fin n => x i) i * sortDesc (fun i : Fin n => x' i) i) / n) := by
    intro n
    rw [hexp n]
    positivity
  have hDlim := tendsto_of_liminf_ofReal hmpos hA hA' hC hD0
  simp_rw [hexp] at hDlim
  refine (tendsto_sqrt_div_sqrt hmpos hDlim hA).congr' ?_
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
  have hemp : ∀ᵐ ω ∂P, ∀ q : ℚ, Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < X i ω).card : ℝ) / n)
      atTop (𝓝 (γ.real (Set.Ioi (q:ℝ)))) := by
    rw [ae_all_iff]
    intro q
    have h := tendsto_empirical_survival hX hXi (fun i => by rw [hXl i, hXl 0]) (q:ℝ)
    rwa [hXl 0] at h
  have hemp' : ∀ᵐ ω ∂P, ∀ q : ℚ, Tendsto
      (fun n => (((range n).filter fun i => (q:ℝ) < X' i ω).card : ℝ) / n)
      atTop (𝓝 (γ.real (Set.Ioi (q:ℝ)))) := by
    rw [ae_all_iff]
    intro q
    have h := tendsto_empirical_survival hX' hX'i (fun i => by rw [hX'l i, hX'l 0]) (q:ℝ)
    rwa [hX'l 0] at h
  have hsq := tendsto_mean_sq hX hXi hXl hM
  have hsq' := tendsto_mean_sq hX' hX'i hX'l hM
  filter_upwards [hpos, hpos', hemp, hemp', hsq, hsq'] with ω hpos hpos' hemp hemp' hsq hsq'
  have hdet := tendsto_sortDesc_diff_div_of_seq (x := fun i => X i ω) (x' := fun i => X' i ω)
    hγ0 hM hMpos hpos hpos'
    (le_liminf_empirical_of_rat hemp) (le_liminf_empirical_of_rat hemp') hsq hsq'
  exact hdet

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

/-- The instance of Lemma (lem:proportional-lobes) used in the proof of
Theorem (thm:accumulation) of the main text: `X` with law
`γ_{ν,k} = gammaMeasure k ((k−1)/ν)`, `v = 1` and `u = ρ = |μ|/ν`, so
that `uX'` has law `γ_{|μ|,k} = gammaMeasure k ((k−1)/|μ|)`.

Encoding: `X i` are the samples from `γ_{ν,k}` and `Y i` the samples
from `γ_{|μ|,k}`; `hindep` says that all the samples
`X 0, X 1, …, Y 0, Y 1, …` are independent (so in particular the two
samples are independent of each other);
`α_n = sortDesc (fun i : Fin n => X i ω)`,
`β_n = sortDesc (fun i : Fin n => Y i ω)`, `ρ = |μ|/ν`, and the norms
are written out as `√(∑ …²)`. -/
theorem proportional_lobes_gamma {k μ ν : ℝ} (hk : 1 < k) (hμ : μ < 0) (hν : 0 < ν)
    {X Y : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) (hY : ∀ i, Measurable (Y i))
    (hindep : iIndepFun (Sum.elim X Y) P)
    (hXlaw : ∀ i, Measure.map (X i) P = gammaMeasure k ((k - 1) / ν))
    (hYlaw : ∀ i, Measure.map (Y i) P = gammaMeasure k ((k - 1) / |μ|)) :
    ∀ᵐ ω ∂P, Tendsto (fun n =>
        √(∑ i : Fin n,
            (sortDesc (fun i => Y i ω) i - (|μ| / ν) * sortDesc (fun i => X i ω) i) ^ 2)
          / √(∑ i : Fin n, (sortDesc (fun i => X i ω) i) ^ 2))
      atTop (𝓝 0) := by
  have hμ0 : 0 < |μ| := abs_pos.mpr hμ.ne
  have hk0 : 0 < k := by linarith
  have hk1 : 0 < k - 1 := by linarith
  set ρ := |μ| / ν with hρ
  have hρ0 : 0 < ρ := div_pos hμ0 hν
  set c := ν / |μ| with hc
  have hc0 : 0 < c := div_pos hν hμ0
  have hcρ : ρ * c = 1 := by
    rw [hρ, hc]; field_simp
  -- the samples `X' i = c * Y i` have law `γ_{ν,k}`, and `Y i = ρ * X' i`
  set X' : ℕ → Ω → ℝ := fun i ω => c * Y i ω with hX'
  have hX'm : ∀ i, Measurable (X' i) := fun i => (hY i).const_mul c
  have hXi : iIndepFun X P :=
    iIndepFun.precomp (f := Sum.elim X Y) (g := Sum.inl) Sum.inl_injective hindep
  have hYi : iIndepFun Y P :=
    iIndepFun.precomp (f := Sum.elim X Y) (g := Sum.inr) Sum.inr_injective hindep
  have hX'i : iIndepFun X' P :=
    hYi.comp (fun _ y => c * y) (fun _ => measurable_const_mul c)
  have hX'l : ∀ i, Measure.map (X' i) P = gammaMeasure k ((k - 1) / ν) := by
    intro i
    have h : X' i = (fun y => c * y) ∘ Y i := rfl
    rw [h, ← Measure.map_map (measurable_const_mul c) (hY i), hYlaw i,
      map_mul_left_gammaMeasure hk0 (div_pos hk1 hμ0) hc0]
    congr 1
    rw [hc]
    field_simp
  have hr : 0 < (k - 1) / ν := div_pos hk1 hν
  have hmain := tendsto_sortDesc_diff_div_of_iid hX hX'm hXi hX'i hXlaw hX'l
    (gammaMeasure_Iic_zero k ((k - 1) / ν)) (lintegral_sq_gammaMeasure_lt_top hk0 hr).ne
    (lintegral_sq_gammaMeasure_pos hk0 hr)
  filter_upwards [hmain] with ω hω
  have hkey : ∀ n : ℕ,
      √(∑ i : Fin n, (sortDesc (fun i => Y i ω) i - ρ * sortDesc (fun i => X i ω) i) ^ 2)
        = ρ * √(∑ i : Fin n,
            (sortDesc (fun i => X' i ω) i - sortDesc (fun i => X i ω) i) ^ 2) := by
    intro n
    have hY' : (fun i : Fin n => Y i ω) = fun i : Fin n => ρ * X' i ω := by
      funext i
      show Y i ω = ρ * (c * Y i ω)
      rw [← mul_assoc, hcρ, one_mul]
    rw [hY', sortDesc_const_mul hρ0.le]
    have hs : ∑ i : Fin n,
          (ρ * sortDesc (fun i => X' i ω) i - ρ * sortDesc (fun i => X i ω) i) ^ 2
        = ρ ^ 2 * ∑ i : Fin n,
            (sortDesc (fun i => X' i ω) i - sortDesc (fun i => X i ω) i) ^ 2 := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [hs, Real.sqrt_mul (sq_nonneg ρ), Real.sqrt_sq hρ0.le]
  have h := hω.const_mul ρ
  rw [mul_zero] at h
  refine h.congr fun n => ?_
  rw [hkey n, mul_div_assoc]

end Main

end Appendices
