/-
Appendix "Random matrices",
Corollary (Iid square random matrix baseline).

Let `L ∈ M_N(ℝ)` be a random matrix whose entries are independent,
mean-zero, with common variance `σ²`.  Then

    E‖S‖² = N(N+1)/2 · σ²,    E‖T‖² = N(N-1)/2 · σ²,

so

    E‖S‖² / E‖L‖² = (N+1)/(2N),    E‖T‖² / E‖L‖² = (N-1)/(2N).

If the entries of `L` are iid Gaussian, then also
`E[‖S‖²/‖L‖²] = (N+1)/(2N)` and `E[‖T‖²/‖L‖²] = (N-1)/(2N)`.

Here `S = ½(L + Lᵀ)`, `T = ½(L - Lᵀ)`, and `‖·‖` is the Frobenius norm;
we state everything for the squared norm `frobSq`.

Modelling: a random matrix is a map `L : Ω → Matrix (Fin N) (Fin N) ℝ`
on a probability space; "independent entries" is `iIndepFun` of the
family of entry maps indexed by `Fin N × Fin N`; "mean zero with common
variance σ²" is `∫ L·ij = 0` and `∫ (L·ij)² = σ²` together with
square-integrability `MemLp _ 2 μ` of each entry.  The ratio statements
divide by `E‖L‖² = N²σ²`, so they require `σ ≠ 0` and `0 < N` (the
paper implicitly assumes this nondegeneracy).
-/
import Appendices.Common
import Appendices.RandomMatrices.EnergyIdentities

-- Some section hypotheses (e.g. `IsProbabilityMeasure`) are not needed by
-- every lemma in the section; silence the corresponding linter.
set_option linter.unusedSectionVars false

namespace Appendices

open Matrix MeasureTheory ProbabilityTheory BigOperators

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}
  [IsProbabilityMeasure μ] {N : ℕ}
  {L : Ω → Matrix (Fin N) (Fin N) ℝ} {σ : ℝ}

/-! ### Entry-level bookkeeping -/

private lemma frobSq_eq_sum_prod (A : Matrix (Fin N) (Fin N) ℝ) :
    frobSq A = ∑ p : Fin N × Fin N, A p.1 p.2 ^ 2 := by
  rw [frobSq_eq_sum_sq]
  exact (Fintype.sum_prod_type (f := fun p : Fin N × Fin N => A p.1 p.2 ^ 2)).symm

private lemma trace_mul_self_eq_sum_prod (A : Matrix (Fin N) (Fin N) ℝ) :
    Matrix.trace (A * A) = ∑ p : Fin N × Fin N, A p.1 p.2 * A p.2 p.1 := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply]
  exact (Fintype.sum_prod_type (f := fun p : Fin N × Fin N => A p.1 p.2 * A p.2 p.1)).symm

section MomentBaseline

variable (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
  (hMemLp : ∀ i j, MemLp (fun ω => L ω i j) 2 μ)
  (hMean : ∀ i j, ∫ ω, L ω i j ∂μ = 0)
  (hVar : ∀ i j, ∫ ω, (L ω i j) ^ 2 ∂μ = σ ^ 2)

include hMemLp in
private lemma integrable_frobSq : Integrable (fun ω => frobSq (L ω)) μ := by
  simp_rw [frobSq_eq_sum_prod]
  exact integrable_finset_sum _ fun p _ => (hMemLp p.1 p.2).integrable_sq

include hIndep hMemLp in
private lemma integrable_trace_mul_self :
    Integrable (fun ω => Matrix.trace (L ω * L ω)) μ := by
  simp_rw [trace_mul_self_eq_sum_prod]
  refine integrable_finset_sum _ fun p _ => ?_
  by_cases h : p.1 = p.2
  · have := (hMemLp p.1 p.2).integrable_sq
    simp_rw [sq] at this
    convert this using 2 with ω
    rw [h]
  · have hind : (fun ω => L ω p.1 p.2) ⟂ᵢ[μ] (fun ω => L ω p.2 p.1) :=
      hIndep.indepFun (i := (p.1, p.2)) (j := (p.2, p.1))
        (by simp [Prod.ext_iff]; exact fun h' => absurd h' h)
    exact hind.integrable_mul ((hMemLp _ _).integrable one_le_two)
      ((hMemLp _ _).integrable one_le_two)

include hMemLp hVar in
/-- `E‖L‖² = N² σ²` for a random matrix with mean-zero entries of
common variance `σ²`. -/
theorem integral_frobSq_iid : ∫ ω, frobSq (L ω) ∂μ = N ^ 2 * σ ^ 2 := by
  simp_rw [frobSq_eq_sum_prod]
  rw [integral_finset_sum _ fun p _ => (hMemLp p.1 p.2).integrable_sq]
  simp only [hVar]
  simp [Finset.card_univ, sq]

include hIndep hMemLp hMean hVar in
/-- `E tr(L²) = N σ²`: only the diagonal entries contribute. -/
theorem integral_trace_mul_self_iid :
    ∫ ω, Matrix.trace (L ω * L ω) ∂μ = N * σ ^ 2 := by
  simp_rw [trace_mul_self_eq_sum_prod]
  rw [integral_finset_sum]
  · have hterm : ∀ p : Fin N × Fin N,
        (∫ ω, L ω p.1 p.2 * L ω p.2 p.1 ∂μ) = if p.1 = p.2 then σ ^ 2 else 0 := by
      intro p
      by_cases h : p.1 = p.2
      · rw [if_pos h, ← hVar p.1 p.2]
        obtain ⟨i, j⟩ := p
        cases h
        simp [sq]
      · rw [if_neg h]
        have hind : (fun ω => L ω p.1 p.2) ⟂ᵢ[μ] (fun ω => L ω p.2 p.1) :=
          hIndep.indepFun (i := (p.1, p.2)) (j := (p.2, p.1))
            (by simp [Prod.ext_iff]; exact fun h' => absurd h' h)
        rw [hind.integral_fun_mul_eq_mul_integral
          (hMemLp _ _).aestronglyMeasurable (hMemLp _ _).aestronglyMeasurable]
        simp only [hMean, zero_mul]
    simp_rw [hterm]
    rw [Fintype.sum_prod_type]
    simp [Finset.sum_ite_eq]
  · intro p _
    by_cases h : p.1 = p.2
    · have := (hMemLp p.1 p.2).integrable_sq
      simp_rw [sq] at this
      convert this using 2 with ω
      rw [h]
    · have hind : (fun ω => L ω p.1 p.2) ⟂ᵢ[μ] (fun ω => L ω p.2 p.1) :=
        hIndep.indepFun (i := (p.1, p.2)) (j := (p.2, p.1))
          (by simp [Prod.ext_iff]; exact fun h' => absurd h' h)
      exact hind.integrable_mul ((hMemLp _ _).integrable one_le_two)
        ((hMemLp _ _).integrable one_le_two)

include hIndep hMemLp hMean hVar in
/-- `E‖S‖² = N(N+1)/2 · σ²`. -/
theorem integral_frobSq_symPart_iid :
    ∫ ω, frobSq (symPart (L ω)) ∂μ = N * (N + 1) / 2 * σ ^ 2 := by
  have hpt : ∀ ω, frobSq (symPart (L ω))
      = frobSq (L ω) / 2 + Matrix.trace (L ω * L ω) / 2 :=
    fun ω => frobSq_symPart (L ω)
  simp_rw [hpt]
  rw [integral_add ((integrable_frobSq hMemLp).div_const 2)
    ((integrable_trace_mul_self hIndep hMemLp).div_const 2),
    integral_div, integral_div, integral_frobSq_iid hMemLp hVar,
    integral_trace_mul_self_iid hIndep hMemLp hMean hVar]
  ring

include hIndep hMemLp hMean hVar in
/-- `E‖T‖² = N(N-1)/2 · σ²`. -/
theorem integral_frobSq_skewPart_iid :
    ∫ ω, frobSq (skewPart (L ω)) ∂μ = N * (N - 1) / 2 * σ ^ 2 := by
  have hpt : ∀ ω, frobSq (skewPart (L ω))
      = frobSq (L ω) / 2 - Matrix.trace (L ω * L ω) / 2 :=
    fun ω => frobSq_skewPart (L ω)
  simp_rw [hpt]
  rw [integral_sub ((integrable_frobSq hMemLp).div_const 2)
    ((integrable_trace_mul_self hIndep hMemLp).div_const 2),
    integral_div, integral_div, integral_frobSq_iid hMemLp hVar,
    integral_trace_mul_self_iid hIndep hMemLp hMean hVar]
  ring

include hIndep hMemLp hMean hVar in
/-- `E‖S‖² / E‖L‖² = (N+1)/(2N)`.  (The division requires the
nondegeneracy `σ ≠ 0`, `0 < N`, which the paper leaves implicit.) -/
theorem integral_frobSq_symPart_div_integral_frobSq_iid
    (hσ : σ ≠ 0) (hN : 0 < N) :
    (∫ ω, frobSq (symPart (L ω)) ∂μ) / (∫ ω, frobSq (L ω) ∂μ)
      = (N + 1) / (2 * N) := by
  rw [integral_frobSq_symPart_iid hIndep hMemLp hMean hVar,
    integral_frobSq_iid hMemLp hVar]
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  field_simp

include hIndep hMemLp hMean hVar in
/-- `E‖T‖² / E‖L‖² = (N-1)/(2N)`. -/
theorem integral_frobSq_skewPart_div_integral_frobSq_iid
    (hσ : σ ≠ 0) (hN : 0 < N) :
    (∫ ω, frobSq (skewPart (L ω)) ∂μ) / (∫ ω, frobSq (L ω) ∂μ)
      = (N - 1) / (2 * N) := by
  rw [integral_frobSq_skewPart_iid hIndep hMemLp hMean hVar,
    integral_frobSq_iid hMemLp hVar]
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  field_simp

end MomentBaseline


/-! ### The Gaussian case

For iid Gaussian entries the *expectation of the ratio* equals the
ratio of expectations.  Following the paper's proof, we transfer the
integral to the canonical product space `(Fin N × Fin N) → ℝ` carrying
the product of `N²` copies of `gaussianReal 0 v`, where the claim
follows from invariance of the product measure under permutations of
the coordinates and under a sign flip of a single coordinate. -/

section GaussianRatio

open scoped NNReal

/-- The joint law of the entries: the product of `N²` Gaussian factors
`gaussianReal 0 v` on `(Fin N × Fin N) → ℝ`. -/
private noncomputable abbrev gaussPi (N : ℕ) (v : ℝ≥0) :
    Measure ((Fin N × Fin N) → ℝ) :=
  Measure.pi fun _ => gaussianReal 0 v

/-- The matrix assembled from a point of the canonical space. -/
private def matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i j => x (i, j)

private lemma matOf_entries {N : ℕ} (A : Matrix (Fin N) (Fin N) ℝ) :
    matOf (fun p => A p.1 p.2) = A := rfl

private lemma frobSq_matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    frobSq (matOf x) = ∑ p, x p ^ 2 := by
  rw [frobSq_eq_sum_prod]
  exact Finset.sum_congr rfl fun p _ => by simp [matOf]

private lemma trace_matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    Matrix.trace (matOf x * matOf x) = ∑ p, x p * x p.swap := by
  rw [trace_mul_self_eq_sum_prod]
  refine Finset.sum_congr rfl fun p _ => ?_
  simp only [matOf, Matrix.of_apply, Prod.mk.eta]
  rfl

private lemma frobSq_symPart_matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    frobSq (symPart (matOf x)) = ∑ p, ((x p + x p.swap) / 2) ^ 2 := by
  rw [frobSq_eq_sum_prod]
  refine Finset.sum_congr rfl fun p _ => ?_
  simp only [symPart, Matrix.smul_apply, Matrix.add_apply, Matrix.transpose_apply,
    smul_eq_mul, matOf, Matrix.of_apply, Prod.mk.eta]
  ring_nf
  congr 1

private lemma frobSq_skewPart_matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    frobSq (skewPart (matOf x)) = ∑ p, ((x p - x p.swap) / 2) ^ 2 := by
  rw [frobSq_eq_sum_prod]
  refine Finset.sum_congr rfl fun p _ => ?_
  simp only [skewPart, Matrix.smul_apply, Matrix.sub_apply, Matrix.transpose_apply,
    smul_eq_mul, matOf, Matrix.of_apply, Prod.mk.eta]
  ring_nf
  congr 1

/-- Invariance of `gaussPi` under a permutation of the coordinates. -/
private lemma integral_gaussPi_comp_perm {N : ℕ} {v : ℝ≥0}
    (e : (Fin N × Fin N) ≃ (Fin N × Fin N))
    (g : ((Fin N × Fin N) → ℝ) → ℝ) :
    ∫ x, g (fun p => x (e p)) ∂(gaussPi N v) = ∫ x, g x ∂(gaussPi N v) := by
  have hmp := MeasureTheory.measurePreserving_piCongrLeft
    (fun _ : Fin N × Fin N => gaussianReal (0 : ℝ) v) e.symm
  have happ : ∀ (x : (Fin N × Fin N) → ℝ),
      (MeasurableEquiv.piCongrLeft (fun _ : Fin N × Fin N => ℝ) e.symm) x
        = fun p => x (e p) := by
    intro x
    funext p
    conv_lhs => rw [show p = e.symm (e p) by simp]
    exact MeasurableEquiv.piCongrLeft_apply_apply
      (β := fun _ : Fin N × Fin N => ℝ) e.symm x (e p)
  calc ∫ x, g (fun p => x (e p)) ∂(gaussPi N v)
      = ∫ x, g ((MeasurableEquiv.piCongrLeft (fun _ : Fin N × Fin N => ℝ) e.symm) x)
          ∂(gaussPi N v) := by
        refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
        simp only [happ]
    _ = ∫ x, g x ∂(gaussPi N v) := hmp.integral_comp' g

/-- Invariance of `gaussPi` under a sign flip of one coordinate. -/
private lemma integral_gaussPi_comp_flip {N : ℕ} {v : ℝ≥0}
    (p₀ : Fin N × Fin N) (g : ((Fin N × Fin N) → ℝ) → ℝ) :
    ∫ x, g (fun p => if p = p₀ then -x p else x p) ∂(gaussPi N v)
      = ∫ x, g x ∂(gaussPi N v) := by
  classical
  set e : ∀ _ : Fin N × Fin N, ℝ ≃ᵐ ℝ :=
    fun p => if p = p₀ then (Homeomorph.neg ℝ).toMeasurableEquiv
      else MeasurableEquiv.refl ℝ with he
  have hmpf : ∀ p, MeasurePreserving (e p) (gaussianReal (0 : ℝ) v)
      (gaussianReal (0 : ℝ) v) := by
    intro p
    by_cases h : p = p₀
    · simp only [he, if_pos h]
      refine ⟨(Homeomorph.neg ℝ).toMeasurableEquiv.measurable, ?_⟩
      have hfun : ((Homeomorph.neg ℝ).toMeasurableEquiv : ℝ → ℝ)
          = fun x => (-1 : ℝ) * x := by
        funext x
        have hx : ((Homeomorph.neg ℝ).toMeasurableEquiv : ℝ → ℝ) x = -x := rfl
        rw [hx]
        ring
      rw [hfun, gaussianReal_map_const_mul]
      norm_num
    · simp only [he, if_neg h]
      exact ⟨measurable_id, Measure.map_id⟩
  have hmp : MeasurePreserving
      (fun (x : (Fin N × Fin N) → ℝ) p => e p (x p)) (gaussPi N v) (gaussPi N v) :=
    MeasureTheory.measurePreserving_pi _ _ hmpf
  have hemb : MeasurableEmbedding
      (fun (x : (Fin N × Fin N) → ℝ) p => e p (x p)) :=
    (MeasurableEquiv.piCongrRight e).measurableEmbedding
  calc ∫ x, g (fun p => if p = p₀ then -x p else x p) ∂(gaussPi N v)
      = ∫ x, g (fun p => e p (x p)) ∂(gaussPi N v) := by
        refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
        show g (fun p => if p = p₀ then -x p else x p) = g (fun p => e p (x p))
        congr 1
        funext p
        by_cases h : p = p₀
        · simp only [he, if_pos h]
          rfl
        · simp only [he, if_neg h]
          rfl
    _ = ∫ x, g x ∂(gaussPi N v) := hmp.integral_comp hemb g

/-- Almost surely `∑ p, x p ² ≠ 0` (for `N ≥ 1` and `v ≠ 0`). -/
private lemma ae_sum_sq_ne_zero {N : ℕ} {v : ℝ≥0} (hv : v ≠ 0) (hN : 0 < N) :
    ∀ᵐ x ∂(gaussPi N v), ∑ p, x p ^ 2 ≠ 0 := by
  haveI : NoAtoms (gaussianReal (0 : ℝ) v) := noAtoms_gaussianReal hv
  have h := Measure.ae_eval_ne (fun _ : Fin N × Fin N => gaussianReal (0 : ℝ) v)
    (⟨⟨0, hN⟩, ⟨0, hN⟩⟩ : Fin N × Fin N) 0
  filter_upwards [h] with x hx h0
  exact hx (pow_eq_zero_iff (n := 2) (by norm_num) |>.mp
    ((Finset.sum_eq_zero_iff_of_nonneg fun p _ => sq_nonneg (x p)).mp h0 _
      (Finset.mem_univ _)))

/-- The integrands below are bounded by `1` in absolute value. -/
private lemma abs_ratio_le_one {N : ℕ} (x : (Fin N × Fin N) → ℝ)
    (q r : Fin N × Fin N) :
    |x q * x r / ∑ p, x p ^ 2| ≤ 1 := by
  set S := ∑ p, x p ^ 2 with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun p _ => sq_nonneg _
  rcases eq_or_lt_of_le hS0 with h | h
  · simp [← h]
  · rw [abs_div, abs_of_pos h, div_le_one h]
    have h1 : |x q * x r| ≤ (x q ^ 2 + x r ^ 2) / 2 := by
      rw [abs_mul]
      nlinarith [sq_nonneg (|x q| - |x r|), sq_abs (x q), sq_abs (x r)]
    have h2 : x q ^ 2 ≤ S := Finset.single_le_sum
      (f := fun p => x p ^ 2) (fun p _ => sq_nonneg _) (Finset.mem_univ q)
    have h3 : x r ^ 2 ≤ S := Finset.single_le_sum
      (f := fun p => x p ^ 2) (fun p _ => sq_nonneg _) (Finset.mem_univ r)
    nlinarith

private lemma integrable_ratio {N : ℕ} {v : ℝ≥0} (q r : Fin N × Fin N) :
    Integrable (fun x => x q * x r / ∑ p, x p ^ 2) (gaussPi N v) := by
  have hmeas : Measurable
      (fun x : (Fin N × Fin N) → ℝ => x q * x r / ∑ p, x p ^ 2) := by
    fun_prop
  refine Integrable.mono' (integrable_const 1) hmeas.aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs]
  exact abs_ratio_le_one x q r

/-- The expected fraction of the squared norm carried by one coordinate
is `1/N²`, by exchangeability of the coordinates. -/
private lemma integral_sq_ratio {N : ℕ} {v : ℝ≥0} (hv : v ≠ 0) (hN : 0 < N)
    (q : Fin N × Fin N) :
    ∫ x, x q ^ 2 / ∑ p, x p ^ 2 ∂(gaussPi N v) = 1 / (N ^ 2 : ℝ) := by
  classical
  have hswap : ∀ q' : Fin N × Fin N,
      (∫ x, x q' ^ 2 / ∑ p, x p ^ 2 ∂(gaussPi N v))
        = ∫ x, x q ^ 2 / ∑ p, x p ^ 2 ∂(gaussPi N v) := by
    intro q'
    have h := integral_gaussPi_comp_perm (N := N) (v := v) (Equiv.swap q q')
      (fun y => y q ^ 2 / ∑ p, y p ^ 2)
    rw [← h]
    refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
    have hden : (∑ p, x (Equiv.swap q q' p) ^ 2) = ∑ p, x p ^ 2 :=
      Fintype.sum_equiv (Equiv.swap q q') _ _ fun p => rfl
    simp only
    rw [Equiv.swap_apply_left, hden]
  have hsum : (∑ q' : Fin N × Fin N,
      ∫ x, x q' ^ 2 / ∑ p, x p ^ 2 ∂(gaussPi N v)) = 1 := by
    rw [← integral_finset_sum _ fun q' _ => by
      simpa [pow_two] using integrable_ratio (N := N) (v := v) q' q']
    have hone : ∀ᵐ x ∂(gaussPi N v),
        (∑ q' : Fin N × Fin N, x q' ^ 2 / ∑ p, x p ^ 2) = 1 := by
      filter_upwards [ae_sum_sq_ne_zero hv hN] with x hx
      rw [← Finset.sum_div, div_self hx]
    rw [integral_congr_ae hone]
    simp
  rw [Finset.sum_congr rfl fun q' _ => hswap q', Finset.sum_const,
    Finset.card_univ, Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul] at hsum
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  rw [eq_div_iff (by positivity)]
  push_cast at hsum ⊢
  nlinarith [hsum]

/-- The cross terms integrate to zero, by the sign-flip symmetry. -/
private lemma integral_cross_ratio_eq_zero {N : ℕ} {v : ℝ≥0}
    {i j : Fin N} (hij : i ≠ j) :
    ∫ x, x (i, j) * x (j, i) / ∑ p, x p ^ 2 ∂(gaussPi N v) = 0 := by
  classical
  have hflip := integral_gaussPi_comp_flip (N := N) (v := v) (i, j)
    (fun y => y (i, j) * y (j, i) / ∑ p, y p ^ 2)
  have hji : ((j, i) : Fin N × Fin N) ≠ (i, j) := by
    simp only [ne_eq, Prod.mk.injEq, not_and]
    intro h
    exact absurd h.symm hij
  have hneg : ∀ x : (Fin N × Fin N) → ℝ,
      (fun y => y (i, j) * y (j, i) / ∑ p, y p ^ 2)
          (fun p => if p = (i, j) then -x p else x p)
        = -(x (i, j) * x (j, i) / ∑ p, x p ^ 2) := by
    intro x
    simp [hji]
    try ring
  rw [funext hneg, integral_neg] at hflip
  linarith [hflip]

/-- `E[tr(L²)/‖L‖²] = 1/N` on the canonical space: the diagonal
contributes `N` copies of `1/N²` and the off-diagonal terms vanish. -/
private lemma integral_crossSum_ratio {N : ℕ} {v : ℝ≥0}
    (hv : v ≠ 0) (hN : 0 < N) :
    ∫ x, (∑ p, x p * x p.swap) / ∑ p, x p ^ 2 ∂(gaussPi N v) = 1 / N := by
  classical
  have hpt : ∀ x : (Fin N × Fin N) → ℝ,
      ((∑ p, x p * x p.swap) / ∑ p, x p ^ 2)
        = ∑ p, x p * x p.swap / ∑ p', x p' ^ 2 :=
    fun x => Finset.sum_div _ _ _
  simp_rw [hpt]
  rw [integral_finset_sum _ fun p _ => integrable_ratio p p.swap]
  have hterm : ∀ p : Fin N × Fin N,
      (∫ x, x p * x p.swap / ∑ p', x p' ^ 2 ∂(gaussPi N v))
        = if p.1 = p.2 then 1 / (N ^ 2 : ℝ) else 0 := by
    intro p
    obtain ⟨i, j⟩ := p
    by_cases h : i = j
    · subst h
      rw [if_pos rfl]
      simpa [pow_two] using integral_sq_ratio hv hN ((i, i) : Fin N × Fin N)
    · rw [if_neg h]
      simpa using integral_cross_ratio_eq_zero (v := v) h
  simp_rw [hterm]
  rw [Fintype.sum_prod_type]
  have hinner : ∀ i : Fin N,
      (∑ j : Fin N, if i = j then 1 / (N ^ 2 : ℝ) else 0) = 1 / (N ^ 2 : ℝ) := by
    intro i
    simp
  simp_rw [hinner]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  field_simp

variable {L : Ω → Matrix (Fin N) (Fin N) ℝ} {v : ℝ≥0}

/-- The Gaussian clause of the corollary: if the entries of `L` are iid
Gaussian, then `E[‖S‖²/‖L‖²] = (N+1)/(2N)`.

Hypotheses: the entries are jointly independent with common law
`gaussianReal 0 v`; nondegeneracy `v ≠ 0` and `0 < N` (implicit in the
paper) is required since otherwise `‖L‖ = 0`. -/
theorem integral_frobSq_symPart_div_frobSq_gaussian
    (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
    (hMeas : ∀ i j, AEMeasurable (fun ω => L ω i j) μ)
    (hLaw : ∀ i j, μ.map (fun ω => L ω i j) = gaussianReal 0 v)
    (hv : v ≠ 0) (hN : 0 < N) :
    ∫ ω, frobSq (symPart (L ω)) / frobSq (L ω) ∂μ = (N + 1) / (2 * N) := by
  classical
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hjoint : μ.map (fun ω (p : Fin N × Fin N) => L ω p.1 p.2) = gaussPi N v := by
    have h1 := (iIndepFun_iff_map_fun_eq_pi_map
      (f := fun (p : Fin N × Fin N) => fun ω => L ω p.1 p.2)
      (fun p => hMeas p.1 p.2)).mp hIndep
    calc μ.map (fun ω (p : Fin N × Fin N) => L ω p.1 p.2)
        = Measure.pi (fun p : Fin N × Fin N =>
            μ.map (fun ω => L ω p.1 p.2)) := h1
      _ = gaussPi N v := by
          congr 1
          funext p
          exact hLaw p.1 p.2
  have hAE : AEMeasurable (fun ω (p : Fin N × Fin N) => L ω p.1 p.2) μ :=
    aemeasurable_pi_lambda _ fun p => hMeas p.1 p.2
  set G : ((Fin N × Fin N) → ℝ) → ℝ := fun x =>
    (∑ p, ((x p + x p.swap) / 2) ^ 2) / ∑ p, x p ^ 2 with hG
  have hGmeas : Measurable G := by fun_prop
  have hGeq : ∀ x : (Fin N × Fin N) → ℝ,
      G x = frobSq (symPart (matOf x)) / frobSq (matOf x) := by
    intro x
    rw [hG, frobSq_symPart_matOf, frobSq_matOf]
  have step1 : ∫ ω, frobSq (symPart (L ω)) / frobSq (L ω) ∂μ
      = ∫ x, G x ∂(gaussPi N v) := by
    rw [← hjoint, integral_map hAE hGmeas.aestronglyMeasurable]
    refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
    show frobSq (symPart (L ω)) / frobSq (L ω) = G (fun p => L ω p.1 p.2)
    rw [hGeq, matOf_entries]
  rw [step1]
  have hae : ∀ᵐ x ∂(gaussPi N v), G x
      = 1 / 2 + (∑ p, x p * x p.swap) / (∑ p, x p ^ 2) / 2 := by
    filter_upwards [ae_sum_sq_ne_zero hv hN] with x hx
    have hM : matOf x ≠ 0 := by
      intro h0
      apply hx
      rw [← frobSq_matOf, h0]
      simp [frobSq]
    rw [hGeq, frobSq_symPart_div (matOf x) hM, trace_matOf, frobSq_matOf]
    ring
  rw [integral_congr_ae hae, integral_add (integrable_const _)
    (((integrable_finset_sum _ fun p _ =>
      integrable_ratio (N := N) (v := v) p p.swap).congr
        (Filter.Eventually.of_forall fun x => (Finset.sum_div _ _ _).symm)).div_const 2)]
  have h2 : ∫ x, ((∑ p, x p * x p.swap) / (∑ p, x p ^ 2)) / 2 ∂(gaussPi N v)
      = (1 / N : ℝ) / 2 := by
    rw [integral_div, integral_crossSum_ratio hv hN]
  rw [h2]
  simp only [integral_const, probReal_univ, smul_eq_mul, one_mul]
  field_simp

/-- The Gaussian clause of the corollary, skew part:
`E[‖T‖²/‖L‖²] = (N-1)/(2N)`. -/
theorem integral_frobSq_skewPart_div_frobSq_gaussian
    (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
    (hMeas : ∀ i j, AEMeasurable (fun ω => L ω i j) μ)
    (hLaw : ∀ i j, μ.map (fun ω => L ω i j) = gaussianReal 0 v)
    (hv : v ≠ 0) (hN : 0 < N) :
    ∫ ω, frobSq (skewPart (L ω)) / frobSq (L ω) ∂μ = (N - 1) / (2 * N) := by
  classical
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hjoint : μ.map (fun ω (p : Fin N × Fin N) => L ω p.1 p.2) = gaussPi N v := by
    have h1 := (iIndepFun_iff_map_fun_eq_pi_map
      (f := fun (p : Fin N × Fin N) => fun ω => L ω p.1 p.2)
      (fun p => hMeas p.1 p.2)).mp hIndep
    calc μ.map (fun ω (p : Fin N × Fin N) => L ω p.1 p.2)
        = Measure.pi (fun p : Fin N × Fin N =>
            μ.map (fun ω => L ω p.1 p.2)) := h1
      _ = gaussPi N v := by
          congr 1
          funext p
          exact hLaw p.1 p.2
  have hAE : AEMeasurable (fun ω (p : Fin N × Fin N) => L ω p.1 p.2) μ :=
    aemeasurable_pi_lambda _ fun p => hMeas p.1 p.2
  set G : ((Fin N × Fin N) → ℝ) → ℝ := fun x =>
    (∑ p, ((x p - x p.swap) / 2) ^ 2) / ∑ p, x p ^ 2 with hG
  have hGmeas : Measurable G := by fun_prop
  have hGeq : ∀ x : (Fin N × Fin N) → ℝ,
      G x = frobSq (skewPart (matOf x)) / frobSq (matOf x) := by
    intro x
    rw [hG, frobSq_skewPart_matOf, frobSq_matOf]
  have step1 : ∫ ω, frobSq (skewPart (L ω)) / frobSq (L ω) ∂μ
      = ∫ x, G x ∂(gaussPi N v) := by
    rw [← hjoint, integral_map hAE hGmeas.aestronglyMeasurable]
    refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
    show frobSq (skewPart (L ω)) / frobSq (L ω) = G (fun p => L ω p.1 p.2)
    rw [hGeq, matOf_entries]
  rw [step1]
  have hae : ∀ᵐ x ∂(gaussPi N v), G x
      = 1 / 2 - (∑ p, x p * x p.swap) / (∑ p, x p ^ 2) / 2 := by
    filter_upwards [ae_sum_sq_ne_zero hv hN] with x hx
    have hM : matOf x ≠ 0 := by
      intro h0
      apply hx
      rw [← frobSq_matOf, h0]
      simp [frobSq]
    rw [hGeq, frobSq_skewPart_div (matOf x) hM, trace_matOf, frobSq_matOf]
    ring
  rw [integral_congr_ae hae, integral_sub (integrable_const _)
    (((integrable_finset_sum _ fun p _ =>
      integrable_ratio (N := N) (v := v) p p.swap).congr
        (Filter.Eventually.of_forall fun x => (Finset.sum_div _ _ _).symm)).div_const 2)]
  have h2 : ∫ x, ((∑ p, x p * x p.swap) / (∑ p, x p ^ 2)) / 2 ∂(gaussPi N v)
      = (1 / N : ℝ) / 2 := by
    rw [integral_div, integral_crossSum_ratio hv hN]
  rw [h2]
  simp only [integral_const, probReal_univ, smul_eq_mul, one_mul]
  field_simp

end GaussianRatio

end Appendices

