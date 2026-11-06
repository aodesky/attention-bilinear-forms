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

The proof follows the paper's.

* Moments.  For `i ≠ j`, `S_ij = ½(L_ij + L_ji)` with `L_ij`, `L_ji`
  independent of variance `σ²`, so `E S_ij² = ½σ²`
  (`integral_symPart_sq_of_ne`); the diagonal entries give
  `E S_ii² = E L_ii² = σ²`.  Summing, the `N(N-1)/2` off-diagonal pairs
  contribute `σ²` each and the diagonal `Nσ²`.  The skew case is the
  same with `E T_ij² = ½σ²` off the diagonal and no diagonal
  contribution.
* Gaussian case.  `L` is an isotropic Gaussian vector in the
  `N²`-dimensional space `M_N(ℝ)` with the Frobenius inner product, and
  the expected fraction of its squared norm in a subspace `W` is
  `dim W / N²` (`integral_sum_sq_proj_div`, stated for an iid Gaussian
  vector in `ℝ^ι` and the orthogonal projection onto any subspace).
  The symmetric and antisymmetric matrices form orthogonal subspaces
  (Lemma 5, `frobInner_eq_zero_of_isSymm_of_transpose_eq_neg`), so `S`
  and `T` are the orthogonal projections of `L` onto them, and their
  dimensions are `N(N+1)/2` and `N(N-1)/2` (`finrank_symSub`,
  `finrank_skewSub`).

  The paper quotes the fact about isotropic Gaussian vectors without
  proof; the Lean proof supplies it.  Write the squared norm of the
  projection `Px` as `⟨x, Px⟩ = ∑ᵢⱼ Pᵢⱼ xᵢ xⱼ`.  Isotropy (invariance of
  the law under orthogonal maps), applied to the coordinate
  permutations and the sign change of one coordinate, gives
  `E[xᵢ xⱼ / ‖x‖²] = δᵢⱼ / n`, so the expected fraction is
  `tr P / n = dim W / n`.
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

/-! ### The moments of `S` and `T` -/

section MomentBaseline

variable (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
  (hMemLp : ∀ i j, MemLp (fun ω => L ω i j) 2 μ)
  (hMean : ∀ i j, ∫ ω, L ω i j ∂μ = 0)
  (hVar : ∀ i j, ∫ ω, (L ω i j) ^ 2 ∂μ = σ ^ 2)

include hIndep in
private lemma indepFun_entry_transpose {i j : Fin N} (hij : i ≠ j) :
    (fun ω => L ω i j) ⟂ᵢ[μ] (fun ω => L ω j i) :=
  hIndep.indepFun (i := (i, j)) (j := (j, i))
    (by simp [Prod.ext_iff]; exact fun h' => absurd h' hij)

include hIndep hMemLp in
private lemma integrable_mul_transpose {i j : Fin N} (hij : i ≠ j) :
    Integrable (fun ω => L ω i j * L ω j i) μ :=
  (indepFun_entry_transpose hIndep hij).integrable_mul
    ((hMemLp _ _).integrable one_le_two) ((hMemLp _ _).integrable one_le_two)

include hIndep hMemLp hMean in
/-- `E[L_ij L_ji] = 0` for `i ≠ j`: the two entries are independent and
mean zero. -/
private lemma integral_mul_transpose {i j : Fin N} (hij : i ≠ j) :
    ∫ ω, L ω i j * L ω j i ∂μ = 0 := by
  rw [(indepFun_entry_transpose hIndep hij).integral_fun_mul_eq_mul_integral
    (hMemLp _ _).aestronglyMeasurable (hMemLp _ _).aestronglyMeasurable]
  simp only [hMean, zero_mul]

include hIndep hMemLp hMean hVar in
/-- `E[(L_ij + c L_ji)²] = 2σ²` for `i ≠ j` and `c = ±1`. -/
private lemma integral_sq_add_mul_transpose {i j : Fin N} (hij : i ≠ j) {c : ℝ}
    (hc : c ^ 2 = 1) :
    ∫ ω, (L ω i j + c * L ω j i) ^ 2 ∂μ = 2 * σ ^ 2 := by
  have hpt : ∀ ω, (L ω i j + c * L ω j i) ^ 2
      = L ω i j ^ 2 + 2 * c * (L ω i j * L ω j i) + L ω j i ^ 2 := fun ω => by
    linear_combination (L ω j i) ^ 2 * hc
  simp_rw [hpt]
  have h1 := (hMemLp i j).integrable_sq
  have h2 := (integrable_mul_transpose hIndep hMemLp hij).const_mul (2 * c)
  have h3 := (hMemLp j i).integrable_sq
  rw [integral_add (f := fun ω => L ω i j ^ 2 + 2 * c * (L ω i j * L ω j i))
      (h1.add h2) h3, integral_add h1 h2, integral_const_mul,
    integral_mul_transpose hIndep hMemLp hMean hij, hVar, hVar]
  ring

include hIndep hMemLp hMean hVar in
/-- `E S_ij² = ½σ²` for `i ≠ j`. -/
theorem integral_symPart_sq_of_ne {i j : Fin N} (hij : i ≠ j) :
    ∫ ω, symPart (L ω) i j ^ 2 ∂μ = σ ^ 2 / 2 := by
  have hpt : ∀ ω, symPart (L ω) i j ^ 2 = (L ω i j + 1 * L ω j i) ^ 2 / 4 := fun ω => by
    simp only [symPart, Matrix.smul_apply, Matrix.add_apply, Matrix.transpose_apply,
      smul_eq_mul]
    ring
  simp_rw [hpt]
  rw [integral_div, integral_sq_add_mul_transpose hIndep hMemLp hMean hVar hij (by norm_num)]
  ring

include hIndep hMemLp hMean hVar in
/-- `E T_ij² = ½σ²` for `i ≠ j`. -/
theorem integral_skewPart_sq_of_ne {i j : Fin N} (hij : i ≠ j) :
    ∫ ω, skewPart (L ω) i j ^ 2 ∂μ = σ ^ 2 / 2 := by
  have hpt : ∀ ω, skewPart (L ω) i j ^ 2 = (L ω i j + (-1) * L ω j i) ^ 2 / 4 := fun ω => by
    simp only [skewPart, Matrix.smul_apply, Matrix.sub_apply, Matrix.transpose_apply,
      smul_eq_mul]
    ring
  simp_rw [hpt]
  rw [integral_div, integral_sq_add_mul_transpose hIndep hMemLp hMean hVar hij (by norm_num)]
  ring

/-- `S_ii = L_ii`. -/
private lemma symPart_apply_self (A : Matrix (Fin N) (Fin N) ℝ) (i : Fin N) :
    symPart A i i = A i i := by
  simp only [symPart, Matrix.smul_apply, Matrix.add_apply, Matrix.transpose_apply, smul_eq_mul]
  ring

/-- `T_ii = 0`. -/
private lemma skewPart_apply_self (A : Matrix (Fin N) (Fin N) ℝ) (i : Fin N) :
    skewPart A i i = 0 := by
  simp [skewPart]

include hMemLp in
private lemma integrable_symPart_sq (i j : Fin N) :
    Integrable (fun ω => symPart (L ω) i j ^ 2) μ := by
  have hfun : ∀ ω, symPart (L ω) i j = (1 / 2 : ℝ) * (L ω i j + L ω j i) := by
    intro ω; simp [symPart]
  simp_rw [hfun]
  exact (((hMemLp i j).add (hMemLp j i)).const_mul _).integrable_sq

include hMemLp in
private lemma integrable_skewPart_sq (i j : Fin N) :
    Integrable (fun ω => skewPart (L ω) i j ^ 2) μ := by
  have hfun : ∀ ω, skewPart (L ω) i j = (1 / 2 : ℝ) * (L ω i j - L ω j i) := by
    intro ω; simp [skewPart]
  simp_rw [hfun]
  exact (((hMemLp i j).sub (hMemLp j i)).const_mul _).integrable_sq

/-- `∑ᵢⱼ (if i = j then a else b) = N a + (N² - N) b`, split as the diagonal
and the off-diagonal entries. -/
private lemma sum_ite_eq_diag_offdiag (a b : ℝ) :
    ∑ i : Fin N, ∑ j : Fin N, (if i = j then a else b) = N * a + (N * (N - 1)) * b := by
  have h : ∀ i j : Fin N, (if i = j then a else b) = b + (if i = j then a - b else 0) := by
    intro i j; split_ifs <;> ring
  simp_rw [h, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ, if_true,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

include hMemLp hVar in
/-- `E‖L‖² = N² σ²` for a random matrix with mean-zero entries of
common variance `σ²`. -/
theorem integral_frobSq_iid : ∫ ω, frobSq (L ω) ∂μ = N ^ 2 * σ ^ 2 := by
  simp_rw [frobSq_eq_sum_sq]
  rw [integral_finset_sum _ fun i _ =>
    integrable_finset_sum _ fun j _ => (hMemLp i j).integrable_sq]
  simp_rw [integral_finset_sum _ fun j _ => (hMemLp _ j).integrable_sq, hVar]
  simp [Finset.card_univ]
  ring

include hIndep hMemLp hMean hVar in
/-- `E‖S‖² = N(N+1)/2 · σ²`: the diagonal contributes `Nσ²` and each of the
`N(N-1)/2` off-diagonal pairs `{i, j}` contributes `E S_ij² + E S_ji² = σ²`. -/
theorem integral_frobSq_symPart_iid :
    ∫ ω, frobSq (symPart (L ω)) ∂μ = N * (N + 1) / 2 * σ ^ 2 := by
  have hterm : ∀ i j : Fin N,
      ∫ ω, symPart (L ω) i j ^ 2 ∂μ = if i = j then σ ^ 2 else σ ^ 2 / 2 := by
    intro i j
    split_ifs with hij
    · subst hij
      simp_rw [symPart_apply_self]
      exact hVar i i
    · exact integral_symPart_sq_of_ne hIndep hMemLp hMean hVar hij
  simp_rw [frobSq_eq_sum_sq]
  rw [integral_finset_sum _ fun i _ =>
    integrable_finset_sum _ fun j _ => integrable_symPart_sq hMemLp i j]
  simp_rw [integral_finset_sum _ fun j _ => integrable_symPart_sq hMemLp _ j, hterm,
    sum_ite_eq_diag_offdiag]
  ring

include hIndep hMemLp hMean hVar in
/-- `E‖T‖² = N(N-1)/2 · σ²`: the same count, with no diagonal contribution. -/
theorem integral_frobSq_skewPart_iid :
    ∫ ω, frobSq (skewPart (L ω)) ∂μ = N * (N - 1) / 2 * σ ^ 2 := by
  have hterm : ∀ i j : Fin N,
      ∫ ω, skewPart (L ω) i j ^ 2 ∂μ = if i = j then 0 else σ ^ 2 / 2 := by
    intro i j
    split_ifs with hij
    · subst hij
      simp [skewPart_apply_self]
    · exact integral_skewPart_sq_of_ne hIndep hMemLp hMean hVar hij
  simp_rw [frobSq_eq_sum_sq]
  rw [integral_finset_sum _ fun i _ =>
    integrable_finset_sum _ fun j _ => integrable_skewPart_sq hMemLp i j]
  simp_rw [integral_finset_sum _ fun j _ => integrable_skewPart_sq hMemLp _ j, hterm,
    sum_ite_eq_diag_offdiag]
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

/-! ### Isotropic Gaussian vectors

An iid Gaussian vector `x ∈ ℝ^ι` (law `gaussPi ι v`) is isotropic.  The
expected fraction of its squared norm in a subspace `W` is
`dim W / |ι|`.  This is the fact the paper quotes; we prove it from the
invariance of the law under two kinds of orthogonal maps, coordinate
permutations and the sign change of one coordinate. -/

section Isotropic

open scoped NNReal

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The law of an iid Gaussian vector: the product of copies of
`gaussianReal 0 v` on `ι → ℝ`. -/
noncomputable abbrev gaussPi (ι : Type*) [Fintype ι] (v : ℝ≥0) : Measure (ι → ℝ) :=
  Measure.pi fun _ => gaussianReal 0 v

/-- Invariance of `gaussPi` under a permutation of the coordinates. -/
private lemma integral_gaussPi_comp_perm {v : ℝ≥0} (e : ι ≃ ι) (g : (ι → ℝ) → ℝ) :
    ∫ x, g (fun p => x (e p)) ∂(gaussPi ι v) = ∫ x, g x ∂(gaussPi ι v) := by
  have hmp := MeasureTheory.measurePreserving_piCongrLeft
    (fun _ : ι => gaussianReal (0 : ℝ) v) e.symm
  have happ : ∀ (x : ι → ℝ),
      (MeasurableEquiv.piCongrLeft (fun _ : ι => ℝ) e.symm) x = fun p => x (e p) := by
    intro x
    funext p
    conv_lhs => rw [show p = e.symm (e p) by simp]
    exact MeasurableEquiv.piCongrLeft_apply_apply (β := fun _ : ι => ℝ) e.symm x (e p)
  calc ∫ x, g (fun p => x (e p)) ∂(gaussPi ι v)
      = ∫ x, g ((MeasurableEquiv.piCongrLeft (fun _ : ι => ℝ) e.symm) x) ∂(gaussPi ι v) := by
        refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
        simp only [happ]
    _ = ∫ x, g x ∂(gaussPi ι v) := hmp.integral_comp' g

/-- Invariance of `gaussPi` under the sign change of one coordinate. -/
private lemma integral_gaussPi_comp_flip {v : ℝ≥0} (p₀ : ι) (g : (ι → ℝ) → ℝ) :
    ∫ x, g (fun p => if p = p₀ then -x p else x p) ∂(gaussPi ι v)
      = ∫ x, g x ∂(gaussPi ι v) := by
  set e : ∀ _ : ι, ℝ ≃ᵐ ℝ :=
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
  have hmp : MeasurePreserving (fun (x : ι → ℝ) p => e p (x p)) (gaussPi ι v) (gaussPi ι v) :=
    MeasureTheory.measurePreserving_pi _ _ hmpf
  have hemb : MeasurableEmbedding (fun (x : ι → ℝ) p => e p (x p)) :=
    (MeasurableEquiv.piCongrRight e).measurableEmbedding
  calc ∫ x, g (fun p => if p = p₀ then -x p else x p) ∂(gaussPi ι v)
      = ∫ x, g (fun p => e p (x p)) ∂(gaussPi ι v) := by
        refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
        show g (fun p => if p = p₀ then -x p else x p) = g (fun p => e p (x p))
        congr 1
        funext p
        by_cases h : p = p₀
        · simp only [he, if_pos h]
          rfl
        · simp only [he, if_neg h]
          rfl
    _ = ∫ x, g x ∂(gaussPi ι v) := hmp.integral_comp hemb g

/-- Almost surely `∑ p, x p ² ≠ 0` (for `ι` nonempty and `v ≠ 0`). -/
private lemma ae_sum_sq_ne_zero [Nonempty ι] {v : ℝ≥0} (hv : v ≠ 0) :
    ∀ᵐ x ∂(gaussPi ι v), ∑ p, x p ^ 2 ≠ 0 := by
  haveI : NoAtoms (gaussianReal (0 : ℝ) v) := noAtoms_gaussianReal hv
  have h := Measure.ae_eval_ne (fun _ : ι => gaussianReal (0 : ℝ) v) (Classical.arbitrary ι) 0
  filter_upwards [h] with x hx h0
  exact hx (pow_eq_zero_iff (n := 2) (by norm_num) |>.mp
    ((Finset.sum_eq_zero_iff_of_nonneg fun p _ => sq_nonneg (x p)).mp h0 _
      (Finset.mem_univ _)))

/-- The integrands below are bounded by `1` in absolute value. -/
private lemma abs_ratio_le_one (x : ι → ℝ) (q r : ι) :
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

private lemma integrable_ratio {v : ℝ≥0} (q r : ι) :
    Integrable (fun x => x q * x r / ∑ p, x p ^ 2) (gaussPi ι v) := by
  have hmeas : Measurable (fun x : ι → ℝ => x q * x r / ∑ p, x p ^ 2) := by
    fun_prop
  refine Integrable.mono' (integrable_const 1) hmeas.aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs]
  exact abs_ratio_le_one x q r

/-- The expected fraction of the squared norm carried by one coordinate
is `1/|ι|`, by invariance under the coordinate permutations. -/
private lemma integral_sq_ratio [Nonempty ι] {v : ℝ≥0} (hv : v ≠ 0) (q : ι) :
    ∫ x, x q * x q / ∑ p, x p ^ 2 ∂(gaussPi ι v) = 1 / (Fintype.card ι : ℝ) := by
  have hswap : ∀ q' : ι,
      (∫ x, x q' * x q' / ∑ p, x p ^ 2 ∂(gaussPi ι v))
        = ∫ x, x q * x q / ∑ p, x p ^ 2 ∂(gaussPi ι v) := by
    intro q'
    have h := integral_gaussPi_comp_perm (v := v) (Equiv.swap q q')
      (fun y => y q * y q / ∑ p, y p ^ 2)
    rw [← h]
    refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
    have hden : (∑ p, x (Equiv.swap q q' p) ^ 2) = ∑ p, x p ^ 2 :=
      Fintype.sum_equiv (Equiv.swap q q') _ _ fun p => rfl
    simp only
    rw [Equiv.swap_apply_left, hden]
  have hsum : (∑ q' : ι, ∫ x, x q' * x q' / ∑ p, x p ^ 2 ∂(gaussPi ι v)) = 1 := by
    rw [← integral_finset_sum _ fun q' _ => integrable_ratio q' q']
    have hone : ∀ᵐ x ∂(gaussPi ι v), (∑ q' : ι, x q' * x q' / ∑ p, x p ^ 2) = 1 := by
      filter_upwards [ae_sum_sq_ne_zero hv] with x hx
      rw [← Finset.sum_div, div_eq_one_iff_eq hx]
      simp [sq]
    rw [integral_congr_ae hone]
    simp
  rw [Finset.sum_congr rfl fun q' _ => hswap q', Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul] at hsum
  have hn : (Fintype.card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  rw [eq_div_iff hn]
  linarith

/-- The cross terms integrate to zero, by invariance under the sign change of
one coordinate. -/
private lemma integral_cross_ratio_eq_zero {v : ℝ≥0} {q r : ι} (hqr : q ≠ r) :
    ∫ x, x q * x r / ∑ p, x p ^ 2 ∂(gaussPi ι v) = 0 := by
  have hflip := integral_gaussPi_comp_flip (v := v) q (fun y => y q * y r / ∑ p, y p ^ 2)
  have hneg : ∀ x : ι → ℝ,
      (fun y => y q * y r / ∑ p, y p ^ 2) (fun p => if p = q then -x p else x p)
        = -(x q * x r / ∑ p, x p ^ 2) := by
    intro x
    have hsq : (∑ p, (if p = q then -x p else x p) ^ 2) = ∑ p, x p ^ 2 :=
      Finset.sum_congr rfl fun p _ => by split_ifs <;> ring
    simp only [if_neg (Ne.symm hqr), hsq, if_true]
    ring
  rw [funext hneg, integral_neg] at hflip
  linarith [hflip]

/-- `E[x_q x_r / ‖x‖²] = δ_qr / |ι|`. -/
private lemma integral_coord_ratio [Nonempty ι] {v : ℝ≥0} (hv : v ≠ 0) (q r : ι) :
    ∫ x, x q * x r / ∑ p, x p ^ 2 ∂(gaussPi ι v)
      = if q = r then 1 / (Fintype.card ι : ℝ) else 0 := by
  split_ifs with h
  · subst h; exact integral_sq_ratio hv q
  · exact integral_cross_ratio_eq_zero h

/-- **Expected fraction of the squared norm in a subspace.**  If `P` is the
orthogonal projection of `ℝ^ι` onto a subspace `W` (`P` is a projection onto
`W` and `x - P x ⊥ W`), then for an iid Gaussian vector `x`,
`E[‖P x‖² / ‖x‖²] = dim W / |ι|`. -/
theorem integral_sum_sq_proj_div [Nonempty ι] {v : ℝ≥0} (hv : v ≠ 0)
    (W : Submodule ℝ (ι → ℝ)) (P : (ι → ℝ) →ₗ[ℝ] (ι → ℝ)) (hP : LinearMap.IsProj W P)
    (horth : ∀ x, ∀ w ∈ W, ∑ i, w i * (x i - P x i) = 0) :
    ∫ x, (∑ i, P x i ^ 2) / ∑ i, x i ^ 2 ∂(gaussPi ι v)
      = Module.finrank ℝ W / Fintype.card ι := by
  -- `‖P x‖² = ⟨x, P x⟩ = ∑ᵢⱼ Pᵢⱼ xᵢ xⱼ`, with `Pᵢⱼ = (P eⱼ)ᵢ`
  have hnorm : ∀ x : ι → ℝ, ∑ i, P x i ^ 2 = ∑ i, ∑ j, P (Pi.single j 1) i * (x i * x j) := by
    intro x
    have h0 := horth x (P x) (hP.map_mem x)
    have hPx : ∀ i, P x i = ∑ j, x j * P (Pi.single j 1) i := by
      intro i
      conv_lhs => rw [← Finset.univ_sum_single x]
      simp only [map_sum, Finset.sum_apply]
      refine Finset.sum_congr rfl fun j _ => ?_
      have hs : (Pi.single j (x j) : ι → ℝ) = x j • (Pi.single j (1 : ℝ) : ι → ℝ) := by
        ext k; by_cases h : k = j <;> simp [Pi.single_apply, h]
      rw [hs, map_smul, Pi.smul_apply, smul_eq_mul]
    have h1 : ∑ i, P x i ^ 2 = ∑ i, P x i * x i := by
      have : ∑ i, P x i * (x i - P x i) = 0 := h0
      simp only [mul_sub, Finset.sum_sub_distrib, sub_eq_zero] at this
      simp only [sq]
      exact this.symm
    rw [h1]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hPx i, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  have hpt : ∀ x : ι → ℝ, (∑ i, P x i ^ 2) / ∑ i, x i ^ 2
      = ∑ i, ∑ j, P (Pi.single j 1) i * (x i * x j / ∑ p, x p ^ 2) := by
    intro x
    rw [hnorm, Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun j _ => by ring
  simp_rw [hpt]
  rw [integral_finset_sum _ fun i _ => integrable_finset_sum _ fun j _ =>
    (integrable_ratio i j).const_mul _]
  simp_rw [integral_finset_sum _ fun j _ => (integrable_ratio _ j).const_mul _,
    integral_const_mul, integral_coord_ratio hv, mul_ite, mul_zero, Finset.sum_ite_eq,
    Finset.mem_univ, if_true]
  -- `∑ᵢ Pᵢᵢ = tr P = dim W`
  have htr : ∑ i, P (Pi.single i 1) i = Module.finrank ℝ W := by
    rw [← hP.trace, LinearMap.trace_eq_matrix_trace ℝ (Pi.basisFun ℝ ι), Matrix.trace]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Matrix.diag_apply, LinearMap.toMatrix_apply, Pi.basisFun_apply, Pi.basisFun_repr]
  rw [← Finset.sum_mul, htr]
  ring

end Isotropic

/-! ### The Gaussian case

`M_N(ℝ)` with the Frobenius inner product is `ℝ^(N × N)` with the dot
product.  The symmetric and antisymmetric matrices form orthogonal
subspaces of dimensions `N(N+1)/2` and `N(N-1)/2`; `S` and `T` are the
orthogonal projections of `L` onto them. -/

section GaussianRatio

open scoped NNReal

/-- The matrix assembled from a point of `ℝ^(N × N)`. -/
private def matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) : Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i j => x (i, j)

private lemma matOf_entries {N : ℕ} (A : Matrix (Fin N) (Fin N) ℝ) :
    matOf (fun p => A p.1 p.2) = A := rfl

/-- The Frobenius inner product is the dot product of the entries. -/
private lemma frobInner_matOf {N : ℕ} (x y : (Fin N × Fin N) → ℝ) :
    frobInner (matOf x) (matOf y) = ∑ p, x p * y p := by
  simp only [frobInner, Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply,
    matOf, Matrix.of_apply]
  exact (Fintype.sum_prod_type (f := fun p : Fin N × Fin N => x p * y p)).symm

private lemma frobSq_matOf {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    frobSq (matOf x) = ∑ p, x p ^ 2 := by
  rw [frobSq_eq_frobInner, frobInner_matOf]
  simp [sq]

/-- The subspace of symmetric matrices. -/
private def symSub (N : ℕ) : Submodule ℝ ((Fin N × Fin N) → ℝ) where
  carrier := {x | ∀ p, x p.swap = x p}
  add_mem' hx hy p := by simp [hx p, hy p]
  zero_mem' p := rfl
  smul_mem' c x hx p := by simp [hx p]

/-- The subspace of antisymmetric matrices. -/
private def skewSub (N : ℕ) : Submodule ℝ ((Fin N × Fin N) → ℝ) where
  carrier := {x | ∀ p, x p.swap = -x p}
  add_mem' hx hy p := by simp [hx p, hy p]; ring
  zero_mem' p := by simp
  smul_mem' c x hx p := by simp [hx p]

/-- `L ↦ S`, as a linear map on `ℝ^(N × N)`. -/
private noncomputable def symProj (N : ℕ) : ((Fin N × Fin N) → ℝ) →ₗ[ℝ] ((Fin N × Fin N) → ℝ) where
  toFun x p := (x p + x p.swap) / 2
  map_add' x y := by funext p; simp only [Pi.add_apply]; ring
  map_smul' c x := by funext p; simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]; ring

/-- `L ↦ T`, as a linear map on `ℝ^(N × N)`. -/
private noncomputable def skewProj (N : ℕ) : ((Fin N × Fin N) → ℝ) →ₗ[ℝ] ((Fin N × Fin N) → ℝ) where
  toFun x p := (x p - x p.swap) / 2
  map_add' x y := by funext p; simp only [Pi.add_apply]; ring
  map_smul' c x := by funext p; simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]; ring

private lemma matOf_symProj {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    matOf (symProj N x) = symPart (matOf x) := by
  ext i j
  simp [matOf, symProj, symPart]
  ring

private lemma matOf_skewProj {N : ℕ} (x : (Fin N × Fin N) → ℝ) :
    matOf (skewProj N x) = skewPart (matOf x) := by
  ext i j
  simp [matOf, skewProj, skewPart]
  ring

private lemma isSymm_matOf {N : ℕ} {w : (Fin N × Fin N) → ℝ} (hw : w ∈ symSub N) :
    (matOf w).IsSymm := by
  ext i j
  simp only [Matrix.transpose_apply, matOf, Matrix.of_apply]
  exact hw (i, j)

private lemma transpose_matOf {N : ℕ} {w : (Fin N × Fin N) → ℝ} (hw : w ∈ skewSub N) :
    (matOf w)ᵀ = -matOf w := by
  ext i j
  simp only [Matrix.transpose_apply, matOf, Matrix.of_apply, Matrix.neg_apply]
  exact hw (i, j)

private lemma isProj_symProj (N : ℕ) : LinearMap.IsProj (symSub N) (symProj N) where
  map_mem x p := by simp [symProj, add_comm]
  map_id x hx := by funext p; simp only [symProj, LinearMap.coe_mk, AddHom.coe_mk, hx p]; ring

private lemma isProj_skewProj (N : ℕ) : LinearMap.IsProj (skewSub N) (skewProj N) where
  map_mem x p := by simp [skewProj]; ring
  map_id x hx := by funext p; simp only [skewProj, LinearMap.coe_mk, AddHom.coe_mk, hx p]; ring

/-- `L - S = T` is orthogonal to the symmetric subspace (Lemma 5). -/
private lemma orth_symProj (N : ℕ) (x : (Fin N × Fin N) → ℝ) (w) (hw : w ∈ symSub N) :
    ∑ p, w p * (x p - symProj N x p) = 0 := by
  have hT : x - symProj N x = skewProj N x := by
    funext p; simp [symProj, skewProj]; ring
  have h := frobInner_eq_zero_of_isSymm_of_transpose_eq_neg (isSymm_matOf hw)
    (transpose_matOf ((isProj_skewProj N).map_mem x))
  rw [frobInner_matOf, ← hT] at h
  exact h

/-- `L - T = S` is orthogonal to the antisymmetric subspace (Lemma 5). -/
private lemma orth_skewProj (N : ℕ) (x : (Fin N × Fin N) → ℝ) (w) (hw : w ∈ skewSub N) :
    ∑ p, w p * (x p - skewProj N x p) = 0 := by
  have hS : x - skewProj N x = symProj N x := by
    funext p; simp [symProj, skewProj]; ring
  have h := frobInner_eq_zero_of_isSymm_of_transpose_eq_neg
    (isSymm_matOf ((isProj_symProj N).map_mem x)) (transpose_matOf hw)
  rw [frobInner_comm, frobInner_matOf, ← hS] at h
  exact h

/-- The trace of `L ↦ ½(L ± Lᵀ)` on `ℝ^(N × N)`: `∑_p ½(1 ± [p = pᵀ])`. -/
private lemma sum_single_swap {N : ℕ} (c : ℝ) :
    ∑ p : Fin N × Fin N, ((Pi.single p 1 : (Fin N × Fin N) → ℝ) p
      + c * (Pi.single p 1 : (Fin N × Fin N) → ℝ) p.swap) / 2 = (N ^ 2 + c * N) / 2 := by
  have h : ∀ p : Fin N × Fin N, ((Pi.single p 1 : (Fin N × Fin N) → ℝ) p
      + c * (Pi.single p 1 : (Fin N × Fin N) → ℝ) p.swap) / 2
        = 1 / 2 + c / 2 * (if p.1 = p.2 then 1 else 0) := by
    intro p
    have hs : (Pi.single p 1 : (Fin N × Fin N) → ℝ) p.swap = if p.1 = p.2 then 1 else 0 := by
      by_cases h : p.1 = p.2
      · obtain ⟨i, j⟩ := p
        simp only at h
        subst h
        simp
      · rw [if_neg h, Pi.single_apply, if_neg]
        intro h'
        simp only [Prod.ext_iff, Prod.fst_swap, Prod.snd_swap] at h'
        exact h h'.2
    rw [hs, Pi.single_eq_same]
    ring
  simp_rw [h, Finset.sum_add_distrib, ← Finset.mul_sum, Fintype.sum_prod_type,
    Finset.sum_ite_eq, Finset.mem_univ, if_true, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  ring

/-- `dim Sym_N = N(N+1)/2`, computed as the trace of the projection `L ↦ S`. -/
theorem finrank_symSub (N : ℕ) : (Module.finrank ℝ (symSub N) : ℝ) = N * (N + 1) / 2 := by
  rw [← (isProj_symProj N).trace, LinearMap.trace_eq_matrix_trace ℝ (Pi.basisFun ℝ _),
    Matrix.trace]
  have := sum_single_swap (N := N) 1
  simp only [one_mul] at this
  simp only [Matrix.diag_apply, LinearMap.toMatrix_apply, Pi.basisFun_apply, Pi.basisFun_repr]
  convert this using 1
  ring

/-- `dim Skew_N = N(N-1)/2`, computed as the trace of the projection `L ↦ T`. -/
theorem finrank_skewSub (N : ℕ) : (Module.finrank ℝ (skewSub N) : ℝ) = N * (N - 1) / 2 := by
  rw [← (isProj_skewProj N).trace, LinearMap.trace_eq_matrix_trace ℝ (Pi.basisFun ℝ _),
    Matrix.trace]
  have := sum_single_swap (N := N) (-1)
  simp only [Matrix.diag_apply, LinearMap.toMatrix_apply, Pi.basisFun_apply, Pi.basisFun_repr]
  convert this using 1
  · refine Finset.sum_congr rfl fun p _ => ?_
    simp only [skewProj, LinearMap.coe_mk, AddHom.coe_mk]
    ring
  · ring

variable {L : Ω → Matrix (Fin N) (Fin N) ℝ} {v : ℝ≥0}

/-- The law of the entries of `L` is `gaussPi`. -/
private lemma map_entries_eq_gaussPi
    (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
    (hMeas : ∀ i j, AEMeasurable (fun ω => L ω i j) μ)
    (hLaw : ∀ i j, μ.map (fun ω => L ω i j) = gaussianReal 0 v) :
    μ.map (fun ω (p : Fin N × Fin N) => L ω p.1 p.2) = gaussPi (Fin N × Fin N) v := by
  rw [(iIndepFun_iff_map_fun_eq_pi_map (f := fun (p : Fin N × Fin N) => fun ω => L ω p.1 p.2)
    (fun p => hMeas p.1 p.2)).mp hIndep]
  congr 1
  funext p
  exact hLaw p.1 p.2

/-- Transfer of `E[‖P L‖²/‖L‖²]` to `gaussPi`. -/
private lemma integral_ratio_eq_gaussPi
    (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
    (hMeas : ∀ i j, AEMeasurable (fun ω => L ω i j) μ)
    (hLaw : ∀ i j, μ.map (fun ω => L ω i j) = gaussianReal 0 v)
    (P : ((Fin N × Fin N) → ℝ) →ₗ[ℝ] ((Fin N × Fin N) → ℝ))
    (f : Matrix (Fin N) (Fin N) ℝ → Matrix (Fin N) (Fin N) ℝ)
    (hf : ∀ x, matOf (P x) = f (matOf x)) :
    ∫ ω, frobSq (f (L ω)) / frobSq (L ω) ∂μ
      = ∫ x, (∑ p, P x p ^ 2) / ∑ p, x p ^ 2 ∂(gaussPi (Fin N × Fin N) v) := by
  have hAE : AEMeasurable (fun ω (p : Fin N × Fin N) => L ω p.1 p.2) μ :=
    aemeasurable_pi_lambda _ fun p => hMeas p.1 p.2
  have hGmeas : Measurable fun x : (Fin N × Fin N) → ℝ => (∑ p, P x p ^ 2) / ∑ p, x p ^ 2 := by
    have : Measurable P := P.continuous_of_finiteDimensional.measurable
    refine Measurable.div (Finset.measurable_sum _ fun p _ => ?_) (by fun_prop)
    exact ((measurable_pi_apply p).comp this).pow_const 2
  rw [← map_entries_eq_gaussPi hIndep hMeas hLaw, integral_map hAE hGmeas.aestronglyMeasurable]
  refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
  simp only
  rw [← frobSq_matOf, ← frobSq_matOf, hf, matOf_entries]

/-- The Gaussian clause of the corollary: if the entries of `L` are iid
Gaussian, then `E[‖S‖²/‖L‖²] = dim Sym_N / N² = (N+1)/(2N)`.

Hypotheses: the entries are jointly independent with common law
`gaussianReal 0 v`; nondegeneracy `v ≠ 0` and `0 < N` (implicit in the
paper) is required since otherwise `‖L‖ = 0`. -/
theorem integral_frobSq_symPart_div_frobSq_gaussian
    (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
    (hMeas : ∀ i j, AEMeasurable (fun ω => L ω i j) μ)
    (hLaw : ∀ i j, μ.map (fun ω => L ω i j) = gaussianReal 0 v)
    (hv : v ≠ 0) (hN : 0 < N) :
    ∫ ω, frobSq (symPart (L ω)) / frobSq (L ω) ∂μ = (N + 1) / (2 * N) := by
  haveI : Nonempty (Fin N × Fin N) := ⟨(⟨0, hN⟩, ⟨0, hN⟩)⟩
  rw [integral_ratio_eq_gaussPi hIndep hMeas hLaw (symProj N) symPart matOf_symProj,
    integral_sum_sq_proj_div hv (symSub N) (symProj N) (isProj_symProj N) (orth_symProj N),
    finrank_symSub, Fintype.card_prod, Fintype.card_fin]
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  push_cast
  field_simp

/-- The Gaussian clause of the corollary, skew part:
`E[‖T‖²/‖L‖²] = dim Skew_N / N² = (N-1)/(2N)`. -/
theorem integral_frobSq_skewPart_div_frobSq_gaussian
    (hIndep : iIndepFun (fun (p : Fin N × Fin N) ω => L ω p.1 p.2) μ)
    (hMeas : ∀ i j, AEMeasurable (fun ω => L ω i j) μ)
    (hLaw : ∀ i j, μ.map (fun ω => L ω i j) = gaussianReal 0 v)
    (hv : v ≠ 0) (hN : 0 < N) :
    ∫ ω, frobSq (skewPart (L ω)) / frobSq (L ω) ∂μ = (N - 1) / (2 * N) := by
  haveI : Nonempty (Fin N × Fin N) := ⟨(⟨0, hN⟩, ⟨0, hN⟩)⟩
  rw [integral_ratio_eq_gaussPi hIndep hMeas hLaw (skewProj N) skewPart matOf_skewProj,
    integral_sum_sq_proj_div hv (skewSub N) (skewProj N) (isProj_skewProj N) (orth_skewProj N),
    finrank_skewSub, Fintype.card_prod, Fintype.card_fin]
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  push_cast
  field_simp

end GaussianRatio

end Appendices
