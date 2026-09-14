/-
Appendix "Random matrices",
Proposition (Expected symmetric energy for random low-rank QK products).

Let `W_K, W_Q ∈ M_{n×N}(ℝ)` be independent random matrices with iid
centered Gaussian entries, where `1 ≤ n ≤ N`.  Set `L = W_KᵀW_Q ∈ M_N(ℝ)`,
`S = ½(L + Lᵀ)`, `T = ½(L - Lᵀ)`, and define

    𝒮 = ‖S‖²/‖L‖²,    𝒜 = ‖T‖²/‖L‖².

Then

    𝔼[𝒮] = (N+1)/(2N),    𝔼[𝒜] = (N-1)/(2N),

independently of the head dimension `n`.

Formalization notes.
* "Independent with iid centered Gaussian entries" is stated as: the
  `2nN` entry variables (indexed by `(Fin n × Fin N) ⊕ (Fin n × Fin N)`,
  see `qkEntry`) are jointly independent, the entries of `W_K` have law
  `gaussianReal 0 v_K` and the entries of `W_Q` have law
  `gaussianReal 0 v_Q` (`expected_symmetric_energy_low_rank_qk`,
  `expected_skew_energy_low_rank_qk`).  The core computation is done
  for a common variance `v` (the private `…_common_variance` steps); the general case
  reduces to it by rescaling `W_K`, which leaves `𝒮` and `𝒜` unchanged.
* The hypotheses `v_K, v_Q ≠ 0` are required: for `v = 0` we would have
  `L = 0` almost surely and `𝒮` would be the junk value `0/0 = 0`.  The
  paper's phrase "iid centered Gaussian entries" implicitly assumes a
  nondegenerate Gaussian.
* Almost surely `‖L‖ ≠ 0` (proved below from `1 ≤ n` and `v ≠ 0`), so
  the ratio `𝒮` is genuinely defined almost everywhere; on the null set
  `{L = 0}` the Lean convention `x/0 = 0` is immaterial to the integral.
* The hypothesis `n ≤ N` is part of the paper's statement (heads have
  `n ≤ N`); it is not needed by the proof beyond `1 ≤ N`, in accordance
  with the paper's remark that the expectation is independent of `n`.
-/
import Appendices.RandomMatrices.EnergyIdentities

namespace Appendices

open MeasureTheory ProbabilityTheory Matrix BigOperators
open scoped NNReal ENNReal

namespace LowRankQK

variable {n N : ℕ} {v : ℝ≥0}

/-- Index type for the entries of the pair `(W_K, W_Q)`: `.inl (r, a)`
indexes the entry `(W_K)_{ra}` and `.inr (r, b)` the entry `(W_Q)_{rb}`. -/
abbrev EntryIndex (n N : ℕ) := (Fin n × Fin N) ⊕ (Fin n × Fin N)

/-- The canonical sample space: one real coordinate per matrix entry. -/
abbrev EntrySpace (n N : ℕ) := EntryIndex n N → ℝ

/-- The matrix `L = W_Kᵀ W_Q` as a function of the joint entries:
`L_{ab} = ∑ r (W_K)_{ra} (W_Q)_{rb}`. -/
def LMat (x : EntrySpace n N) : Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun a b => ∑ r, x (.inl (r, a)) * x (.inr (r, b))

lemma LMat_apply (x : EntrySpace n N) (a b : Fin N) :
    LMat x a b = ∑ r, x (.inl (r, a)) * x (.inr (r, b)) := rfl

/-- The joint law of the `2nN` iid centered Gaussian entries: the product
Gaussian measure on `EntrySpace n N`. -/
noncomputable def P (n N : ℕ) (v : ℝ≥0) : Measure (EntrySpace n N) :=
  Measure.pi fun _ => gaussianReal 0 v

instance : IsProbabilityMeasure (P n N v) := by
  unfold P; infer_instance

lemma measurable_LMat_apply (a b : Fin N) :
    Measurable fun x : EntrySpace n N => LMat x a b := by
  simp only [LMat_apply]
  exact Finset.measurable_sum _ fun r _ =>
    (measurable_pi_apply _).mul (measurable_pi_apply _)

lemma measurable_frobSq_LMat :
    Measurable fun x : EntrySpace n N => frobSq (LMat x) := by
  simp only [frobSq_eq_sum_sq]
  exact Finset.measurable_sum _ fun a _ => Finset.measurable_sum _ fun b _ =>
    (measurable_LMat_apply a b).pow_const 2

/-! ### Invariance of the product Gaussian measure -/

/-- Precomposition of the coordinates with a bijection of the entry
indices preserves `P`. -/
lemma integral_comp_perm (e : EntryIndex n N ≃ EntryIndex n N)
    {g : EntrySpace n N → ℝ} (hg : AEStronglyMeasurable g (P n N v)) :
    ∫ x, g (fun i => x (e i)) ∂(P n N v) = ∫ x, g x ∂(P n N v) := by
  have hcoe : ⇑(MeasurableEquiv.piCongrLeft (fun _ : EntryIndex n N => ℝ) e.symm)
      = fun x (i : EntryIndex n N) => x (e i) := by
    rw [MeasurableEquiv.coe_piCongrLeft]
    funext x i
    have h := Equiv.piCongrLeft_apply_apply (fun _ : EntryIndex n N => ℝ) e.symm x (e i)
    rwa [Equiv.symm_apply_apply] at h
  have hmp : MeasurePreserving (fun x : EntrySpace n N => fun i => x (e i))
      (P n N v) (P n N v) := by
    have h1 := measurePreserving_piCongrLeft
      (fun _ : EntryIndex n N => gaussianReal 0 v) e.symm
    rwa [hcoe] at h1
  calc ∫ x, g (fun i => x (e i)) ∂(P n N v)
      = ∫ y, g y ∂((P n N v).map (fun x (i : EntryIndex n N) => x (e i))) :=
        (integral_map hmp.aemeasurable (by rw [hmp.map_eq]; exact hg)).symm
    _ = ∫ y, g y ∂(P n N v) := by rw [hmp.map_eq]

/-- Multiplying each coordinate by a fixed sign `±1` preserves `P`. -/
lemma integral_comp_flip (ε : EntryIndex n N → ℝ)
    (hε : ∀ i, ε i = 1 ∨ ε i = -1)
    {g : EntrySpace n N → ℝ} (hg : AEStronglyMeasurable g (P n N v)) :
    ∫ x, g (fun i => ε i * x i) ∂(P n N v) = ∫ x, g x ∂(P n N v) := by
  have hmap : ∀ i : EntryIndex n N,
      (gaussianReal 0 v).map (ε i * ·) = gaussianReal 0 v := by
    intro i
    rcases hε i with h | h <;> rw [h]
    · simp
    · rw [gaussianReal_map_const_mul]
      congr 1
      · ring
      · rw [show (⟨(-1 : ℝ) ^ 2, sq_nonneg _⟩ : ℝ≥0) = 1 by ext; norm_num, one_mul]
  have hmapped := Measure.pi_map_pi (μ := fun _ : EntryIndex n N => gaussianReal 0 v)
    (f := fun i => (ε i * ·)) (fun i => (measurable_id.const_mul (ε i)).aemeasurable)
  simp only [hmap] at hmapped
  have hmp : MeasurePreserving (fun x : EntrySpace n N => fun i => ε i * x i)
      (P n N v) (P n N v) :=
    ⟨measurable_pi_lambda _ fun i => measurable_const.mul (measurable_pi_apply i), hmapped⟩
  calc ∫ x, g (fun i => ε i * x i) ∂(P n N v)
      = ∫ y, g y ∂((P n N v).map (fun x (i : EntryIndex n N) => ε i * x i)) :=
        (integral_map hmp.aemeasurable (by rw [hmp.map_eq]; exact hg)).symm
    _ = ∫ y, g y ∂(P n N v) := by rw [hmp.map_eq]

/-! ### Almost-sure nonvanishing of `L` -/

/-- The law of `q ↦ ∑_{j ∈ t} c_j q_j` under a product Gaussian measure
is `gaussianReal 0 ((∑_{j ∈ t} c_j²) v)`. -/
lemma map_dot_pi_gaussian {ι : Type*} [Fintype ι] (c : ι → ℝ) (t : Finset ι) :
    (Measure.pi fun _ : ι => gaussianReal 0 v).map (fun q => ∑ j ∈ t, c j * q j)
      = gaussianReal 0 ((∑ j ∈ t, (c j ^ 2).toNNReal) * v) := by
  classical
  set Q := Measure.pi fun _ : ι => gaussianReal 0 v with hQ
  have hIndep : iIndepFun (fun j (q : ι → ℝ) => c j * q j) Q :=
    iIndepFun_pi (μ := fun _ : ι => gaussianReal 0 v) (X := fun j => (c j * ·))
      (fun j => (measurable_id.const_mul (c j)).aemeasurable)
  have hMeasY : ∀ j, Measurable fun q : ι → ℝ => c j * q j :=
    fun j => measurable_const.mul (measurable_pi_apply j)
  have hLaw : ∀ j, Q.map (fun q : ι → ℝ => c j * q j)
      = gaussianReal 0 ((c j ^ 2).toNNReal * v) := by
    intro j
    have h1 : (fun q : ι → ℝ => c j * q j) = (c j * ·) ∘ Function.eval j := rfl
    rw [h1, ← Measure.map_map (measurable_const_mul (c j)) (measurable_pi_apply j),
      (measurePreserving_eval (fun _ : ι => gaussianReal 0 v) j).map_eq,
      gaussianReal_map_const_mul, mul_zero,
      Real.toNNReal_of_nonneg (sq_nonneg (c j))]
  induction t using Finset.induction_on with
  | empty =>
      simp only [Finset.sum_empty, zero_mul, gaussianReal_zero_var]
      rw [show (fun q : ι → ℝ => (0 : ℝ)) = fun _ => (0 : ℝ) from rfl,
        Measure.map_const, measure_univ, one_smul]
  | insert i s his ih =>
      have hfun : (fun q : ι → ℝ => ∑ j ∈ insert i s, c j * q j)
          = (fun q : ι → ℝ => c i * q i) + fun q => ∑ j ∈ s, c j * q j := by
        funext q
        simp [Finset.sum_insert his]
      have hind : IndepFun (fun q : ι → ℝ => c i * q i)
          (fun q => ∑ j ∈ s, c j * q j) Q := by
        have h := (hIndep.indepFun_finset_sum_of_notMem hMeasY his).symm
        have h2 : (∑ j ∈ s, fun q : ι → ℝ => c j * q j)
            = fun q : ι → ℝ => ∑ j ∈ s, c j * q j := by
          funext q
          simp [Finset.sum_apply]
        rwa [h2] at h
      rw [hfun, Finset.sum_insert his,
        gaussianReal_add_gaussianReal_of_indepFun hind (hLaw i) ih,
        zero_add, add_mul]

/-- For a coefficient vector `c` not identically zero on `t`, the
hyperplane `{q | ∑_{j ∈ t} c_j q_j = 0}` is null for the product
Gaussian measure. -/
lemma pi_gaussian_dot_eq_zero_null {ι : Type*} [Fintype ι] (hv : v ≠ 0)
    (c : ι → ℝ) (t : Finset ι) (hc : ∃ j ∈ t, c j ≠ 0) :
    Measure.pi (fun _ : ι => gaussianReal 0 v)
      {q | ∑ j ∈ t, c j * q j = 0} = 0 := by
  have hmeasSum : Measurable fun q : ι → ℝ => ∑ j ∈ t, c j * q j :=
    Finset.measurable_sum _ fun j _ => measurable_const.mul (measurable_pi_apply j)
  have hV : (∑ j ∈ t, (c j ^ 2).toNNReal) * v ≠ 0 := by
    obtain ⟨j₀, hj₀t, hj₀⟩ := hc
    refine mul_ne_zero (fun h => hj₀ ?_) hv
    have h0 := Finset.sum_eq_zero_iff.mp h j₀ hj₀t
    have hsq : c j₀ ^ 2 = 0 :=
      le_antisymm (Real.toNNReal_eq_zero.mp h0) (sq_nonneg _)
    exact pow_eq_zero_iff (two_ne_zero) |>.mp hsq
  haveI := noAtoms_gaussianReal (μ := 0) hV
  have hset : {q : ι → ℝ | ∑ j ∈ t, c j * q j = 0}
      = (fun q : ι → ℝ => ∑ j ∈ t, c j * q j) ⁻¹' {0} := rfl
  rw [hset, ← Measure.map_apply hmeasSum (measurableSet_singleton 0),
    map_dot_pi_gaussian c t, measure_singleton]

/-- Almost surely `L ≠ 0` (for `n ≥ 1`, `N ≥ 1` and a nondegenerate
Gaussian). -/
lemma ae_LMat_ne_zero (hn : 0 < n) (hN : 0 < N) (hv : v ≠ 0) :
    ∀ᵐ x ∂(P n N v), LMat x ≠ 0 := by
  haveI := noAtoms_gaussianReal (μ := 0) hv
  set a₀ : Fin N := ⟨0, hN⟩
  set r₀ : Fin n := ⟨0, hn⟩
  set A := Fin n × Fin N
  set PA : Measure (A → ℝ) := Measure.pi fun _ : A => gaussianReal 0 v with hPA
  haveI : IsProbabilityMeasure PA := by rw [hPA]; infer_instance
  -- the event written on the product of the two entry blocks
  set W : Set ((A → ℝ) × (A → ℝ)) :=
    {kq | ∑ r, kq.1 (r, a₀) * kq.2 (r, a₀) = 0} with hW
  have hWmeas : MeasurableSet W := by
    have hm : Measurable fun kq : (A → ℝ) × (A → ℝ) =>
        ∑ r, kq.1 (r, a₀) * kq.2 (r, a₀) :=
      Finset.measurable_sum _ fun r _ =>
        ((measurable_pi_apply _).comp measurable_fst).mul
          ((measurable_pi_apply _).comp measurable_snd)
    exact hm (measurableSet_singleton 0)
  -- each slice at fixed W_K-entries `k` is null unless the `a₀`-th
  -- column of `W_K` vanishes
  have hslice : ∀ k : A → ℝ, PA (Prod.mk k ⁻¹' W)
      ≤ Set.indicator {k' : A → ℝ | k' (r₀, a₀) = 0} (fun _ => 1) k := by
    intro k
    by_cases hk : k (r₀, a₀) = 0
    · have h1 : ({k' : A → ℝ | k' (r₀, a₀) = 0}).indicator
          (fun _ => (1 : ℝ≥0∞)) k = 1 :=
        Set.indicator_of_mem (show k ∈ {k' : A → ℝ | k' (r₀, a₀) = 0} from hk) _
      rw [h1]
      exact prob_le_one
    · have h1 : ({k' : A → ℝ | k' (r₀, a₀) = 0}).indicator
          (fun _ => (1 : ℝ≥0∞)) k = 0 :=
        Set.indicator_of_notMem (show k ∉ {k' : A → ℝ | k' (r₀, a₀) = 0} from hk) _
      rw [h1]
      set t : Finset A := Finset.univ.image fun r : Fin n => (r, a₀) with ht
      have hinj : Function.Injective fun r : Fin n => ((r, a₀) : A) := by
        intro r r' h
        exact congrArg Prod.fst h
      have hsum : ∀ q : A → ℝ,
          ∑ j ∈ t, k j * q j = ∑ r, k (r, a₀) * q (r, a₀) := by
        intro q
        rw [ht, Finset.sum_image fun r _ r' _ h => hinj h]
      have hpre : Prod.mk k ⁻¹' W = {q : A → ℝ | ∑ j ∈ t, k j * q j = 0} := by
        ext q
        simp only [hW, Set.mem_preimage, Set.mem_setOf_eq, hsum]
      rw [hpre, hPA,
        pi_gaussian_dot_eq_zero_null hv k t
          ⟨(r₀, a₀), Finset.mem_image_of_mem _ (Finset.mem_univ r₀), hk⟩]
  -- reduce `{L = 0}` to the vanishing of the `(a₀, a₀)` entry
  rw [ae_iff]
  simp only [ne_eq, not_not]
  have hsub : {x : EntrySpace n N | LMat x = 0}
      ⊆ {x | ∑ r, x (.inl (r, a₀)) * x (.inr (r, a₀)) = 0} := by
    intro x hx
    have h := congrFun (congrFun hx a₀) a₀
    simpa [LMat_apply] using h
  refine measure_mono_null hsub (le_antisymm ?_ (zero_le _))
  -- split the entries into the two blocks and integrate out the slices
  have hmp := measurePreserving_sumPiEquivProdPi
    (μ := fun _ : EntryIndex n N => gaussianReal 0 v)
  have hpre : (MeasurableEquiv.sumPiEquivProdPi fun _ : EntryIndex n N => ℝ) ⁻¹' W
      = {x : EntrySpace n N | ∑ r, x (.inl (r, a₀)) * x (.inr (r, a₀)) = 0} := rfl
  calc (P n N v) {x : EntrySpace n N |
        ∑ r, x (.inl (r, a₀)) * x (.inr (r, a₀)) = 0}
      ≤ (PA.prod PA) W := le_of_eq <| by
        rw [← hpre]
        exact hmp.measure_preimage hWmeas.nullMeasurableSet
    _ = ∫⁻ k, PA (Prod.mk k ⁻¹' W) ∂PA := Measure.prod_apply hWmeas
    _ ≤ ∫⁻ k, Set.indicator {k' : A → ℝ | k' (r₀, a₀) = 0} (fun _ => 1) k ∂PA :=
        lintegral_mono hslice
    _ = PA {k' : A → ℝ | k' (r₀, a₀) = 0} := by
        have hms : MeasurableSet {k' : A → ℝ | k' (r₀, a₀) = 0} :=
          measurable_pi_apply _ (measurableSet_singleton 0)
        rw [lintegral_indicator hms]
        simp
    _ = 0 := by
        rw [hPA]
        exact Measure.pi_hyperplane _ _ 0

/-! ### The moments of the normalized entries -/

/-- Permuting rows and columns of `L` corresponds to permuting the
column indices of `W_K` and of `W_Q`. -/
lemma LMat_perm (σ τ : Equiv.Perm (Fin N)) (x : EntrySpace n N) :
    LMat (fun i => x (Equiv.sumCongr ((Equiv.refl (Fin n)).prodCongr σ)
      ((Equiv.refl (Fin n)).prodCongr τ) i)) = (LMat x).submatrix σ τ := rfl

/-- The squared Frobenius norm is invariant under row and column
permutations. -/
lemma frobSq_submatrix (M : Matrix (Fin N) (Fin N) ℝ) (σ τ : Equiv.Perm (Fin N)) :
    frobSq (M.submatrix σ τ) = frobSq M := by
  rw [frobSq_eq_sum_sq, frobSq_eq_sum_sq]
  exact Fintype.sum_equiv σ _ _ fun c => Fintype.sum_equiv τ _ _ fun d => rfl

/-- Each squared entry is at most the squared Frobenius norm. -/
lemma sq_entry_le_frobSq (M : Matrix (Fin N) (Fin N) ℝ) (a b : Fin N) :
    M a b ^ 2 ≤ frobSq M := by
  rw [frobSq_eq_sum_sq]
  calc M a b ^ 2 ≤ ∑ d, M a d ^ 2 :=
        Finset.single_le_sum (fun d _ => sq_nonneg _) (Finset.mem_univ b)
    _ ≤ ∑ c, ∑ d, M c d ^ 2 :=
        Finset.single_le_sum (f := fun c => ∑ d, M c d ^ 2)
          (fun c _ => Finset.sum_nonneg fun d _ => sq_nonneg _) (Finset.mem_univ a)

/-- All expectations `𝔼[L_{ab}²/‖L‖²]` are equal, by row and column
permutation invariance. -/
lemma integral_entry_sq_ratio_eq (a b a' b' : Fin N) :
    ∫ x, LMat x a b ^ 2 / frobSq (LMat x) ∂(P n N v)
      = ∫ x, LMat x a' b' ^ 2 / frobSq (LMat x) ∂(P n N v) := by
  have hmeas : AEStronglyMeasurable
      (fun y : EntrySpace n N => LMat y a b ^ 2 / frobSq (LMat y)) (P n N v) :=
    (((measurable_LMat_apply a b).pow_const 2).div
      measurable_frobSq_LMat).aestronglyMeasurable
  have h := integral_comp_perm (v := v)
    (Equiv.sumCongr ((Equiv.refl (Fin n)).prodCongr (Equiv.swap a a'))
      ((Equiv.refl (Fin n)).prodCongr (Equiv.swap b b'))) hmeas
  simp only [LMat_perm, frobSq_submatrix, Matrix.submatrix_apply,
    Equiv.swap_apply_left] at h
  exact h.symm

/-- `𝔼[L_{ab} L_{ba}/‖L‖²] = 0` for `a ≠ b`, by flipping the sign of the
`a`-th column of `W_K`. -/
lemma integral_entry_offdiag_ratio_eq_zero {a b : Fin N} (hab : a ≠ b) :
    ∫ x, LMat x a b * LMat x b a / frobSq (LMat x) ∂(P n N v) = 0 := by
  classical
  set ε : EntryIndex n N → ℝ :=
    Sum.elim (fun p => if p.2 = a then -1 else 1) (fun _ => 1) with hε
  have hε1 : ∀ i, ε i = 1 ∨ ε i = -1 := by
    rintro (p | p)
    · simp only [hε, Sum.elim_inl]
      split_ifs
      · exact Or.inr rfl
      · exact Or.inl rfl
    · exact Or.inl rfl
  have hL : ∀ (x : EntrySpace n N) (c d : Fin N),
      LMat (fun i => ε i * x i) c d
        = (if c = a then (-1 : ℝ) else 1) * LMat x c d := by
    intro x c d
    simp only [LMat_apply, hε, Sum.elim_inl, Sum.elim_inr, one_mul,
      Finset.mul_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    ring
  have hfrob : ∀ x : EntrySpace n N,
      frobSq (LMat (fun i => ε i * x i)) = frobSq (LMat x) := by
    intro x
    rw [frobSq_eq_sum_sq, frobSq_eq_sum_sq]
    refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun d _ => ?_
    rw [hL]
    split_ifs <;> ring
  have hmeas : AEStronglyMeasurable (fun y : EntrySpace n N =>
      LMat y a b * LMat y b a / frobSq (LMat y)) (P n N v) :=
    (((measurable_LMat_apply a b).mul (measurable_LMat_apply b a)).div
      measurable_frobSq_LMat).aestronglyMeasurable
  have h := integral_comp_flip (v := v) ε hε1 hmeas
  have hint : (fun x : EntrySpace n N =>
      LMat (fun i => ε i * x i) a b * LMat (fun i => ε i * x i) b a
        / frobSq (LMat (fun i => ε i * x i)))
      = fun x => -(LMat x a b * LMat x b a / frobSq (LMat x)) := by
    funext x
    rw [hL, hL, hfrob, if_pos rfl, if_neg (Ne.symm hab)]
    ring
  rw [hint, integral_neg] at h
  linarith

/-- The entry ratios are bounded by `1` and hence integrable. -/
lemma integrable_entry_sq_ratio (a b : Fin N) :
    Integrable (fun x : EntrySpace n N => LMat x a b ^ 2 / frobSq (LMat x))
      (P n N v) := by
  refine (integrable_const (1 : ℝ)).mono'
    (((measurable_LMat_apply a b).pow_const 2).div
      measurable_frobSq_LMat).aestronglyMeasurable
    (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (sq_nonneg _),
    abs_of_nonneg (frobSq_nonneg _)]
  exact div_le_one_of_le₀ (sq_entry_le_frobSq _ a b) (frobSq_nonneg _)

lemma integrable_entry_offdiag_ratio (a b : Fin N) :
    Integrable (fun x : EntrySpace n N =>
      LMat x a b * LMat x b a / frobSq (LMat x)) (P n N v) := by
  refine (integrable_const (1 : ℝ)).mono'
    (((measurable_LMat_apply a b).mul (measurable_LMat_apply b a)).div
      measurable_frobSq_LMat).aestronglyMeasurable
    (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (frobSq_nonneg _)]
  refine div_le_one_of_le₀ ?_ (frobSq_nonneg _)
  have h1 := sq_entry_le_frobSq (LMat x) a b
  have h2 := sq_entry_le_frobSq (LMat x) b a
  have h3 : |LMat x a b * LMat x b a|
      ≤ (LMat x a b ^ 2 + LMat x b a ^ 2) / 2 := by
    rw [abs_mul]
    nlinarith [sq_nonneg (|LMat x a b| - |LMat x b a|), sq_abs (LMat x a b),
      sq_abs (LMat x b a), abs_nonneg (LMat x a b), abs_nonneg (LMat x b a)]
  linarith

/-- `𝔼[L_{ab}²/‖L‖²] = 1/N²`. -/
lemma integral_entry_sq_ratio (hn : 0 < n) (hN : 0 < N) (hv : v ≠ 0)
    (a b : Fin N) :
    ∫ x, LMat x a b ^ 2 / frobSq (LMat x) ∂(P n N v) = 1 / (N : ℝ) ^ 2 := by
  -- the sum of all `N²` entry ratios is `1` almost everywhere
  have hsum1 : ∫ x, ∑ c, ∑ d, LMat x c d ^ 2 / frobSq (LMat x) ∂(P n N v)
      = 1 := by
    have hae : ∀ᵐ x ∂(P n N v),
        ∑ c, ∑ d, LMat x c d ^ 2 / frobSq (LMat x) = 1 := by
      filter_upwards [ae_LMat_ne_zero hn hN hv] with x hx
      have hfz : frobSq (LMat x) ≠ 0 := fun h => hx ((frobSq_eq_zero_iff _).mp h)
      simp only [← Finset.sum_div]
      rw [← frobSq_eq_sum_sq, div_self hfz]
    rw [integral_congr_ae hae, integral_const]
    simp
  have hsplit : ∫ x, ∑ c, ∑ d, LMat x c d ^ 2 / frobSq (LMat x) ∂(P n N v)
      = ∑ c, ∑ d, ∫ x, LMat x c d ^ 2 / frobSq (LMat x) ∂(P n N v) := by
    rw [integral_finset_sum _ fun c _ =>
      integrable_finset_sum _ fun d _ => integrable_entry_sq_ratio c d]
    exact Finset.sum_congr rfl fun c _ =>
      integral_finset_sum _ fun d _ => integrable_entry_sq_ratio c d
  have hone : (1 : ℝ) = (N : ℝ) ^ 2
      * ∫ x, LMat x a b ^ 2 / frobSq (LMat x) ∂(P n N v) := by
    rw [← hsum1, hsplit]
    have hall : ∀ c d : Fin N,
        ∫ x, LMat x c d ^ 2 / frobSq (LMat x) ∂(P n N v)
          = ∫ x, LMat x a b ^ 2 / frobSq (LMat x) ∂(P n N v) :=
      fun c d => integral_entry_sq_ratio_eq c d a b
    simp only [hall, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    ring
  have hN2 : ((N : ℝ) ^ 2) ≠ 0 :=
    pow_ne_zero 2 (Nat.cast_ne_zero.mpr hN.ne')
  rw [eq_div_iff hN2]
  linarith

/-- The trace ratio is the sum of the `N²` products `L_{cd} L_{dc}/‖L‖²`. -/
lemma trace_ratio_eq_sum (x : EntrySpace n N) :
    Matrix.trace (LMat x * LMat x) / frobSq (LMat x)
      = ∑ c, ∑ d, LMat x c d * LMat x d c / frobSq (LMat x) := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, ← Finset.sum_div]

/-- `𝔼[tr(L²)/‖L‖²] = 1/N`. -/
lemma integral_trace_sq_ratio (hn : 0 < n) (hN : 0 < N) (hv : v ≠ 0) :
    ∫ x, Matrix.trace (LMat x * LMat x) / frobSq (LMat x) ∂(P n N v)
      = 1 / (N : ℝ) := by
  classical
  simp only [trace_ratio_eq_sum]
  rw [integral_finset_sum _ fun c _ =>
    integrable_finset_sum _ fun d _ => integrable_entry_offdiag_ratio c d]
  have hsplit2 : ∀ c : Fin N,
      ∫ x, ∑ d, LMat x c d * LMat x d c / frobSq (LMat x) ∂(P n N v)
        = ∑ d, ∫ x, LMat x c d * LMat x d c / frobSq (LMat x) ∂(P n N v) :=
    fun c => integral_finset_sum _ fun d _ => integrable_entry_offdiag_ratio c d
  simp only [hsplit2]
  have hval : ∀ c d : Fin N,
      ∫ x, LMat x c d * LMat x d c / frobSq (LMat x) ∂(P n N v)
        = if c = d then 1 / (N : ℝ) ^ 2 else 0 := by
    intro c d
    by_cases hcd : c = d
    · subst hcd
      rw [if_pos rfl]
      have hsq : (fun x : EntrySpace n N =>
          LMat x c c * LMat x c c / frobSq (LMat x))
          = fun x => LMat x c c ^ 2 / frobSq (LMat x) := by
        funext x
        rw [sq]
      rw [hsq]
      exact integral_entry_sq_ratio hn hN hv c c
    · rw [if_neg hcd]
      exact integral_entry_offdiag_ratio_eq_zero hcd
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hinner : ∀ c : Fin N,
      ∑ d, ∫ x, LMat x c d * LMat x d c / frobSq (LMat x) ∂(P n N v)
        = 1 / (N : ℝ) ^ 2 := by
    intro c
    simp only [hval, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  simp only [hinner, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  field_simp

lemma measurable_frobSq_symPart_LMat :
    Measurable fun x : EntrySpace n N => frobSq (symPart (LMat x)) := by
  simp only [frobSq_eq_sum_sq, symPart, Matrix.smul_apply, Matrix.add_apply,
    Matrix.transpose_apply, smul_eq_mul]
  exact Finset.measurable_sum _ fun c _ => Finset.measurable_sum _ fun d _ =>
    (measurable_const.mul
      ((measurable_LMat_apply c d).add (measurable_LMat_apply d c))).pow_const 2

lemma integrable_trace_ratio :
    Integrable (fun x : EntrySpace n N =>
      Matrix.trace (LMat x * LMat x) / frobSq (LMat x)) (P n N v) := by
  simp only [trace_ratio_eq_sum]
  exact integrable_finset_sum _ fun c _ =>
    integrable_finset_sum _ fun d _ => integrable_entry_offdiag_ratio c d

lemma integrable_symPart_ratio :
    Integrable (fun x : EntrySpace n N =>
      frobSq (symPart (LMat x)) / frobSq (LMat x)) (P n N v) := by
  refine (integrable_const (1 : ℝ)).mono'
    (measurable_frobSq_symPart_LMat.div measurable_frobSq_LMat).aestronglyMeasurable
    (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (frobSq_nonneg _),
    abs_of_nonneg (frobSq_nonneg _)]
  refine div_le_one_of_le₀ ?_ (frobSq_nonneg _)
  have h := frobSq_eq_frobSq_symPart_add_frobSq_skewPart (LMat x)
  have h2 := frobSq_nonneg (skewPart (LMat x))
  linarith

/-- The core computation: `𝔼[‖S‖²/‖L‖²] = (N+1)/(2N)` over the canonical
product Gaussian measure. -/
lemma integral_symPart_ratio (hn : 0 < n) (hN : 0 < N) (hv : v ≠ 0) :
    ∫ x, frobSq (symPart (LMat x)) / frobSq (LMat x) ∂(P n N v)
      = ((N : ℝ) + 1) / (2 * N) := by
  have hae : ∀ᵐ x ∂(P n N v),
      frobSq (symPart (LMat x)) / frobSq (LMat x)
        = 1 / 2 + Matrix.trace (LMat x * LMat x) / frobSq (LMat x) / 2 := by
    filter_upwards [ae_LMat_ne_zero hn hN hv] with x hx
    have hfz : frobSq (LMat x) ≠ 0 := fun h => hx ((frobSq_eq_zero_iff _).mp h)
    rw [frobSq_symPart]
    field_simp
  rw [integral_congr_ae hae,
    integral_add (integrable_const _) (integrable_trace_ratio.div_const 2),
    integral_const, integral_div, integral_trace_sq_ratio hn hN hv]
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  simp only [probReal_univ, smul_eq_mul, one_mul]
  field_simp

/-- The core computation: `𝔼[‖T‖²/‖L‖²] = (N-1)/(2N)` over the canonical
product Gaussian measure. -/
lemma integral_skewPart_ratio (hn : 0 < n) (hN : 0 < N) (hv : v ≠ 0) :
    ∫ x, frobSq (skewPart (LMat x)) / frobSq (LMat x) ∂(P n N v)
      = ((N : ℝ) - 1) / (2 * N) := by
  have hae : ∀ᵐ x ∂(P n N v),
      frobSq (skewPart (LMat x)) / frobSq (LMat x)
        = 1 - frobSq (symPart (LMat x)) / frobSq (LMat x) := by
    filter_upwards [ae_LMat_ne_zero hn hN hv] with x hx
    have hfz : frobSq (LMat x) ≠ 0 := fun h => hx ((frobSq_eq_zero_iff _).mp h)
    have h := frobSq_eq_frobSq_symPart_add_frobSq_skewPart (LMat x)
    field_simp
    linarith
  rw [integral_congr_ae hae,
    integral_sub (integrable_const _) integrable_symPart_ratio,
    integral_const, integral_symPart_ratio hn hN hv]
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  simp only [probReal_univ, smul_eq_mul, one_mul]
  field_simp
  ring

end LowRankQK

/-! ### The proposition, on an abstract probability space -/

/-- The family of the `2nN` matrix entries of the pair `(W_K, W_Q)`, as
real random variables: `.inl (r, a)` is the entry `(W_K)_{ra}` and
`.inr (r, b)` is `(W_Q)_{rb}`. -/
def qkEntry {Ω : Type*} {n N : ℕ} (WK WQ : Ω → Matrix (Fin n) (Fin N) ℝ) :
    (Fin n × Fin N) ⊕ (Fin n × Fin N) → Ω → ℝ :=
  Sum.elim (fun p ω => WK ω p.1 p.2) (fun p ω => WQ ω p.1 p.2)

section MainStatement

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {n N : ℕ} {v : ℝ≥0} {WK WQ : Ω → Matrix (Fin n) (Fin N) ℝ}

/-- Step of the proof of `expected_symmetric_energy_low_rank_qk`, the
common-variance case: all `2nN` entries are
jointly independent with the same law `gaussianReal 0 v`, `v ≠ 0`, and
`1 ≤ n ≤ N`.  With `L = W_Kᵀ W_Q`, `S = ½(L + Lᵀ)` and
`𝒮 = ‖S‖²/‖L‖²`:

    `𝔼[𝒮] = (N+1)/(2N)`,

independently of the head dimension `n`. -/
private theorem expected_symmetric_energy_low_rank_qk_common_variance
    (hn : 1 ≤ n) (hnN : n ≤ N) (hv : v ≠ 0)
    (hmeas : ∀ i, AEMeasurable (qkEntry WK WQ i) μ)
    (hindep : iIndepFun (qkEntry WK WQ) μ)
    (hlaw : ∀ i, μ.map (qkEntry WK WQ i) = gaussianReal 0 v) :
    ∫ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω) ∂μ
      = ((N : ℝ) + 1) / (2 * N) := by
  have hN : 0 < N := lt_of_lt_of_le hn hnN
  have hJmeas : AEMeasurable (fun ω i => qkEntry WK WQ i ω) μ :=
    aemeasurable_pi_lambda _ fun i => hmeas i
  have hJlaw : μ.map (fun ω i => qkEntry WK WQ i ω) = LowRankQK.P n N v := by
    rw [(iIndepFun_iff_map_fun_eq_pi_map hmeas).mp hindep]
    exact congrArg Measure.pi (funext fun i => hlaw i)
  have hLL : ∀ ω, (WK ω)ᵀ * WQ ω
      = LowRankQK.LMat (fun i => qkEntry WK WQ i ω) := by
    intro ω
    ext a b
    simp [LowRankQK.LMat_apply, Matrix.mul_apply, qkEntry]
  calc ∫ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω) ∂μ
      = ∫ ω, frobSq (symPart (LowRankQK.LMat (fun i => qkEntry WK WQ i ω)))
          / frobSq (LowRankQK.LMat (fun i => qkEntry WK WQ i ω)) ∂μ := by
        simp_rw [hLL]
    _ = ∫ x, frobSq (symPart (LowRankQK.LMat x)) / frobSq (LowRankQK.LMat x)
          ∂(μ.map (fun ω i => qkEntry WK WQ i ω)) :=
        (integral_map hJmeas ((LowRankQK.measurable_frobSq_symPart_LMat.div
          LowRankQK.measurable_frobSq_LMat).aestronglyMeasurable)).symm
    _ = ∫ x, frobSq (symPart (LowRankQK.LMat x)) / frobSq (LowRankQK.LMat x)
          ∂(LowRankQK.P n N v) := by rw [hJlaw]
    _ = ((N : ℝ) + 1) / (2 * N) := LowRankQK.integral_symPart_ratio hn hN hv

/-- Step of the proof of `expected_skew_energy_low_rank_qk`, the
common-variance case, skew part: in the setting of
`expected_symmetric_energy_low_rank_qk_common_variance`, with
`T = ½(L - Lᵀ)` and `𝒜 = ‖T‖²/‖L‖²`:

    `𝔼[𝒜] = (N-1)/(2N)`,

independently of the head dimension `n`. -/
private theorem expected_skew_energy_low_rank_qk_common_variance
    (hn : 1 ≤ n) (hnN : n ≤ N) (hv : v ≠ 0)
    (hmeas : ∀ i, AEMeasurable (qkEntry WK WQ i) μ)
    (hindep : iIndepFun (qkEntry WK WQ) μ)
    (hlaw : ∀ i, μ.map (qkEntry WK WQ i) = gaussianReal 0 v) :
    ∫ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω) ∂μ
      = ((N : ℝ) - 1) / (2 * N) := by
  have hN : 0 < N := lt_of_lt_of_le hn hnN
  have hJmeas : AEMeasurable (fun ω i => qkEntry WK WQ i ω) μ :=
    aemeasurable_pi_lambda _ fun i => hmeas i
  have hJlaw : μ.map (fun ω i => qkEntry WK WQ i ω) = LowRankQK.P n N v := by
    rw [(iIndepFun_iff_map_fun_eq_pi_map hmeas).mp hindep]
    exact congrArg Measure.pi (funext fun i => hlaw i)
  have hLL : ∀ ω, (WK ω)ᵀ * WQ ω
      = LowRankQK.LMat (fun i => qkEntry WK WQ i ω) := by
    intro ω
    ext a b
    simp [LowRankQK.LMat_apply, Matrix.mul_apply, qkEntry]
  have hmeasSkew : Measurable fun x : LowRankQK.EntrySpace n N =>
      frobSq (skewPart (LowRankQK.LMat x)) := by
    simp only [frobSq_eq_sum_sq, skewPart, Matrix.smul_apply, Matrix.sub_apply,
      Matrix.transpose_apply, smul_eq_mul]
    exact Finset.measurable_sum _ fun c _ => Finset.measurable_sum _ fun d _ =>
      (measurable_const.mul ((LowRankQK.measurable_LMat_apply c d).sub
        (LowRankQK.measurable_LMat_apply d c))).pow_const 2
  calc ∫ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω) ∂μ
      = ∫ ω, frobSq (skewPart (LowRankQK.LMat (fun i => qkEntry WK WQ i ω)))
          / frobSq (LowRankQK.LMat (fun i => qkEntry WK WQ i ω)) ∂μ := by
        simp_rw [hLL]
    _ = ∫ x, frobSq (skewPart (LowRankQK.LMat x)) / frobSq (LowRankQK.LMat x)
          ∂(μ.map (fun ω i => qkEntry WK WQ i ω)) :=
        (integral_map hJmeas ((hmeasSkew.div
          LowRankQK.measurable_frobSq_LMat).aestronglyMeasurable)).symm
    _ = ∫ x, frobSq (skewPart (LowRankQK.LMat x)) / frobSq (LowRankQK.LMat x)
          ∂(LowRankQK.P n N v) := by rw [hJlaw]
    _ = ((N : ℝ) - 1) / (2 * N) := LowRankQK.integral_skewPart_ratio hn hN hv

/-! ### Different variances for `W_K` and `W_Q`

The paper's hypothesis "independent random matrices with iid centered
Gaussian entries" allows the entries of `W_K` and of `W_Q` to have
different variances `v_K` and `v_Q`.  Since `𝒮` and `𝒜` are invariant
under rescaling `W_K` by a nonzero constant, this reduces to the
common-variance case by replacing `W_K` with `√(v_Q/v_K) W_K`. -/

private lemma symPart_smul (c : ℝ) (L : Matrix (Fin N) (Fin N) ℝ) :
    symPart (c • L) = c • symPart L := by
  unfold symPart
  rw [Matrix.transpose_smul, ← smul_add, smul_comm]

private lemma skewPart_smul (c : ℝ) (L : Matrix (Fin N) (Fin N) ℝ) :
    skewPart (c • L) = c • skewPart L := by
  unfold skewPart
  rw [Matrix.transpose_smul, ← smul_sub, smul_comm]

private lemma frobSq_smul (c : ℝ) (L : Matrix (Fin N) (Fin N) ℝ) :
    frobSq (c • L) = c ^ 2 * frobSq L := by
  simp only [frobSq_eq_sum_sq, Matrix.smul_apply, smul_eq_mul, mul_pow,
    Finset.mul_sum]

omit [IsProbabilityMeasure μ] in
/-- Rescaling `W_K` by `√(v_Q/v_K)` gives a pair with common variance
`v_Q` whose product `L` is the rescaled product. -/
private lemma exists_rescaled_WK {vK vQ : ℝ≥0} (hvK : vK ≠ 0)
    (hmeas : ∀ i, AEMeasurable (qkEntry WK WQ i) μ)
    (hindep : iIndepFun (qkEntry WK WQ) μ)
    (hlawK : ∀ p, μ.map (qkEntry WK WQ (Sum.inl p)) = gaussianReal 0 vK)
    (hlawQ : ∀ p, μ.map (qkEntry WK WQ (Sum.inr p)) = gaussianReal 0 vQ) :
    ∃ WK' : Ω → Matrix (Fin n) (Fin N) ℝ,
      (∀ i, AEMeasurable (qkEntry WK' WQ i) μ) ∧
      iIndepFun (qkEntry WK' WQ) μ ∧
      (∀ i, μ.map (qkEntry WK' WQ i) = gaussianReal 0 vQ) ∧
      ∀ ω, (WK' ω)ᵀ * WQ ω = Real.sqrt ((vQ : ℝ) / vK) • ((WK ω)ᵀ * WQ ω) := by
  set c : ℝ := Real.sqrt ((vQ : ℝ) / vK) with hc
  have hvK' : (vK : ℝ) ≠ 0 := by exact_mod_cast hvK
  have hc2 : c ^ 2 = (vQ : ℝ) / vK := Real.sq_sqrt (div_nonneg vQ.2 vK.2)
  set s : (Fin n × Fin N) ⊕ (Fin n × Fin N) → ℝ → ℝ :=
    Sum.elim (fun _ y => c * y) (fun _ y => y) with hs
  have hsmeas : ∀ i, Measurable (s i) := by
    rintro (p | p)
    · exact measurable_const_mul c
    · exact measurable_id
  have hq : qkEntry (fun ω => c • WK ω) WQ = fun i => s i ∘ qkEntry WK WQ i := by
    funext i ω
    rcases i with p | p <;> rfl
  refine ⟨fun ω => c • WK ω, ?_, ?_, ?_, ?_⟩
  · intro i
    rw [hq]
    exact (hsmeas i).comp_aemeasurable (hmeas i)
  · rw [hq]
    exact hindep.comp s hsmeas
  · rintro (p | p)
    · rw [hq]
      show μ.map ((c * ·) ∘ qkEntry WK WQ (Sum.inl p)) = _
      rw [← AEMeasurable.map_map_of_aemeasurable
          (measurable_const_mul c).aemeasurable (hmeas _),
        hlawK p, gaussianReal_map_const_mul, mul_zero]
      congr 1
      ext
      rw [NNReal.coe_mul, NNReal.coe_mk, hc2, div_mul_cancel₀ _ hvK']
    · rw [hq]
      exact hlawQ p
  · intro ω
    rw [Matrix.transpose_smul, Matrix.smul_mul]

/-- **Expected symmetric energy for random low-rank QK products.**
Let `W_K, W_Q ∈ M_{n×N}(ℝ)` be independent random matrices with iid
centered Gaussian entries (joint independence of all `2nN` entries; the
entries of `W_K` have law `gaussianReal 0 v_K` and those of `W_Q` have
law `gaussianReal 0 v_Q`, with `v_K, v_Q ≠ 0`), where `1 ≤ n ≤ N`.  With
`L = W_Kᵀ W_Q`, `S = ½(L + Lᵀ)` and `𝒮 = ‖S‖²/‖L‖²`:

    `𝔼[𝒮] = (N+1)/(2N)`,

independently of the head dimension `n`. -/
theorem expected_symmetric_energy_low_rank_qk
    (hn : 1 ≤ n) (hnN : n ≤ N) {vK vQ : ℝ≥0} (hvK : vK ≠ 0) (hvQ : vQ ≠ 0)
    (hmeas : ∀ i, AEMeasurable (qkEntry WK WQ i) μ)
    (hindep : iIndepFun (qkEntry WK WQ) μ)
    (hlawK : ∀ p, μ.map (qkEntry WK WQ (Sum.inl p)) = gaussianReal 0 vK)
    (hlawQ : ∀ p, μ.map (qkEntry WK WQ (Sum.inr p)) = gaussianReal 0 vQ) :
    ∫ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω) ∂μ
      = ((N : ℝ) + 1) / (2 * N) := by
  obtain ⟨WK', hmeas', hindep', hlaw', heq⟩ :=
    exists_rescaled_WK hvK hmeas hindep hlawK hlawQ
  have hvKpos : (0 : ℝ) < vK := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvK)
  have hvQpos : (0 : ℝ) < vQ := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvQ)
  have hc : Real.sqrt ((vQ : ℝ) / vK) ≠ 0 :=
    (Real.sqrt_pos.mpr (div_pos hvQpos hvKpos)).ne'
  have hpt : ∀ ω, frobSq (symPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω)
      = frobSq (symPart ((WK' ω)ᵀ * WQ ω)) / frobSq ((WK' ω)ᵀ * WQ ω) := by
    intro ω
    rw [heq ω, symPart_smul, frobSq_smul, frobSq_smul,
      mul_div_mul_left _ _ (pow_ne_zero 2 hc)]
  simp_rw [hpt]
  exact expected_symmetric_energy_low_rank_qk_common_variance hn hnN hvQ
    hmeas' hindep' hlaw'

/-- **Expected skew energy for random low-rank QK products.**
In the setting of `expected_symmetric_energy_low_rank_qk`, with
`T = ½(L - Lᵀ)` and `𝒜 = ‖T‖²/‖L‖²`:

    `𝔼[𝒜] = (N-1)/(2N)`,

independently of the head dimension `n`. -/
theorem expected_skew_energy_low_rank_qk
    (hn : 1 ≤ n) (hnN : n ≤ N) {vK vQ : ℝ≥0} (hvK : vK ≠ 0) (hvQ : vQ ≠ 0)
    (hmeas : ∀ i, AEMeasurable (qkEntry WK WQ i) μ)
    (hindep : iIndepFun (qkEntry WK WQ) μ)
    (hlawK : ∀ p, μ.map (qkEntry WK WQ (Sum.inl p)) = gaussianReal 0 vK)
    (hlawQ : ∀ p, μ.map (qkEntry WK WQ (Sum.inr p)) = gaussianReal 0 vQ) :
    ∫ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω) ∂μ
      = ((N : ℝ) - 1) / (2 * N) := by
  obtain ⟨WK', hmeas', hindep', hlaw', heq⟩ :=
    exists_rescaled_WK hvK hmeas hindep hlawK hlawQ
  have hvKpos : (0 : ℝ) < vK := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvK)
  have hvQpos : (0 : ℝ) < vQ := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvQ)
  have hc : Real.sqrt ((vQ : ℝ) / vK) ≠ 0 :=
    (Real.sqrt_pos.mpr (div_pos hvQpos hvKpos)).ne'
  have hpt : ∀ ω, frobSq (skewPart ((WK ω)ᵀ * WQ ω)) / frobSq ((WK ω)ᵀ * WQ ω)
      = frobSq (skewPart ((WK' ω)ᵀ * WQ ω)) / frobSq ((WK' ω)ᵀ * WQ ω) := by
    intro ω
    rw [heq ω, skewPart_smul, frobSq_smul, frobSq_smul,
      mul_div_mul_left _ _ (pow_ne_zero 2 hc)]
  simp_rw [hpt]
  exact expected_skew_energy_low_rank_qk_common_variance hn hnN hvQ
    hmeas' hindep' hlaw'

end MainStatement

end Appendices
