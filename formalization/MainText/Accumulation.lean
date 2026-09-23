/-
Main text, §"Profiles of trained attention heads": Theorem `thm:accumulation`.

    Let `X` be a positive random variable with finite, nonzero second
    moment.  Fix `u, v > 0` and set `ρ = u/v`.  For each `n` let `N_n ≥ 2n`
    and let `S_n ∈ M_{N_n}(ℝ)` be a symmetric matrix with positive
    eigenvalues `vX₁, …, vX_n` and negative eigenvalues `-uX'₁, …, -uX'_n`,
    where the `X_i` and `X'_i` are independent copies of `X`.  Then the
    profile `π(S_n)` converges almost surely to

        (1/(1 + ρ²)) (0, 2ρ, (1 - ρ)₊², (ρ - 1)₊²)  ∈ p(Θ).

    As `ρ` ranges over `(0, 1]` the limit traces the edge `{d = 0}` of `Δ`
    from `(0,1,0)` to `(1,0,0)`, and as `ρ` ranges over `[1, ∞)` it traces the
    edge `{c = 0}` from `(1,0,0)` to `(0,0,1)`.

Encoding.
  * The random variables are as in `Appendices.proportional_lobes`
    (Lemma 6): `X i`, `X' i` are measurable, jointly independent, each with
    law `γ`, where `γ(-∞, 0] = 0` (positivity) and `0 < ∫ y² dγ < ∞`.
  * "`S_n` is symmetric with positive eigenvalues `vX_i`, negative
    eigenvalues `-uX'_i`" and, since `N_n ≥ 2n`, all remaining eigenvalues
    zero (as the introduction's statement of the theorem says explicitly):
    the multiset of roots of the characteristic polynomial of `S_n` (its
    eigenvalues with multiplicity) is
    `{vX_i} + {-uX'_i} + {0^(N_n - 2n)}`.
  * `p(Θ)` with `p(a,b,c,d) = (0, a + b, c, d)`, and `Θ` as in
    `MainText.RankOneLocus`.
  * "Traces the edge from … to …" is formalized as: the limit lies on the
    edge, and its coordinate `b = 2ρ/(1+ρ²)` is a strictly monotone bijection
    from `(0,1]` onto `(0,1]` (increasing), respectively from `[1,∞)` onto
    `(0,1]` (decreasing); `b → 0` is the vertex `c = 1` (resp. `d = 1`), and
    `b = 1` the vertex `(1,0,0)` of `Δ`.

The proof follows the paper's: with `α_n`, `β_n` the sorted lobes and
`β_n = ρα_n + ε_n`, `δ_n = ‖ε_n‖/‖α_n‖ → 0` almost surely by Lemma 6; the
Cauchy–Schwarz and triangle inequalities, and the `1`-Lipschitz property of
`x ↦ x_±`, bound `⟨α,β⟩/‖α‖²`, `‖β‖²/‖α‖²` and `‖(α-β)_±‖/‖α‖` within
`δ_n`, `δ_n² + 2ρδ_n`, `δ_n` of `ρ`, `ρ²`, `(1-ρ)_±`
(`accumulation_estimates`); dividing by `‖S_n‖²/‖α_n‖² = 1 + ‖β‖²/‖α‖²` gives
the limit.
-/
import MainText.RankOneLocus
import Appendices.RandomMatrices.ProportionalLobes
import Appendices.BalancedBimodalSpectra.SimplexFaces

namespace Appendices.MainText

open Matrix Filter Topology

/-! ### Padding with zeros -/

section Pad

variable {n k : ℕ}

/-- A tuple of length `n` padded with zeros to length `k ≥ n`. -/
def pad (k : ℕ) (f : Fin n → ℝ) : Fin k → ℝ :=
  fun j => if h : (j : ℕ) < n then f ⟨j, h⟩ else 0

lemma map_pad (hk : n ≤ k) (f : Fin n → ℝ) :
    Multiset.map (pad k f) Finset.univ.val
      = Multiset.map f Finset.univ.val + Multiset.replicate (k - n) 0 := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hk
  rw [Fin.univ_val_map, Fin.univ_val_map, List.ofFn_add, ← Multiset.coe_add]
  congr 1
  · congr 1
    exact List.ofFn_inj.mpr (funext fun i => by simp [pad])
  · have : (fun j : Fin m => pad (n + m) f (Fin.natAdd n j)) = fun _ => 0 :=
      funext fun j => by simp [pad]
    rw [this, List.ofFn_const, Multiset.coe_replicate, Nat.add_sub_cancel_left]

lemma sum_pad (hk : n ≤ k) (f g : Fin n → ℝ) (F : ℝ → ℝ → ℝ) (hF : F 0 0 = 0) :
    ∑ j, F (pad k f j) (pad k g j) = ∑ i, F (f i) (g i) := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hk
  rw [Fin.sum_univ_add]
  simp [pad, hF]

lemma pad_antitone {f : Fin n → ℝ} (hf : Antitone f) (hf0 : ∀ i, 0 ≤ f i) :
    Antitone (pad k f) := by
  intro i j hij
  simp only [pad]
  by_cases hj : (j : ℕ) < n
  · have hi : (i : ℕ) < n := lt_of_le_of_lt (Fin.le_def.mp hij) hj
    rw [dif_pos hi, dif_pos hj]
    exact hf (Fin.le_def.mpr (Fin.le_def.mp hij))
  · rw [dif_neg hj]
    split_ifs
    · exact hf0 _
    · exact le_rfl

end Pad

/-! ### The profile of a symmetric matrix with prescribed spectrum -/

section Spectrum

variable {n k : ℕ}

lemma symPart_of_isSymm {S : Matrix (Fin k) (Fin k) ℝ} (hS : S.IsSymm) : symPart S = S := by
  rw [symPart, hS.eq, ← two_smul ℝ S, smul_smul]; norm_num

/-- A symmetric matrix whose eigenvalues are `x₁, …, x_n > 0`,
`-y₁, …, -y_n < 0` and `k - 2n` zeros has `α = sortDesc x` and
`β = sortDesc y`, padded with zeros. -/
lemma alphaList_betaList_of_spectrum (hk : 2 * n ≤ k) {S : Matrix (Fin k) (Fin k) ℝ}
    (hS : S.IsSymm) {x y : Fin n → ℝ} (hx : ∀ i, 0 < x i) (hy : ∀ i, 0 < y i)
    (hroots : S.charpoly.roots = Multiset.map x Finset.univ.val
      + Multiset.map (fun i => -y i) Finset.univ.val + Multiset.replicate (k - 2 * n) 0) :
    alphaList (symPart S) (symPart_isHermitian S) = pad k (sortDesc x) ∧
      betaList (symPart S) (symPart_isHermitian S) = pad k (sortDesc y) := by
  have hnk : n ≤ k := by omega
  have hsym := symPart_of_isSymm hS
  have hroots' : (symPart S).charpoly.roots = _ := hsym ▸ hroots
  have hanti : ∀ z : Fin n → ℝ, (∀ i, 0 < z i) → Antitone (pad k (sortDesc z)) :=
    fun z hz => pad_antitone (sortDesc_antitone z) fun i => by
      rw [sortDesc_eq_comp_perm]; exact (hz _).le
  have hrep : ∀ m : ℕ, Multiset.map (fun r : ℝ => max r 0) (Multiset.replicate m 0)
      = Multiset.replicate m 0 := fun m => by simp [Multiset.map_replicate]
  have hsplit : n + (k - 2 * n) = k - n := by omega
  constructor
  · rw [← sortDesc_eq_self (hanti x hx)]
    refine sortDesc_eq_of_map_eq ?_
    change Multiset.map ((fun r => max r 0) ∘ (symPart_isHermitian S).eigenvalues) _ = _
    rw [← Multiset.map_map, map_eigenvalues_eq_roots, hroots', map_pad hnk, map_sortDesc,
      Multiset.map_add, Multiset.map_add, Multiset.map_map, Multiset.map_map, hrep]
    have h1 : ((fun r : ℝ => max r 0) ∘ x) = x := funext fun i => max_eq_left (hx i).le
    have h2 : ((fun r : ℝ => max r 0) ∘ fun i => -y i) = fun _ => 0 :=
      funext fun i => max_eq_right (by simp [(hy i).le])
    rw [h1, h2, Multiset.map_const', Finset.card_val, Finset.card_univ, Fintype.card_fin,
      add_assoc, ← Multiset.replicate_add, hsplit]
  · rw [← sortDesc_eq_self (hanti y hy)]
    refine sortDesc_eq_of_map_eq ?_
    change Multiset.map ((fun r => max (-r) 0) ∘ (symPart_isHermitian S).eigenvalues) _ = _
    rw [← Multiset.map_map, map_eigenvalues_eq_roots, hroots', map_pad hnk, map_sortDesc,
      Multiset.map_add, Multiset.map_add, Multiset.map_map, Multiset.map_map]
    have h1 : ((fun r : ℝ => max (-r) 0) ∘ x) = fun _ => 0 :=
      funext fun i => max_eq_right (by simp [(hx i).le])
    have h2 : ((fun r : ℝ => max (-r) 0) ∘ fun i => -y i) = y :=
      funext fun i => by simp [(hy i).le]
    have h3 : Multiset.map (fun r : ℝ => max (-r) 0) (Multiset.replicate (k - 2 * n) 0)
        = Multiset.replicate (k - 2 * n) 0 := by simp [Multiset.map_replicate]
    rw [h1, h2, h3, Multiset.map_const', Finset.card_val, Finset.card_univ, Fintype.card_fin,
      add_comm (Multiset.replicate n 0), add_assoc, ← Multiset.replicate_add, hsplit]

end Spectrum

/-! ### The estimates of the paper's proof -/

section Estimates

variable {n : ℕ}

/-- The entrywise positive part `x₊ = max(x, 0)` of a vector. -/
noncomputable def posPartVec (x : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 fun j => max (x j) 0

/-- `x ↦ x₊` is `1`-Lipschitz in each entry, hence for the Euclidean norm. -/
lemma norm_posPartVec_sub_le (x y : EuclideanSpace ℝ (Fin n)) :
    ‖posPartVec x - posPartVec y‖ ≤ ‖x - y‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  refine Real.sqrt_le_sqrt (Finset.sum_le_sum fun j _ => ?_)
  simp only [posPartVec, PiLp.sub_apply, Real.norm_eq_abs, sq_abs]
  have := abs_max_sub_max_le_abs (x j) (y j) 0
  nlinarith [abs_nonneg (max (x j) 0 - max (y j) 0), abs_nonneg (x j - y j),
    sq_abs (max (x j) 0 - max (y j) 0), sq_abs (x j - y j)]

/-- `((c α))₊ = c₊ α` for `α ≥ 0`. -/
lemma posPartVec_smul {a : EuclideanSpace ℝ (Fin n)} (ha0 : ∀ j, 0 ≤ a j) (c : ℝ) :
    posPartVec (c • a) = max c 0 • a := by
  ext j
  simp only [posPartVec, PiLp.smul_apply, smul_eq_mul]
  rw [max_mul_of_nonneg _ _ (ha0 j), zero_mul]

/-- **The estimates of the proof of `thm:accumulation`.**  For `α ≥ 0`,
`α ≠ 0`, `β = ρα + ε` and `δ = ‖ε‖/‖α‖`:
`|⟨α,β⟩/‖α‖² - ρ| ≤ δ`, `|‖β‖²/‖α‖² - ρ²| ≤ δ² + 2ρδ`, and
`|‖(α - β)_±‖/‖α‖ - (1 - ρ)_±| ≤ δ`. -/
theorem accumulation_estimates (a b : EuclideanSpace ℝ (Fin n)) {ρ : ℝ} (hρ : 0 ≤ ρ)
    (ha : a ≠ 0) (ha0 : ∀ j, 0 ≤ a j) :
    |inner ℝ a b / ‖a‖ ^ 2 - ρ| ≤ ‖b - ρ • a‖ / ‖a‖ ∧
      |‖b‖ ^ 2 / ‖a‖ ^ 2 - ρ ^ 2|
        ≤ (‖b - ρ • a‖ / ‖a‖) ^ 2 + 2 * ρ * (‖b - ρ • a‖ / ‖a‖) ∧
      |‖posPartVec (a - b)‖ / ‖a‖ - max (1 - ρ) 0| ≤ ‖b - ρ • a‖ / ‖a‖ ∧
      |‖posPartVec (b - a)‖ / ‖a‖ - max (ρ - 1) 0| ≤ ‖b - ρ • a‖ / ‖a‖ := by
  set ε := b - ρ • a with hε
  have hb : b = ρ • a + ε := by rw [hε]; abel
  have hna : 0 < ‖a‖ := norm_pos_iff.mpr ha
  have hna2 : 0 < ‖a‖ ^ 2 := by positivity
  have hCS : |inner ℝ a ε| ≤ ‖a‖ * ‖ε‖ := abs_real_inner_le_norm a ε
  have hinner : inner ℝ a b = ρ * ‖a‖ ^ 2 + inner ℝ a ε := by
    rw [hb, inner_add_right, real_inner_smul_right, real_inner_self_eq_norm_sq]
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- Cauchy–Schwarz
    rw [hinner, show (ρ * ‖a‖ ^ 2 + inner ℝ a ε) / ‖a‖ ^ 2 - ρ = inner ℝ a ε / ‖a‖ ^ 2 by
      field_simp; ring, abs_div, abs_of_pos hna2, div_le_div_iff₀ hna2 hna]
    nlinarith [norm_nonneg ε]
  · -- `‖β‖² = ρ²‖α‖² + 2ρ⟨α,ε⟩ + ‖ε‖²`
    have hbb : ‖b‖ ^ 2 = ρ ^ 2 * ‖a‖ ^ 2 + 2 * ρ * inner ℝ a ε + ‖ε‖ ^ 2 := by
      rw [hb, norm_add_sq_real, norm_smul, real_inner_smul_left, Real.norm_eq_abs,
        abs_of_nonneg hρ]
      ring
    rw [hbb, show (ρ ^ 2 * ‖a‖ ^ 2 + 2 * ρ * inner ℝ a ε + ‖ε‖ ^ 2) / ‖a‖ ^ 2 - ρ ^ 2
        = (2 * ρ * inner ℝ a ε + ‖ε‖ ^ 2) / ‖a‖ ^ 2 by field_simp; ring,
      abs_div, abs_of_pos hna2, div_le_iff₀ hna2]
    have h1 : |2 * ρ * inner ℝ a ε + ‖ε‖ ^ 2| ≤ 2 * ρ * (‖a‖ * ‖ε‖) + ‖ε‖ ^ 2 := by
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ 2 * ρ),
        abs_of_nonneg (sq_nonneg ‖ε‖)]
      gcongr
    refine h1.trans (le_of_eq ?_)
    field_simp
    ring
  · -- `x ↦ x₊` is `1`-Lipschitz and `((1-ρ)α)₊ = (1-ρ)₊ α`
    have hlip := norm_posPartVec_sub_le (a - b) ((1 - ρ) • a)
    rw [posPartVec_smul ha0, show a - b - (1 - ρ) • a = -ε by rw [hε]; module,
      norm_neg] at hlip
    have hnorm : ‖max (1 - ρ) 0 • a‖ = max (1 - ρ) 0 * ‖a‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
    have htri := abs_norm_sub_norm_le (posPartVec (a - b)) (max (1 - ρ) 0 • a)
    rw [hnorm] at htri
    rw [show ‖posPartVec (a - b)‖ / ‖a‖ - max (1 - ρ) 0
        = (‖posPartVec (a - b)‖ - max (1 - ρ) 0 * ‖a‖) / ‖a‖ by field_simp,
      abs_div, abs_of_pos hna]
    exact div_le_div_of_nonneg_right (htri.trans hlip) hna.le
  · have hlip := norm_posPartVec_sub_le (b - a) ((ρ - 1) • a)
    rw [posPartVec_smul ha0, show b - a - (ρ - 1) • a = ε by rw [hε]; module] at hlip
    have hnorm : ‖max (ρ - 1) 0 • a‖ = max (ρ - 1) 0 * ‖a‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
    have htri := abs_norm_sub_norm_le (posPartVec (b - a)) (max (ρ - 1) 0 • a)
    rw [hnorm] at htri
    rw [show ‖posPartVec (b - a)‖ / ‖a‖ - max (ρ - 1) 0
        = (‖posPartVec (b - a)‖ - max (ρ - 1) 0 * ‖a‖) / ‖a‖ by field_simp,
      abs_div, abs_of_pos hna]
    exact div_le_div_of_nonneg_right (htri.trans hlip) hna.le

end Estimates

/-! ### The profile of `S_n` in terms of the lobes -/

section ProfileLobes

variable {n k : ℕ}

lemma inner_toLp (f g : Fin n → ℝ) :
    inner ℝ (WithLp.toLp 2 f : EuclideanSpace ℝ (Fin n)) (WithLp.toLp 2 g) = ∑ i, f i * g i := by
  rw [EuclideanSpace.inner_toLp_toLp]
  simp [dotProduct, mul_comm]

lemma norm_sq_toLp (f : Fin n → ℝ) :
    ‖(WithLp.toLp 2 f : EuclideanSpace ℝ (Fin n))‖ ^ 2 = ∑ i, f i ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq]; simp [Real.norm_eq_abs, sq_abs]

/-- For a symmetric `S` with spectrum `{x_i} + {-y_i} + {0^(k-2n)}`, the
profile in terms of `α = sortDesc x`, `β = sortDesc y` (as Euclidean vectors
`a`, `b`): `π(S) = (0, 2⟨α,β⟩, ‖(α-β)₊‖², ‖(α-β)₋‖²)/‖S‖²` with
`‖S‖² = ‖α‖² + ‖β‖²`. -/
lemma profile_of_spectrum (hk : 2 * n ≤ k) {S : Matrix (Fin k) (Fin k) ℝ}
    (hS : S.IsSymm) {x y : Fin n → ℝ} (hx : ∀ i, 0 < x i) (hy : ∀ i, 0 < y i)
    (hroots : S.charpoly.roots = Multiset.map x Finset.univ.val
      + Multiset.map (fun i => -y i) Finset.univ.val + Multiset.replicate (k - 2 * n) 0) :
    let a : EuclideanSpace ℝ (Fin n) := WithLp.toLp 2 (sortDesc x)
    let b : EuclideanSpace ℝ (Fin n) := WithLp.toLp 2 (sortDesc y)
    profile S = (0, 2 * inner ℝ a b / (‖a‖ ^ 2 + ‖b‖ ^ 2),
      ‖posPartVec (a - b)‖ ^ 2 / (‖a‖ ^ 2 + ‖b‖ ^ 2),
      ‖posPartVec (b - a)‖ ^ 2 / (‖a‖ ^ 2 + ‖b‖ ^ 2)) := by
  intro a b
  have hnk : n ≤ k := by omega
  obtain ⟨hα, hβ⟩ := alphaList_betaList_of_spectrum hk hS hx hy hroots
  have hsym := symPart_of_isSymm hS
  -- `‖S‖² = ‖α‖² + ‖β‖²`
  have hF : frobSq S = ‖a‖ ^ 2 + ‖b‖ ^ 2 := by
    have h := sum_alpha_sq_add_beta_sq (symPart_isHermitian S)
    rw [hα, hβ, sum_pad hnk (sortDesc x) (sortDesc x) (fun r _ => r ^ 2) (by simp),
      sum_pad hnk (sortDesc y) (sortDesc y) (fun r _ => r ^ 2) (by simp),
      ← frobSq_of_isHermitian (symPart_isHermitian S), hsym] at h
    rw [← h, norm_sq_toLp, norm_sq_toLp]
  have hskew : skewPart S = 0 := by
    rw [skewPart, show Sᵀ = S from hS.eq, sub_self, smul_zero]
  simp only [profile, Prod.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [profileA, hskew]; simp [frobSq]
  · rw [profileB, hα, hβ, sum_pad hnk _ _ (fun r s => r * s) (by simp), hF, inner_toLp]
  · have hP : ‖posPartVec (a - b)‖ ^ 2 = ∑ i, max (sortDesc x i - sortDesc y i) 0 ^ 2 := by
      rw [posPartVec, norm_sq_toLp]; rfl
    rw [profileC, hα, hβ, sum_pad hnk _ _ (fun r s => max (r - s) 0 ^ 2) (by simp), hF, hP]
  · have hP : ‖posPartVec (b - a)‖ ^ 2 = ∑ i, max (sortDesc y i - sortDesc x i) 0 ^ 2 := by
      rw [posPartVec, norm_sq_toLp]; rfl
    rw [profileD, hα, hβ, sum_pad hnk _ _ (fun r s => max (s - r) 0 ^ 2) (by simp), hF, hP]

end ProfileLobes

/-! ### The theorem -/

section Theorem

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- **Theorem (accumulation).**  With `X_i`, `X'_i` independent copies of a
positive random variable `X` of finite, nonzero second moment, `u, v > 0`,
`ρ = u/v`, `N_n ≥ 2n`, and `S_n` symmetric with eigenvalues `vX_1, …, vX_n`,
`-uX'_1, …, -uX'_n` and `N_n - 2n` zeros, the profile `π(S_n)` converges
almost surely to `(1/(1+ρ²)) (0, 2ρ, (1-ρ)₊², (ρ-1)₊²)`. -/
theorem accumulation {u v : ℝ} (hu : 0 < u) (hv : 0 < v)
    {X X' : ℕ → Ω → ℝ} (hX : ∀ i, Measurable (X i)) (hX' : ∀ i, Measurable (X' i))
    (hindep : iIndepFun (Sum.elim X X') P)
    {γ : Measure ℝ} (hXlaw : ∀ i, Measure.map (X i) P = γ)
    (hX'law : ∀ i, Measure.map (X' i) P = γ)
    (hγpos : γ (Set.Iic 0) = 0)
    (hM : ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ ≠ ⊤)
    (hMpos : 0 < ∫⁻ y, ENNReal.ofReal (y ^ 2) ∂γ)
    (Nn : ℕ → ℕ) (hNn : ∀ n, 2 * n ≤ Nn n)
    (S : (n : ℕ) → Ω → Matrix (Fin (Nn n)) (Fin (Nn n)) ℝ)
    (hSsymm : ∀ n ω, (S n ω).IsSymm)
    (hspec : ∀ n ω, (S n ω).charpoly.roots
      = Multiset.map (fun i : Fin n => v * X i ω) Finset.univ.val
        + Multiset.map (fun i : Fin n => -(u * X' i ω)) Finset.univ.val
        + Multiset.replicate (Nn n - 2 * n) 0) :
    ∀ᵐ ω ∂P, Tendsto (fun n => profile (S n ω)) atTop
      (𝓝 ((1 / (1 + (u / v) ^ 2)) •
        ((0 : ℝ), 2 * (u / v), max (1 - u / v) 0 ^ 2, max (u / v - 1) 0 ^ 2))) := by
  set ρ := u / v with hρdef
  have hρ : 0 < ρ := div_pos hu hv
  -- the copies of `X` are almost surely positive
  have hposX : ∀ (Y : ℕ → Ω → ℝ), (∀ i, Measurable (Y i)) →
      (∀ i, Measure.map (Y i) P = γ) → ∀ᵐ ω ∂P, ∀ i, 0 < Y i ω := by
    intro Y hY hYlaw
    rw [ae_all_iff]
    intro i
    have h0 : P (Y i ⁻¹' Set.Iic 0) = 0 := by
      rw [← Measure.map_apply (hY i) measurableSet_Iic, hYlaw i, hγpos]
    filter_upwards [measure_eq_zero_iff_ae_notMem.mp h0] with ω hω
    simpa using hω
  -- Lemma 6: `δ_n = ‖β_n - ρα_n‖/‖α_n‖ → 0`
  have hlobes := proportional_lobes hu hv hX hX' hindep hXlaw hX'law hγpos hM hMpos
  filter_upwards [hlobes, hposX X hX hXlaw, hposX X' hX' hX'law] with ω hδ hXpos hX'pos
  -- the lobes as Euclidean vectors
  let a : (n : ℕ) → EuclideanSpace ℝ (Fin n) := fun n =>
    WithLp.toLp 2 (sortDesc fun i : Fin n => v * X i ω)
  let b : (n : ℕ) → EuclideanSpace ℝ (Fin n) := fun n =>
    WithLp.toLp 2 (sortDesc fun i : Fin n => u * X' i ω)
  have ha0 : ∀ n j, 0 ≤ a n j := fun n j => by
    simp only [a, sortDesc_eq_comp_perm, Function.comp_apply]
    exact (mul_pos hv (hXpos _)).le
  have hane : ∀ n, 1 ≤ n → a n ≠ 0 := fun n hn h => by
    have := congrArg (fun w : EuclideanSpace ℝ (Fin n) => w ⟨0, hn⟩) h
    simp only [a, PiLp.zero_apply, sortDesc_eq_comp_perm,
      Function.comp_apply] at this
    exact (mul_pos hv (hXpos _)).ne' this
  -- `δ_n` is Lemma 6's quantity
  have hδ' : Tendsto (fun n => ‖b n - ρ • a n‖ / ‖a n‖) atTop (𝓝 0) := by
    refine hδ.congr fun n => ?_
    simp only [a, b, EuclideanSpace.norm_eq, PiLp.sub_apply, PiLp.smul_apply,
      smul_eq_mul, Real.norm_eq_abs, sq_abs, ρ]
  -- the four ratios, and their limits
  set r1 := fun n => inner ℝ (a n) (b n) / ‖a n‖ ^ 2
  set r2 := fun n => ‖b n‖ ^ 2 / ‖a n‖ ^ 2
  set r3 := fun n => ‖posPartVec (a n - b n)‖ / ‖a n‖
  set r4 := fun n => ‖posPartVec (b n - a n)‖ / ‖a n‖
  have hev : ∀ᶠ n in atTop, 1 ≤ n := eventually_ge_atTop 1
  have hlim : ∀ (r : ℕ → ℝ) (L : ℝ) (f : ℝ → ℝ), Tendsto f (𝓝 0) (𝓝 0) →
      (∀ n, 1 ≤ n → |r n - L| ≤ f (‖b n - ρ • a n‖ / ‖a n‖)) →
      Tendsto r atTop (𝓝 L) := by
    intro r L f hf hr
    rw [tendsto_iff_norm_sub_tendsto_zero]
    refine squeeze_zero' (Eventually.of_forall fun n => norm_nonneg _)
      (hev.mono fun n hn => by rw [Real.norm_eq_abs]; exact hr n hn) (hf.comp hδ')
  have h1 : Tendsto r1 atTop (𝓝 ρ) :=
    hlim r1 ρ id tendsto_id fun n hn =>
      (accumulation_estimates (a n) (b n) hρ.le (hane n hn) (ha0 n)).1
  have h2 : Tendsto r2 atTop (𝓝 (ρ ^ 2)) :=
    hlim r2 (ρ ^ 2) (fun δ => δ ^ 2 + 2 * ρ * δ)
      ((by fun_prop : Continuous fun δ : ℝ => δ ^ 2 + 2 * ρ * δ).tendsto' 0 0 (by simp))
      fun n hn => (accumulation_estimates (a n) (b n) hρ.le (hane n hn) (ha0 n)).2.1
  have h3 : Tendsto r3 atTop (𝓝 (max (1 - ρ) 0)) :=
    hlim r3 _ id tendsto_id fun n hn =>
      (accumulation_estimates (a n) (b n) hρ.le (hane n hn) (ha0 n)).2.2.1
  have h4 : Tendsto r4 atTop (𝓝 (max (ρ - 1) 0)) :=
    hlim r4 _ id tendsto_id fun n hn =>
      (accumulation_estimates (a n) (b n) hρ.le (hane n hn) (ha0 n)).2.2.2
  -- the profile, divided through by `‖α_n‖²`
  have hprof : ∀ᶠ n in atTop, profile (S n ω)
      = (0, 2 * r1 n / (1 + r2 n), r3 n ^ 2 / (1 + r2 n), r4 n ^ 2 / (1 + r2 n)) := by
    filter_upwards [hev] with n hn
    have hna : 0 < ‖a n‖ := norm_pos_iff.mpr (hane n hn)
    rw [profile_of_spectrum (hNn n) (hSsymm n ω) (fun i => mul_pos hv (hXpos i))
      (fun i => mul_pos hu (hX'pos i)) (hspec n ω)]
    change ((0 : ℝ), 2 * inner ℝ (a n) (b n) / (‖a n‖ ^ 2 + ‖b n‖ ^ 2),
      ‖posPartVec (a n - b n)‖ ^ 2 / (‖a n‖ ^ 2 + ‖b n‖ ^ 2),
      ‖posPartVec (b n - a n)‖ ^ 2 / (‖a n‖ ^ 2 + ‖b n‖ ^ 2)) = _
    simp only [r1, r2, r3, r4, Prod.mk.injEq]
    refine ⟨trivial, ?_, ?_, ?_⟩ <;> field_simp
  have hden : (1 + ρ ^ 2) ≠ 0 := by positivity
  refine Tendsto.congr' (EventuallyEq.symm hprof) ?_
  have hlimit : (1 / (1 + ρ ^ 2)) • ((0 : ℝ), 2 * ρ, max (1 - ρ) 0 ^ 2, max (ρ - 1) 0 ^ 2)
      = (0, 2 * ρ / (1 + ρ ^ 2), max (1 - ρ) 0 ^ 2 / (1 + ρ ^ 2),
          max (ρ - 1) 0 ^ 2 / (1 + ρ ^ 2)) := by
    simp only [Prod.smul_mk, smul_eq_mul, mul_zero]
    refine Prod.ext rfl (Prod.ext ?_ (Prod.ext ?_ ?_)) <;> simp only <;> field_simp
  rw [hlimit]
  refine tendsto_const_nhds.prodMk_nhds (Tendsto.prodMk_nhds ?_ (Tendsto.prodMk_nhds ?_ ?_))
  · exact ((tendsto_const_nhds.mul h1).div (tendsto_const_nhds.add h2) hden)
  · exact ((h3.pow 2).div (tendsto_const_nhds.add h2) hden)
  · exact ((h4.pow 2).div (tendsto_const_nhds.add h2) hden)

end Theorem

/-! ### Where the limit lies -/

section Limit

/-- The limit profile `(1/(1+ρ²)) (0, 2ρ, (1-ρ)₊², (ρ-1)₊²)`. -/
noncomputable def limitProfile (ρ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (1 / (1 + ρ ^ 2)) • ((0 : ℝ), 2 * ρ, max (1 - ρ) 0 ^ 2, max (ρ - 1) 0 ^ 2)

/-- The projection `p : Δ₃ → Δ`, `p(a,b,c,d) = (0, a + b, c, d)`. -/
def projP (x : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := (0, x.1 + x.2.1, x.2.2.1, x.2.2.2)

lemma limitProfile_eq (ρ : ℝ) :
    limitProfile ρ = (0, 2 * ρ / (1 + ρ ^ 2), max (1 - ρ) 0 ^ 2 / (1 + ρ ^ 2),
      max (ρ - 1) 0 ^ 2 / (1 + ρ ^ 2)) := by
  simp only [limitProfile, Prod.smul_mk, smul_eq_mul, mul_zero, Prod.mk.injEq]
  refine ⟨trivial, ?_, ?_, ?_⟩ <;> ring

/-- **The limit lies in `p(Θ)`** (for `ρ ≥ 0`). -/
theorem limitProfile_mem_projP_Theta {ρ : ℝ} (hρ : 0 ≤ ρ) :
    limitProfile ρ ∈ projP '' Theta := by
  have hden : 0 < 1 + ρ ^ 2 := by positivity
  refine ⟨(ρ / (1 + ρ ^ 2), ρ / (1 + ρ ^ 2), max (1 - ρ) 0 ^ 2 / (1 + ρ ^ 2),
    max (ρ - 1) 0 ^ 2 / (1 + ρ ^ 2)), ⟨by positivity, by positivity, by positivity,
    by positivity, ?_, rfl, ?_⟩, ?_⟩
  · rcases le_total ρ 1 with h | h
    · rw [max_eq_left (by linarith : 0 ≤ 1 - ρ), max_eq_right (by linarith : ρ - 1 ≤ 0)]
      field_simp; ring
    · rw [max_eq_right (by linarith : 1 - ρ ≤ 0), max_eq_left (by linarith : 0 ≤ ρ - 1)]
      field_simp; ring
  · rcases le_total ρ 1 with h | h
    · rw [max_eq_right (by linarith : ρ - 1 ≤ 0)]; ring
    · rw [max_eq_right (by linarith : 1 - ρ ≤ 0)]; ring
  · rw [limitProfile_eq, projP]
    refine Prod.ext rfl (Prod.ext ?_ rfl)
    simp only
    ring

/-- The coordinate `b(ρ) = 2ρ/(1+ρ²)` of the limit. -/
noncomputable def limitB (ρ : ℝ) : ℝ := 2 * ρ / (1 + ρ ^ 2)

/-- **The limit traces the edge `{d = 0}` for `ρ ∈ (0, 1]`**: there it equals
`(0, b, 1 - b, 0)` with `b = b(ρ)`, and `b` is a strictly increasing
bijection from `(0, 1]` onto `(0, 1]`, from the vertex `(b,c,d) = (0,1,0)`
(as `ρ → 0`) to `(1,0,0)` (at `ρ = 1`). -/
theorem limitProfile_edge_d {ρ : ℝ} (h0 : 0 < ρ) (h1 : ρ ≤ 1) :
    limitProfile ρ = (0, limitB ρ, 1 - limitB ρ, 0) := by
  have hden : 0 < 1 + ρ ^ 2 := by positivity
  rw [limitProfile_eq, limitB, max_eq_left (by linarith : 0 ≤ 1 - ρ),
    max_eq_right (by linarith : ρ - 1 ≤ 0)]
  simp only [Prod.mk.injEq]
  refine ⟨trivial, trivial, ?_, by simp⟩
  field_simp; ring

/-- **The limit traces the edge `{c = 0}` for `ρ ∈ [1, ∞)`**: there it equals
`(0, b, 0, 1 - b)` with `b = b(ρ)`. -/
theorem limitProfile_edge_c {ρ : ℝ} (h1 : 1 ≤ ρ) :
    limitProfile ρ = (0, limitB ρ, 0, 1 - limitB ρ) := by
  have hden : 0 < 1 + ρ ^ 2 := by positivity
  rw [limitProfile_eq, limitB, max_eq_right (by linarith : 1 - ρ ≤ 0),
    max_eq_left (by linarith : 0 ≤ ρ - 1)]
  simp only [Prod.mk.injEq]
  refine ⟨trivial, trivial, by simp, ?_⟩
  field_simp; ring

lemma limitB_sub (ρ σ : ℝ) :
    limitB ρ - limitB σ = 2 * (ρ - σ) * (1 - ρ * σ) / ((1 + ρ ^ 2) * (1 + σ ^ 2)) := by
  rw [limitB, limitB]
  field_simp
  ring

theorem limitB_strictMonoOn : StrictMonoOn limitB (Set.Ioc 0 1) := by
  intro ρ hρ σ hσ hlt
  have h := limitB_sub σ ρ
  have hpos : 0 < 2 * (σ - ρ) * (1 - σ * ρ) / ((1 + σ ^ 2) * (1 + ρ ^ 2)) := by
    apply div_pos _ (by positivity)
    apply mul_pos (by linarith)
    nlinarith [hρ.1, hρ.2, hσ.1, hσ.2]
  linarith

theorem limitB_strictAntiOn : StrictAntiOn limitB (Set.Ici 1) := by
  intro ρ hρ σ hσ hlt
  have h := limitB_sub ρ σ
  have hpos : 0 < 2 * (ρ - σ) * (1 - ρ * σ) / ((1 + ρ ^ 2) * (1 + σ ^ 2)) := by
    apply div_pos _ (by positivity)
    have h1 : ρ - σ < 0 := by linarith
    have h2 : 1 - ρ * σ < 0 := by nlinarith [Set.mem_Ici.mp hρ, Set.mem_Ici.mp hσ]
    nlinarith
  linarith

/-- `b` maps `(0,1]` onto `(0,1]`. -/
theorem limitB_image_Ioc : limitB '' Set.Ioc 0 1 = Set.Ioc 0 1 := by
  ext β
  constructor
  · rintro ⟨ρ, ⟨h0, h1⟩, rfl⟩
    have hden : 0 < 1 + ρ ^ 2 := by positivity
    refine ⟨by rw [limitB]; positivity, ?_⟩
    rw [limitB, div_le_one hden]
    nlinarith [sq_nonneg (ρ - 1)]
  · rintro ⟨h0, h1⟩
    -- `ρ = (1 - √(1 - β²))/β` solves `βρ² - 2ρ + β = 0`
    set r := Real.sqrt (1 - β ^ 2)
    have hr0 : 0 ≤ r := Real.sqrt_nonneg _
    have hrr : r ^ 2 = 1 - β ^ 2 := Real.sq_sqrt (by nlinarith)
    have hr1 : r < 1 := by nlinarith
    refine ⟨(1 - r) / β, ⟨div_pos (by linarith) h0, ?_⟩, ?_⟩
    · rw [div_le_one h0]; nlinarith
    · rw [limitB]
      have hden : 0 < 1 + ((1 - r) / β) ^ 2 := by positivity
      rw [div_eq_iff hden.ne']
      field_simp
      nlinarith [hrr]

/-- `b` maps `[1,∞)` onto `(0,1]`. -/
theorem limitB_image_Ici : limitB '' Set.Ici 1 = Set.Ioc 0 1 := by
  ext β
  constructor
  · rintro ⟨ρ, h1, rfl⟩
    have h1 : 1 ≤ ρ := h1
    have hden : 0 < 1 + ρ ^ 2 := by positivity
    refine ⟨by rw [limitB]; positivity, ?_⟩
    rw [limitB, div_le_one hden]
    nlinarith [sq_nonneg (ρ - 1)]
  · rintro ⟨h0, h1⟩
    -- `ρ = (1 + √(1 - β²))/β`
    set r := Real.sqrt (1 - β ^ 2)
    have hr0 : 0 ≤ r := Real.sqrt_nonneg _
    have hrr : r ^ 2 = 1 - β ^ 2 := Real.sq_sqrt (by nlinarith)
    refine ⟨(1 + r) / β, ?_, ?_⟩
    · show 1 ≤ (1 + r) / β
      rw [le_div_iff₀ h0]; linarith
    · rw [limitB]
      have hden : 0 < 1 + ((1 + r) / β) ^ 2 := by positivity
      rw [div_eq_iff hden.ne']
      field_simp
      nlinarith [hrr]

end Limit

end Appendices.MainText
