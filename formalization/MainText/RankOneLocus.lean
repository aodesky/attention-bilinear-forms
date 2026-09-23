/-
Main text, §"The profile of a real bilinear form": Proposition `prop:rankOneLocus`.

    Let `L` be a rank-one matrix of unit norm.  Then

        π(L) = ((1 - t²)/2, (1 - t²)/2, t₊², t₋²),   where t = tr(L).

    The set of profiles of rank-one matrices is equal to
    `Θ = {a = b, cd = 0} ⊂ Δ₃`.

Here `x₊ = max(x, 0)` and `x₋ = max(-x, 0)`, so `t₊² = max(t,0)²`.

The paper's proof: write `L = u vᵀ` with unit vectors `u, v`; then
`t = ⟨u, v⟩`.  If `u = ± v`, an orthogonal change of coordinates bringing `u`
to the first basis vector makes `L = ± E₁₁`.  If `|t| < 1`, the vectors
`e = (u + v)/√(2 + 2t)` and `f = (v - u)/√(2 - 2t)` are orthonormal; in an
orthonormal basis extending them, `L` is the block
`½ [[1 + t, √(1 - t²)], [-√(1 - t²), t - 1]]` padded with zeros, whose
symmetric part is `diag((1+t)/2, (t-1)/2)`, and the rest follows.

We follow it.  "An orthogonal change of coordinates" is `L ↦ Uᵀ L U` for a
matrix `U` with `Uᵀ U = 1` whose first columns are the prescribed vectors
(`exists_orthogonal_cols`), under which the profile is invariant
(`profile_conj`); in the new coordinates `L` becomes `vecMulVec p q` with
`p = Uᵀ u`, `q = Uᵀ v`.  The paper's "the rest follows easily" is
`profile_of_diag`.  The core computation is carried out for `s · u vᵀ` with
`s > 0` (`profile_smul_vecMulVec`), which gives both the formula (`s = 1`)
and, since the profile of a nonzero `L` is that of `L/‖L‖`, the set of
profiles of all rank-one matrices.

Implicit hypothesis made explicit: the set equality `{π(L) : rank L = 1} = Θ`
needs `N ≥ 2`.  For `N = 1` a rank-one matrix is a nonzero scalar and its
profile is `(0,0,1,0)` or `(0,0,0,1)`, so only two points of `Θ` occur;
realizing an interior point of `Θ` requires `u` and `v` not parallel.

Encoding.  The profile is the quadruple `(profileA L, …, profileD L)` of
`Appendices.Profile`.  `Δ₃` is the standard simplex
`{(a,b,c,d) : a,b,c,d ≥ 0, a + b + c + d = 1}`.
-/
import MainText.ProfileInvariance
import Appendices.RandomMatrices.EnergyIdentities

namespace Appendices.MainText

open Matrix

variable {N : ℕ}

/-- The profile `π(L) = (a, b, c, d)`. -/
noncomputable def profile (L : Matrix (Fin N) (Fin N) ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (profileA L, profileB L, profileC L, profileD L)

/-- `Θ = {a = b, cd = 0} ⊂ Δ₃`. -/
def Theta : Set (ℝ × ℝ × ℝ × ℝ) :=
  {x | 0 ≤ x.1 ∧ 0 ≤ x.2.1 ∧ 0 ≤ x.2.2.1 ∧ 0 ≤ x.2.2.2 ∧
    x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 1 ∧ x.1 = x.2.1 ∧ x.2.2.1 * x.2.2.2 = 0}

/-! ### Orthogonal changes of coordinates with prescribed columns -/

/-- An orthonormal family of vectors, indexed by `s`, extends to the columns of
an orthogonal matrix. -/
lemma exists_orthogonal_cols (s : Set (Fin N)) (v : Fin N → Fin N → ℝ)
    (hv : ∀ i ∈ s, ∀ j ∈ s, v i ⬝ᵥ v j = if i = j then 1 else 0) :
    ∃ U : Matrix (Fin N) (Fin N) ℝ, Uᵀ * U = 1 ∧ ∀ i ∈ s, ∀ k, U k i = v i k := by
  let v' : Fin N → EuclideanSpace ℝ (Fin N) := fun i => WithLp.toLp 2 (v i)
  have hv' : Orthonormal ℝ (s.restrict v') := by
    rw [orthonormal_iff_ite]
    rintro ⟨i, hi⟩ ⟨j, hj⟩
    simp only [Set.restrict_apply, v', EuclideanSpace.inner_toLp_toLp, star_trivial,
      Subtype.mk.injEq]
    rw [dotProduct_comm, hv i hi j hj]
  obtain ⟨b, hb⟩ := hv'.exists_orthonormalBasis_extension_of_card_eq
    (by simp : Module.finrank ℝ (EuclideanSpace ℝ (Fin N)) = Fintype.card (Fin N))
  refine ⟨Matrix.of fun k j => b j k, ?_, fun i hi k => by simp [hb i hi, v']⟩
  ext i j
  have h := orthonormal_iff_ite.mp b.orthonormal j i
  rw [EuclideanSpace.inner_eq_star_dotProduct, star_trivial] at h
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply, Matrix.one_apply]
  simp only [dotProduct] at h
  rw [h]
  by_cases hij : i = j
  · simp [hij]
  · simp [hij, Ne.symm hij]

/-- In the coordinates given by the columns of `U`, `u vᵀ` becomes `p qᵀ`
with `p = Uᵀ u`, `q = Uᵀ v`. -/
lemma conj_vecMulVec (U : Matrix (Fin N) (Fin N) ℝ) (u v : Fin N → ℝ) :
    Uᵀ * vecMulVec u v * U = vecMulVec (Uᵀ *ᵥ u) (Uᵀ *ᵥ v) := by
  ext i j
  simp only [Matrix.mul_apply, vecMulVec_apply, Matrix.transpose_apply, mulVec, dotProduct,
    Finset.sum_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
  ring

/-! ### The profile when the symmetric part is diagonal -/

/-- The profile read off from a diagonal symmetric part `diag(d)` whose positive
parts are `P` at the first index and whose negative parts are, up to
rearrangement, `Q` at one index.  This is the paper's "the rest follows". -/
lemma profile_of_diag (hN : 0 < N) (M : Matrix (Fin N) (Fin N) ℝ) (d : Fin N → ℝ)
    (hd : symPart M = diagonal d) {P Q : ℝ} (hP : 0 ≤ P) (hQ : 0 ≤ Q)
    (hα : (fun i => max (d i) 0) = fun i => if i = (⟨0, hN⟩ : Fin N) then P else 0)
    (hβ : Multiset.map (fun i => max (-d i) 0) Finset.univ.val
      = Multiset.map (fun i => if i = (⟨0, hN⟩ : Fin N) then Q else 0) Finset.univ.val) :
    profileB M = 2 * (P * Q) / frobSq M ∧
      profileC M = max (P - Q) 0 ^ 2 / frobSq M ∧
      profileD M = max (Q - P) 0 ^ 2 / frobSq M := by
  set i₀ : Fin N := ⟨0, hN⟩
  have hanti : ∀ R : ℝ, 0 ≤ R → Antitone (fun i : Fin N => if i = i₀ then R else 0) := by
    intro R hR i j hij
    by_cases hj : j = i₀
    · have hi : i = i₀ := by
        apply Fin.ext
        have := Fin.le_def.mp hij
        rw [hj] at this
        simp [i₀] at this ⊢
        omega
      simp [hi, hj]
    · simp only [hj, if_false]
      split_ifs <;> linarith
  have ha : alphaList (symPart M) (symPart_isHermitian M)
      = fun i => if i = i₀ then P else 0 := by
    rw [alphaList_diagonal _ hd, hα]
    exact sortDesc_eq_self (hanti P hP)
  have hb : betaList (symPart M) (symPart_isHermitian M)
      = fun i => if i = i₀ then Q else 0 := by
    rw [betaList_diagonal _ hd, sortDesc_eq_of_map_eq hβ]
    exact sortDesc_eq_self (hanti Q hQ)
  have hsum : ∀ g : ℝ → ℝ → ℝ, g 0 0 = 0 →
      ∑ j : Fin N, g ((fun i => if i = i₀ then P else 0) j)
        ((fun i => if i = i₀ then Q else 0) j) = g P Q := by
    intro g hg
    rw [Fintype.sum_eq_single i₀ (fun j hj => by simp [hj, hg])]
    simp
  refine ⟨?_, ?_, ?_⟩
  · rw [profileB, ha, hb, hsum (fun x y => x * y) (by simp)]
  · rw [profileC, ha, hb, hsum (fun x y => max (x - y) 0 ^ 2) (by simp)]
  · rw [profileD, ha, hb, hsum (fun x y => max (y - x) 0 ^ 2) (by simp)]

/-! ### Auxiliary identities -/

lemma frobSq_smul_vecMulVec (s : ℝ) (u v : Fin N → ℝ) :
    frobSq (s • vecMulVec u v) = s ^ 2 * ((u ⬝ᵥ u) * (v ⬝ᵥ v)) := by
  rw [frobSq_eq_sum_sq, dotProduct, dotProduct, Finset.sum_mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [smul_apply, vecMulVec_apply, smul_eq_mul]
  ring

lemma frobSq_diagonal (d : Fin N → ℝ) : frobSq (diagonal d) = ∑ i, d i ^ 2 := by
  rw [frobSq_eq_sum_sq]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Fintype.sum_eq_single i (fun j hj => by simp [diagonal_apply_ne _ (Ne.symm hj)])]
  simp

/-- `Uᵀ` sends the `j`-th column of an orthogonal `U` to the `j`-th standard
basis vector. -/
lemma transpose_mulVec_col {U : Matrix (Fin N) (Fin N) ℝ} (hU : Uᵀ * U = 1) (j : Fin N) :
    Uᵀ *ᵥ (fun k => U k j) = fun i => if i = j then 1 else 0 := by
  funext i
  have h := congrFun (congrFun hU i) j
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply] at h
  simp only [mulVec, dotProduct, Matrix.transpose_apply]
  exact h

lemma conj_smul (U A : Matrix (Fin N) (Fin N) ℝ) (s : ℝ) :
    Uᵀ * (s • A) * U = s • (Uᵀ * A * U) := by
  rw [Matrix.mul_smul, Matrix.smul_mul]

/-- `a = ‖T‖²/‖L‖² = (‖L‖² - ‖S‖²)/‖L‖²`. -/
lemma profileA_eq_of_symPart (M : Matrix (Fin N) (Fin N) ℝ) (d : Fin N → ℝ)
    (hd : symPart M = diagonal d) :
    profileA M = (frobSq M - ∑ i, d i ^ 2) / frobSq M := by
  rw [profileA, frobSq_eq_frobSq_symPart_add_frobSq_skewPart M, hd, frobSq_diagonal]
  ring_nf

/-! ### The core computation -/

/-- **The profile of `s · u vᵀ`** for unit vectors `u, v` and `s > 0`:
`((1 - t²)/2, (1 - t²)/2, t₊², t₋²)` with `t = ⟨u, v⟩`. -/
theorem profile_smul_vecMulVec (u v : Fin N → ℝ) (hu : u ⬝ᵥ u = 1) (hv : v ⬝ᵥ v = 1)
    {s : ℝ} (hs : 0 < s) :
    profile (s • vecMulVec u v)
      = ((1 - (u ⬝ᵥ v) ^ 2) / 2, (1 - (u ⬝ᵥ v) ^ 2) / 2,
          max (u ⬝ᵥ v) 0 ^ 2, max (-(u ⬝ᵥ v)) 0 ^ 2) := by
  set t := u ⬝ᵥ v with ht
  have hF : frobSq (s • vecMulVec u v) = s ^ 2 := by
    rw [frobSq_smul_vecMulVec, hu, hv]; ring
  have hs2 : s ^ 2 ≠ 0 := by positivity
  have hN : 0 < N := by
    rcases Nat.eq_zero_or_pos N with rfl | h
    · simp [dotProduct] at hu
    · exact h
  set i₀ : Fin N := ⟨0, hN⟩
  -- `|t| ≤ 1`, with `t = ±1` exactly when `v = t u`
  have hsub : (v - u) ⬝ᵥ (v - u) = 2 - 2 * t := by
    simp only [sub_dotProduct, dotProduct_sub, hu, hv, ht, dotProduct_comm v u]; ring
  have hadd : (v + u) ⬝ᵥ (v + u) = 2 + 2 * t := by
    simp only [add_dotProduct, dotProduct_add, hu, hv, ht, dotProduct_comm v u]; ring
  have hle : t ≤ 1 := by
    have : 0 ≤ (v - u) ⬝ᵥ (v - u) := Finset.sum_nonneg fun i _ => mul_self_nonneg _
    linarith
  have hge : -1 ≤ t := by
    have : 0 ≤ (v + u) ⬝ᵥ (v + u) := Finset.sum_nonneg fun i _ => mul_self_nonneg _
    linarith
  by_cases hpm : t = 1 ∨ t = -1
  · /- `u = ± v`: bring `u` to the first basis vector. -/
    have hvu : v = t • u := by
      rcases hpm with h | h
      · have h0 : (v - u) ⬝ᵥ (v - u) = 0 := by rw [hsub, h]; ring
        have := dotProduct_self_eq_zero.mp h0
        rw [h, one_smul]; exact sub_eq_zero.mp this
      · have h0 : (v + u) ⬝ᵥ (v + u) = 0 := by rw [hadd, h]; ring
        have := dotProduct_self_eq_zero.mp h0
        rw [h, neg_one_smul]; exact eq_neg_of_add_eq_zero_left this
    obtain ⟨U, hU, hcol⟩ := exists_orthogonal_cols {i₀} (fun _ => u)
      (by rintro i rfl j rfl; simp [hu])
    have hp : Uᵀ *ᵥ u = fun i => if i = i₀ then 1 else 0 := by
      rw [← transpose_mulVec_col hU i₀]
      congr 1; funext k; rw [hcol i₀ rfl k]
    obtain ⟨hA, hB, hC, hD⟩ := profile_conj hU (s • vecMulVec u v)
    have hFM := frobSq_conj hU (s • vecMulVec u v)
    obtain ⟨M, hM⟩ : ∃ M, Uᵀ * (s • vecMulVec u v) * U = M := ⟨_, rfl⟩
    rw [hM] at hA hB hC hD hFM
    rw [hF] at hFM
    have hM' : M = s • vecMulVec (fun i => if i = i₀ then 1 else 0)
        (t • fun i => if i = i₀ then (1 : ℝ) else 0) := by
      rw [← hM, conj_smul, conj_vecMulVec, hvu, mulVec_smul, hp]
    set d : Fin N → ℝ := fun i => if i = i₀ then s * t else 0
    have hd : symPart M = diagonal d := by
      ext i j
      rw [hM', symPart]
      by_cases hi : i = i₀ <;> by_cases hj : j = i₀ <;>
        simp [hi, hj, d, diagonal, vecMulVec_apply, Ne.symm]
      all_goals ring
    obtain ⟨hB', hC', hD'⟩ := profile_of_diag hN M d hd (le_max_right (s * t) 0)
      (le_max_right (-(s * t)) 0)
      (by funext i; by_cases hi : i = i₀ <;> simp [d, hi, i₀])
      (by congr 1; funext i; by_cases hi : i = i₀ <;> simp [d, hi, i₀])
    have hsumd : ∑ i, d i ^ 2 = s ^ 2 * t ^ 2 := by
      rw [Fintype.sum_eq_single i₀ (fun j hj => by simp [d, hj])]
      simp [d]; ring
    have hA' := profileA_eq_of_symPart M d hd
    rw [profile, ← hA, ← hB, ← hC, ← hD, hA', hB', hC', hD', hFM, hsumd]
    rcases hpm with h | h <;> rw [h] <;> simp [hs.le, hs2]
  · /- `|t| < 1`: the orthonormal pair `e, f`. -/
    push_neg at hpm
    have hlt : t < 1 := lt_of_le_of_ne hle hpm.1
    have hgt : -1 < t := lt_of_le_of_ne hge (Ne.symm hpm.2)
    have hx2 : 0 < (1 + t) / 2 := by linarith
    have hy2 : 0 < (1 - t) / 2 := by linarith
    set x := Real.sqrt ((1 + t) / 2) with hxdef
    set y := Real.sqrt ((1 - t) / 2) with hydef
    have hx : 0 < x := Real.sqrt_pos.mpr hx2
    have hy : 0 < y := Real.sqrt_pos.mpr hy2
    have hxx : x ^ 2 = (1 + t) / 2 := Real.sq_sqrt hx2.le
    have hyy : y ^ 2 = (1 - t) / 2 := Real.sq_sqrt hy2.le
    -- `u` and `v` are not parallel, so `N ≥ 2`
    have hN2 : 1 < N := by
      by_contra hcon
      have hsingle : ∀ w w' : Fin N → ℝ, w ⬝ᵥ w' = w i₀ * w' i₀ := fun w w' =>
        Fintype.sum_eq_single i₀ fun j hj => absurd (Fin.ext (by simp [i₀]; omega)) hj
      have h1 : t ^ 2 = 1 := by
        calc t ^ 2 = (u i₀ * u i₀) * (v i₀ * v i₀) := by rw [ht, hsingle]; ring
          _ = 1 := by rw [← hsingle u u, ← hsingle v v, hu, hv]; ring
      rcases sq_eq_one_iff.mp h1 with h | h
      · exact hpm.1 h
      · exact hpm.2 h
    set i₁ : Fin N := ⟨1, hN2⟩
    have h01 : i₀ ≠ i₁ := by simp [i₀, i₁, Fin.ext_iff]
    -- the orthonormal pair `e = (u + v)/√(2 + 2t)`, `f = (v - u)/√(2 - 2t)`
    obtain ⟨e, he⟩ : ∃ e : Fin N → ℝ, e = (1 / (2 * x)) • (u + v) := ⟨_, rfl⟩
    obtain ⟨f, hf⟩ : ∃ f : Fin N → ℝ, f = (1 / (2 * y)) • (v - u) := ⟨_, rfl⟩
    have hee : e ⬝ᵥ e = 1 := by
      rw [he, smul_dotProduct, dotProduct_smul, add_comm u v, hadd, smul_eq_mul, smul_eq_mul]
      field_simp; nlinarith [hxx]
    have hff : f ⬝ᵥ f = 1 := by
      rw [hf, smul_dotProduct, dotProduct_smul, hsub, smul_eq_mul, smul_eq_mul]
      field_simp; nlinarith [hyy]
    have hef : e ⬝ᵥ f = 0 := by
      rw [he, hf, smul_dotProduct, dotProduct_smul]
      simp only [add_dotProduct, dotProduct_sub, hu, hv, dotProduct_comm v u, smul_eq_mul]
      ring
    have hfe : f ⬝ᵥ e = 0 := by rw [dotProduct_comm, hef]
    have hue : u = x • e - y • f := by
      funext k; simp only [he, hf, Pi.sub_apply, Pi.smul_apply, Pi.add_apply, smul_eq_mul]
      field_simp; ring
    have hve : v = x • e + y • f := by
      funext k; simp only [he, hf, Pi.add_apply, Pi.smul_apply, Pi.sub_apply, smul_eq_mul]
      field_simp; ring
    -- an orthonormal basis extending `e, f`
    obtain ⟨U, hU, hcol⟩ := exists_orthogonal_cols {i₀, i₁}
      (fun i => if i = i₀ then e else f) (by
        rintro i hi j hj
        rcases hi with rfl | rfl <;> rcases hj with rfl | rfl <;>
          simp [h01, h01.symm, hee, hff, hef, hfe])
    have hpe : Uᵀ *ᵥ e = fun i => if i = i₀ then 1 else 0 := by
      rw [← transpose_mulVec_col hU i₀]
      congr 1; funext k; rw [hcol i₀ (by simp) k]; simp
    have hpf : Uᵀ *ᵥ f = fun i => if i = i₁ then 1 else 0 := by
      rw [← transpose_mulVec_col hU i₁]
      congr 1; funext k; rw [hcol i₁ (by simp) k]; simp [h01.symm]
    obtain ⟨hA, hB, hC, hD⟩ := profile_conj hU (s • vecMulVec u v)
    have hFM := frobSq_conj hU (s • vecMulVec u v)
    obtain ⟨M, hM⟩ : ∃ M, Uᵀ * (s • vecMulVec u v) * U = M := ⟨_, rfl⟩
    rw [hM] at hA hB hC hD hFM
    rw [hF] at hFM
    -- in these coordinates `L` is `s (x e₀ - y e₁)(x e₀ + y e₁)ᵀ`, the paper's block
    have hM' : M = s • vecMulVec
        (fun i => x * (if i = i₀ then 1 else 0) - y * (if i = i₁ then 1 else 0))
        (fun i => x * (if i = i₀ then 1 else 0) + y * (if i = i₁ then 1 else 0)) := by
      rw [← hM, conj_smul, conj_vecMulVec, hue, hve, mulVec_sub, mulVec_add, mulVec_smul,
        mulVec_smul, hpe, hpf]
      rfl
    -- whose symmetric part is `diag(s(1+t)/2, s(t-1)/2, 0, …)`
    set d : Fin N → ℝ := fun i =>
      if i = i₀ then s * x ^ 2 else if i = i₁ then -(s * y ^ 2) else 0
    have hd : symPart M = diagonal d := by
      ext i j
      rw [hM', symPart]
      by_cases hi0 : i = i₀ <;> by_cases hi1 : i = i₁ <;> by_cases hj0 : j = i₀ <;>
        by_cases hj1 : j = i₁ <;>
        first
        | exact absurd (hi0.symm.trans hi1) h01
        | exact absurd (hj0.symm.trans hj1) h01
        | simp [hi0, hi1, hj0, hj1, h01, h01.symm, d, diagonal, vecMulVec_apply,
            eq_comm (a := i₀) (b := j), eq_comm (a := i₁) (b := j)] <;> ring
    obtain ⟨hB', hC', hD'⟩ := profile_of_diag hN M d hd (P := s * x ^ 2) (Q := s * y ^ 2)
      (by positivity) (by positivity)
      (by
        funext i
        simp only [d]
        by_cases hi0 : i = i₀
        · rw [if_pos hi0, if_pos hi0]; exact max_eq_left (by positivity)
        · by_cases hi1 : i = i₁
          · rw [if_neg hi0, if_pos hi1, if_neg hi0]
            exact max_eq_right (neg_nonpos.mpr (by positivity))
          · rw [if_neg hi0, if_neg hi1, if_neg hi0, max_self])
      (by
        have : (fun i => max (-d i) 0)
            = (fun i => if i = (⟨0, hN⟩ : Fin N) then s * y ^ 2 else 0) ∘ Equiv.swap i₀ i₁ := by
          funext i
          simp only [d, Function.comp_apply]
          by_cases hi0 : i = i₀
          · rw [if_pos hi0, hi0, Equiv.swap_apply_left, if_neg h01.symm]
            exact max_eq_right (neg_nonpos.mpr (by positivity))
          · by_cases hi1 : i = i₁
            · rw [if_neg hi0, if_pos hi1, hi1, Equiv.swap_apply_right, if_pos rfl, neg_neg]
              exact max_eq_left (by positivity)
            · rw [if_neg hi0, if_neg hi1, Equiv.swap_apply_of_ne_of_ne hi0 hi1, if_neg hi0,
                neg_zero, max_self]
        rw [this, ← Multiset.map_map, Multiset.map_univ_val_equiv])
    have hsumd : ∑ i, d i ^ 2 = s ^ 2 * (x ^ 2) ^ 2 + s ^ 2 * (y ^ 2) ^ 2 := by
      rw [Fintype.sum_eq_add i₀ i₁ h01 (fun j hj => by simp [d, hj.1, hj.2])]
      simp [d, h01.symm]; ring
    have hA' := profileA_eq_of_symPart M d hd
    rw [profile, ← hA, ← hB, ← hC, ← hD, hA', hB', hC', hD', hFM, hsumd]
    have hPQ : s * x ^ 2 - s * y ^ 2 = s * t := by rw [hxx, hyy]; ring
    have hQP : s * y ^ 2 - s * x ^ 2 = s * -t := by rw [hxx, hyy]; ring
    have hmax : ∀ r : ℝ, max (s * r) 0 = s * max r 0 := fun r => by
      rw [mul_max_of_nonneg _ _ hs.le, mul_zero]
    rw [hPQ, hQP, hmax, hmax, hxx, hyy]
    simp only [Prod.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_⟩ <;> field_simp <;> ring

/-! ### Rank-one matrices -/

/-- A rank-one matrix is `s · u vᵀ` with `u`, `v` unit vectors and `s > 0`
(the paper's "write `L = u vᵀ`"). -/
lemma exists_smul_vecMulVec_of_rank_eq_one {L : Matrix (Fin N) (Fin N) ℝ}
    (hL : L.rank = 1) :
    ∃ (u v : Fin N → ℝ) (s : ℝ), u ⬝ᵥ u = 1 ∧ v ⬝ᵥ v = 1 ∧ 0 < s ∧
      L = s • vecMulVec u v := by
  -- the range of `L` is spanned by one vector `w`, so every column is a multiple of `w`
  have hr : Module.finrank ℝ (LinearMap.range L.mulVecLin) = 1 := hL
  obtain ⟨w, hw, hspan⟩ := finrank_eq_one_iff'.mp hr
  have hcol : ∀ j, ∃ c : ℝ, c • (w : Fin N → ℝ) = L *ᵥ Pi.single j 1 := by
    intro j
    obtain ⟨c, hc⟩ := hspan ⟨L *ᵥ Pi.single j 1, LinearMap.mem_range_self L.mulVecLin _⟩
    exact ⟨c, congrArg Subtype.val hc⟩
  choose c hc using hcol
  have hLwc : L = vecMulVec (w : Fin N → ℝ) c := by
    ext i j
    have := congrFun (hc j) i
    simp only [Pi.smul_apply, smul_eq_mul, mulVec_single_one] at this
    rw [vecMulVec_apply, mul_comm, this]
    rfl
  have hw0 : (w : Fin N → ℝ) ≠ 0 := fun h => hw (Subtype.ext h)
  have hc0 : c ≠ 0 := by
    rintro rfl
    have : L = 0 := by rw [hLwc]; ext i j; simp [vecMulVec_apply]
    rw [this, Matrix.rank_zero] at hL
    exact zero_ne_one hL
  have hpos : ∀ z : Fin N → ℝ, z ≠ 0 → 0 < z ⬝ᵥ z := by
    intro z hz
    obtain ⟨i, hi⟩ := Function.ne_iff.mp hz
    exact lt_of_lt_of_le (mul_self_pos.mpr hi)
      (Finset.single_le_sum (fun j _ => mul_self_nonneg (z j)) (Finset.mem_univ i))
  set a := Real.sqrt ((w : Fin N → ℝ) ⬝ᵥ w)
  set b := Real.sqrt (c ⬝ᵥ c)
  have ha : 0 < a := Real.sqrt_pos.mpr (hpos _ hw0)
  have hb : 0 < b := Real.sqrt_pos.mpr (hpos _ hc0)
  have haa : a ^ 2 = (w : Fin N → ℝ) ⬝ᵥ w := Real.sq_sqrt (hpos _ hw0).le
  have hbb : b ^ 2 = c ⬝ᵥ c := Real.sq_sqrt (hpos _ hc0).le
  refine ⟨(1 / a) • (w : Fin N → ℝ), (1 / b) • c, a * b, ?_, ?_, by positivity, ?_⟩
  · rw [smul_dotProduct, dotProduct_smul, ← haa, smul_eq_mul, smul_eq_mul]; field_simp
  · rw [smul_dotProduct, dotProduct_smul, ← hbb, smul_eq_mul, smul_eq_mul]; field_simp
  · refine hLwc.trans ?_
    ext i j
    simp only [smul_apply, vecMulVec_apply, Pi.smul_apply, smul_eq_mul]
    field_simp

/-- **Proposition (rank-one profiles), formula.**  If `L` has rank one and unit
norm, then `π(L) = ((1 - t²)/2, (1 - t²)/2, t₊², t₋²)` with `t = tr(L)`. -/
theorem profile_rank_one (L : Matrix (Fin N) (Fin N) ℝ) (hL : L.rank = 1)
    (hnorm : frobSq L = 1) :
    profile L = ((1 - L.trace ^ 2) / 2, (1 - L.trace ^ 2) / 2,
      max L.trace 0 ^ 2, max (-L.trace) 0 ^ 2) := by
  obtain ⟨u, v, s, hu, hv, hs, rfl⟩ := exists_smul_vecMulVec_of_rank_eq_one hL
  -- unit norm forces `s = 1`, and then `t = tr(u vᵀ) = ⟨u, v⟩`
  have hs1 : s = 1 := by
    rw [frobSq_smul_vecMulVec, hu, hv, mul_one, mul_one] at hnorm
    nlinarith [hnorm]
  subst hs1
  rw [one_smul, trace_vecMulVec, ← one_smul ℝ (vecMulVec u v)]
  exact profile_smul_vecMulVec u v hu hv one_pos

/-- `|⟨u, v⟩| ≤ 1` for unit vectors. -/
lemma abs_dotProduct_le_one {u v : Fin N → ℝ} (hu : u ⬝ᵥ u = 1) (hv : v ⬝ᵥ v = 1) :
    (u ⬝ᵥ v) ^ 2 ≤ 1 := by
  have h1 : 0 ≤ (v - u) ⬝ᵥ (v - u) := Finset.sum_nonneg fun i _ => mul_self_nonneg _
  have h2 : 0 ≤ (v + u) ⬝ᵥ (v + u) := Finset.sum_nonneg fun i _ => mul_self_nonneg _
  simp only [sub_dotProduct, dotProduct_sub, add_dotProduct, dotProduct_add, hu, hv,
    dotProduct_comm v u] at h1 h2
  nlinarith

/-- **Proposition (rank-one profiles), the locus.**  For `N ≥ 2`, the set of
profiles of rank-one matrices is `Θ = {a = b, cd = 0} ⊂ Δ₃`. -/
theorem profiles_rank_one_eq_Theta (hN : 2 ≤ N) :
    {p | ∃ L : Matrix (Fin N) (Fin N) ℝ, L.rank = 1 ∧ profile L = p} = Theta := by
  ext p
  constructor
  · rintro ⟨L, hL, rfl⟩
    obtain ⟨u, v, s, hu, hv, hs, rfl⟩ := exists_smul_vecMulVec_of_rank_eq_one hL
    rw [profile_smul_vecMulVec u v hu hv hs]
    have ht := abs_dotProduct_le_one hu hv
    set t := u ⬝ᵥ v
    refine ⟨by linarith, by linarith, sq_nonneg _, sq_nonneg _, ?_, rfl, ?_⟩
    · rcases le_total 0 t with h | h
      · rw [max_eq_left h, max_eq_right (by linarith)]; ring
      · rw [max_eq_right h, max_eq_left (by linarith)]; ring
    · rcases le_total 0 t with h | h
      · rw [max_eq_right (by linarith : -t ≤ 0)]; ring
      · rw [max_eq_right h]; ring
  · rintro ⟨ha, hb, hc, hd, hsum, hab, hcd⟩
    obtain ⟨a, b, c, d⟩ := p
    simp only at ha hb hc hd hsum hab hcd ⊢
    -- `t = √c - √d`, and `u = e₀`, `v = t e₀ + √(1 - t²) e₁`
    set t := Real.sqrt c - Real.sqrt d
    have hsq : t ^ 2 = c + d := by
      rcases mul_eq_zero.mp hcd with h | h
      · simp [t, h, Real.sq_sqrt hd]
      · simp [t, h, Real.sq_sqrt hc]
    have ht1 : t ^ 2 ≤ 1 := by rw [hsq]; linarith
    set i₀ : Fin N := ⟨0, by omega⟩
    set i₁ : Fin N := ⟨1, by omega⟩
    have h01 : i₀ ≠ i₁ := by simp [i₀, i₁, Fin.ext_iff]
    set r := Real.sqrt (1 - t ^ 2)
    have hrr : r ^ 2 = 1 - t ^ 2 := Real.sq_sqrt (by linarith)
    let u : Fin N → ℝ := fun i => if i = i₀ then 1 else 0
    let v : Fin N → ℝ := fun i => if i = i₀ then t else if i = i₁ then r else 0
    have hdot : ∀ w : Fin N → ℝ, u ⬝ᵥ w = w i₀ := fun w => by
      rw [dotProduct, Fintype.sum_eq_single i₀ (fun j hj => by simp [u, hj])]; simp [u]
    have hu : u ⬝ᵥ u = 1 := by rw [hdot]; simp [u]
    have huv : u ⬝ᵥ v = t := by rw [hdot]; simp [v]
    have hv : v ⬝ᵥ v = 1 := by
      rw [dotProduct, Fintype.sum_eq_add i₀ i₁ h01 (fun j hj => by simp [v, hj.1, hj.2])]
      simp [v, h01.symm]; nlinarith [hrr]
    refine ⟨vecMulVec u v, ?_, ?_⟩
    · -- `u vᵀ` has rank one: rank at most one, and `(u vᵀ) v = u ≠ 0`
      refine le_antisymm (Matrix.rank_vecMulVec_le u v) ?_
      rw [Nat.one_le_iff_ne_zero]
      intro h0
      have hbot : LinearMap.range (vecMulVec u v).mulVecLin = ⊥ :=
        Submodule.finrank_eq_zero.mp h0
      have hmem : (vecMulVec u v) *ᵥ v ∈ LinearMap.range (vecMulVec u v).mulVecLin :=
        LinearMap.mem_range_self _ v
      rw [hbot, Submodule.mem_bot] at hmem
      have := congrFun hmem i₀
      simp only [mulVec, dotProduct, vecMulVec_apply, Pi.zero_apply] at this
      have hvv : ∑ j, v j * v j = 1 := hv
      have hsum' : ∑ j, u i₀ * v j * v j = u i₀ * ∑ j, v j * v j := by
        rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by ring
      rw [hsum', hvv] at this
      simp [u] at this
    · rw [← one_smul ℝ (vecMulVec u v), profile_smul_vecMulVec u v hu hv one_pos, huv, hsq]
      have hsc := Real.sqrt_nonneg c
      have hsd := Real.sqrt_nonneg d
      simp only [Prod.mk.injEq]
      rcases mul_eq_zero.mp hcd with h | h
      · subst h
        simp only [t, Real.sqrt_zero, zero_sub, neg_neg]
        rw [max_eq_right (by linarith : -Real.sqrt d ≤ 0), max_eq_left hsd, Real.sq_sqrt hd]
        refine ⟨by linarith, by linarith, by norm_num, rfl⟩
      · subst h
        simp only [t, Real.sqrt_zero, sub_zero]
        rw [max_eq_left hsc, max_eq_right (by linarith : -Real.sqrt c ≤ 0), Real.sq_sqrt hc]
        refine ⟨by linarith, by linarith, rfl, by norm_num⟩

end Appendices.MainText
