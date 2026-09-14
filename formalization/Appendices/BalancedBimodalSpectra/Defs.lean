/-
Appendix "Balanced bimodal spectra": the pairing geometry definitions.

Fix `m ≥ 1` and set `n = 2m`.  `Λ_m` denotes the set of unit vectors
`λ = (λ_1 ≤ ⋯ ≤ λ_n) ∈ ℝ^n` with `λ_m < 0 < λ_{m+1}`.  For `λ ∈ Λ_m`
define `λ₊, λ₋ ∈ ℝ^m` by

    (λ₊)_j = λ_{n+1-j},    (λ₋)_j = -λ_j,    j = 1, …, m,

so that `λ₊` and `λ₋` list the positive entries and the absolute values
of the negative entries of `λ`, both in decreasing order.  Define

    ℐ(λ) = ‖λ₊‖² - ‖λ₋‖²,   C(λ) = ⟨λ₊, λ₋⟩,   𝒫(λ) = 1 - ‖λ₊ - λ₋‖,

and the linear map `ι : ℝ^n → ℝ^n` by `(ιλ)_i = -λ_{n+1-i}`.

Vectors in `ℝ^n` are `Fin n → ℝ`, 0-indexed: the paper's `λ_j` is
`x ⟨j-1⟩`, and the paper's index `n+1-j` is `Fin.rev ⟨j-1⟩`.
-/
import Mathlib

namespace Appendices

open BigOperators

/-- The set `Λ_m` of unit vectors `λ ∈ ℝ^{2m}` with nondecreasing
entries and `λ_m < 0 < λ_{m+1}` (1-indexed as in the paper).
The membership predicate is stated for all `m`; for `m = 0` the middle
entries do not exist and the set is empty (no unit vectors in `ℝ^0`). -/
structure MemLambdaSet (m : ℕ) (x : Fin (2 * m) → ℝ) : Prop where
  mono : Monotone x
  unit : ∑ i, x i ^ 2 = 1
  middle_neg : ∀ h : 0 < m, x ⟨m - 1, by omega⟩ < 0
  middle_pos : ∀ h : 0 < m, 0 < x ⟨m, by omega⟩

/-- `Λ_m` is empty for `m = 0`: membership forces `0 < m`. -/
lemma MemLambdaSet.m_pos {m : ℕ} {x : Fin (2 * m) → ℝ} (hx : MemLambdaSet m x) :
    0 < m := by
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · exact absurd hx.unit (by simp)
  · exact hm

/-- `(λ₊)_j = λ_{n+1-j}`: the positive entries of `λ`, in decreasing
order (for `λ ∈ Λ_m`). -/
def lamPlus {m : ℕ} (x : Fin (2 * m) → ℝ) : Fin m → ℝ :=
  fun j => x (Fin.rev ⟨j, by omega⟩)

/-- `(λ₋)_j = -λ_j`: the absolute values of the negative entries of
`λ`, in decreasing order (for `λ ∈ Λ_m`). -/
def lamMinus {m : ℕ} (x : Fin (2 * m) → ℝ) : Fin m → ℝ :=
  fun j => -x ⟨j, by omega⟩

/-- The signature imbalance `ℐ(λ) = ‖λ₊‖² - ‖λ₋‖²`. -/
def imbalance {m : ℕ} (x : Fin (2 * m) → ℝ) : ℝ :=
  ∑ j, lamPlus x j ^ 2 - ∑ j, lamMinus x j ^ 2

/-- `C(λ) = ⟨λ₊, λ₋⟩`. -/
def statC {m : ℕ} (x : Fin (2 * m) → ℝ) : ℝ :=
  ∑ j, lamPlus x j * lamMinus x j

/-- The pairing score `𝒫(λ) = 1 - ‖λ₊ - λ₋‖`. -/
noncomputable def pairingScore {m : ℕ} (x : Fin (2 * m) → ℝ) : ℝ :=
  1 - Real.sqrt (∑ j, (lamPlus x j - lamMinus x j) ^ 2)

/-- The linear map `ι : ℝ^n → ℝ^n`, `(ιλ)_i = -λ_{n+1-i}`
(defined here for every `n`). -/
def iotaMap {n : ℕ} (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => -x (Fin.rev i)

end Appendices
