/-
Appendix "Moments": the statistics `O`, `E`, `R` of a spectrum.

For a vector `λ` (all of whose entries satisfy `|λ_i| < 1` in the
intended use), the paper defines

    O(λ) = Σ_i λ_i/(1-λ_i²),   E(λ) = Σ_i λ_i²/(1-λ_i²),
    R(λ) = Σ_i 1/(1-λ_i).

`O` is the sum of all odd moments of the spectrum and `E` the sum of
all even moments (Lemma OE-equivalences).
-/
import Mathlib

namespace Appendices

open BigOperators

/-- `O(λ) = Σ_i λ_i/(1-λ_i²)`. -/
noncomputable def statO {n : ℕ} (x : Fin n → ℝ) : ℝ :=
  ∑ i, x i / (1 - x i ^ 2)

/-- `E(λ) = Σ_i λ_i²/(1-λ_i²)`. -/
noncomputable def statE {n : ℕ} (x : Fin n → ℝ) : ℝ :=
  ∑ i, x i ^ 2 / (1 - x i ^ 2)

/-- `R(λ) = Σ_i 1/(1-λ_i)`. -/
noncomputable def statR {n : ℕ} (x : Fin n → ℝ) : ℝ :=
  ∑ i, 1 / (1 - x i)

end Appendices
