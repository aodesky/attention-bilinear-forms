import Appendices
import MainText

-- Check the same complete set of project declarations as the formalization's
-- check_axioms.sh, sharing the traversal cache across their dependencies.
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let collectAll : CollectAxioms.M Unit := do
    for (name, _) in env.constants.toList do
      if let some mod := env.getModuleFor? name then
        if (`Appendices).isPrefixOf mod || (`MainText).isPrefixOf mod then
          CollectAxioms.collect name
  let (_, state) := (collectAll.run env).run {}
  for axiomName in state.axioms do
    unless allowed.contains axiomName do
      throwError m!"Project uses non-standard axiom {axiomName}"
  logInfo "AXIOM CHECK PASSED: only propext, Classical.choice, Quot.sound"
