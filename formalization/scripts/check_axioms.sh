#!/bin/bash
# Verify: no sorries anywhere, and every declaration in the project
# uses only the three standard axioms.
set -e
cd "$(dirname "$0")/.."
if grep -rn "sorry" Appendices Appendices.lean MainText MainText.lean --include='*.lean'; then
  echo "FAIL: sorry found"; exit 1
fi
if grep -rn "^axiom\|[^_[:alnum:]]axiom " Appendices Appendices.lean MainText MainText.lean --include='*.lean'; then
  echo "FAIL: axiom declaration found"; exit 1
fi
cat > /tmp/AxiomCheck.lean <<'LEAN'
import Appendices
import MainText
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let mut bad := #[]
  for (n, _) in env.constants.toList do
    if let some mod := env.getModuleFor? n then
      if (`Appendices).isPrefixOf mod || (`MainText).isPrefixOf mod then
        let (_, s) := ((CollectAxioms.collect n).run env).run {}
        for ax in s.axioms do
          unless allowed.contains ax do
            bad := bad.push (n, ax)
  if bad.isEmpty then
    logInfo "AXIOM CHECK PASSED: only propext, Classical.choice, Quot.sound"
  else
    for (n, ax) in bad do
      logError m!"{n} uses non-standard axiom {ax}"
LEAN
lake env lean /tmp/AxiomCheck.lean
