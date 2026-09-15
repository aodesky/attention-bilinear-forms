# Formalization blueprint

The [published blueprint](https://aodesky.github.io/attention-bilinear-forms/blueprint/)
covers the audited public formalization: Definition 6, Lemmas 5–10,
Corollaries 1–2, Proposition 6 and Theorem 4, plus their shared definitions.
The dependency graph records mathematical dependencies between these nodes.

Edit `src/content.tex` to change statements, proof outlines, declaration
links (`\lean`) or dependencies (`\uses`). The `\leanok` markers indicate
completed formal statements and proofs. The theorem numbering follows the
paper; the auxiliary definitions have their own blueprint entries.

## Build

Install Graphviz (including its development headers), a TeX installation,
and the Python dependencies in an isolated environment:

```sh
python3 -m venv .venv-blueprint
.venv-blueprint/bin/pip install -r blueprint/requirements.txt
python3 scripts/build_blueprint.py --plastex .venv-blueprint/bin/plastex
```

The Lean toolchain is pinned in `formalization/lean-toolchain`. On a new
checkout, first run `lake exe cache get` from `formalization/` to download
the mathlib cache. The build checks the Lean project, every linked
declaration, and the allowed axioms before rendering. An existing cached
project can be supplied using `--lean-project /path/to/formalization`;
its public source files and dependency pins must match exactly.
`CheckAxioms.lean` checks the same complete set of project declarations as
the formalization's axiom script, sharing the dependency traversal cache
between declarations to avoid repeatedly checking the same mathlib proofs.

This repository keeps its Lake project in a subdirectory, so the build
invokes the standard `leanblueprint` plasTeX plugin directly. It renders
into the ignored `web/` directory, then copies the static output here for
the existing GitHub Pages branch deployment. The Lean buttons link to
declaration lines in the audited source revision, without requiring a
separate doc-gen build. `build-manifest.json` records those links and the
checks performed. Commit the source and generated files together.
