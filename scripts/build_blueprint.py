#!/usr/bin/env python3
"""Build the Lean blueprint and publish its static files into blueprint/.

Install blueprint/requirements.txt and Graphviz first. The renderer is the
standard leanblueprint plasTeX plugin, invoked directly because this repository
keeps its Lake project in formalization/ rather than at the repository root.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT = ROOT / "blueprint"
SOURCE = BLUEPRINT / "src"
LEAN = ROOT / "formalization"
DOC_BASE = "https://aodesky.github.io/attention-bilinear-forms/blueprint/lean"
REPOSITORY = "https://github.com/aodesky/attention-bilinear-forms"


def declaration_locations():
    """Index public declaration headers and their enclosing namespaces."""
    locations = {}
    for path in sorted((LEAN / "Appendices").rglob("*.lean")):
        scopes = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            scope = re.match(r"^(namespace|section)\s*(\S*)", line)
            if scope:
                scopes.append((scope[1], scope[2]))
                continue
            if re.match(r"^end(?:\s|$)", line):
                if scopes:
                    scopes.pop()
                continue
            declaration = re.match(
                r"^(?:(?:noncomputable|protected)\s+)*(?:theorem|lemma|def|structure|abbrev)\s+([^\s(:]+)", line)
            if declaration:
                namespace = [name for kind, name in scopes if kind == "namespace"]
                name = ".".join([*namespace, declaration[1]])
                if name in locations:
                    raise ValueError(f"Duplicate declaration: {name}")
                locations[name] = (path.relative_to(ROOT).as_posix(), number)
    return locations


def blueprint_declarations():
    text = (SOURCE / "content.tex").read_text()
    declarations = sorted({name.strip() for group in re.findall(r"\\lean\{([^}]+)\}", text)
                           for name in group.split(",")})
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate blueprint labels")
    for group in re.findall(r"\\uses\{([^}]+)\}", text):
        for label in group.split(","):
            if label.strip() not in labels:
                raise ValueError(f"Unknown dependency: {label}")
    return declarations, labels


def check_lean(project, declarations):
    # A cached local checkout may be used only when its public Lean sources and
    # pinned dependencies match the files being published.
    for path in [LEAN / "Appendices.lean", LEAN / "lean-toolchain", LEAN / "lake-manifest.json",
                 LEAN / "lakefile.toml", *(LEAN / "Appendices").rglob("*.lean")]:
        other = project / path.relative_to(LEAN)
        if not other.is_file() or other.read_bytes() != path.read_bytes():
            raise ValueError(f"Lean check project differs from public source: {other}")
    subprocess.run(["lake", "build"], cwd=project, check=True)
    checks = "import Appendices\n" + "\n".join(f"#check {name}" for name in declarations) + "\n"
    with tempfile.TemporaryDirectory(prefix="blueprint-check-") as directory:
        path = Path(directory) / "BlueprintCheck.lean"
        path.write_text(checks)
        subprocess.run(["lake", "env", "lean", str(path)], cwd=project, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    subprocess.run(["lake", "env", "lean", str(BLUEPRINT / "CheckAxioms.lean")],
                   cwd=project, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-project", type=Path, default=LEAN,
                        help="Matching built Lean project (default: formalization/)")
    parser.add_argument("--plastex", default="plastex")
    args = parser.parse_args()
    executable = shutil.which(args.plastex)
    if executable is None:
        raise FileNotFoundError(f"plasTeX executable not found: {args.plastex}")
    executable = str(Path(executable).resolve())
    declarations, labels = blueprint_declarations()
    locations = declaration_locations()
    for name in declarations:
        if name not in locations:
            raise ValueError(f"Declaration not found in public Lean sources: {name}")
    check_lean(args.lean_project.resolve(), declarations)
    revision = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "formalization/Appendices", "formalization/Appendices.lean"],
        cwd=ROOT, text=True).strip()
    # Refuse to publish links to a committed source different from the local one.
    subprocess.run(["git", "diff", "--exit-code", revision, "--", "formalization/Appendices",
                    "formalization/Appendices.lean"], cwd=ROOT, check=True, stdout=subprocess.PIPE)
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment["PATH"]
    subprocess.run([executable, "-c", "plastex.cfg", "web.tex"], cwd=SOURCE,
                   env=environment, check=True)
    output = BLUEPRINT / "web"
    if not (output / "index.html").is_file() or not (output / "dep_graph_document.html").is_file():
        raise RuntimeError("Renderer did not produce the blueprint and dependency graph")
    # Convert doc-gen lookup URLs into exact, revision-pinned source links.
    links = {f"{DOC_BASE}/find/#doc/{name}":
             f"{REPOSITORY}/blob/{revision}/{locations[name][0]}#L{locations[name][1]}"
             for name in declarations}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".json", ".js", ".svg"}:
            text = path.read_text()
            for old in sorted(links, key=len, reverse=True):
                text = text.replace(old, links[old])
            path.write_text(text)
    for path in output.iterdir():
        destination = BLUEPRINT / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(path, destination)
    manifest = {
        "lean_source_revision": revision,
        "lean_toolchain": (LEAN / "lean-toolchain").read_text().strip(),
        "nodes": len(labels), "declarations": len(declarations),
        "checks": ["lake build", "all blueprint declarations exist", "standard axioms only"],
        "source_sha256": hashlib.sha256((SOURCE / "content.tex").read_bytes()).hexdigest(),
        "source_links": {name: links[f"{DOC_BASE}/find/#doc/{name}"] for name in declarations},
    }
    (BLUEPRINT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {len(labels)} blueprint nodes with {len(declarations)} checked Lean declarations.")


if __name__ == "__main__":
    main()
