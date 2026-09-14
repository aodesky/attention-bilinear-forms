#!/usr/bin/env python3
"""Build a model-selectable HTML gallery for bilinear-form analyses.

The dashboard discovers each model's ``latest.json`` beneath
``analyses/symmetric_spectra``. Static plots are shown at full resolution and
the Plotly simplex pages remain draggable inside the dashboard. Re-run after
analyzing another model; no model list is hard-coded.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PLOT_ORDER = [
    "cluster_extreme_histograms_k3.png",
    "layer_cluster_heatmap_k3.png",
    "simplex_weights_k3.png",
    "symmetric_simplex_weights_k3.png",
]

HIDDEN_PLOTS = {
    "projection_k3.png",
    "cluster_mean_spectra_k3.png",
}

TITLES = {
    "projection": "Spectral projection",
    "cluster_mean_spectra": "Whole-cluster average spectra",
    "cluster_extreme_histograms": "Extreme-member mean histograms",
    "layer_cluster_heatmap": "Clusters by layer",
    "simplex_weights": "Full weight simplex",
    "symmetric_simplex_weights": "Projection to a = 0",
}


def title_for(filename: str) -> str:
    stem = Path(filename).stem
    for prefix, title in TITLES.items():
        if stem.startswith(prefix):
            suffix = stem.removeprefix(prefix).strip("_").replace("k", "k = ")
            return f"{title} ({suffix})" if suffix else title
    return stem.replace("_", " ")


def relative_url(target: Path, dashboard: Path) -> str:
    return os.path.relpath(target, dashboard.parent).replace(os.sep, "/")


def discover(dataset: Path, dashboard: Path) -> list[dict[str, Any]]:
    root = dataset / "analyses" / "symmetric_spectra"
    models: list[dict[str, Any]] = []
    for latest_path in sorted(root.glob("*/*/latest.json")):
        latest = json.loads(latest_path.read_text())
        analysis = dataset / latest["relative_path"]
        completion_path = analysis / "completion.json"
        if not completion_path.is_file():
            continue
        summary = json.loads(completion_path.read_text())
        repo_id = summary["model"]["repo_id"]
        positional_method = summary.get("positional_method", "unknown")
        primary_k = int(summary["primary_k"])

        pngs = {
            path.name: path for path in analysis.glob("*.png")
            if path.name not in HIDDEN_PLOTS
        }
        ordered_names = [name for name in PLOT_ORDER if name in pngs]
        ordered_names.extend(sorted(set(pngs) - set(ordered_names)))
        plots = [
            {
                "title": title_for(name),
                "url": relative_url(pngs[name], dashboard),
                "pdf_url": relative_url(pngs[name].with_suffix(".pdf"), dashboard)
                if pngs[name].with_suffix(".pdf").is_file()
                else None,
            }
            for name in ordered_names
        ]
        interactive = []
        for k in (primary_k,):
            page = analysis / f"simplex_weights_k{k}_interactive.html"
            if page.is_file():
                interactive.append(
                    {
                        "title": (
                            "Draggable full simplex with RoPE displacement"
                            if "rope_weight_trajectory" in summary.get("files", {})
                            else f"Draggable full simplex (k = {k})"
                        ),
                        "url": relative_url(page, dashboard),
                    }
                )
        models.append(
            {
                "id": repo_id,
                "revision": summary["model"]["revision"][:12],
                "positional_method": positional_method,
                "heads": summary["head_count"],
                "spectrum_dimension": summary["spectrum_dimension"],
                "primary_k": primary_k,
                "classification": summary.get("classification", "three classes"),
                "component_share": 100
                * sum(summary["singular_value_energy_shares_first_three"]),
                "plots": plots,
                "interactive": interactive,
                "analysis_url": relative_url(analysis, dashboard) + "/",
            }
        )
    return sorted(models, key=lambda model: model["id"])


def render(models: list[dict[str, Any]]) -> str:
    data = json.dumps(models, separators=(",", ":")).replace("</", "<\\/")
    options = "".join(
        f'<option value="{html.escape(model["id"], quote=True)}">'
        f'{html.escape(model["id"])}</option>'
        for model in models
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bilinear-form analysis dashboard</title>
<style>
  :root {{ color-scheme: light; --ink:#171717; --muted:#666; --line:#d8d8d8; --paper:#fafafa; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; color:var(--ink); background:var(--paper); }}
  header {{ position:sticky; top:0; z-index:10; padding:16px 24px; background:rgba(250,250,250,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
  .bar {{ max-width:1500px; margin:auto; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
  h1 {{ margin:0; font-size:20px; font-weight:650; }}
  label {{ color:var(--muted); font-size:14px; }}
  select {{ margin-left:7px; padding:7px 30px 7px 10px; font-size:15px; border:1px solid #aaa; border-radius:6px; background:white; }}
  main {{ max-width:1500px; margin:0 auto; padding:22px 24px 50px; }}
  #summary {{ display:flex; gap:24px; flex-wrap:wrap; margin:0 0 20px; padding:13px 16px; background:white; border:1px solid var(--line); border-radius:8px; font-size:14px; }}
  #summary span {{ color:var(--muted); }} #summary strong {{ color:var(--ink); font-weight:600; }}
  .section-title {{ margin:28px 0 12px; font-size:17px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:18px; align-items:start; }}
  .card {{ background:white; border:1px solid var(--line); border-radius:9px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .card h2 {{ margin:0; padding:11px 14px; font-size:15px; font-weight:600; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; }}
  .card h2 a {{ color:#555; font-size:12px; font-weight:500; text-decoration:none; }}
  .card img {{ display:block; width:100%; height:auto; background:white; cursor:zoom-in; }}
  iframe {{ display:block; width:100%; height:680px; border:0; background:white; }}
  .empty {{ color:var(--muted); padding:30px; text-align:center; }}
  @media(max-width:650px) {{ .grid {{ grid-template-columns:1fr; }} main,header {{ padding-left:10px; padding-right:10px; }} iframe {{ height:520px; }} }}
</style>
</head>
<body>
<header><div class="bar"><h1>Bilinear-form analyses</h1>
<label>Model<select id="model">{options}</select></label></div></header>
<main><div id="summary"></div><div id="content"></div></main>
<script>
const MODELS={data};
const select=document.getElementById('model');
const content=document.getElementById('content');
const summary=document.getElementById('summary');
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function card(plot){{
  const link=plot.pdf_url?`<a href="${{plot.pdf_url}}" target="_blank">PDF</a>`:'';
  return `<article class="card"><h2>${{esc(plot.title)}}${{link}}</h2><a href="${{plot.url}}" target="_blank"><img loading="lazy" src="${{plot.url}}" alt="${{esc(plot.title)}}"></a></article>`;
}}
function interactiveCard(plot){{return `<article class="card"><h2>${{esc(plot.title)}}<a href="${{plot.url}}" target="_blank">open separately</a></h2><iframe loading="lazy" src="${{plot.url}}" title="${{esc(plot.title)}}"></iframe></article>`;}}
function show(updateHash=true){{
  const m=MODELS.find(x=>x.id===select.value)||MODELS[0];
  if(!m){{summary.innerHTML='';content.innerHTML='<p class="empty">No completed analyses found.</p>';return;}}
  summary.innerHTML=`<div><span>model</span> <strong>${{esc(m.id)}}</strong></div><div><span>revision</span> <strong>${{m.revision}}</strong></div><div><span>position method</span> <strong>${{esc(m.positional_method)}}</strong></div><div><span>heads</span> <strong>${{m.heads.toLocaleString()}}</strong></div><div><span>spectrum length</span> <strong>${{m.spectrum_dimension}}</strong></div><div><span>classification</span> <strong>three profile types</strong></div><div><span>first 3 components</span> <strong>${{m.component_share.toFixed(1)}}%</strong></div>`;
  let body='';
  if(m.interactive.length) body+=`<h2 class="section-title">Interactive 3-D views</h2><div class="grid">${{m.interactive.map(interactiveCard).join('')}}</div>`;
  body+=`<h2 class="section-title">All generated figures</h2><div class="grid">${{m.plots.map(card).join('')}}</div>`;
  content.innerHTML=body;
  if(updateHash) history.replaceState(null,'','#'+encodeURIComponent(m.id));
}}
const requested=decodeURIComponent(location.hash.slice(1));
if(MODELS.some(model=>model.id===requested)) select.value=requested;
select.addEventListener('change',()=>show(true)); show(false);
</script>
</body></html>"""


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", help="Default: DATASET/dashboard.html")
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    output = Path(args.out).resolve() if args.out else dataset / "dashboard.html"
    models = discover(dataset, output)
    if not models:
        raise SystemExit("No completed model analyses found")
    atomic_write(output, render(models))
    print(f"Wrote {output} with {len(models)} models")
    for model in models:
        print(f"  {model['id']}: {len(model['plots'])} figures, "
              f"{len(model['interactive'])} interactive views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
