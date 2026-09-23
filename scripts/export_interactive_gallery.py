#!/usr/bin/env python3
"""Export the latest local analyses to the existing GitHub Pages gallery.

Usage: python3 scripts/export_interactive_gallery.py --dataset /path/to/bilinear-forms
Only completed analyses and the four public figures are copied.
"""

import argparse
import ast
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

from site_navigation import project_navigation


ROOT = Path(__file__).resolve().parent.parent
FIGURES = (
    ("simplex_weights_k3_interactive.html", "simplex.html"),
    ("symmetric_simplex_weights_k3.png", "delta.png"),
    ("cluster_extreme_histograms_k3.png", "extreme-histograms.png"),
    ("layer_cluster_heatmap_k3.png", "clusters-by-layer.png"),
)


def model_set():
    # Read the canonical model list without importing numerical dependencies.
    tree = ast.parse((ROOT / "scripts/model_set.py").read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name) and node.target.id == "MODELS")


def page(tag, models):
    versions = {name: hashlib.sha256((ROOT / "interactive" / name).read_bytes()).hexdigest()[:12]
                for name in ("gallery.css", "gallery.js")}
    options = "".join(
        f'<option value="{html.escape(key, quote=True)}"'
        f'{" selected" if key == tag else ""}>{html.escape(key)}</option>'
        for key, _ in models
    )
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(tag)} · Profiles of attention heads</title>
  <meta name="description" content="Explore attention-head profiles: interactive 3-D simplex, projection to Δ, spectral distributions by parity, and clusters by layer.">
  <link rel="stylesheet" href="gallery.css?v={versions['gallery.css']}">
  <link rel="stylesheet" href="../assets/project-nav.css?v=7">
  <script src="../assets/theme.js"></script>
  <script src="gallery.js?v={versions['gallery.js']}" defer></script>
</head>
<body>
  {project_navigation(current="plots")}
  <header class="model-controls">
    <label for="model">Model</label>
    <select id="model">{options}</select>
  </header>
  <main>
    <div class="caption">
      <h1 id="figure-title">Interactive 3-D simplex</h1>
      <a id="open-figure" href="assets/{tag}/simplex.html" target="_blank" rel="noopener">Open figure ↗</a>
    </div>
    <p id="figure-help">Drag to rotate, scroll to zoom, and hover for layer and head details.</p>
    <div id="viewer" role="tabpanel" aria-labelledby="figure-tab-0">
      <iframe id="interactive-figure" title="Interactive 3-D simplex for {html.escape(tag)}"></iframe>
      <img id="static-figure" alt="" hidden>
      <p id="figure-error" role="alert" hidden>This figure could not be loaded. Try opening it separately.</p>
    </div>
  </main>
  <footer>
    <nav class="figure-labels" role="tablist" aria-label="Figures">
      <button id="figure-tab-0" type="button" role="tab" aria-controls="viewer" data-figure="0" aria-selected="true">3-D simplex</button>
      <button id="figure-tab-1" type="button" role="tab" aria-controls="viewer" data-figure="1" aria-selected="false" tabindex="-1">Projection to Δ</button>
      <button id="figure-tab-2" type="button" role="tab" aria-controls="viewer" data-figure="2" aria-selected="false" tabindex="-1">Spectral distributions by parity</button>
      <button id="figure-tab-3" type="button" role="tab" aria-controls="viewer" data-figure="3" aria-selected="false" tabindex="-1">Clusters by layer</button>
    </nav>
  </footer>
  <noscript>Enable JavaScript to use the model dropdown and figure tabs. <a href="assets/{tag}/simplex.html">Open the 3-D figure</a>.</noscript>
</body>
</html>
'''


def export(dataset, output):
    models = model_set()
    analyses = {}
    for latest in sorted((dataset / "analyses/symmetric_spectra").glob("*/*/latest.json")):
        analysis = dataset / json.loads(latest.read_text())["relative_path"]
        completion = analysis / "completion.json"
        if completion.is_file():
            repo = json.loads(completion.read_text())["model"]["repo_id"]
            if repo in analyses:
                raise ValueError(f"Multiple current revisions for {repo}")
            analyses[repo] = analysis
    # Validate the entire export before changing any published assets.
    for _, repo in models:
        if repo not in analyses:
            raise FileNotFoundError(f"No completed analysis for {repo}")
        for source, _ in FIGURES:
            if not (analyses[repo] / source).is_file():
                raise FileNotFoundError(analyses[repo] / source)

    for tag, repo in models:
        destination = output / "assets" / tag
        destination.mkdir(parents=True, exist_ok=True)
        for source, target in FIGURES:
            source_path = analyses[repo] / source
            if target != "simplex.html":
                shutil.copyfile(source_path, destination / target)
                continue
            document = source_path.read_text()
            # Share the exact bundled Plotly version locally across all models.
            match = re.search(r'<script>(/\*\*\s*\n\* plotly\.js.*?)(</script>)', document, re.S)
            if match is None:
                raise ValueError(f"Bundled Plotly script not found: {source_path}")
            library = match.group(1)
            library_name = f"plotly-{hashlib.sha256(library.encode()).hexdigest()[:12]}.min.js"
            (output / "assets" / library_name).write_text(library)
            document = document[:match.start()] + f'<script src="../{library_name}"></script>' + document[match.end():]
            # The outer gallery supplies figure navigation. The embedded plot
            # retains only the model's optional RoPE displacement control.
            embed_style = '''<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html,body { height:100%; margin:0; }
body { display:flex; flex-direction:column; background:transparent; }
/* Embedded in the gallery the page shows through; opened on its own there is
   nothing behind it, so paint the ground to match the chosen mode. */
:root[data-standalone][data-theme="dark"] body { background:#14171a; }
:root[data-standalone][data-theme="light"] body { background:#ffffff; }
body > div:not(.plot-control) { flex:1; min-height:0; }
.plot-control { flex-wrap:wrap; padding:8px 12px; gap:8px; color:#111; }
@media(max-width:600px) { .plot-control { font-size:13px; } }
:root[data-theme="dark"] .plot-control { color:#e8e6e3; }
</style>
<script>
/* The figure is generated with a light Plotly template. The gallery tells this
   frame which mode it is being viewed in, and we restyle the plot to match:
   transparent backgrounds so the page shows through, and legible tick, title
   and legend text. Runs again whenever the parent toggles. */
(function () {
  var DARK = '#e8e6e3', LIGHT = '#111111';
  function paint(mode) {
    document.documentElement.setAttribute('data-theme', mode);
    var fg = mode === 'dark' ? DARK : LIGHT;
    var grid = mode === 'dark' ? 'rgba(232,230,227,.14)' : 'rgba(0,0,0,.12)';
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd || !window.Plotly) return;
    /* Leave the 3-D scene alone. Its axes, walls and grid are part of the
       figure as generated; overriding them exposed grid lines the original
       never showed. Only the page-level surfaces and text are themed.
       The 2-D figures (histograms, clusters) do use these axis colors. */
    var axis = { color: fg, gridcolor: grid, zerolinecolor: grid,
                 linecolor: grid, tickfont: { color: fg },
                 title: { font: { color: fg } } };
    var update = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: fg },
      'legend.font.color': fg,
      'title.font.color': fg,
      xaxis: axis, yaxis: axis
    };
    try { window.Plotly.relayout(gd, update); } catch (e) { /* older bundle */ }
  }
  window.addEventListener('message', function (e) {
    if (e.data && e.data.type === 'theme' && (e.data.mode === 'dark' || e.data.mode === 'light')) {
      paint(e.data.mode);
    }
  });
  /* Opened on its own (the gallery's "Open figure" link, or a bookmark) there
     is no parent to ask, so fall back to ?theme=, then the stored preference,
     then the OS setting. */
  function standalone() {
    var q = null;
    try { q = new URLSearchParams(location.search).get('theme'); } catch (e) {}
    if (q === 'dark' || q === 'light') return q;
    try {
      var v = localStorage.getItem('theme');
      if (v === 'dark' || v === 'light') return v;
    } catch (e) {}
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  }

  /* Ask the parent once the plot exists; retry briefly while Plotly boots. */
  var tries = 0;
  (function ready() {
    if (document.querySelector('.js-plotly-plot') && window.Plotly) {
      if (window.parent !== window) {
        try { parent.postMessage({ type: 'theme-request' }, '*'); } catch (e) {}
      } else {
        document.documentElement.setAttribute('data-standalone', '');
        paint(standalone());
      }
      return;
    }
    if (tries++ < 60) setTimeout(ready, 100);
  })();
})();
</script>'''
            document = document.replace("</head>", embed_style + "</head>", 1)
            (destination / target).write_text(document)
        (output / f"{tag}.html").write_text(page(tag, models))
    print(f"Exported four figures for {len(models)} models to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "interactive")
    args = parser.parse_args()
    export(args.dataset.resolve(), args.out.resolve())
