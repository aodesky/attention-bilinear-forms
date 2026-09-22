# On attention heads and bilinear forms

Experimental code, data and Lean formalizations accompanying the paper
*On attention heads and bilinear forms*.

Visit the [project website](https://aodesky.github.io/attention-bilinear-forms/) for the plots, formalization blueprint, and links to code and data.

Every attention head of a decoder-only transformer carries a bilinear
form `L = W_K^T W_Q` on the residual stream. This repository contains
the code that extracts those forms from fourteen pretrained models,
the derived per-head measurements the paper reports, an interactive
view of the results, and the Lean formalizations of the appendices.

## Layout

    scripts/               the experiments (Python)
    scripts/survey_results/  per-head statistics and spectra
    interactive/           model-selectable gallery with four figures per model
    formalization/         the Lean formalizations of the appendices

`scripts/README.md` describes the experiments and how to rerun them.
The appendix "Code and experimental data" of the paper lists each
experiment, the script that runs it, and where its results appear.

To refresh the public gallery from completed local bilinear-form analyses:

```sh
python3 scripts/export_interactive_gallery.py --dataset /path/to/data/bilinear-forms
```

This preserves the model page URLs and exports the interactive 3-D simplex
with the local markers, the 2-D projection to Δ, spectral distributions by parity
(k = 3), and clusters by layer (k = 3). The top dropdown changes
models while retaining the figure selected by the bottom tabs. Figure
selections can be linked directly using the page's URL fragment. Plotly is
served from a shared local asset, so the gallery needs no external CDN.

## The two stages

**Extraction** reads model weights. `scripts/extract_qk_bilinear_db.py`
downloads each model from the Hugging Face hub, forms the bilinear form
of every head, restricts it to `im(W_K^T) + im(W_Q^T)`, and stores the
result in a content-addressed database:

    python scripts/extract_qk_bilinear_db.py run --out data/bilinear-forms

With no `--models` argument it extracts the fourteen models the paper
reports. The resulting database is about 3 GB and is not included
here.

**Analysis** reads the extracted forms. The per-head measurements the
paper quotes are included in `scripts/survey_results/`, so the figures
and statistics can be reproduced without repeating the extraction:

    python scripts/head_statistics.py     # rebuild the measurements
    python scripts/score_survey.py        # the score figures
    python scripts/type_partition.py      # the three-type partition

## The models

The set of models reported in the paper is listed in
`scripts/model_set.py`:
`distilgpt2`, `gpt2`, `gpt2-medium`, `gpt2-large`, the OPT models at
125m, 350m, 1.3b, 2.7b, 6.7b and 13b, `bloom-3b`, `bloom-7b1`,
`pythia-70m-deduped` and `Mistral-Small-24B-Base-2501`. Together they
carry 9,512 attention heads, and span learned absolute, rotary and
ALiBi positional encodings, head dimensions 64, 80 and 128, and both
multi-head and grouped-query attention.

## Interactive plots

`interactive/` holds one page per model, plotting the profile of every
head on the simplex of profiles, coloured by spectral type. Open
any of them in a browser; the Plotly library is included as a shared local asset.

## Formalization

Read the [formalization blueprint](https://aodesky.github.io/attention-bilinear-forms/blueprint/)
for mathematical statements, proof outlines and links to the audited Lean
declarations, or explore its
[dependency graph](https://aodesky.github.io/attention-bilinear-forms/blueprint/dep_graph_document.html).

`formalization/` contains Lean 4 formalizations of the mathematical
results of the appendices, together with the definition of the profile
of a bilinear form and the main-text theorem on the faces of the
simplex, whose proof is given there. Building requires the Lean toolchain named in
`formalization/lean-toolchain`, which `elan` installs automatically:

    cd formalization
    lake exe cache get
    lake build

The formalizations use no `sorry`s and no axioms beyond `propext`,
`Classical.choice` and `Quot.sound`; `formalization/scripts/check_axioms.sh`
checks this.
`formalization/README.md` maps each file to the result it formalizes.

## Requirements

Python 3.10 or later with the packages in `requirements.txt`. The
analysis needs `numpy`, `scipy`, `pandas`, `matplotlib` and
`scikit-learn`; the extraction additionally needs `torch`,
`transformers`, `safetensors` and `huggingface_hub`. Regenerating the
figures needs a LaTeX installation, since they are typeset with it.

## Project website

`index.html` and `assets/site.css` provide the GitHub Pages landing page.
The navigation links to the paper on arXiv.
`python scripts/build_site_artwork.py` regenerates the illustration and
social card from the included GPT-2 head measurements (requires Pillow).
The gallery exporter and blueprint builder preserve navigation back to
the project home and between the hosted components.
