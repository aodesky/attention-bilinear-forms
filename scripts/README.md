# Code and experimental data

This directory holds the code for the experiments of *On attention heads
and bilinear forms*. Every figure, table and quoted statistic in the
paper is produced by one of these scripts. The experiments are numbered
in the appendix "Code and experimental data"; each entry there names the
script that runs it and the data it writes.

## Requirements

Python 3.10 or later, with `numpy`, `scipy`, `pandas`, `matplotlib` and
`scikit-learn`. The extraction step additionally needs `torch`,
`transformers` and `safetensors`; nothing else requires model weights or
a network connection.

    pip install -r ../requirements.txt

Figures are typeset with LaTeX (`text.usetex`), so a TeX installation is
needed to regenerate them.

## The two stages

**Extraction.** `extract_qk_bilinear_db.py` downloads each model,
computes the bilinear form `L = W_K^T W_Q` of every attention head,
restricts it to the reduced subspace `im(W_K^T) + im(W_Q^T)`, and stores
the result in a content-addressed database under `data/bilinear-forms`.
This is the only step that reads model weights. It has already been run
for the models of the paper, and the database ships with this archive,
so the analysis can be reproduced without repeating it.

**Analysis.** Everything else reads that database. `model_set.py` fixes
the set of models reported in the paper and opens the stored forms;
`head_statistics.py` computes the per-head scores, profiles and spectra
once into `survey_results/`; the remaining scripts read those files and
produce the figures and tables.

    python head_statistics.py          # rebuild survey_results/
    python score_survey.py             # the score figures
    python gamma_goodness_of_fit.py    # the goodness-of-fit tables

## The model set

`model_set.py` is the single place that decides which models the paper
reports. The extraction database contains two further models (the Qwen3
pair) that are deliberately not part of the reported set; adding or
removing a model means editing `MODELS` there and rerunning the analysis,
not editing the individual scripts.

## Layout

    model_set.py             the reported models, and access to their forms
    head_statistics.py       per-head scores, profiles and spectra
    extract_qk_bilinear_db.py   model weights -> bilinear forms (stage one)
    analyze_bilinear_spectra.py spectra, clustering and type assignment
    survey_results/          per-head statistics and spectra (written)
    cluster_results/         sorted spectra used by the cluster figures
    qk_geometry_results/     per-model summaries from the earlier survey

Scripts whose names begin with `plot_` or `regenerate_` draw a figure of
the paper and write it to `figures/qk_geometry/`. Scripts beginning with
`verify_` check a stated result numerically and print the comparison.

The repository also contains an `exploratory/` directory of scripts
written while the work was in progress. Those refer to results that are
not part of the paper, and are not included in this archive.
