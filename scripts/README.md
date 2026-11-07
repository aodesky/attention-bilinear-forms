# Code and experimental data

This directory holds the code for *On attention heads and bilinear
forms*. Each empirical observation has one numbered entry point:

    python observation_01_balance.py
    python observation_02_head_profiles.py
    python observation_03_bimodal_gamma.py
    python observation_04_symmetric_bias.py
    python observation_05_pairing.py
    python observation_06_full_gamma.py
    python observation_07_modes.py
    python observation_08_shapes.py
    python observation_09_sigma_lambda.py

The observation files import reusable analysis classes and components.
`ScoreSurvey` supplies Observations 1, 4 and 5; `SpectralGeometry`
supplies Observations 7, 8 and 9; and `ObservationPipeline` composes the
analysis and plotting components needed by Observations 2, 3 and 6.
Observation 1 reports positive and negative eigenvalue counts directly.
Observation 5 reports lobe proportionality errors and pairing-bound gaps,
connecting the pairing figures to Section 5.2.1.

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
once into `survey_results/`; the observation entry points read those
files and produce the figures, tables and summaries.

    python head_statistics.py          # rebuild survey_results/
    python observation_04_symmetric_bias.py
    python observation_06_full_gamma.py

## The model set

`model_set.py` is the single place that decides which models the paper
reports. The extraction database contains one further model
(`Qwen3-0.6B-Base`) that is deliberately not part of the reported set;
adding or removing a model means editing `MODELS` there and rerunning the
analysis, not editing the individual scripts.

Three models carry a per-token normalization on the query/key path that
the stored forms omit: `Qwen3-14B-Base` (QK-norm; the gains and per-token
scalars are dropped) and `Moonlight-16B-A3B` (multi-head latent attention;
the gain of the latent RMSNorm is absorbed into the key, its per-token
scalar dropped). Both the extractor and `analyze_bilinear_spectra.py`
refuse such checkpoints unless `--allow-qk-norm` is passed.
`verify_extraction.py` recomputes one layer of every model from the raw
checkpoint, independently of the extractor, and compares singular values.

## Layout

    model_set.py             the reported models, and access to their forms
    head_statistics.py       per-head scores, profiles and spectra
    effective_rank.py        participation ratio and stable rank of L, from
                             the singular values saved by head_statistics.py
    symmetric_bias.py        the diagonal-block share delta and the two
                             inequalities (companion paper on symmetric bias)
    diagonal_blocks.py       the diagonal blocks in the metric of S, and the
                             random-solution baseline (companion paper)
    extract_qk_bilinear_db.py   model weights -> bilinear forms (stage one)
    analyze_bilinear_spectra.py spectra, clustering and type assignment
    observation_common.py       base classes for observation entry points
    observation_*.py            one runnable file per observation
    survey_results/          per-head statistics and spectra (written)
    cluster_results/         sorted spectra used by the cluster figures
    qk_geometry_results/     per-model summaries from the earlier survey

Scripts whose names begin with `plot_` or `regenerate_` draw a figure of
the paper and write it to `figures/qk_geometry/`. Scripts beginning with
`verify_` check a stated result numerically and print the comparison.

The repository also contains an `exploratory/` directory of scripts
written while the work was in progress. Those refer to results that are
not part of the paper, and are not included in this archive.
