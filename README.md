# NMDA QSAR reproducibility files

This archive is the machine-readable supporting repository for the NMDA
activity-cliff analysis. It intentionally contains only analysis scripts,
postprocessed input data, and tabular results; figures, trained-model
checkpoints, caches, and other generated binaries are excluded.

## SALI definition used for this release

All activity-cliff analyses use **SALI > 5.5**. At this threshold, the
recomputed analysis identifies 140 pIC50 cliff pairs (95 of 332 compounds)
and 261 pKi cliff pairs (196 of 511 compounds). Ordinary cross-validation
model fits do not depend on the SALI cutoff; the threshold changes the
cliff-subset, directionality, cross-endpoint, and leave-cliff-out reporting.

The four prospective compounds use the final concentration-response estimates
in `data/external_concentration_response.csv` (six cell-level fits per
compound). Their measured para-to-meta potency differences are 1.571 pIC50
units for the N-ethyl pair (37.2-fold; Welch 95% CI 21.5-64.5-fold) and 1.159
pIC50 units for the N-propyl pair (14.4-fold; 95% CI 10.0-20.8-fold). With the
primary ECFP4 similarities, the corresponding SALI point estimates are 5.236
and 4.153. Thus neither external pair is classified as SALI > 5.5 by its point
estimate; `results/external_pair_statistics.csv` retains the experimental
intervals and a prespecified sensitivity analysis excluding the single HC49
trace flagged as a poor fit in the source workbook.

All four external compounds were synthesized and tested as racemates. Their
machine-readable structures are deliberately non-isomeric SMILES. ECFP4 and
FCFP4 were generated without chirality, and no assigned atom configuration was
available to the descriptor or graph workflows. The released predictions are
therefore racemate-level predictions and cannot resolve enantiomer-specific
activity.

## Contents

- `environment.yml`: pinned Conda/Python environment for the final analysis.
- `data/`: curated postprocessed pIC50 and pKi input data.
- `data/external_concentration_response.csv`: final compound-level
  concentration-response estimates used throughout the prospective analysis.
- `data/external_individual_fits.csv`: the 24 cell-level pIC50 and Hill fits,
  including the source workbook's trace-level note.
- `scripts/external_data.py`: central validated loader for the four external
  measurements; analysis scripts do not carry independent hard-coded copies.
- `scripts/external_pair_statistics.py`: experimental contrasts, uncertainty,
  fold changes, similarities, SALI values, and flagged-trace sensitivity.
- `scripts/qsar_nmda.py`: main QSAR workflow and figure-generation utilities.
- `scripts/reproduce_threshold55.py`: reruns all threshold-dependent analyses
  and writes the primary CSV files in `results/`.
- `scripts/chemspace_viz.py`: chemical-space and SALI visualization workflow.
- `scripts/toc_graphic.py`: table-of-contents graphic workflow.
- `scripts/applicability_domain.py`: training-only descriptor PCA and empirical
  applicability-domain assessment for the four external compounds.
- `scripts/y_randomization.py`: fixed-fold, 100-repeat target-permutation
  control for five prespecified representative pIC50 configurations.
- `scripts/rerun_seeded_graph_models.py`: deterministic five-seed Chemprop and
  CheMeleon-initialized analysis for both endpoints.
- `scripts/assemble_seeded_graph_outputs.py`: merges graph summaries with the
  fixed non-graph result tables.
- `scripts/audit_seeded_graph_outputs.py`: checks seeds, folds, checkpoint
  loading, predictions, and aggregate files.
- `results/`: machine-readable cross-validation, directionality,
  cross-endpoint, leave-cliff-out, prospective-pair, sensitivity, and Figure 6
  attribution outputs.

## Software environment

Create and activate the pinned environment from the top-level archive
directory:

```bash
conda env create -f environment.yml
conda activate nmda-qsar-final
python scripts/audit_final_release.py
```

The environment targets Linux x86-64 and pins the direct dependencies used by
the final measured-IC50 analysis, including Python 3.12.12, scikit-learn 1.4.2,
RDKit 2025.9.3, PyTorch 2.9.1, Chemprop 2.2.3, and TabPFN 2.1.0. The PyTorch
wheel supplies the CUDA 12.8 user-space libraries on Linux; the NVIDIA driver
is a host requirement and is not part of the Conda environment. Conventional
models, audits, and most figure-generation commands can run without a GPU.

The numerical analyses were finalized with scikit-learn 1.4.2. A later UMAP
0.5.11 installation used for the archived Figure 2 declares scikit-learn >=1.6,
although it executed with 1.4.2 in the original runtime. To keep a fresh
installation solver-consistent without changing the analysis-critical
scikit-learn version, `environment.yml` pins UMAP 0.5.7. UMAP affects only the
two-dimensional visualization; SALI values, model fits, predictions, and all
reported metrics are independent of the UMAP embedding.

Model weights are not embedded in the environment. TabPFN downloads its v2.1
regressor weights on first use. The CheMeleon checkpoint must be supplied as
described under **Reproducing the graph-model analysis** below.

## Reproducing the threshold-dependent CSVs

Use a Python environment with RDKit, NumPy, pandas, scikit-learn,
matplotlib, XGBoost, and LightGBM installed. From the top-level archive
directory, run:

```bash
python scripts/reproduce_threshold55.py
```

The script overwrites the threshold-dependent result CSVs in `results/` with
the same fixed-fold, seeded analysis used for this release. It may take a
few minutes.

To reproduce the final experimental contrasts independently, run:

```bash
python scripts/external_pair_statistics.py --out_dir results
```

The complete conventional-model rerun (including deterministic 5x5
cross-validation, leave-cliff-out evaluation, prospective predictions,
training-sample bootstrap intervals, and SHAP outputs) is:

```bash
python scripts/qsar_nmda.py --skip_graph --sali_sweep --cv5x5
```

TabPFN uses the v2.1 regressor with eight estimators and random state 0. PCA is
fitted inside each cross-validation pipeline, so validation-fold structures do
not influence the representation fitted to the corresponding training fold.

## Y-randomization control

The target-permutation control keeps the molecular features, five fixed folds,
preprocessing, model settings and evaluation metrics unchanged. In each of 100
null repeats, the pIC50 labels are shuffled across the 332 records before any
model fitting. Hyperparameters are not reselected on the randomized labels.
The five configurations were selected before running the control to represent
the descriptor workflow, the highest overall fixed-fold pIC50 R2, the strongest
cliff-subset configurations, and the TabPFN foundation-model workflow.

```bash
python scripts/y_randomization.py --n-permutations 100
```

Machine-readable repeat-level values, summaries and metadata are written to
`results/y_randomization/`. The empirical one-sided p value uses the finite-null
correction `(1 + count(null R2 >= observed R2)) / (1 + n_permutations)`.
The observed R2 values were 0.237–0.344, whereas the mean randomized-label R2
values were −0.407 to −0.030. No null repeat reached its configuration's
observed R2 (empirical p = 0.0099 for each); the largest null R2 was 0.048.
This is a model-sanity control, not a substitute for grouped-structure
validation or evidence of accurate local effect-size prediction.

## Regenerating figures

After the result CSVs are present, run:

```bash
python scripts/qsar_nmda.py --figures_only --out_dir results
python scripts/chemspace_viz.py --figure2 --sali_threshold 5.5 --out_dir results
python scripts/toc_graphic.py --out_dir results
```

`chemspace_viz.py` additionally requires `umap-learn`. Figure files are
deliberately not included in this supplementary archive.

To refit the pIC50 models and regenerate the SHAP figures, including Figure 6,
run:

```bash
python scripts/qsar_nmda.py --shap_only --out_dir results
```

For Ridge, the workflow uses exact, deterministic interventional values from
`shap.LinearExplainer` with the complete transformed training set as the
background. The plotted feature-level values and panel correlations are saved
to `results/shap/ridge_shap_feature_values.csv` and
`results/shap/ridge_shap_correlations.csv`, respectively.

To regenerate the descriptor-space applicability-domain figure and its
machine-readable values, run:

```bash
python scripts/applicability_domain.py \
  --input data/pic50_clean.csv \
  --out_dir results/applicability_domain
```

The scaler and PCA are fitted to the training structures only. The external
compounds are projected into that fixed space, and their score and orthogonal
distances are compared with empirical training-set 95th-percentile limits.

## Reproducing the graph-model analysis

The graph rerun used Chemprop 2.2.3, PyTorch 2.9.1, five fixed row-wise folds,
40 epochs, batch size 64, and master seeds 42–46. Scratch Chemprop uses the
default hidden dimension 300 and depth 3. The CheMeleon-initialized model is
constructed from the checkpoint hyperparameters (hidden dimension 2048,
depth 6), loads the message-passing encoder strictly, and fine-tunes all
parameters with a newly initialized regression head. Both configurations use
mean aggregation, fold-local target normalization, MSE loss, Adam, and the
Chemprop default Noam learning-rate schedule.

The checkpoint binary is deliberately excluded from the archive. Place the
official file at `checkpoints/chemeleon_mp.pt` and verify SHA-256
`c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`, then run:

```bash
python scripts/rerun_seeded_graph_models.py --device gpu --threads 8
python scripts/audit_seeded_graph_outputs.py
python scripts/assemble_seeded_graph_outputs.py
```

The first command is resumable at the endpoint/model/seed level. The released
CSV files include every repeat, aggregate means, standard deviations and
ranges, full-data prospective predictions, out-of-fold predictions, and
per-cliff-pair directionality calls.

## License

The code is made available under the MIT License in `LICENSE`. The input
data and analysis outputs remain subject to the manuscript's citation and
data-availability terms.
