## Experimental matched-pair effects

| Pair (meta-Cl minus para-Cl) | Observed ΔpIC50 (Welch 95% CI) | Potency ratio (95% CI) | ECFP4 Tanimoto | ECFP4 SALI (95% CI) | Point estimate > 5.5? |
|---|---:|---:|---:|---:|:---:|
| N-ethyl: HC49 − HC22 | 1.571 (1.332–1.810) | 37.2-fold (21.5–64.5) | 0.700 | 5.236 (4.439–6.032) | No |
| N-propyl: HC46 − HC52 | 1.159 (1.000–1.318) | 14.4-fold (10.0–20.8) | 0.721 | 4.153 (3.582–4.723) | No |


## Training-set validation at SALI > 5.5

- pIC50: 140 cliff pairs involving 95 of 332 compounds.
- pKi: 261 cliff pairs involving 196 of 511 compounds.
- Best fixed-fold conventional pIC50 MAE: RF/FP 0.589 (R² 0.344) and
  XGB/FCFP 0.593 (R² 0.325).
- Best 5×5 mean pIC50 MAE: XGB/FCFP 0.576 ± 0.021.
- Best training-cliff directionality: SVM/FP 83.6%; SVM/FCFP 80.0%.

## Seeded graph models (five repeats; seeds 42–46)

| Endpoint | Model | MAE, mean ± SD (range) | R², mean ± SD | Cliff MAE, mean ± SD | Cliff directionality, mean ± SD |
|---|---|---:|---:|---:|---:|
| pIC50 | ChemProp | 0.761 ± 0.003 (0.757–0.764) | 0.117 ± 0.011 | 1.051 ± 0.008 | 54.9 ± 3.0% |
| pIC50 | ChemProp + CheMeleon | 0.594 ± 0.004 (0.590–0.599) | 0.281 ± 0.005 | 0.745 ± 0.008 | 79.7 ± 0.8% |
| pKi | ChemProp | 0.687 ± 0.007 (0.678–0.698) | 0.186 ± 0.018 | 0.949 ± 0.010 | 51.1 ± 1.7% |
| pKi | ChemProp + CheMeleon | 0.665 ± 0.005 (0.659–0.673) | 0.157 ± 0.009 | 0.834 ± 0.009 | 77.5 ± 1.1% |

## Prospective paired predictions

- Ridge/ECFP4 predicts +0.695 pIC50 for both pairs; the training-resample 95%
  interval for the mean paired effect is −0.344 to +1.436.
- Scratch ChemProp predicts +0.015 (ethyl) and +0.015 (propyl); the five-seed
  range for the mean effect is +0.013 to +0.017. Both directions are positive
  in all five seeds, but the observed effect size is severely underestimated.
- ChemProp + CheMeleon predicts +0.104 (ethyl) and +0.053 (propyl); the
  five-seed range for the mean effect is −0.037 to +0.189. Both directions are
  positive in four of five seeds, and the observed effect size is again
  severely underestimated.

The primary prospective endpoint should therefore be described as recovery of
the signed within-pair ordering, with substantial attenuation and uncertainty,
not as quantitatively accurate prediction of the measured potency shifts.

## Applicability domain

All four analogs are inside both empirical 95% descriptor-PCA limits. Their
score-distance fractions of the limit are 0.641–0.704 and their orthogonal-
distance fractions are 0.384–0.668. Their maximum ECFP4 similarities to the
training set are 0.438–0.535, indicating descriptor-space inclusion despite
only moderate nearest-neighbour fingerprint similarity.
