# Reduced coupling between prestimulus alpha and response speed in autistic children

Code and figure data for the paper *Reduced coupling between prestimulus alpha and response speed in autistic
children* by Amirmasoud Ahmadi and Fatemeh Ghorbani Lohesara (full citation will be added on publication).

## The question

A child's brain does not arrive at every stimulus in the same state. Alpha activity over the back of the head rises
and falls from one second to the next, and in adults those fluctuations are linked to how the next stimulus is
handled: high alpha just before a target tends to go with weaker detection and, in a number of studies, slower
responses. That makes prestimulus alpha a small, measurable window into moment-to-moment readiness.

We asked whether this link between ongoing alpha and the speed of the next response looks the same in autistic and
typically developing children. Using open EEG recorded while 8- to 14-year-olds did three simple detection tasks, we
related alpha in the half second before each stimulus to the reaction time that followed, trial by trial and within
each child.

## What we found

In the visual detection task (FAST), typically developing children responded more slowly after moments of high
alpha. In autistic children this coupling was essentially absent.

| | Coupling (% change in RT per SD of alpha) | 95% CI |
|---|---|---|
| Typically developing (n = 38) | +1.42 | 0.74 to 2.09 |
| Autistic (n = 49) | -0.22 | -0.91 to 0.47 |
| Difference | -1.64 (P = 0.001) | -2.60 to -0.68 |

A few things make us take the difference seriously:

- It is not carried by a handful of children. Per-child slopes were positive in 32 of 38 typically developing
  children and in 25 of 49 autistic children.
- It sits at the alpha peak. Resolving the coupling by frequency puts the group difference in a cluster around each
  child's individual alpha frequency (IAF ± 1 Hz, cluster P = 0.019), and over posterior electrodes.
- It survives a long list of reasonable alternative choices. Across 34 analytic variants (other spectral estimators,
  windows, bands and ROIs, extra covariates, different trial and child selections, different model structures) the
  difference stays between -1.40% and -2.03%, and every confidence interval excludes zero.
- Shuffling group labels 10,000 times almost never produces a difference this large (P < 0.001), and pairing each
  child's alpha with another child's reaction times does not reproduce it either.

The picture is less uniform across tasks. The difference was smaller and uncertain in the illusory-contour task
(-1.35%, P = 0.14) and absent in the audiovisual reaction-time task (-0.28%, P = 0.63); a joint model of the three
tasks says the group difference is not the same everywhere (χ²(2) = 8.48, P = 0.014). This is one dataset, and the
result needs to be replicated in an independent sample.

## Data

All analyses use OpenNeuro [ds006780](https://openneuro.org/datasets/ds006780) v1.0.0 (Molholm, Vanneau & Foxe,
2025): 64-channel BioSemi EEG at 512 Hz from autistic children, typically developing children and unaffected
siblings, recorded during an audiovisual simple reaction-time task (AVSRT), a visual detection stream of faces and
objects (FAST) and an illusory-contour task (IC). The EEG itself is not redistributed here; download it from OpenNeuro.

## How the measure is built

The central quantity is simple to state. For every trial we take a window that ends 50 ms before the stimulus
(500 ms long in FAST and IC, 800 ms in AVSRT), estimate power with a DPSS taper and average it over each child's
IAF ± 2 Hz across eight parieto-occipital electrodes (PO7, PO8, PO3, PO4, POz, O1, O2, Oz). Log alpha is then split,
within each child and task, into a slow part (the block mean) and a fast part (the deviation from that mean, scaled
by the child's own SD). Only the fast part is the predictor, so the analysis asks whether a moment that is high in
alpha *for that child* is followed by a slower response, and comparisons between children with very different overall
alpha power do not enter.

Reaction times are modelled on the log scale in linear mixed models with alpha × group, trial history, timing,
time-on-task splines, eye-movement covariates, age and sex, with child-specific intercepts and alpha slopes.
Inference uses Satterthwaite degrees of freedom and cluster-robust (CR2) standard errors, equivalence tests against a
smallest effect of interest for each task, Benjamini-Hochberg and Holm corrections, and cluster-based permutation
over electrodes and frequencies.

## Finding your way around

The code is grouped by method, not by the order in which things run.

| Folder | What it contains |
|---|---|
| `trials/` | Building trial tables from the BioSemi status channel and `events.tsv`: matching stimuli to responses, RT windows, trial history and timing, the rules specific to each task, and child-level performance criteria |
| `preprocessing/` | 1-40 Hz filtering within recording segments, bad channels with PyPREP, Picard ICA with ICLabel (and a check that protects posterior alpha), an unfiltered copy for the eye channels, blink and saccade detection |
| `feature_extraction/` | Individual alpha frequency, single-trial prestimulus alpha, the within-child decomposition, reliability of alpha, trial-level eye-movement covariates and alpha-burst occupancy |
| `modeling/` | Model data and formulas, the R engine (lme4/lmerTest, glmmTMB, clubSandwich, brms), spline bases, sensitivity variants, reliability calibration, equivalence tests, and a second implementation in statsmodels used as a cross-check |
| `statistical_analysis/` | The group comparisons and everything that tests them: two-stage slopes, permutation and pairing nulls, disjoint trial subsets, RT quantiles, misses, siblings, age, sex, IQ and medication, evoked-response paths, saccades, scalp and frequency maps with cluster permutation, and the Holm family of follow-up tests |
| `utils/` | Configuration loading, the recording index, participant covariates and keyed child identifiers |
| `configs/` | The parameters behind all of the above, in YAML |
| `figures/` | The numbers behind every data panel of Figures 2-6 and S1-S9, and the code that draws each panel |

If you want to read the core of the paper in code, a good route is `feature_extraction/prestimulus_alpha.py` and
`feature_extraction/decomposition.py` for the predictor, `modeling/model_specs.py` and `modeling/R/fit_lmm.R` for the
models, and `statistical_analysis/coupling_analyses.py` for the group comparison. Labels such as R1-R14 and X1-X4 in
docstrings refer to the analyses listed in Additional file 1 of the paper.

This is not a one-command pipeline. The modules document how each reported step was done and can be reused on their
own, but they expect the derived trial tables that the full analysis produces.

## Figures

`figures/data` has one CSV per panel with the values that were plotted: group estimates, confidence intervals, P
values, scalp-map values and per-child slopes, with no child identifiers. `figure2.py` to `figure6.py` and
`supplementary.py` redraw every data panel from these files, one panel at a time, into `figures/panels`. Figure 1,
the overview of the tasks and the measure, is not included. The scalp-map panels use MNE and take a minute or two.

## A note on child identifiers

Analysis tables identify children by a keyed hash of the subject ID (`utils/child_ids.py`), never by the ID itself.
Because children are ordered by this hash whenever an analysis resamples them, the exact permutation and bootstrap
draws, and so the last digits of permutation and bootstrap results, depend on a private key that is not shared.

## Requirements

Python 3.11 with MNE-Python 1.13.2, mne-icalabel 0.10.0 (onnxruntime 1.31.0), PyPREP 0.9.0, python-picard 0.8.2,
NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6, pyarrow 26.0.0, statsmodels 0.15.0, patsy 1.0.3, specparam 2.0.0rc7,
joblib 1.6.0, PyYAML 6.0.3 and matplotlib 3.11.2 (`requirements.txt`).

R 4.4.2 with lme4 1.1-37, lmerTest 3.1-3, glmmTMB 1.1.11, clubSandwich 0.6.0, nlme 3.1-166, brms 2.22.0
(rstan 2.32.7), survival 3.7-0, data.table 1.17.0 and jsonlite 1.8.9. The R scripts are called through `Rscript`;
set the `RSCRIPT` environment variable if it is not on your path.

## Citation

If you use this code, please cite the paper. The reference will appear here once it is published.

## Licence

GNU General Public License v3.0 (`LICENSE`). `feature_extraction/iaf.py` is a port of restingIAF (Corcoran et al.,
2018) and `feature_extraction/bursts.py` builds on BOSC and eBOSC; both are GPL-3.0 and keep their original notices.

## Contact

Amirmasoud Ahmadi, Max Planck Institute for Biological Intelligence. Amirmasoud.Ahmadi@bi.mpg.de
