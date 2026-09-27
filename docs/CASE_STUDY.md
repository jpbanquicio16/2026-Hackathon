# Real-data case study: where white-wine predictions fail

## Question and source

Can physicochemical measurements distinguish wines with a reported sensory quality
score of **7 or more**? This binary threshold was fixed before fitting. It is an
illustrative screening task; it does not measure wine prices or identify causes of quality.

Source: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), Cortez, Cerdeira, Almeida, Matos and Reis (2009),
DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The white-wine file contains
4,898 rows, eleven physicochemical features, and the sensory quality target.
The original semicolon-separated file is in `data/winequality-white.csv`.

## Protocol fixed before examining test errors

* Remove 937 exact repeated full rows; 3,961 remain.
* Preserve source identity with `example_id`. Keep identical feature profiles in the
  same `sample_group`, including profiles with different reported quality scores.
* Convert quality to `quality_7_plus` / `quality_below_7`; exclude original quality,
  example ID and group ID from the eleven training features.
* Seed **42**, **20% of groups** held out. This produces
  **3,168 training / 793 test rows**.
* Compare logistic regression and a 100-tree random forest using **5-fold
  stratified group cross-validation on training rows**, ranked by balanced accuracy.
  No parameter search. Fold-local imputation and scaling; no class rebalancing.
* Lock **Random forest** using validation scores, then evaluate it once on test
  rows. The other candidate's test result is never calculated. A majority-class
  dummy baseline uses the same training/test partition.
* Inspect alcohol × volatile acidity with four quantile bins per axis. These axes
  were specified before scoring. The highlighted cell below was selected after
  observing the map and is an exploratory finding.

## Results

| Candidate | Training CV balanced accuracy | Fold standard deviation |
| --- | ---: | ---: |
| Random forest | 0.6852 | 0.0093 |
| Logistic regression | 0.6300 | 0.0302 |

The selected model's **held-out balanced accuracy is 0.6639**,
versus **0.5000** for the majority-class baseline. Its accuracy is
**82.72%** (656 correct / 793 evaluated),
and macro F1 is **0.6878**. Fold standard deviations are
not confidence intervals. See the exported per-class table for minority-class recall.

## Failure-map finding and decision

The highest-error cell with at least 20 examples is **alcohol
[11.4, 14.2]** × **volatile acidity
[0.27, 0.33)**. It has **20 errors in
54 rows (37.0%)**, compared with
**17.3%** across mapped rows. Its descriptive 95% Wilson
interval is **25.4%–50.4%**.
The exact examples are in `selected_cell_rows.csv`; every map cell is independently
reconciled against its serialized prediction rows in `independent_audit.json`.

This identifies a measurement region for additional data collection and review of
false positives versus false negatives. A next experiment would pre-register this
slice, collect independent wines in it, and check whether the error concentration
persists. Do not tune the current model to this test slice and report the same test
score as an independent improvement. No causal effect of alcohol or acidity is established.

## Limitations

This historical dataset is not a random sample of every wine market. The sensory
score is ordinal and thresholding loses information. Removing repetitions changes
the population being described. Feature-profile grouping is a proxy for dependence;
producer, collection time and true sample identity are unavailable. Wilson intervals
and the app's cell tests assume independent examples, so they are descriptive here,
especially for remaining repeated profiles. Holm correction covers cells in one map,
not repeated feature, threshold or model exploration. These results are not a claim
about deployment performance or commercial grading reliability.

## Reproduce and inspect

```bash
python scripts/make_wine_case_study.py
python scripts/reproduce_run.py examples/wine_case_study/wine_quality.atlas.zip --output /tmp/atlas-wine-replay
streamlit run app.py
```

Use the locked Python environment described in the README. In the app, select
**Analyze an existing prediction CSV → White wine case study**, choose alcohol and
volatile acidity, and choose Quantiles. Inspect any cell and compare its example IDs
with the audit. To reproduce the complete fit, use **Train a model and evaluate it →
Restore a saved run** with `wine_quality.atlas.zip`.

Full outputs: [examples/wine_case_study](../examples/wine_case_study/).
Source SHA-256: `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836`.
