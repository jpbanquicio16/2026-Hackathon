# AI Error Heatmap

**The model has an overall score, but which kinds of examples does it get wrong?**

Model Failure Atlas is a Streamlit app for exploring where a model goes wrong.
Choose one of two workflows at the top of the page: analyze existing predictions,
or train a model and generate held-out predictions. Choose **Classification** or
**Regression** as the prediction task. Then pick two numeric features. The app reports
overall performance, splits both features into ranges, and colours each combination
of ranges by its error rate (classification) or error magnitude (regression). Select
a cell to see the exact rows behind its numbers.

Both workflows use the same calculations, heatmap and row inspection. Training mode
labels all results **Held-out test performance** and never substitutes training
performance for a test result.

## Features at a glance

| Area | What the app does |
| --- | --- |
| Tasks | Classification and regression, in both workflows |
| Models | 7 classifiers and 6 regressors (scikit-learn), compared on identical rows |
| Splits | Stratified random, grouped (group-disjoint) and time-ordered held-out splits |
| Validation | Cross-validation on training rows only, optional repeats and hyperparameter search |
| Imbalance | None, class weights or oversampling, applied inside each training fit |
| Checks | Target-type guidance, dataset quality summary, target-leakage detection |
| Metrics | Accuracy, balanced accuracy, precision/recall/F1, normalised confusion matrices, MAE/RMSE/R², ROC-AUC |
| Probabilities | Confidence analysis, binary threshold exploration, multiclass ROC-AUC (macro/weighted, OvR/OvO) |
| Explanations | Model-based and permutation feature importance; exact row explanations for linear models and single trees |
| Provenance | Dataset fingerprint, split ID, full run metadata, session experiment history, report bundle |

For the local Iris dataset, `iris.csv` has 150 flower measurements and true
`species` labels, but no model predictions. Selecting two measurements as
actual and predicted labels produces an invalid model evaluation. The app warns
when label choices look like continuous measurements; numeric class codes can
still be valid labels when they represent classes.

Run `python scripts/make_iris_evaluation.py` with `pandas`, `numpy`, and
`scikit-learn` installed to regenerate
`examples/iris_heldout_predictions.csv`. The script uses a stratified 80/20
train/test split with seed 42 (120 training rows and 30 held-out rows). It fits
`StandardScaler` and a 3-nearest-neighbors classifier on **training rows only**,
using all four measurements. It predicts only the held-out rows and exports
their source-row `example_id`, four features, `actual_species`, and
`predicted_species`. Upload that file, choose the two species columns as labels,
and use `sepal_length` and `sepal_width` as the default map axes. The 30 exported
rows are the complete test set; the other 120 rows are never included in the
app's performance numbers.

## Quick start

Requires Python 3.10 or newer (tested on 3.12).

```bash
git clone https://github.com/jpbanquicio16/2026-Hackathon.git
cd 2026-Hackathon
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501> with the sample dataset loaded. To use
your own data, choose **Upload a CSV** in the sidebar.

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

The suite includes headless Streamlit `AppTest` journeys, so it needs no browser.
Compare the old and current dataset fingerprints on a large synthetic frame with
`python scripts/benchmark_fingerprint.py [rows] [columns] [models]` (defaults
200,000 × 12, three models). Timings are printed, not asserted, because they
depend on the machine.

## Two workflows

### Analyze an existing prediction CSV

1. Select **Analyze an existing prediction CSV** at the top of the page.
2. Use the synthetic sample, or choose **Upload a CSV** in the sidebar.
3. Choose actual and predicted label columns. Both must describe class outcomes
   for the same examples. Review the missing-label and diagnostic warnings.
4. Choose two numeric map axes. Click a cell or use the range selectors to inspect
   its rows, and optionally filter the detail table to errors only.

Numeric class codes are supported. Incompatible numeric/text label kinds,
high-cardinality numeric labels and class sets with no overlap produce advisory
warnings, not rejection. Predictions of classes absent from the actual test labels
can be legitimate; the app lists these classes. The app cannot establish from
column values alone whether predictions came from a real model or held-out data.
It never merges distinct labels such as `01`, `1`, and `1.0`.

With **Regression** selected, the actual and predicted columns must be numeric. The
app reports MAE, RMSE and R², plots residuals, and maps mean absolute error or the
share of rows above an absolute-error tolerance you choose.

Uploaded probabilities are used only after you say what they mean, under
**Class probabilities and binary thresholds**:

- **Two classes:** map one column as P(positive class) and confirm it. This enables
  ROC-AUC, a precision/recall-by-threshold chart and an exploratory decision
  threshold (the original prediction is kept in an extra column).
- **More than two classes:** tick **Map class probability columns**, choose one
  column per class (columns named after a class are pre-selected) and confirm.
  The mapping is rejected if a class has no column, a column is used twice, values
  are non-numeric, blank or outside 0–1, or rows do not sum to 1 within 0.02. Small
  rounding deviations are renormalised with a warning. The app also warns when a
  column's name mentions a different class, or when the most probable class rarely
  matches the predicted label. Missing scores are never invented.

### Train a model and evaluate it

1. Select **Train a model and evaluate it**, choose Classification or Regression,
   then upload a **Labelled CSV**. The file needs a target and at least one numeric
   feature. For Iris, upload `iris.csv`, choose `species`, and retain the four
   measurement features. For regression, choose a numeric target such as
   `petal_length`.
2. Review the target guidance, the **Dataset quality summary** and any leakage
   warnings (see [Checks before training](#checks-before-training)).
3. Choose the split method, features, test-set proportion (10–50%), random seed and
   model. The defaults are a random split, 20%, seed 42, and 3-nearest neighbors,
   with 3-fold cross-validation on the training rows.
4. Under **Validation, balancing and model comparison**, optionally change the folds
   and repeats, turn on hyperparameter search, choose a class-imbalance strategy,
   and pick extra models to compare.
5. Select **Train and evaluate**. A progress bar reports each validation fit as it
   finishes. The app reports training and held-out row counts, model, seed and split.
   All performance metrics and map rows use only the test set.
6. Inspect a map cell. Use **Download held-out prediction CSV** to export the
   original selected features, retained identifiers, `source_row_id`, actual labels,
   predictions and, when available, class probabilities (`probability_0`, … in the
   order of the sorted class names) and `predicted_confidence`. Upload this CSV in
   the first mode to reproduce the analysis. The generated label names are normally
   `actual_label` and `predicted_label`; a suffix is added if a source column already
   uses either name.

#### Models

| Classifiers | Regressors |
| --- | --- |
| 3-nearest neighbors, Logistic regression, Decision tree (`max_depth=5`), Random forest (100 trees), Support vector machine (RBF), Gradient boosting (histogram), Calibrated logistic regression | Ridge regression, 3-nearest neighbors, Decision tree, Random forest, Support vector machine (RBF), Gradient boosting |

Every model runs inside the same pipeline: training-only median imputation, then
standard scaling, then the estimator. Stochastic estimators receive the selected
seed. **Hyperparameter search** (grid search, scored by balanced accuracy for
classification and MAE for regression) needs cross-validation and never sees the
test rows. Compared models always share identical training and test rows.

**Support vector machine probabilities.** The classifier SVM is wrapped in sigmoid
(Platt) calibration on 2 stratified folds of the training rows. Predictions are
the most probable calibrated class, so predicted labels, confidence and probabilities
agree. Its probabilities feed ROC-AUC, confidence analysis and binary thresholds.
They are unavailable, with an explicit message, for grouped and time-ordered splits
(internal random calibration folds would mix groups or periods) and when a class has
fewer than 2 training rows in a fit.

#### Splits and validation

- **Random:** stratified for classification. Each class needs at least two rows, and
  both sets must contain every class. If class counts or the proportion make this
  impossible, the app explains why and shows no results. There is no unstratified
  fallback.
- **Grouped:** all rows from a group stay on one side. For classification, the app
  searches up to 100 group partitions from the same seed for one that puts every
  class in both sets (the first partition is used unchanged when it already does).
  When no partition can, it warns that the test set is missing classes, so recall,
  ROC-AUC and the confusion matrix may be incomplete. It also explains why, for
  example when every class lives in a single group. Cross-validation uses
  `StratifiedGroupKFold` (classes balanced across folds, groups never split), or
  `GroupKFold` for regression.
- **Time ordered:** train on earlier rows, test on later rows. Equal timestamps stay
  together, and validation uses expanding time windows (`TimeSeriesSplit`).
- Random cross-validation uses `StratifiedKFold` / `KFold`, or their repeated
  versions when repeats are above 1.

#### Class imbalance (classification)

**None**, **Class weights** (`class_weight="balanced"`) or **Oversampling** (random
minority oversampling). Either is applied inside every training fit, including
validation and calibration folds; test rows are never reweighted or resampled. Class
weights are offered only for models that accept them: every classifier except
3-nearest neighbors. The strategy is recorded in the run metadata, the comparison
table and the experiment history.

#### Data preparation

- The target, detected ID columns and the group or time column cannot be training
  features. ID detection uses ID names, leading-zero numeric codes and row-counter
  patterns.
- **Features must be numeric.** A candidate needs at least 90% finite numbers among
  its non-missing values and more than one distinct value. Text, categorical and
  date strings are **not encoded**; excluded columns are listed with the reason.
- Target labels use the existing whitespace/missing-value rules. Missing targets
  are counted and removed before splitting; numeric class codes remain text labels.
- Exported feature values remain as uploaded, so missing map-axis values still
  cause map omissions even though the model can predict after imputation. Those
  rows remain in test accuracy. `source_row_id` is the 1-based data-row position in
  the labelled input; the detail table's `row` is the position in the exported CSV.
- Changing training settings removes old results until **Train and evaluate** is
  selected again. Inspecting cells or changing map settings reuses the fitted result.

With the checked-in Iris data, the default split and 3-nearest neighbors, the split
is 120 training and 30 test rows (10 per species). There are **28 correct and 2
incorrect predictions, 93.3% accuracy**. Source rows 135 and 139 are virginica
predicted as versicolor. The tests check the exported CSV independently, all 16
default map cells against their inspected rows, round-trip upload, and isolation of
preprocessing from test values. Reproduction was verified with Python 3.12, NumPy
1.26.4, pandas 3.0, scikit-learn 1.9.0 and Streamlit 1.58.

## Checks before training

### Target guidance

The app classifies the chosen target from its distinct values, their share of rows,
whether they are numeric or whole numbers, and the sample size:

- **Binary** and **multiclass** targets are treated as classes. A few whole-number
  codes (up to 10) count as multiclass, with a note that an ordinal score could
  also suit Regression.
- **Likely continuous**: many distinct decimals, or many whole numbers relative to
  the rows. With Classification selected, the app warns, for example: *“sulphates”
  contains 68 distinct numeric values and appears continuous. Classification is
  unlikely to be appropriate for this target. Consider switching the task type to
  Regression.*
- **Ambiguous**: for example 15 whole numbers, a rating scale, or text where most
  rows have their own value.

Class counts and imbalance warnings are shown only for binary and multiclass
targets. For a continuous or ambiguous target they appear only after you tick
**Treat the values as classes anyway**. If training such a target fails, the error
says how many values occur only once, that the target appears continuous, and to
try Regression.

### Dataset quality summary

An expander lists each check with a status (Problem, Warning, Review, Info, OK) and
puts the ones that need attention first:

- missing values, duplicate rows, column types and possible identifier columns;
- **target type** and **target balance**: number of classes, largest and smallest
  class with counts and percentages, and the imbalance ratio;
- **potential leakage**: how many suspicious columns there are, with their names and
  a short reason for each.

It also shows the per-column table and the class-balance table.

### Leakage detection

Every column is compared with the target. Findings are clues, not proof, and every
flagged column comes with plain-language reasons. The selected training features get
a warning above the **Train** button; the run metadata records which flagged
features were used.

- **Probable leakage (high risk):**
  - exact copies, including the same numbers written differently (`1` vs `1.00`);
  - label-encoded or renamed copies of a categorical target (a one-to-one mapping,
    for example `0 → setosa`);
  - columns that determine the target, or are determined by it (`is_setosa`), and
    near-deterministic mappings with a few exceptions;
  - numeric offsets, scalar multiples, linear transformations, strictly monotonic
    transformations and near-perfect correlation (|r| ≥ 0.98).
- **Review the name:** names are split into words, so snake_case, kebab-case,
  spaces, camelCase, PascalCase and acronyms all work. For example `predictedLabel`
  becomes “predicted label” and `PREDScore` becomes “pred score”. The app flags
  prediction words (pred, prediction, predicted, yhat, y_pred, ypred), probability
  words (prob, proba, probability), outcome words (target, label, outcome) and timing
  words (future, post, after, excluding post code / after tax). The words `score` and
  `result` are flagged only next to the target's name or a class name, so
  `credit_score` is not flagged.
- Not flagged: identifiers, unique high-cardinality codes, and features that are
  strongly but not near-perfectly correlated with the target.

## Demo walkthrough (about two minutes)

The sample file, `sample_predictions.csv`, is **synthetic**: results from an
imaginary traffic-sign classifier on 1,200 images (see [Sample data](#sample-data)).

1. **Data.** The app reports: *1,200 rows uploaded. 8 rows excluded because the
   actual or predicted label was missing. Analysis uses 1,192 rows.* The label
   columns are detected automatically.
2. **Overall score.** Accuracy is **85.0%** (1,013 correct out of 1,192 evaluated
   rows). The confusion matrix shows that the most common mistake is `speed_30`
   read as `speed_50`. It also flags `speed_80` as a class that was predicted but
   never occurs as an actual label.
3. **Failure map.** With brightness on X and blur on Y, one corner stands out.
   Dark, blurry images (brightness 0.02–0.26, blur 6–8 px) are wrong
   **37 times out of 53 (69.8%)**. Across all mapped rows the error rate is
   15.2%, and the best cell is wrong 1 time in 94.
4. **Inspect.** Click that cell, or choose its two ranges. The table lists exactly
   those 53 rows; switch on **Errors only** to see the 37 mistakes. If you open
   **Optional columns** and confirm what the confidence column means, the app also
   shows that the model's average confidence on those mistakes is still 0.62.
5. **The caution.** Switch the Y axis to `speed_kmh`. Dark, fast images also look
   bad (45 of 72 wrong), even though the generator never uses speed directly: blur
   increases with speed. The map shows **where** errors concentrate, not **why**.

## Existing-prediction CSV requirements

- A header row, then **one row per evaluated example** (one prediction per row).
- **An actual-label column and a predicted-label column.** They can have any names:
  you choose them in the app. Names such as `actual`/`predicted`, `label`/`prediction`
  and `y_true`/`y_pred` are pre-selected.
- **At least two numeric feature columns** to use as the map's axes. Without them,
  the overview still works.
- Optional: **identifier columns** (for example `image_id`), which are never offered
  as axes, and a **confidence column**. See [Confidence](#confidence).
- Comma-separated UTF-8 is expected. Semicolon- or tab-separated files and Latin-1
  text are detected, and the app notes that it did so.

```text
image_id,actual,predicted,confidence,brightness,blur_px,speed_kmh
img_00001,no_entry,speed_30,0.299,0.04,2.9,38
img_00002,no_entry,no_entry,0.789,0.61,6.7,35
```

A second example with different column names and deliberately messy values is in
`examples/churn_predictions.csv`. Upload it to see column mapping and the
validation messages.

## How the numbers are calculated

### Labels and excluded rows

- Cells are read as **text exactly as written**, so `01` is never converted to `1`.
- Labels are trimmed of surrounding whitespace, and nothing else changes. `Stop`
  and `stop`, or `1` and `1.0`, remain **different classes**. When labels like these
  appear, the app warns you rather than silently merging them.
- A label is **missing** if it is blank or one of `NA`, `N/A`, `n/a`, `NaN`, `nan`,
  `null`, `NULL`, `<NA>` or `#N/A`. `None` is *not* treated as missing, because it
  can be a real class name.
- A row missing either label cannot be judged. It is **excluded** from every number
  and counted in the exclusion notice and the "Excluded rows" tile.
- The remaining rows are the **evaluated rows**. Every row counts equally.

### Overall performance

- An **error** is any row where `actual ≠ predicted`.
- **Accuracy** = correct ÷ evaluated rows. The denominator is always shown.
- The **confusion matrix** uses every class that appears as an actual *or* a
  predicted label. Classes that are only ever predicted, or never predicted, are
  listed under it. It can show counts or be normalised by actual class (recall) or
  predicted class (precision).
- **Precision, recall, F1 and class balance** give macro and weighted averages,
  balanced accuracy and a per-class table (support, precision, recall, F1).
- **Regression** reports MAE, RMSE and R² over every numeric actual/predicted pair.
  R² is undefined for a constant actual target.

### The failure map

- **Axes.** A column can be an axis if at least 90% of its non-blank values are
  numbers and it has at least two distinct values. Label, identifier and confidence
  columns are never axes. The app lists every column it doesn't offer, with the
  reason.
- **Mapped rows.** A row with a blank or non-numeric value in either selected
  feature is left off the map. It still counts in the overall accuracy. The
  number omitted, and why, is shown above the map.
- **Ranges.** Each feature is split into equal-width ranges (4 by default,
  adjustable from 2 to 8) spanning the mapped rows' minimum to maximum. The sidebar
  also offers quantile ranges and custom boundaries, three colour palettes and
  sample-count overlays. **Filter map by class** restricts the map and its rows to
  chosen actual or predicted classes; overall metrics keep every row.
  - Edges are rounded to readable values, and the rounded edges are the ones rows
    are assigned by. A value printed on a boundary therefore lands where the labels
    say.
  - Decimal ranges include their lower edge and exclude their upper edge, except
    the last range, which includes the maximum. `[0.26, 0.49)` means
    0.26 ≤ x < 0.49.
  - Whole-number features get whole-number ranges written inclusively (`20–44`).
    If there are fewer distinct whole values than ranges, each value gets its own
    range.
  - A feature with a single value among the mapped rows can't be split, and the
    app says so.
- **Cells.** Each row is assigned its two ranges **once**. The heatmap, the detail
  table and the "Map as a table" view all read those same assignments.
  - **Error rate** = errors ÷ examples in the cell.
  - **Colour** is error rate on a fixed 0–100% scale, so a colour always means the
    same rate.
  - Each cell shows its rate and "errors of examples".
  - **Empty cells** are grey and say "no examples". They have no rate and are never
    shown as 0%.
  - Cells with fewer examples than the **small-sample threshold** (default 10,
    adjustable) are marked "small", not hidden.

### Inspecting a cell

- Select a cell by clicking it on the map or with the two range selectors. The
  summary states the ranges, the counts and the error rate. The table lists every
  source row in the cell, with its row number (1 = first row after the header),
  its result and all original columns.
- **Errors only** filters the table but never changes the selected cell or its
  headline counts.
- The selection is **cleared** when you load a different file, change the label
  columns, change either axis, or change the number of ranges. It is kept when you
  toggle "Errors only" or change the small-sample threshold.
- "Class mix in this cell" breaks the cell down by actual class, because a cell's
  rate can reflect which classes land in it.

### Confidence

Confidence means different things in different pipelines, so the app uses it only
after you confirm that the column is **the probability the model gave to the class
it predicted**, between 0 and 1. That rules out a positive-class probability and a
raw score. Once confirmed and validated, confidence appears as a bar in the tables,
and each cell reports the average confidence on wrong and correct predictions.
Values outside 0–1 are rejected with a message. Accuracy and the map never depend
on confidence. In training mode, confidence is the fitted model's probability for
its predicted class, when the model provides probabilities.

**Confidence and ROC-AUC** compares confidence on correct and incorrect predictions
and shows a calibration table by confidence bin.

### ROC-AUC

ROC-AUC needs class probabilities. These come from a probabilistic model in
training mode, or from confirmed uploaded columns (see
[the analyze workflow](#analyze-an-existing-prediction-csv)).

- **Binary:** the standard ROC-AUC for the second class in sorted order. It is the
  same under every averaging mode.
- **Multiclass:** choose **macro** or **weighted** averaging and **one-vs-rest** or
  **one-vs-one**. Weighted one-vs-rest weights each class by its actual support.
  Weighted one-vs-one (Hand & Till) weights each class pair by its prevalence. Both
  match scikit-learn's `roc_auc_score`.
- When ROC-AUC is unavailable, the app says why: no probabilities (for example an
  SVM on a grouped split, or a prediction CSV with no confirmed probability
  columns), invalid values, rows that do not sum to 1, or classes without a
  probability.

### Explanations

**Feature importance and row explanations** shows:

- **Model-based importance:** impurity importance for trees and forests, or mean
  absolute standardised coefficient for linear models. Models without built-in
  importance say so.
- **Permutation importance** on the held-out rows, for any model.
- **Exact row explanations:** coefficient × value contributions for uncalibrated
  linear models, and the decision path for a single decision tree. For other models
  the app states *"Local row explanations are not available for this model. Global
  feature importance is shown instead."* and computes permutation importance
  automatically when the model has no built-in importance. There is no SHAP
  dependency.

### Experiment history and comparison

Each fitted model is added to **Session experiment history** (the latest 10 runs in
this browser session). Each row shows: run, UTC timestamp, task, model, target, test
proportion, split method, group or time column, seed, CV method and folds, whether
it was tuned, imbalance strategy, primary metric, CV score, held-out score, row
counts, features and split ID.

You can download the table as CSV and every run's full metadata as JSON, or open one
run's metadata in the page. **Compare saved runs** shows a side-by-side table only
when the runs share the same split ID (same data fingerprint, task, target and
train/test rows). Otherwise it explains that their scores are not a controlled
comparison.

The dataset fingerprint is a SHA-256 of the column names, dtypes, index and shape
plus pandas' per-row hashes. It is computed once per upload, cached, and reused by
every model and rerun.

### Reports and downloads

Downloads include the held-out prediction CSV, training metadata JSON (seed, split
row IDs, fitted parameters, CV method and folds, imbalance strategy, probability
method, package versions and timestamp), the displayed confusion matrix, per-class
metrics, the map table and, after **Prepare evaluation report**, a ZIP bundle, an
HTML report and audit metadata. Report files reflect the active threshold, tolerance
and map filters.

### Consistency checks

The footer of the page re-checks, on the live data:
- correct + errors = evaluated rows;
- the confusion-matrix total = evaluated rows;
- every mapped row sits in exactly one cell;
- the cell counts and cell errors add up to the mapped rows and their errors;
- mapped + omitted = evaluated rows;
- for the selected cell, the rows listed match the counts on the map.

## Reading the map responsibly

- **A high error rate marks a group to investigate. It does not show that the
  feature causes the errors.** Features are often correlated; in the sample,
  speed looks risky only because it drives blur.
- **Small cells are noisy.** Two errors out of three examples is 67%, but it could
  easily be 33% with more data.
- **Class mix matters.** A cell full of an inherently hard class will look bad for
  that reason alone. Check "Class mix in this cell".
- Ranges are computed from the mapped rows. Changing the axes or the number of
  ranges redraws the boundaries, so compare cells within one map, not across maps.

## Assumptions and decisions

The project follows `Hackathon Planning Doc.md` and
`Model-Failure-Atlas-Build-Plan.md`. Where they left room:

- **Column mapping** rather than fixed column names, with name-based
  pre-selection so the demo needs no clicks.
- **Row identity** is the position in the file. An ID column, if present, is
  shown too.
- **Identifier detection** pre-fills columns whose name contains an ID word, or
  whose values are zero-padded codes or a 1..n counter. In existing-prediction mode,
  you can edit it under "Optional columns". Training mode excludes detected IDs.
- **Confidence** is suggested only for columns named like `confidence`. Names like
  `probability` or `score` are ambiguous (they may be positive-class scores).
- **Equal-width ranges** (4 × 4 by default), as both documents recommend.
  Quantiles and custom boundaries are optional.
- **Per-class map filtering** is available under **Filter map by class**. The
  class-mix table still covers the most important caveat for unfiltered maps.
- **Selection** uses two range selectors, as the documents recommend. Clicking a
  cell on the map fills the same selectors, so both paths use the same state.
- **Sequential colour scales** (blue by default), light to dark (flipped in dark
  mode), with labels on every cell and a table view, so no reading depends on
  colour alone.

## Limitations

- **No categorical encoding.** Training features must be numeric; text, categorical
  and date columns are listed as excluded rather than one-hot or ordinal encoded.
  Map axes are numeric too.
- Every row is weighted equally in the overall metrics. Skewed features can leave
  most equal-width ranges nearly empty; use quantile ranges for those.
- Leakage detection is heuristic. It catches copies, encodings, deterministic
  mappings, simple numeric transformations and suspicious names, but not leakage
  through combinations of features, aggregates computed over the whole dataset, or
  columns with innocent names recorded after the outcome. Findings are clues, not
  proof.
- Grouped splits cannot always cover every class. When groups align with classes the
  app warns rather than splitting groups. Time-ordered splits are never rebalanced.
- SVM and calibrated logistic-regression probabilities come from internal random
  calibration folds, so they are available for random splits only. Calibrated
  logistic regression is not offered for grouped or time-ordered splits.
- Exact row explanations cover uncalibrated linear models and single decision trees
  only; other models get global importance.
- Experiment history lives in the browser session (latest 10 runs) and is lost when
  the session ends; download it to keep it. Choosing between models by repeatedly
  looking at the same test results makes that test set less useful as an independent
  check. Prefer the cross-validation scores for selection.
- Target-type guidance uses value counts and ratios. It can misjudge unusual targets,
  so the app lets you confirm that values are classes.
- One cell can be selected at a time, and ranges are recalculated from scratch for
  each choice of axes. There is no saved state between sessions.
- Labels are compared exactly after trimming. Near-duplicates are reported, not
  fixed.
- A row with more fields than the header makes the whole file unreadable (with a
  message naming the line). Blank lines are skipped, so row numbers count data
  rows, not file lines.
- Analysis recalculates on each interaction; training runs only on an explicit
  button press and its result stays in the current session. Large datasets may be
  slow, especially with nearest neighbors, SVMs and hyperparameter search. The
  dataset fingerprint, quality summary and leakage review are cached per upload.
- The dark-mode colour flip follows the theme Streamlit reports. If you switch
  themes mid-session, the map may keep its previous colours until the page reruns.
- Uploads are limited to 25 MB (roughly 400,000 rows) so that a shared deployment
  stays responsive. Change `maxUploadSize` in `.streamlit/config.toml` to raise it.

## Sample data

`sample_predictions.csv` and `examples/churn_predictions.csv` are **synthetic**.
They are generated by `scripts/make_sample_data.py` with fixed random seeds, so
running `python scripts/make_sample_data.py` reproduces them exactly. No real
model, images or customers were involved.

In the sign data, each image's chance of being misclassified is set by a formula
that rises with darkness and blur, rises most when both are present, and rises a
little for small signs. Wrong predictions follow plausible confusions (for
example `speed_30` ↔ `speed_50`). Blur partly depends on speed, which is why speed
looks risky on the map without appearing in the formula. The generator also plants
8 unlabelled rows and 12 rows with no blur measurement, to show how exclusions are
reported.

## Deploying

The app is a standard Streamlit app with no secrets, databases or system packages,
so it runs on [Streamlit Community Cloud](https://share.streamlit.io) as-is:

1. Push `main` to GitHub (`git push origin main`).
2. Sign in at <https://share.streamlit.io> with your GitHub account and allow it to
   access the repository.
3. Choose **Create app**, then deploy from GitHub with:
   - Repository: `jpbanquicio16/2026-Hackathon`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: any free subdomain, for example `model-failure-atlas`
4. Under **Advanced settings**, choose Python **3.12** (the tested version). No
   secrets are needed.
5. Select **Deploy**. The first build installs `requirements.txt` (not
   `requirements-dev.txt`) and takes a few minutes.
6. Open the app URL in a private window and run through the
   [demo walkthrough](#demo-walkthrough-about-two-minutes) once.

After that, every push to `main` redeploys the app. The sample data ships in the
repository and `.streamlit/config.toml` is read automatically. Community Cloud puts
apps that get no traffic for a while to sleep, so open the link a few minutes before
presenting.

## Project structure

```text
app.py                      Streamlit page orchestration: workflows, role mapping, overview,
                            probabilities/thresholds, ROC-AUC, heatmap, selection, reports
analysis.py                 Loading, label cleaning, evaluation rows, overview, ranges, cells
training.py                 Splits, cross-validation, tuning, models, imbalance, calibration,
                            fingerprint, metadata, comparison/history tables, explanations
evaluation.py               Metrics: regression, per-class, normalised confusion, thresholds,
                            ROC-AUC averaging, uploaded probability validation
diagnostics.py              Target profile and guidance, class balance, dataset quality,
                            leakage detection
exports.py                  JSON-safe metadata, report files and ZIP bundle
ui_training.py              Training controls, quality/leakage display, experiment history,
                            model comparison and explanation UI
sample_predictions.csv      Synthetic demo data (traffic signs)
iris.csv                    Labelled Iris data for training mode
examples/
  churn_predictions.csv     Second synthetic dataset with different columns and messy values
  iris_heldout_predictions.csv  Held-out Iris predictions (see above)
scripts/
  make_sample_data.py       Regenerates the synthetic CSVs
  make_iris_evaluation.py   Regenerates the Iris held-out predictions
  benchmark_fingerprint.py  Old vs current dataset fingerprint timing
tests/
  test_analysis.py          Hand-worked fixture, ranges, validation, invariants
  test_heatmap.py           Structure of the chart specification
  test_app.py               Original user journeys via AppTest (selection resets, uploads)
  test_training.py          Reproducible split, export, preprocessing isolation and failures
  test_extended_training.py CV/tuning isolation, grouped/time splits, oversampling, regressors
  test_evaluation_features.py  Metrics, thresholds, quantile/custom ranges, confidence
                            bins, quality checks, report files
  test_leakage.py           Encoded targets, numeric transforms, tokenised names, non-leaks
  test_training_hardening.py  Grouped coverage, SVM probabilities, target guidance, class
                            weights, fingerprint, progress, history, explanation fallbacks
  test_probability_auc.py   Multiclass ROC-AUC modes vs scikit-learn, probability mapping
  test_ui_flows.py          AppTest: regression, comparison, thresholds, history, leakage,
                            target guidance, SVM ROC-AUC, probability mapping
requirements.txt            Runtime dependencies
requirements-dev.txt        Adds pytest
.streamlit/config.toml      Upload size limit
```
