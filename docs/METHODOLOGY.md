# Methodology

How [Model Failure Atlas](../README.md) calculates what it shows, how to read the map
responsibly, the interface choices behind it, and its limitations. For step-by-step use,
see the [user guide](USER_GUIDE.md).

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
  also offers quantile ranges and custom boundaries, five colour palettes and
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
  - Every populated cell has a **95% Wilson score interval** for its error rate.
    Empty cells have no interval. For regression this describes the above-tolerance
    fraction, not the mean absolute error. Interval width reflects sampling uncertainty.
  - **Elevated error evidence** requires the minimum sample count and a one-sided
    Fisher exact cell-versus-rest p-value below .05 after Holm correction across all
    populated cells in this map. A missing complement has no test. This assumes
    independent examples; repeated groups invalidate that assumption. Changing axes,
    filters or thresholds creates further comparisons not covered by this correction.
    Wilson intervals are marginal intervals, not simultaneous bounds across the map.
    These are exploratory diagnostics, not causal findings or a guarantee about future rows.

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
[the analyze workflow](USER_GUIDE.md#analyze-an-existing-prediction-csv)).

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

Each locked and evaluated model is added to **Session experiment history** (the latest 10 runs in
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

## Interface choices

Where there was more than one reasonable option:

- **Column mapping** rather than fixed column names, with name-based
  pre-selection so the demo needs no clicks.
- **Row identity** is the position in the file. An ID column, if present, is
  shown too.
- **Identifier detection** pre-fills columns whose name contains an ID word, or
  whose values are zero-padded codes or a 1..n counter. In existing-prediction mode,
  you can edit it under "Optional columns". Training mode excludes detected IDs.
- **Confidence** is suggested only for columns named like `confidence`. Names like
  `probability` or `score` are ambiguous (they may be positive-class scores).
- **Per-class map filtering** is available under **Filter map by class**. The
  class-mix table still covers the most important caveat for unfiltered maps.
- **Selection** uses two range selectors. Clicking a cell on the map fills the
  same selectors, so both paths use the same state.
- **Sequential colour scales** (café brown by default, running from the theme's
  Bone through Tan to Café Noir), light to dark (flipped in dark mode), with labels
  on every cell and a table view, so no reading depends on colour alone. Cell labels
  switch between dark and white text wherever white gives the better contrast.

Binning, split and validation choices are explained under
[Design decisions](../README.md#design-decisions) in the README.

## Limitations

- **Limited categorical encoding.** Repeated text categories with at most 50 values
  are offered for training. Fold-local imputation and one-hot encoding handle missing
  and unseen values. Dates/free text require explicit feature engineering. Axes remain numeric.
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
- Experiment history lives in the browser session (latest 10 revealed runs); download
  complete run archives for later replay. Candidates are ranked by training CV and
  only one locked candidate is evaluated. Repeat reveals on this dataset/target are
  marked exploratory, but reveals outside the current session cannot be tracked.
- Target-type guidance uses value counts and ratios. It can misjudge unusual targets,
  so the app lets you confirm that values are classes.
- One cell can be selected at a time, and ranges are recalculated from scratch for
  each choice of axes. Downloaded reports preserve the view; replay starts from the
  original model predictions so threshold experiments are never confused with fitting.
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
