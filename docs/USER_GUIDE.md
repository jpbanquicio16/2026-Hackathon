# User guide

How to use [Model Failure Atlas](../README.md): both workflows, the checks that run before
training, what an uploaded CSV needs, and the bundled sample data. How each number is
calculated is described in the [methodology notes](METHODOLOGY.md).

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
   then upload a **Labelled CSV**. The file needs a target and at least one usable
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
5. Select **Fit and compare on training data**. A progress bar reports each validation
   fit. Compare CV scores, then choose **Lock model and reveal held-out test performance**.
   Only the selected model generates test predictions. The app reports row counts,
   model, seed and split; all map rows and test metrics use the held-out set.
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

Every model uses the same fold-local preprocessing: numeric features receive median
imputation and standard scaling; categorical features receive most-frequent imputation
and one-hot encoding. Stochastic estimators receive the selected seed.
**Hyperparameter search** (grid search, scored by balanced accuracy for
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
- **Numeric and categorical features.** Numeric columns need at least 90% finite
  numbers among non-missing values. Repeated text categories with at most 50 values
  are offered; numeric codes can be selected under **Treat these features as categories**.
  Numeric imputation/scaling and categorical most-frequent imputation/one-hot encoding
  are fitted inside each fold. Unknown test categories yield all-zero indicators;
  labels such as `01` and `1` stay distinct. Dates and free text require feature engineering.
- Target labels use the existing whitespace/missing-value rules. Missing targets
  are counted and removed before splitting; numeric class codes remain text labels.
- Exported feature values remain as uploaded, so missing map-axis values still
  cause map omissions even though the model can predict after imputation. Those
  rows remain in test accuracy. `source_row_id` is the 1-based data-row position in
  the labelled input; the detail table's `row` is the position in the exported CSV.
- Changing training settings removes old results until candidates are fitted and a
  model is locked again. Inspecting cells reuses the fitted result. A repeat reveal
  for the same dataset/target in this session is marked as exploratory test reuse.
  Restoring a saved result also counts as exposure before any later model choices.

After **Prepare evaluation report**, download the reproducible run archive. It
contains the full source dataset, recipe, predictions and displayed reports. To
resume later, choose **Restore a saved run** as the training source. Exact replay
checks the calculation code, Python major/minor version, ML package versions, split
row IDs and prediction CSV. Environment changes require an explicit exploratory replay.

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

## Existing-prediction CSV requirements

- A header row, then **one row per evaluated example** (one prediction per row).
- **An actual-label column and a predicted-label column.** They can have any names:
  you choose them in the app. Names such as `actual`/`predicted`, `label`/`prediction`
  and `y_true`/`y_pred` are pre-selected.
- **At least two numeric feature columns** to use as the map's axes. Without them,
  the overview still works.
- Optional: **identifier columns** (for example `image_id`), which are never offered
  as axes, and a **confidence column**. See [Confidence](METHODOLOGY.md#confidence).
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

### Iris

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
