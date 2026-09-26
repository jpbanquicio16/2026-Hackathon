# AI Error Heatmap

**The model has an overall score, but which kinds of examples does it get wrong?**

Model Failure Atlas is a Streamlit app for exploring classification errors. Choose
one of two workflows at the top of the page: analyze existing predictions or train
a classifier and generate held-out predictions. Then pick two numeric features. The app
reports overall performance, splits both features into ranges, and colours each
combination of ranges by its error rate. Select a cell to see the exact rows
behind its numbers.

Both workflows use the same calculations, heatmap and row inspection. Training mode
labels all results **Held-out test performance** and never substitutes training
performance for a test result.

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

### Train a model and evaluate it

1. Select **Train a model and evaluate it**, then upload a **Labelled CSV**.
   The file needs a target class and at least one numeric feature. For Iris, upload
   `iris.csv`, choose `species`, and retain the four measurement features.
2. Choose the features, test-set proportion (10–50%), random seed and classifier.
   The defaults are 20%, seed 42, and 3-nearest neighbors.
3. Select **Train and evaluate**. The app reports training and held-out row counts,
   classifier and seed. All performance metrics and map rows use only the test set.
4. Inspect a map cell. Use **Download held-out prediction CSV** to export the
   original selected features, retained identifiers, `source_row_id`, actual labels
   and predictions. Upload this CSV in the first mode to reproduce the analysis.
   The generated label names are normally `actual_label` and `predicted_label`;
   a suffix is added if a source column already uses either name. Check the label
   mapping after re-upload when source names conflict.

**Data preparation and splitting:**

- The target and detected ID columns cannot be training features. Detection uses
  ID names, leading-zero numeric codes and row-counter patterns. Other leakage
  columns (for example measurements taken after the outcome) must be deselected
  by the user.
- Initial training support is numeric only. A candidate needs at least 90% finite
  numbers among its non-missing values and more than one distinct value. Text,
  categorical and date strings are not encoded; excluded columns are listed.
- Target labels use the existing whitespace/missing-value rules. Missing targets
  are counted and removed before splitting; numeric class codes remain text labels.
- A random stratified split is made **before fitting any preprocessing**. Each
  class needs at least two rows, and both sets must contain all classes. If class
  counts, the proportion or the classifier make this impossible, the app explains
  the problem and shows no performance results. There is no unstratified or
  training-set fallback.
- Missing, invalid and infinite feature values are filled using training-set
  medians; scaling is fitted on training rows too. A feature with no usable values
  in the training set is rejected. The three choices are **3-nearest neighbors**
  (`n_neighbors=3`), **Logistic regression** (`max_iter=2000`) and **Decision tree**
  (`max_depth=5`). All share the imputer/scaler pipeline; stochastic estimators
  receive the selected seed.
- Exported feature values remain as uploaded, so missing map-axis values still
  cause map omissions even though the model can predict after imputation. Those
  rows remain in test accuracy. `source_row_id` is the 1-based data-row position in
  the labelled input; the detail table's `row` is the position in the exported CSV.
- Changing training settings removes old results until **Train and evaluate** is
  selected again. Inspecting cells or changing map settings reuses the fitted result.

With the checked-in Iris data and default settings, the split is 120 training and
30 test rows (10 per species). There are **28 correct and 2 incorrect predictions,
93.3% accuracy**. Source rows 135 and 139 are virginica predicted as versicolor.
The tests check the exported CSV independently, all 16 default map cells against
their inspected rows, round-trip upload, and isolation of preprocessing from test
values. Reproduction was verified with Python 3.12, NumPy 1.26.4 and scikit-learn 1.9.0.

Run the tests with:

```bash
pip install -r requirements-dev.txt
pytest
```

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
  listed under it.

### The failure map

- **Axes.** A column can be an axis if at least 90% of its non-blank values are
  numbers and it has at least two distinct values. Label, identifier and confidence
  columns are never axes. The app lists every column it doesn't offer, with the
  reason.
- **Mapped rows.** A row with a blank or non-numeric value in either selected
  feature is left off the map. It still counts in the overall accuracy. The
  number omitted, and why, is shown above the map.
- **Ranges.** Each feature is split into equal-width ranges (4 by default,
  adjustable from 2 to 8) spanning the mapped rows' minimum to maximum.
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
on confidence.

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
- **Per-class filtering** is not included. The planning doc lists it as "if time
  permits"; the build plan defers per-class views until after the hackathon. The
  class-mix table covers the most important caveat instead.
- **Selection** uses two range selectors, as the documents recommend. Clicking a
  cell on the map fills the same selectors, so both paths use the same state.
- **Sequential blue colour scale**, light to dark (flipped in dark mode), with
  labels on every cell and a table view, so no reading depends on colour alone.

## Limitations

- Classification only. Every row is weighted equally; per-class metrics
  (precision, recall) are not computed.
- Numeric features only, split into equal-width ranges. Skewed features can leave
  most rows in one range; categorical features can't be axes yet.
- Training also supports only numeric features. It is a baseline classification
  workflow, with no regression, categorical encoding, cross-validation, tuning,
  group-aware splitting or time-aware splitting. Use independently sampled rows;
  repeated entities and time-ordered data need an external split. Repeatedly tuning
  against the same test results makes that set less useful as an independent check.
- One cell can be selected at a time, and ranges are recalculated from scratch for
  each choice of axes. There is no saved state between sessions.
- Labels are compared exactly after trimming. Near-duplicates are reported, not
  fixed.
- A row with more fields than the header makes the whole file unreadable (with a
  message naming the line). Blank lines are skipped, so row numbers count data
  rows, not file lines.
- Analysis recalculates on each interaction; training runs only on an explicit
  button press and its result stays in the current session. Large datasets may be
  slow, especially with nearest neighbors.
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
app.py                      Streamlit page: layout, widgets, heatmap, selection state
analysis.py                 All calculations: loading, validation, metrics, ranges, cells
training.py                 Split, training-only preprocessing, classifiers, held-out export
sample_predictions.csv      Synthetic demo data (traffic signs)
examples/
  churn_predictions.csv     Second synthetic dataset with different columns and messy values
scripts/
  make_sample_data.py       Regenerates both CSVs
tests/
  test_analysis.py          Hand-worked fixture, ranges, validation, invariants
  test_heatmap.py           Structure of the chart specification
  test_app.py               User journey via Streamlit's AppTest (selection resets, uploads)
  test_training.py          Reproducible split, export, preprocessing isolation and failures
requirements.txt            Runtime dependencies
requirements-dev.txt        Adds pytest
.streamlit/config.toml      Upload size limit
```
