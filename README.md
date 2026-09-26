# AI Error Heatmap

**The model has an overall score, but which kinds of examples does it get wrong?**

Model Failure Atlas is a Streamlit app for exploring classification errors. Load a
CSV of predictions a model has already made (one row per example), choose the
actual-label and predicted-label columns, and pick two numeric features. The app
reports overall performance, splits both features into ranges, and colours each
combination of ranges by its error rate. Select a cell to see the exact rows
behind its numbers.

The app only **analyses existing predictions**. It does not train, load or run a
model.

## Quick start

Requires Python 3.10 or newer (tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501> with the sample dataset loaded. To use
your own data, choose **Upload a CSV** in the sidebar.

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

## CSV requirements

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
  whose values are zero-padded codes or a 1..n counter. It is only a default; you
  can edit it under "Optional columns".
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
- One cell can be selected at a time, and ranges are recalculated from scratch for
  each choice of axes. There is no saved state between sessions.
- Labels are compared exactly after trimming. Near-duplicates are reported, not
  fixed.
- A row with more fields than the header makes the whole file unreadable (with a
  message naming the line). Blank lines are skipped, so row numbers count data
  rows, not file lines.
- The whole page recalculates on every interaction. That takes about half a second
  for 100,000 rows × 12 columns; much larger files will feel slow.
- The dark-mode colour flip follows the theme Streamlit reports. If you switch
  themes mid-session, the map may keep its previous colours until the page reruns.

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

The app is a standard Streamlit app. For example, on Streamlit Community Cloud:
push the repository to GitHub, create an app pointing at `app.py`, and it installs
`requirements.txt`. Open the deployed URL in a private window and run through the
demo once, to check it from a fresh session.

## Project structure

```text
app.py                      Streamlit page: layout, widgets, heatmap, selection state
analysis.py                 All calculations: loading, validation, metrics, ranges, cells
sample_predictions.csv      Synthetic demo data (traffic signs)
examples/
  churn_predictions.csv     Second synthetic dataset with different columns and messy values
scripts/
  make_sample_data.py       Regenerates both CSVs
tests/
  test_analysis.py          Hand-worked fixture, ranges, validation, invariants
  test_heatmap.py           Structure of the chart specification
  test_app.py               User journey via Streamlit's AppTest (selection resets, uploads)
requirements.txt            Runtime dependencies
requirements-dev.txt        Adds pytest
```
