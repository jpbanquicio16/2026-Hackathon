# Portfolio tasks 1–7: verification record

Verified locally on 27 September 2026 with Python 3.12.13 and the pinned dependencies.
The suite, scripts and native Safari checks exercise the working application; a public
Community Cloud deployment is still pending authentication.

| Check | Status | Expected | Observed |
| --- | --- | --- | --- |
| Full pytest suite | Pass | No failures | 244 passed |
| Lint, dependency consistency and whitespace | Pass | No errors | Ruff, `pip check` and `git diff --check` passed |
| Commit hooks, including new files | Pass | All configured hooks pass | File hygiene, YAML/TOML, file size, merge conflict, private-key and Ruff checks passed |
| Dependency lock regeneration | Pass | Identical output from the pinned environment | Both lock files unchanged |
| Wine script and independent audit | Pass | 793 test rows, 656 correct, 137 errors | Exact match; accuracy 82.7238%, balanced accuracy 0.663886 |
| Wine confusion matrix | Pass | Actual rows / predicted columns: `[[63, 100], [37, 593]]`, class order quality ≥7, quality <7 | Export and independent CSV calculation match |
| Wine map and row membership | Pass | Every cell's counts, errors and IDs agree with a separate CSV/boundary loop | All 16 cells match; no train/test group overlap |
| Wine browser metrics | Pass | 793 evaluated, 656 correct, 137 errors, 82.7% displayed accuracy | Exact match in existing-predictions mode |
| Wine browser slice | Pass | Alcohol [11.4,14.2], volatile acidity [0.27,0.33): 54 rows, 20 errors, 37.0%, Wilson interval 25.4–50.4% | Exact match; first 12 displayed example IDs match the audit |
| Deferred test scoring | Pass | No test predictions or scores before locking; only the chosen fitted model evaluates test rows | Prediction-call test and browser flow match |
| Numeric and categorical preprocessing | Pass | Fold-local fitting, explicit categories, unseen test categories ignored | Classification/regression tests pass; `01`, `1`, `1.0` remain distinct |
| Mixed-feature browser training | Pass | Iris plus repeated `site` category: 120 train, 30 test, seed 42, 3-nearest neighbors | 28 correct, 2 errors, 93.3% accuracy; no test metrics before locking |
| Mixed-feature browser cell download | Pass | X [4.4,5.25), Y [2.3,2.925): 2 rows, 0 errors | Download matches independently filtered predictions; source IDs 58 and 94 |
| Browser archive download and restore | Pass | Refit reproduces split IDs and prediction bytes | Browser confirms exact reproduction; downloaded archive also replays independently |
| Wine CLI replay | Pass | Exact predictions and unchanged environment | `exact_predictions: true`, no environment changes |
| Restored test exposure | Pass | Viewing a restored test result makes later choices exploratory | Fresh-session restore → fit → reveal test passes; reuse warning shown |
| Malformed archives | Pass | Clear data error instead of app exception | Invalid manifest fields and missing package versions rejected; corruption and environment-change checks pass |
| Cell uncertainty | Pass | Known Wilson bounds, empty cells have no interval, tiny cells are not declared hotspots | Focused tests pass; Holm-adjusted comparisons exposed with caveats |
| Benchmark correctness smoke | Pass | 1,000 rows, consistent map totals, app renders without exceptions | Passed; full local 1k/10k/50k timings in `benchmarks.md` |
| Hosted CI on Linux/macOS | Unable to verify | Both clean-install matrix jobs pass on the published changes | Workflow configured; execution not yet confirmed |
| Public deployment | Unable to verify | Updated public app loads and both workflows work | Community Cloud still requires user sign-in and Terms of Service acceptance |

The wine audit, including all selected-cell IDs, is in
[`examples/wine_case_study/independent_audit.json`](../examples/wine_case_study/independent_audit.json).
Browser checks cover the bundled wine predictions and a CSV upload through mixed-feature
training, cell inspection, CSV/archive download and archive restoration. Automated
AppTest journeys additionally check selection resets and the other existing workflows.

The mixed-feature browser fixture was made from the checked-in Iris CSV by adding
`site = ["north", "south", ""] * 50`. All four measurements and `site` were selected;
the category was one-hot encoded. Settings were random stratified 20% test, seed 42,
three-fold training CV and 3-nearest neighbors. This fixture tests the interface and
preprocessing; the artificial category has no biological interpretation.

## Fixes made while finishing the handoff

- Validated archive structure before reading recipe fields or package versions.
- Counted restored test results as session exposure before subsequent model choices.
- Corrected upload guidance for categorical inputs and saved ZIP archives.
- Made generated JSON and HTML end with a newline, so regeneration passes file hooks.
- Corrected the user guide's old description of numeric-only preprocessing.

## Design limits

- Persistence uses downloadable archives, not a shared server database. Each archive
  contains the full input dataset. Hashes check integrity, not the author's identity.
- Exact replay requires matching calculation code, Python major/minor and ML packages.
  Changed environments need explicit exploratory replay; prediction differences are reported.
- Test exposure is tracked in the session and retained in exported metadata; independent
  sessions and external experiments cannot be tracked globally.
- Maps still need two numeric axes. Categories support prediction, not categorical axes.
- Wilson/Fisher calculations assume independent observations. Holm correction covers
  cells in one map, not repeated searches across axes, filters or thresholds.
- Benchmarks measure local server work, not browser painting, network latency, hosted
  cold starts or concurrent users. They are measurements rather than service guarantees.

## Re-run the checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python -m pip check
.venv/bin/pre-commit run --all-files
.venv/bin/python scripts/make_wine_case_study.py
.venv/bin/python scripts/reproduce_run.py examples/wine_case_study/wine_quality.atlas.zip --output /tmp/atlas-replay
.venv/bin/python scripts/benchmark_atlas.py --sizes 1000 --repeats 1 --output /tmp/atlas-benchmark.json
git diff --check
```
