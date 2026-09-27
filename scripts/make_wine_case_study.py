"""Reproduce the real-data study; no network access or test-driven model selection."""

import csv
import hashlib
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis
import experiments
import exports
import training

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "winequality-white.csv"
SEED, TEST_SIZE, FOLDS = 42, .2, 5
MODELS = ("Logistic regression", "Random forest")
AXES = ("alcohol", "volatile acidity")
SOURCE_URL = "https://archive.ics.uci.edu/dataset/186/wine+quality"


def prepare():
    source = analysis.load_csv(SOURCE.read_bytes()).frame
    features = tuple(c for c in source if c != "quality")
    # Exact full-row repetitions are redundant. Remaining equal feature profiles stay
    # together even when their sensory labels differ; they never cross split/fold boundaries.
    unique = source.drop_duplicates().copy()
    frame = unique.loc[:, list(features)].copy()
    frame.insert(0, "example_id", [f"white_wine_{i:04d}" for i in unique.index])
    frame["sample_group"] = [hashlib.sha256("|".join(row).encode()).hexdigest()[:16] for row in frame[list(features)].itertuples(index=False, name=None)]
    frame["target"] = np.where(unique.quality.astype(int) >= 7, "quality_7_plus", "quality_below_7")
    # The app reads this exact CSV; row IDs in the run refer to this prepared file.
    frame = analysis.load_csv(frame.to_csv(index=False).encode()).frame
    return source, frame, features


def main():
    source, frame, features = prepare()
    output = ROOT / "examples" / "wine_case_study"
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / "examples" / "wine_training.csv").write_text(frame.to_csv(index=False))
    candidates = training.compare_models(frame, "target", features, MODELS, test_size=TEST_SIZE,
                                         seed=SEED, cv_folds=FOLDS, split_method="Grouped", split_column="sample_group",
                                         evaluate_test=False)
    ranking = training.validation_ranking(candidates)
    chosen = next(r for r in candidates if r.classifier == ranking.iloc[0]["model"])
    assert all(not r.metrics and r.frame.empty for r in candidates)
    result = training.reveal_test(chosen, frame)
    result.metadata.update(source_file="wine_training.csv", run_id="wine-quality-42",
                           selection={"method": "training-fold CV", "candidates": ranking.to_dict(orient="records"),
                                      "previous_test_reveals_for_dataset_target": 0, "exploratory_test_reuse": False})
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(result.y_train), 1)), result.y_train)
    baseline_predictions = baseline.predict(np.zeros((len(result.y_test), 1)))
    baseline_score = float(balanced_accuracy_score(result.y_test, baseline_predictions))
    evaluation = analysis.evaluate(result.frame, result.actual, result.predicted)
    fmap = analysis.build_failure_map(result.frame, evaluation.rows, *AXES, n_bins=4, method="Quantiles")
    chosen_cell = fmap.cells[fmap.cells.total >= 20].sort_values(["error_rate", "total"], ascending=False).iloc[0]
    selected = (int(chosen_cell.x_bin), int(chosen_cell.y_bin))
    files = exports.evaluation_files(result.frame, evaluation, result.metadata, fmap, selected)
    # An independent audit reads serialized predictions and bins directly, without
    # evaluate(), build_failure_map() or cell_index() to compute the expected counts.
    records = list(csv.DictReader(io.StringIO(result.csv_bytes().decode())))
    confusion = Counter((r[result.actual], r[result.predicted]) for r in records)
    correct = sum(r[result.actual] == r[result.predicted] for r in records)
    assignments = {}
    for x in range(len(fmap.x_bins)):
        for y in range(len(fmap.y_bins)):
            def inside(value, edges, i):
                return edges[i] <= value and (value < edges[i+1] or i == len(edges)-2 and value == edges[i+1])
            rows = [r for r in records if inside(float(r[AXES[0]]), fmap.x_bins.edges, x) and inside(float(r[AXES[1]]), fmap.y_bins.edges, y)]
            errors = sum(r[result.actual] != r[result.predicted] for r in rows)
            cell = fmap.cell(x, y)
            assert cell.total == len(rows) and cell.errors == errors
            expected_ids = [r["example_id"] for r in rows]
            assert expected_ids == result.frame.loc[fmap.cell_index(x, y), "example_id"].tolist()
            assignments[f"{x},{y}"] = {"count": len(rows), "errors": errors, "example_ids": expected_ids}
    assert correct / len(records) == result.metrics["accuracy"]
    assert not set(frame.loc[list(result.train_rows), "sample_group"]) & set(frame.loc[list(result.test_rows), "sample_group"])
    audit = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "source_rows": len(source),
             "exact_duplicates_removed": len(source)-len(frame), "prepared_rows": len(frame),
             "evaluated": len(records), "correct": correct, "incorrect": len(records)-correct,
             "accuracy": correct/len(records), "confusion": [{"actual": a, "predicted": p, "count": n} for (a, p), n in sorted(confusion.items())],
             "baseline_balanced_accuracy": baseline_score, "selected_cell": selected, "cells": assignments,
             "checks": {"export_metrics_match": True, "all_cell_counts_match_inspection": True, "train_test_groups_disjoint": True}}
    files["independent_audit.json"] = exports.json_bytes(audit)
    files["validation_candidates.csv"] = ranking.to_csv(index=False).encode()
    for name, data in files.items():
        (output / name).write_bytes(data)
    (ROOT / "examples" / "wine_heldout_predictions.csv").write_bytes(result.csv_bytes())
    (output / "training_metadata.json").write_bytes(result.metadata_bytes())
    # The archive is a reproducible download, not an executable pickled model.
    (output / "wine_quality.atlas.zip").write_bytes(experiments.create_archive(result, files))
    text = f"""# Real-data case study: where white-wine predictions fail

## Question and source

Can physicochemical measurements distinguish wines with a reported sensory quality
score of **7 or more**? This binary threshold was fixed before fitting. It is an
illustrative screening task; it does not measure wine prices or identify causes of quality.

Source: [UCI Wine Quality]({SOURCE_URL}), Cortez, Cerdeira, Almeida, Matos and Reis (2009),
DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The white-wine file contains
{len(source):,} rows, eleven physicochemical features, and the sensory quality target.
The original semicolon-separated file is in `data/winequality-white.csv`.

## Protocol fixed before examining test errors

* Remove {len(source)-len(frame):,} exact repeated full rows; {len(frame):,} remain.
* Preserve source identity with `example_id`. Keep identical feature profiles in the
  same `sample_group`, including profiles with different reported quality scores.
* Convert quality to `quality_7_plus` / `quality_below_7`; exclude original quality,
  example ID and group ID from the eleven training features.
* Seed **{SEED}**, **{TEST_SIZE:.0%} of groups** held out. This produces
  **{len(result.train_rows):,} training / {len(result.test_rows):,} test rows**.
* Compare logistic regression and a 100-tree random forest using **{FOLDS}-fold
  stratified group cross-validation on training rows**, ranked by balanced accuracy.
  No parameter search. Fold-local imputation and scaling; no class rebalancing.
* Lock **{result.classifier}** using validation scores, then evaluate it once on test
  rows. The other candidate's test result is never calculated. A majority-class
  dummy baseline uses the same training/test partition.
* Inspect alcohol × volatile acidity with four quantile bins per axis. These axes
  were specified before scoring. The highlighted cell below was selected after
  observing the map and is an exploratory finding.

## Results

| Candidate | Training CV balanced accuracy | Fold standard deviation |
| --- | ---: | ---: |
"""
    for row in ranking.to_dict(orient="records"):
        text += f"| {row['model']} | {row['CV mean']:.4f} | {row['CV std']:.4f} |\n"
    text += f"""
The selected model's **held-out balanced accuracy is {result.metrics['balanced_accuracy']:.4f}**,
versus **{baseline_score:.4f}** for the majority-class baseline. Its accuracy is
**{result.metrics['accuracy']:.2%}** ({correct:,} correct / {len(records):,} evaluated),
and macro F1 is **{result.metrics['macro_f1']:.4f}**. Fold standard deviations are
not confidence intervals. See the exported per-class table for minority-class recall.

## Failure-map finding and decision

The highest-error cell with at least 20 examples is **alcohol
{fmap.x_bins.labels[selected[0]]}** × **volatile acidity
{fmap.y_bins.labels[selected[1]]}**. It has **{int(chosen_cell.errors)} errors in
{int(chosen_cell.total)} rows ({chosen_cell.error_rate:.1%})**, compared with
**{fmap.rows.is_error.mean():.1%}** across mapped rows. Its descriptive 95% Wilson
interval is **{chosen_cell.error_ci_low:.1%}–{chosen_cell.error_ci_high:.1%}**.
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
Source SHA-256: `{audit['source_sha256']}`.
"""
    (ROOT / "docs" / "CASE_STUDY.md").write_text(text)
    print({"model": result.classifier, **result.metrics, "rows": len(frame), "test": len(result.test_rows)})


if __name__ == "__main__":
    main()
