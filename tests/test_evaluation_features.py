import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import analysis as A
import diagnostics as D
import evaluation as E
import exports as X


def test_classification_metrics_and_normalised_empty_denominators():
    raw = A.load_csv(b"actual,predicted\na,a\na,a\nb,a\nb,b\n").frame
    evaluated = A.evaluate(raw, "actual", "predicted")
    metrics, table = E.classification_metrics(evaluated.rows)
    assert metrics["accuracy"] == .75
    assert metrics["balanced_accuracy"] == .75
    assert metrics["macro_precision"] == pytest.approx((2 / 3 + 1) / 2)
    assert table["support"].tolist() == [2, 2]
    matrix = A.compute_overview(evaluated.rows).confusion.reindex(index=["a", "b", "c"], columns=["a", "b", "c"], fill_value=0)
    row_percent = E.normalise_confusion(matrix, "Actual class (%)")
    col_percent = E.normalise_confusion(matrix, "Predicted class (%)")
    assert row_percent.loc["b"].tolist() == [50, 50, 0]
    assert row_percent.loc["c"].isna().all()
    assert col_percent["c"].isna().all()
    assert col_percent.loc["a", "a"] == pytest.approx(200 / 3)


def test_threshold_counts_auc_and_missing_probability_behavior():
    raw = A.load_csv(b"actual,predicted,p\nN,N,0.1\nP,P,0.9\nP,N,0.4\nN,P,0.8\nP,N,invalid\n").frame
    changed, invalid = E.apply_threshold(raw, "predicted", "p", "P", "N", .3)
    assert invalid == 1
    assert changed["predicted"].tolist() == ["N", "P", "P", "P", ""]
    assert changed["original_prediction"].tolist() == raw["predicted"].tolist()
    evaluated = A.evaluate(changed, "actual", "predicted")
    assert evaluated.n_evaluated == 4 and evaluated.n_excluded == 1
    assert E.classification_metrics(evaluated.rows)[0]["accuracy"] == .75
    p = A.parse_numeric(changed["p"]).values
    auc, _ = E.probability_auc(evaluated.rows, pd.DataFrame({"P": p, "N": 1 - p}))
    assert auc == .75


def test_regression_metrics_exclusions_and_map_are_independent_of_tolerance():
    raw = A.load_csv(b"actual,predicted,x,y\n1,1.5,.1,.2\n2,1,.4,.5\n3,4,.9,.9\nno,4,.8,.8\n").frame
    evaluated = E.regression_evaluation(raw, "actual", "predicted", .5)
    assert evaluated.n_evaluated == 3 and evaluated.n_excluded == 1
    metrics = E.regression_metrics(evaluated.rows)
    assert metrics["mae"] == pytest.approx(2.5 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(.75))
    assert metrics["r2"] == pytest.approx(-.125)
    assert metrics["above_tolerance"] == 2
    other = E.regression_metrics(E.regression_evaluation(raw, "actual", "predicted", 2).rows)
    assert other["mae"] == metrics["mae"] and other["above_tolerance"] == 0
    fmap = A.build_failure_map(raw, evaluated.rows, "x", "y", 2)
    assert fmap.cells["total"].sum() == 3 and fmap.cells["errors"].sum() == 2
    for cell in fmap.cells.itertuples():
        rows = evaluated.rows.loc[fmap.cell_index(cell.x_bin, cell.y_bin)]
        assert len(rows) == cell.total
        if len(rows):
            assert cell.mae == rows["absolute_error"].mean()
        else:
            assert pd.isna(cell.mae)


def test_quantile_ties_boundaries_and_custom_coverage():
    bins = A.flexible_bins([0, 0, 0, 0, 1, 1], 4, "Quantiles")
    assert len(bins) < 4
    assigned = bins.assign([0, 1])
    assert assigned.tolist() == [0, len(bins) - 1]
    custom = A.flexible_bins([0, 1, 2], 4, "Custom boundaries", (0, 1, 2))
    assert custom.assign([0, np.nextafter(1, 0), 1, 2]).tolist() == [0, 0, 1, 1]
    for edges in ((1, 2), (0, 0, 2), (0, float("nan"), 2), (0, float("inf"))):
        with pytest.raises(A.DataError):
            A.flexible_bins([0, 1, 2], 4, "Custom boundaries", edges)


def test_confidence_bins_keep_one_and_do_not_invent_empty_rates():
    confidence = pd.Series([0, 1, .99, np.nan, -1])
    bins = E.confidence_bins(confidence, pd.Series([False, False, True, True, True]))
    assert bins["examples"].sum() == 3
    assert bins.iloc[-1]["examples"] == 2 and bins.iloc[-1]["accuracy"] == .5
    assert bins.loc[bins["examples"] == 0, "accuracy"].isna().all()


def test_quality_target_guidance_and_leakage_are_diagnostic():
    raw = pd.DataFrame({"quality": [str(i % 3) for i in range(60)], "sulphates": [str(i / 10) for i in range(60)], "prediction": [str(i % 3) for i in range(60)], "id": [f"id_{i}" for i in range(60)]})
    assert D.target_suggestion(raw) == "quality"
    assert not D.target_warnings(raw, "quality", "classification")
    assert any("Regression" in warning for warning in D.target_warnings(raw, "sulphates", "classification"))
    clues = D.leakage_warnings(raw, "quality", ("prediction", "sulphates"))
    assert "98% identical" in clues.iloc[0]["review reason"]
    summary = D.dataset_summary(raw).set_index("column")
    assert summary.loc["id", "possible identifier"]
    assert summary.loc["quality", "unique"] == 3


def test_report_matches_active_rows_and_escapes_csv_html():
    raw = A.load_csv(b"actual,predicted,x,y\n<script>,<script>,0.1,0.1\nb,<script>,0.2,0.2\nb,b,0.9,0.9\n").frame
    evaluated = A.evaluate(raw, "actual", "predicted")
    fmap = A.build_failure_map(raw, evaluated.rows, "x", "y", 2)
    files = X.evaluation_files(raw, evaluated, {"source_file": "<script>alert(1)</script>"}, fmap, (0, 0))
    meta = json.loads(files["audit_metadata.json"])
    assert meta["active_metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert meta["map"]["mapped_rows"] == 3
    assert meta["selected_cell"]["row_ids"] == [1, 2]
    assert "<script>" not in files["evaluation_report.html"].decode()
    assert "&lt;script&gt;" in files["evaluation_report.html"].decode()
    with zipfile.ZipFile(io.BytesIO(X.bundle(files))) as archive:
        assert {"predictions.csv", "audit_metadata.json", "evaluation_report.html", "map_cells.csv", "map_row_assignments.csv", "selected_cell_rows.csv"} <= set(archive.namelist())
        assert len(pd.read_csv(archive.open("predictions.csv"))) == 3
