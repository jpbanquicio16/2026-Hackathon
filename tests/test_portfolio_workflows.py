"""Contracts for model selection, mixed features, uncertainty and portable runs."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

import analysis
import experiments
import exports
import training
import uncertainty

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")


@pytest.fixture
def iris():
    return analysis.load_csv((ROOT / "iris.csv").read_bytes()).frame


def test_test_predictions_do_not_exist_until_lock(iris, monkeypatch):
    calls = []
    predict = LogisticRegression.predict

    def tracked(self, X):
        calls.append(len(X))
        return predict(self, X)

    monkeypatch.setattr(LogisticRegression, "predict", tracked)
    candidate = training.train_and_evaluate(iris, "species", FEATURES, classifier="Logistic regression", evaluate_test=False)
    assert candidate.frame.empty and not candidate.metrics
    assert calls == [120]  # training diagnostics only; no prediction on the 30 test rows
    assert "test accuracy" not in training.validation_ranking([candidate])
    with pytest.raises(analysis.DataError, match="reveal"):
        experiments.create_archive(candidate)
    result = training.reveal_test(candidate, iris)
    assert calls == [120, 30]
    assert result.model is candidate.model and len(result.frame) == 30
    assert not candidate.metadata["test_revealed"]  # the hidden candidate is not mutated
    changed = iris.copy()
    changed.iloc[0, 0] = "777"
    with pytest.raises(analysis.DataError, match="differs"):
        training.reveal_test(candidate, changed)


@pytest.mark.parametrize("task,model,target", [("classification", "Logistic regression", "species"), ("regression", "Ridge regression", "petal_length")])
def test_categories_are_fitted_on_train_only_with_unknown_test_values(iris, task, model, target):
    iris["colour"] = ["red", "blue", ""] * 50
    features = ("sepal_length", "sepal_width", "colour")
    first = training.train_and_evaluate(iris, target, features, classifier=model, task=task)
    changed = iris.copy()
    changed.loc[list(first.test_rows), "colour"] = "never_seen_in_training"
    result = training.train_and_evaluate(changed, target, features, classifier=model, task=task)
    cat = result.model["preprocessor"].named_transformers_["categorical"]
    assert "never_seen_in_training" not in cat["encoder"].categories_[0]
    np.testing.assert_array_equal(cat["encoder"].categories_[0], ["blue", "red"])
    np.testing.assert_array_equal(cat.transform(result.X_test[["colour"]]), np.zeros((30, 2)))
    assert len(result.frame) == 30 and result.frame.colour.eq("never_seen_in_training").all()
    _, importance = training.model_importance(result)
    assert len(importance) == 4
    note, explanation = training.row_explanation(result, 1)
    assert len(explanation) == 4 and "sum to" in note


def test_numeric_categories_preserve_spelling(iris):
    iris["code"] = ["01", "1", "1.0"] * 50
    result = training.train_and_evaluate(iris, "species", (*FEATURES, "code"), categorical_features=("code",), classifier="Decision tree", cv_folds=3)
    assert set(result.model["preprocessor"].named_transformers_["categorical"]["encoder"].categories_[0]) == {"01", "1", "1.0"}
    assert result.metadata["categorical_features"] == ["code"]


def test_wilson_bounds_empty_perfect_and_small_cells():
    low, high = uncertainty.wilson_interval([5, 0, 1, 0], [10, 1, 1, 0])
    assert (low[0], high[0]) == pytest.approx((.2365930905, .7634069095))
    assert low[1] == pytest.approx(0) and high[1] == pytest.approx(.7934506856)
    assert low[2] == pytest.approx(.2065493144) and high[2] == pytest.approx(1)
    assert np.isnan(low[3]) and np.isnan(high[3])
    for errors, total in [(-1, 2), (3, 2), (0, -1)]:
        with pytest.raises(ValueError):
            uncertainty.wilson_interval(errors, total)


def test_hotspot_evidence_corrects_for_cells_and_rejects_tiny_samples():
    cells = uncertainty.add_cell_uncertainty(pd.DataFrame({"total": [100, 100, 100, 2, 0], "errors": [95, 5, 5, 2, 0]}))
    assert cells.hotspot_p_adjusted.iloc[0] >= cells.hotspot_p.iloc[0]
    assert uncertainty.evidence_labels(cells).tolist() == ["Elevated error evidence", "No clear elevation", "No clear elevation", "Too few examples", "No examples"]
    one = uncertainty.add_cell_uncertainty(pd.DataFrame({"total": [10], "errors": [10]}))
    assert np.isnan(one.hotspot_p_adjusted.iloc[0])


def test_archive_roundtrip_integrity_and_environment_checks(iris):
    result = training.train_and_evaluate(iris, "species", FEATURES, classifier="Decision tree", cv_folds=3)
    archive = experiments.create_archive(result)
    replay = experiments.reproduce(archive)
    assert replay.csv_bytes() == result.csv_bytes()
    assert replay.metadata["reproduction"]["exact_predictions"]
    assert replay.train_rows == result.train_rows
    manifest, raw, files = experiments.read_archive(archive)
    assert set(files) >= {"dataset.csv", "report/evaluation_report.html", "manifest.json"}
    damaged = io.BytesIO()
    with zipfile.ZipFile(damaged, "w") as z:
        for name, value in files.items():
            z.writestr(name, b"bad" if name == "dataset.csv" else value)
    with pytest.raises(analysis.DataError, match="Integrity check"):
        experiments.read_archive(damaged.getvalue())
    manifest["code_sha256"] = "different calculation code"
    files["manifest.json"] = exports.json_bytes(manifest)
    changed = exports.bundle(files)
    with pytest.raises(analysis.DataError, match="environment differs"):
        experiments.reproduce(changed)
    exploratory = experiments.reproduce(changed, allow_environment_change=True)
    assert exploratory.metadata["reproduction"]["exact_predictions"]
    assert exploratory.metadata["reproduction"]["environment_changes"] == ["Atlas calculation code"]


@pytest.mark.parametrize("field,value", [("files", []), ("packages", {}), ("recipe", []),
                                         ("source_index", "invalid"), ("python", None)])
def test_malformed_archive_schema_produces_user_error(iris, field, value):
    result = training.train_and_evaluate(iris, "species", FEATURES, classifier="Decision tree")
    manifest, _, files = experiments.read_archive(experiments.create_archive(result))
    manifest[field] = value
    files["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(analysis.DataError, match="Cannot restore this run archive"):
        experiments.read_archive(exports.bundle(files))


def test_ui_staged_selection_archive_restore_and_cell_inspection():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.radio(key="workflow").set_value("Train a model and evaluate it").run()
    at.file_uploader(key="train_upload").set_value(("iris.csv", (ROOT / "iris.csv").read_bytes(), "text/csv")).run()
    at.button(key="train_button").click().run()
    assert not at.exception and not at.metric
    assert "training_result" not in at.session_state
    assert not at.session_state["training_batch"][0].metrics
    at.button(key="reveal_test").click().run()
    result = at.session_state["training_result"]
    at.selectbox(key="sel_x").set_value(2)
    at.selectbox(key="sel_y").set_value(1)
    at.run()
    assert any("Wilson interval" in c.value for c in at.caption)
    at.button(key="prepare_report").click().run()
    archive = at.session_state["prepared_run_archive"][1]
    # Replay in a fresh session: a later fit must still warn that test scores were seen.
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.radio(key="workflow").set_value("Train a model and evaluate it").run()
    at.radio(key="training_source").set_value("Restore a saved run").run()
    at.file_uploader(key="run_archive_upload").set_value(("saved.atlas.zip", archive, "application/zip")).run()
    assert not at.exception
    at.button(key="reproduce_run").click().run()
    assert not at.exception
    assert at.session_state["training_result"].csv_bytes() == result.csv_bytes()
    assert any("Reproduction verified" in s.value for s in at.success)
    assert at.selectbox(key="sel_x").value is None
    at.radio(key="training_source").set_value("Upload a CSV").run()
    at.file_uploader(key="train_upload").set_value(("iris.csv", (ROOT / "iris.csv").read_bytes(), "text/csv")).run()
    at.button(key="train_button").click().run()
    at.button(key="reveal_test").click().run()
    assert not at.exception
    selection = at.session_state["training_result"].metadata["selection"]
    assert selection["previous_test_reveals_for_dataset_target"] == 1
    assert selection["exploratory_test_reuse"]
    assert any("Exploratory test reuse" in warning.value for warning in at.warning)
