"""Leakage heuristics: encoded targets, numeric transforms, suspicious names and non-leaks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as A
import diagnostics as D

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def iris():
    return A.load_csv((ROOT / "iris.csv").read_bytes()).frame


@pytest.fixture
def numeric():
    rng = np.random.RandomState(0)
    y = rng.normal(10, 2, 300).round(3)
    frame = pd.DataFrame({"y": y.astype(str)})
    frame.index = pd.RangeIndex(1, 301)
    return frame, y, rng


def finding(raw, target, feature):
    return next((f for f in D.leakage_findings(raw, target, (feature,)) if f.feature == feature), None)


def test_exact_numeric_duplicate_target(numeric):
    raw, _, _ = numeric
    raw["copy"] = raw["y"]
    found = finding(raw, "y", "copy")
    assert found.risk == "high" and "identical to the target" in found.reasons[0]


def test_exact_string_duplicate_target(iris):
    iris["species_again"] = iris["species"]
    found = finding(iris, "species", "species_again")
    assert found.risk == "high" and "100.0%" in found.reasons[0]


def test_label_encoded_categorical_target(iris):
    iris["species_code"] = iris["species"].map({"setosa": "0", "versicolor": "1", "virginica": "2"})
    found = finding(iris, "species", "species_code")
    assert found.risk == "high"
    assert "label-encoded copy of the target" in found.reasons[0]
    assert "0 → setosa" in found.reasons[0]


def test_near_deterministic_and_coarsened_encodings_are_flagged(iris):
    codes = iris["species"].map({"setosa": "0", "versicolor": "1", "virginica": "2"})
    codes.iloc[0] = "1"  # one mislabelled row out of 150
    iris["noisy_code"] = codes
    iris["is_setosa"] = (iris["species"] == "setosa").astype(int).astype(str)
    assert "near-deterministic mapping" in finding(iris, "species", "noisy_code").signals
    assert "determined by the target" in finding(iris, "species", "is_setosa").signals


def test_stringified_numeric_target(numeric):
    raw, y, _ = numeric
    raw["y_text"] = [f"{v:.5f}" for v in y]  # "9.123" written as "9.12300"
    found = finding(raw, "y", "y_text")
    assert found.risk == "high" and "same numbers as the target written differently" in found.reasons[0]


@pytest.mark.parametrize("transform, phrase", [
    (lambda y: y + 100, "constant offset"),
    (lambda y: y * 2.5, "multiplied by 2.5"),
    (lambda y: 3 * y - 7, "linear transformation"),
    (np.log, "monotonic one-to-one transformation"),
])
def test_deterministic_numeric_transforms(numeric, transform, phrase):
    raw, y, _ = numeric
    raw["derived"] = [repr(float(v)) for v in transform(y)]
    found = finding(raw, "y", "derived")
    assert found.risk == "high" and phrase in found.reasons[0]


@pytest.mark.parametrize("column, term", [
    ("predictedLabel", "'predicted'"),
    ("prediction_score", "'prediction'"),
    ("yhat", "'yhat'"),
    ("y_pred", "'pred'"),
    ("yHat", "'y_hat'"),
    ("targetScore", "'target'"),
    ("futureOutcome", "'future'"),
    ("PREDScore", "'pred'"),
    ("model-probability", "'probability'"),
])
def test_suspicious_names_are_tokenised_and_explained(iris, column, term):
    iris[column] = np.random.RandomState(1).rand(len(iris)).round(3).astype(str)
    found = finding(iris, "species", column)
    assert found is not None and found.risk == "review"
    assert term in found.reasons[0] and found.reasons[0].startswith("Column name contains")


def test_camel_case_and_acronym_tokenisation():
    assert D.name_tokens("predictedLabel") == ["predicted", "label"]
    assert D.name_tokens("PREDScore") == ["pred", "score"]
    assert D.name_tokens("HTTPResponse-code 2") == ["http", "response", "code", "2"]
    assert D.name_tokens("prob2") == ["prob", "2"]


@pytest.mark.parametrize("column", ["sepal_width", "credit_score", "post_code", "postal_zone", "predictor_count", "labelled_by", "afternoon_temp"])
def test_ordinary_features_and_unrelated_words_are_not_flagged(iris, column):
    if column not in iris:
        iris[column] = np.random.RandomState(2).rand(len(iris)).round(3).astype(str)
    assert finding(iris, "species", column) is None


def test_context_words_are_flagged_only_with_corroboration(iris):
    iris["score_setosa"] = np.random.RandomState(3).rand(len(iris)).round(3).astype(str)
    iris["species_result"] = np.random.RandomState(4).rand(len(iris)).round(3).astype(str)
    assert "target class “setosa”" in finding(iris, "species", "score_setosa").reasons[0]
    assert "target name “species”" in finding(iris, "species", "species_result").reasons[0]


def test_high_cardinality_identifier_is_not_leakage(iris, numeric):
    iris["record_id"] = [f"r{v}" for v in np.random.RandomState(5).permutation(len(iris))]
    iris["code"] = [str(v) for v in np.random.RandomState(6).permutation(len(iris))]
    assert finding(iris, "species", "record_id") is None
    assert finding(iris, "species", "code") is None
    raw, _, rng = numeric
    raw["row_number"] = [str(v) for v in rng.permutation(len(raw))]
    assert finding(raw, "y", "row_number") is None


def test_strong_but_imperfect_correlation_is_not_flagged(numeric):
    raw, y, rng = numeric
    raw["related"] = (y + rng.normal(0, 1, len(y))).round(3).astype(str)
    raw["very_close"] = (y + rng.normal(0, .2, len(y))).round(3).astype(str)
    r = np.corrcoef(y, raw["related"].astype(float))[0, 1]
    assert .85 < r < .95
    assert finding(raw, "y", "related") is None
    assert "almost perfectly correlated" in finding(raw, "y", "very_close").reasons[0]


def test_leakage_table_keeps_the_review_reason_column(iris):
    iris["species_code"] = iris["species"].map({"setosa": "0", "versicolor": "1", "virginica": "2"})
    table = D.leakage_warnings(iris, "species", ("sepal_length", "species_code"))
    assert table.columns.tolist() == ["feature", "risk", "signals", "review reason"]
    assert table["feature"].tolist() == ["species_code"]


def test_quality_overview_reports_leakage_and_target_balance(iris):
    iris["species_code"] = iris["species"].map({"setosa": "0", "versicolor": "1", "virginica": "2"})
    iris["y_pred"] = iris["species"].sample(frac=1, random_state=0).to_numpy()
    findings = D.leakage_findings(iris, "species")
    overview = D.quality_overview(iris, "species", "classification", findings=findings).set_index("check")
    assert overview.loc["Potential leakage", "status"] == "Warning"
    assert "2 suspicious columns" in overview.loc["Potential leakage", "detail"]
    assert "species_code (label-encoded copy" in overview.loc["Potential leakage", "detail"]
    assert overview.loc["Target balance", "status"] == "OK"
    assert "3 classes" in overview.loc["Target balance", "detail"] and "33.3%" in overview.loc["Target balance", "detail"]
    clean = D.quality_overview(iris.drop(columns=["species_code", "y_pred"]), "species", findings=[]).set_index("check")
    assert clean.loc["Potential leakage", "status"] == "OK"
