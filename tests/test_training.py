"""Held-out isolation, reproducibility, validation and export contracts."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis
import training

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")


@pytest.fixture
def iris():
    return analysis.load_csv((ROOT / "iris.csv").read_bytes()).frame


def test_iris_split_and_export_are_reproducible(iris):
    result = training.train_and_evaluate(iris, "species", FEATURES)
    again = training.train_and_evaluate(iris, "species", FEATURES)
    assert result.csv_bytes() == again.csv_bytes()
    assert len(result.train_rows) == 120 and len(result.test_rows) == 30
    assert not set(result.train_rows) & set(result.test_rows)
    assert set(result.train_rows) | set(result.test_rows) == set(iris.index)
    assert result.frame[result.actual].value_counts().to_dict() == {
        "setosa": 10, "versicolor": 10, "virginica": 10,
    }
    exported = pd.read_csv(io.BytesIO(result.csv_bytes()))
    assert exported["source_row_id"].tolist() == list(result.test_rows)
    # Independent equality/crosstab calculation on the actual exported CSV.
    correct = exported[result.actual] == exported[result.predicted]
    assert int(correct.sum()) == 28
    assert correct.mean() == pytest.approx(28 / 30)
    assert exported.loc[~correct, "source_row_id"].tolist() == [135, 139]
    assert pd.crosstab(exported[result.actual], exported[result.predicted]).to_numpy().tolist() == [
        [10, 0, 0], [0, 10, 0], [0, 2, 8],
    ]


def test_test_values_cannot_affect_fitted_preprocessing(iris):
    baseline = training.train_and_evaluate(iris, "species", FEATURES)
    changed = iris.copy()
    changed.loc[list(baseline.test_rows), "sepal_length"] = "999999"
    result = training.train_and_evaluate(changed, "species", FEATURES)
    np.testing.assert_array_equal(
        baseline.model["imputer"].statistics_, result.model["imputer"].statistics_,
    )
    np.testing.assert_array_equal(baseline.model["scaler"].mean_, result.model["scaler"].mean_)
    expected_train = iris.loc[list(result.train_rows), list(FEATURES)].astype(float)
    np.testing.assert_allclose(result.model["imputer"].statistics_, expected_train.median())
    np.testing.assert_allclose(result.model["scaler"].mean_, expected_train.mean())


def test_missing_features_imputed_for_prediction_but_preserved_for_map(iris):
    baseline = training.train_and_evaluate(iris, "species", FEATURES)
    changed = iris.copy()
    source_row = baseline.test_rows[0]
    changed.loc[source_row, "sepal_length"] = ""
    changed.loc[baseline.train_rows[0], "sepal_length"] = "bad"
    result = training.train_and_evaluate(changed, "species", FEATURES)
    assert result.frame.loc[result.frame["source_row_id"] == source_row, "sepal_length"].iloc[0] == ""
    evaluation = analysis.evaluate(result.frame, result.actual, result.predicted)
    fmap = analysis.build_failure_map(result.frame, evaluation.rows, "sepal_length", "sepal_width")
    assert evaluation.n_evaluated == 30
    assert fmap.n_mapped == 29 and fmap.n_omitted == 1
    expected = pd.to_numeric(changed.loc[list(result.train_rows), "sepal_length"], errors="coerce").median()
    assert result.model["imputer"].statistics_[0] == expected


@pytest.mark.parametrize("classifier", training.CLASSIFIERS)
def test_each_classifier_uses_only_heldout_rows(iris, classifier):
    result = training.train_and_evaluate(iris, "species", FEATURES, classifier=classifier)
    assert len(result.frame) == 30
    assert set(result.frame["source_row_id"]) == set(result.test_rows)
    assert set(result.frame[result.predicted]) <= set(iris["species"])


@pytest.mark.parametrize("forbidden", ["species", "customer_id", "note"])
def test_target_ids_and_text_cannot_be_training_features(iris, forbidden):
    iris["customer_id"] = [f"c_{i}" for i in iris.index]
    iris["note"] = "text"
    assert forbidden not in training.feature_choices(iris, "species").usable
    with pytest.raises(analysis.DataError, match="exclude the target and identifiers"):
        training.train_and_evaluate(iris, "species", ("sepal_length", forbidden))


@pytest.mark.parametrize("targets, test_size, message", [
    (["a"] * 10, .2, "at least two classes"),
    (["a"] * 9 + ["b"], .2, "at least two rows"),
    (["a", "a", "b", "b", "c", "c"], .2, "Both sets need"),
])
def test_impossible_splits_do_not_fall_back_to_training(targets, test_size, message):
    raw = pd.DataFrame({"target": targets, "x": [i / 10 for i in range(len(targets))]})
    with pytest.raises(analysis.DataError, match=message):
        training.train_and_evaluate(raw, "target", ("x",), test_size=test_size)


def test_numeric_labels_keep_distinct_spelling_and_missing_targets_drop(iris):
    iris["species"] = iris["species"].map({"setosa": " 01 ", "versicolor": "1", "virginica": "2"})
    iris.loc[1, "species"] = ""
    result = training.train_and_evaluate(iris, "species", FEATURES)
    assert result.missing_targets == 1
    assert 1 not in result.train_rows + result.test_rows
    assert set(result.frame[result.actual]) == {"01", "1", "2"}
    assert set(result.frame[result.predicted]) <= {"01", "1", "2"}
