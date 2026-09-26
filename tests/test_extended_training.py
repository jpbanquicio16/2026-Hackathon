import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as A
import training as T

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")


@pytest.fixture
def iris():
    return A.load_csv((ROOT / "iris.csv").read_bytes()).frame


def test_tuning_and_cv_do_not_see_test_features(iris):
    result = T.train_and_evaluate(iris, "species", FEATURES, cv_folds=3, cv_repeats=2, tune=True)
    changed = iris.copy()
    changed.loc[list(result.test_rows), "petal_length"] = "99999"
    rerun = T.train_and_evaluate(changed, "species", FEATURES, cv_folds=3, cv_repeats=2, tune=True)
    assert result.metadata["selected_parameters"] == rerun.metadata["selected_parameters"]
    pd.testing.assert_frame_equal(result.cv_results, rerun.cv_results)
    assert result.metadata["cv_splits"] == 6
    for split in result.metadata["cv_row_ids"]:
        assert not set(split["train"]) & set(split["validation"])
        assert not (set(split["train"]) | set(split["validation"])) & set(result.test_rows)
    metadata = json.loads(result.metadata_bytes())
    assert metadata["fitted_parameters"]["imputer__missing_values"] == "NaN"
    assert metadata["test_row_ids"] == list(result.test_rows)
    assert metadata["versions"]["scikit-learn"]


def test_grouped_split_and_cv_keep_groups_disjoint(iris):
    iris["group_id"] = [f"g_{i % 10}" for i in range(len(iris))]
    result = T.train_and_evaluate(iris, "species", FEATURES, classifier="Random forest", split_method="Grouped", split_column="group_id", cv_folds=3)
    def groups(rows):
        return set(iris.loc[rows, "group_id"])
    assert not groups(list(result.train_rows)) & groups(list(result.test_rows))
    for split in result.metadata["cv_row_ids"]:
        assert not groups(split["train"]) & groups(split["validation"])


def test_time_split_and_cv_keep_tied_times_together():
    rng = np.random.RandomState(9)
    raw = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60), "target": rng.normal(size=60), "time": np.repeat(pd.date_range("2024-01-01", periods=30).astype(str), 2)})
    raw.index = pd.RangeIndex(1, 61)
    raw = raw.astype(str)
    result = T.train_and_evaluate(raw, "target", ("x", "z"), classifier="Ridge regression", task="regression", split_method="Time ordered", split_column="time", cv_folds=3)
    assert raw.loc[list(result.train_rows), "time"].max() < raw.loc[list(result.test_rows), "time"].min()
    for split in result.metadata["cv_row_ids"]:
        assert raw.loc[split["train"], "time"].max() < raw.loc[split["validation"], "time"].min()


def test_balancing_occurs_only_in_training_and_each_cv_fit(iris):
    raw = iris.drop(index=range(1, 36))
    result = T.train_and_evaluate(raw, "species", FEATURES, balance=True, cv_folds=3, tune=True)
    classifier = result.model["classifier"]
    assert classifier.n_resampled_ == int(result.y_train.value_counts().max()) * result.y_train.nunique()
    assert classifier.n_resampled_ > len(result.train_rows)
    assert len(result.frame) == len(result.test_rows)
    assert not set(result.train_rows) & set(result.test_rows)


def test_models_compare_on_identical_rows_and_provide_probabilities(iris):
    results = T.compare_models(iris, "species", FEATURES, ["Logistic regression", "Random forest", "Calibrated logistic regression"], cv_folds=3)
    assert len({r.metadata["split_id"] for r in results}) == 1
    for result in results:
        probabilities = result.frame[list(result.probability_columns.values())]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        assert result.frame[result.confidence].between(0, 1).all()


@pytest.mark.parametrize("name", T.REGRESSORS)
def test_every_regressor_exports_numeric_test_predictions(iris, name):
    result = T.train_and_evaluate(iris, "petal_length", ("sepal_length", "sepal_width", "petal_width"), classifier=name, task="regression", cv_folds=3)
    exported = result.frame
    residual = exported[result.predicted].astype(float) - exported[result.actual].astype(float)
    assert result.metrics["mae"] == pytest.approx(residual.abs().mean())
    assert result.metrics["rmse"] == pytest.approx(np.sqrt((residual ** 2).mean()))
    assert result.metadata["cv_metric"] == "MAE"
    assert not result.probability_columns


def test_permutation_and_local_explanation_are_available_for_supported_models(iris):
    result = T.train_and_evaluate(iris, "species", FEATURES, classifier="Logistic regression")
    importance = T.permutation_scores(result)
    assert set(importance["feature"]) == set(FEATURES)
    note, table = T.row_explanation(result, 1)
    assert "contribution" in table and "linear score" in note
    expected = result.model.decision_function(result.X_test.iloc[[0]])[0]
    predicted = result.frame.iloc[0][result.predicted]
    index = list(result.model.classes_).index(predicted)
    assert table["contribution"].sum() + result.model["classifier"].intercept_[index] == pytest.approx(expected[index])
    tree = T.train_and_evaluate(iris, "species", FEATURES, classifier="Decision tree")
    assert "condition" in T.row_explanation(tree, 1)[1]
