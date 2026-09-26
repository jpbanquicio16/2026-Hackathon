"""Grouped class coverage, SVM probabilities, target guidance, imbalance strategies,
fingerprints, progress reporting, experiment history and explanation fallbacks."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import analysis as A
import diagnostics as D
import evaluation as E
import training as T

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
CLASSES = ["setosa", "versicolor", "virginica"]


@pytest.fixture
def iris():
    return A.load_csv((ROOT / "iris.csv").read_bytes()).frame


def groups_of(raw, rows, column):
    return set(raw.loc[list(rows), column])


def probability_frame(result):
    rows = A.evaluate(result.frame, result.actual, result.predicted).rows
    probabilities = pd.DataFrame({label: result.frame[column].astype(float) for label, column in result.probability_columns.items()})
    return rows, probabilities


# ---------------------------------------------------------------------------
# Grouped splits


def test_groups_aligned_with_classes_warn_about_missing_classes(iris):
    iris["site"] = iris["species"].map({"setosa": "north", "versicolor": "south", "virginica": "east"})
    result = T.train_and_evaluate(iris, "species", FEATURES, split_method="Grouped", split_column="site")
    coverage = result.metadata["split_class_coverage"]
    assert not coverage["representative"]
    assert len(coverage["missing_from_test"]) == 2 and len(coverage["missing_from_train"]) == 1
    notes = " ".join(result.notes)
    assert "Warning: The held-out test set does not contain all target classes. Missing from test set:" in notes
    assert "Metrics such as recall, ROC-AUC and confusion matrices may be incomplete or misleading." in notes
    assert "Missing from training set" in notes
    assert "Every target class occurs in only one group, so groups align with the classes" in notes
    assert not groups_of(iris, result.train_rows, "site") & groups_of(iris, result.test_rows, "site")


def test_class_confined_to_one_group_is_explained(iris):
    iris["site"] = ["s" if label == "setosa" else f"g{i % 10}" for i, label in enumerate(iris["species"])]
    result = T.train_and_evaluate(iris, "species", FEATURES, split_method="Grouped", split_column="site")
    assert "setosa occurs in only one group, so a grouped split cannot place it in both training and test sets." in result.notes


def test_too_few_single_class_groups_for_the_test_set_are_explained(iris):
    iris["batch"] = [f"b{(row - 1) // 25}" for row in iris.index]  # six groups, two per class
    result = T.train_and_evaluate(iris, "species", FEATURES, split_method="Grouped", split_column="batch", test_size=.2)
    assert result.metadata["split_class_coverage"]["missing_from_test"]
    assert any("Every group contains a single class" in note and "fewer than the 3 classes" in note for note in result.notes)


def test_grouped_split_searches_for_a_partition_that_covers_every_class(iris):
    iris["batch"] = [f"b{(row - 1) // 10}" for row in iris.index]  # fifteen single-class groups
    first = GroupShuffleSplit(n_splits=1, test_size=.3, random_state=0)

    def first_partition_covers(seed):
        first.random_state = seed
        train, test = next(first.split(iris, iris["species"], iris["batch"]))
        return iris["species"].iloc[train].nunique() == iris["species"].iloc[test].nunique() == 3

    seed = next(s for s in range(100) if not first_partition_covers(s))
    result = T.train_and_evaluate(iris, "species", FEATURES, split_method="Grouped", split_column="batch", test_size=.3, seed=seed)
    coverage = result.metadata["split_class_coverage"]
    assert coverage["representative"] and coverage["group_split_candidate"] > 1
    assert any(f"candidate {coverage['group_split_candidate']} drawn from the same seed" in note for note in result.notes)
    assert set(result.y_test) == set(CLASSES) and set(result.y_train) == set(CLASSES)
    assert not groups_of(iris, result.train_rows, "batch") & groups_of(iris, result.test_rows, "batch")

    covering = next(s for s in range(100) if first_partition_covers(s))
    unchanged = T.train_and_evaluate(iris, "species", FEATURES, split_method="Grouped", split_column="batch", test_size=.3, seed=covering)
    first.random_state = covering
    _, test = next(first.split(iris, iris["species"], iris["batch"]))
    assert unchanged.metadata["split_class_coverage"]["group_split_candidate"] == 1
    assert sorted(unchanged.test_rows) == sorted(iris.index[test].tolist())


def test_grouped_cross_validation_is_stratified_and_group_disjoint(iris):
    iris["site"] = [f"g{i % 15}" for i in range(len(iris))]  # every group spans the classes
    result = T.train_and_evaluate(iris, "species", FEATURES, split_method="Grouped", split_column="site", cv_folds=3)
    assert result.metadata["cv_method"] == "StratifiedGroupKFold"
    for split in result.metadata["cv_row_ids"]:
        assert not groups_of(iris, split["train"], "site") & groups_of(iris, split["validation"], "site")
        assert set(iris.loc[split["validation"], "species"]) == set(CLASSES)
    assert not any("validation folds do not contain every class" in note for note in result.notes)


def test_grouped_regression_uses_group_k_fold(iris):
    iris["site"] = [f"g{i % 15}" for i in range(len(iris))]
    result = T.train_and_evaluate(iris, "petal_length", ("sepal_length", "petal_width"), classifier="Ridge regression", task="regression", split_method="Grouped", split_column="site", cv_folds=3)
    assert result.metadata["cv_method"] == "GroupKFold"
    assert result.metadata["split_class_coverage"] == {}


# ---------------------------------------------------------------------------
# SVM probabilities


def test_svm_probabilities_are_calibrated_and_ordered(iris):
    result = T.train_and_evaluate(iris, "species", FEATURES, classifier="Support vector machine", cv_folds=3)
    assert list(result.probability_columns) == CLASSES == list(result.model.classes_)
    assert list(result.probability_columns.values()) == ["probability_0", "probability_1", "probability_2"]
    rows, probabilities = probability_frame(result)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (probabilities.idxmax(axis=1) == result.frame[result.predicted]).all()
    np.testing.assert_allclose(result.frame[result.confidence], probabilities.max(axis=1))
    assert result.metadata["probability_status"] == "available"
    assert "sigmoid (Platt) calibration" in result.metadata["probability_method"]

    auc, note = E.probability_auc(rows, probabilities)
    expected = roc_auc_score(rows["actual"], probabilities[CLASSES].to_numpy(), multi_class="ovr", labels=CLASSES)
    assert auc == pytest.approx(expected) and "one-vs-rest" in note


def test_binary_svm_supports_threshold_exploration(iris):
    raw = iris.loc[iris["species"] != "setosa"].copy()
    result = T.train_and_evaluate(raw, "species", FEATURES, classifier="Support vector machine")
    assert set(result.probability_columns) == {"versicolor", "virginica"}
    column = result.probability_columns["virginica"]
    frame = result.frame
    at_half, invalid = E.apply_threshold(frame, result.predicted, column, "virginica", "versicolor", .5)
    assert invalid == 0
    assert (at_half[result.predicted] == frame[result.predicted]).mean() >= .95
    strict, _ = E.apply_threshold(frame, result.predicted, column, "virginica", "versicolor", .95)
    assert (strict[result.predicted] == "virginica").sum() <= (at_half[result.predicted] == "virginica").sum()
    rows, probabilities = probability_frame(result)
    auc, note = E.probability_auc(rows, probabilities)
    assert auc == pytest.approx(roc_auc_score(rows["actual"] == "virginica", probabilities["virginica"]))
    assert note.startswith("Binary ROC-AUC")


def test_svm_on_grouped_split_explains_missing_probabilities(iris):
    iris["site"] = [f"g{i % 10}" for i in range(len(iris))]
    result = T.train_and_evaluate(iris, "species", FEATURES, classifier="Support vector machine", split_method="Grouped", split_column="site")
    assert not result.probability_columns and result.confidence is None
    assert "SVM probabilities are unavailable for grouped and time-ordered splits" in result.metadata["probability_status"]
    assert result.metadata["probability_status"] in result.notes
    assert hasattr(result.model, "decision_function") and not hasattr(result.model, "predict_proba")


def test_svm_without_enough_rows_per_class_for_calibration_is_explained(iris):
    raw = iris.drop(index=range(51, 99))  # two versicolor rows: one for training, one for test
    result = T.train_and_evaluate(raw, "species", FEATURES, classifier="Support vector machine")
    assert (result.y_train == "versicolor").sum() == 1
    assert not result.probability_columns
    assert "calibration needs at least 2 training rows of every class" in result.metadata["probability_status"]


# ---------------------------------------------------------------------------
# Target guidance


def frame_of(values, name="target"):
    raw = pd.DataFrame({name: [str(v) for v in values], "x": [str(i) for i in range(len(values))]})
    raw.index = pd.RangeIndex(1, len(raw) + 1)
    return raw


@pytest.mark.parametrize("values, kind", [
    (["yes", "no"] * 50, "binary"),
    ([0, 1] * 50, "binary"),
    (["a", "b", "c"] * 40, "multiclass"),
    ([3, 4, 5, 6, 7, 8] * 30, "multiclass"),
    ([round(.33 + .01 * (i % 68), 2) for i in range(200)], "continuous"),
    ([i % 45 for i in range(200)], "continuous"),
    ([i % 15 for i in range(300)], "ambiguous"),
    ([1.5, 2.5, 3.5] * 40, "ambiguous"),
    ([f"customer {i}" for i in range(100)], "ambiguous"),
    (["only"] * 20, "constant"),
])
def test_target_profile_kinds(values, kind):
    assert D.target_profile(frame_of(values), "target").kind == kind


def test_continuous_classification_target_gets_regression_guidance_not_imbalance():
    values = [.5] * 150 + [round(.6 + .01 * i, 2) for i in range(50)]
    guidance = D.target_guidance(frame_of(values, "sulphates"), "sulphates", "classification")
    assert guidance.profile.kind == "continuous" and not guidance.show_class_diagnostics
    assert guidance.warnings == (
        "“sulphates” contains 51 distinct numeric values and appears continuous. Classification is unlikely to be "
        "appropriate for this target. Consider switching the task type to Regression.",
    )
    confirmed = D.target_guidance(frame_of(values, "sulphates"), "sulphates", "classification", treat_as_classes=True)
    assert confirmed.show_class_diagnostics
    assert any(w.startswith("Class imbalance") for w in confirmed.warnings)
    assert any("only one example" in w for w in confirmed.warnings)


def test_regression_guidance_for_text_and_ordinal_targets():
    text = D.target_guidance(frame_of(["a", "b", "c"] * 20), "target", "regression")
    assert "is not numeric" in text.warnings[0] and not text.show_class_diagnostics
    ordinal = D.target_guidance(frame_of([1, 2, 3] * 20), "target", "regression")
    assert not ordinal.warnings and "Regression treats them as ordered numbers" in ordinal.notes[0]


def test_real_class_imbalance_is_still_reported():
    guidance = D.target_guidance(frame_of(["common"] * 100 + ["rare"] * 5), "target", "classification")
    assert guidance.show_class_diagnostics
    assert any("Class imbalance: the largest class has 100 rows and the smallest has 5" in w for w in guidance.warnings)
    balance = D.class_balance(frame_of(["common"] * 100 + ["rare"] * 5)["target"])
    assert balance["imbalanced"] and balance["ratio"] == 20 and balance["smallest_class"] == "rare"


def test_continuous_classification_target_fails_with_actionable_error(iris):
    with pytest.raises(A.DataError) as error:
        T.train_and_evaluate(iris, "sepal_length", ("sepal_width", "petal_length", "petal_width"))
    message = str(error.value)
    assert "cannot create a stratified train/test split" in message
    assert "“sepal_length” appears continuous" in message and "Try Regression" in message


def test_rare_categorical_class_fails_with_named_classes(iris):
    extra = pd.DataFrame([{"sepal_length": "5", "sepal_width": "3", "petal_length": "1", "petal_width": "0.2", "species": "hybrid"}], index=[151])
    with pytest.raises(A.DataError, match="This class has only one row: “hybrid”"):
        T.train_and_evaluate(pd.concat([iris, extra]), "species", FEATURES)


def test_fit_errors_are_translated_and_keep_details():
    message = T._fit_error(ValueError("The least populated class in y has only 1 members, which is less than n_splits=3."))
    assert "switch to Regression if this is a continuous variable" in message and "least populated class" in message
    assert T._fit_error(ValueError("something else")) == "The model could not fit this data: something else"


# ---------------------------------------------------------------------------
# Class imbalance strategies


@pytest.fixture
def imbalanced(iris):
    return iris.drop(index=range(56, 101)).loc[lambda f: f["species"] != "setosa"].copy()  # 5 versicolor, 50 virginica


@pytest.mark.parametrize("name", T.CLASS_WEIGHT_MODELS)
def test_class_weights_reach_every_supported_estimator(imbalanced, name):
    kwargs = {"test_size": .4} if name in T.CALIBRATED else {}
    result = T.train_and_evaluate(imbalanced, "species", FEATURES, classifier=name, imbalance="Class weights", **kwargs)
    assert T._fitted_estimator(result.model).class_weight == "balanced"
    assert result.metadata["imbalance_strategy"] == "Class weights"
    assert result.metadata["class_balancing"] == "class_weight='balanced' inside each training fit"


def test_class_weights_survive_tuning_and_change_the_fit(imbalanced):
    lr = {"classifier": "Logistic regression", "cv_folds": 2, "tune": True}
    weighted = T.train_and_evaluate(imbalanced, "species", FEATURES, imbalance="Class weights", **lr)
    plain = T.train_and_evaluate(imbalanced, "species", FEATURES, **lr)
    assert weighted.model["classifier"].class_weight == "balanced"
    assert weighted.metadata["fitted_parameters"]["classifier__class_weight"] == "balanced"
    assert plain.model["classifier"].class_weight is None
    assert not np.allclose(weighted.model["classifier"].coef_, plain.model["classifier"].coef_)


def test_oversampling_and_class_weights_are_distinct_strategies(imbalanced):
    oversampled = T.train_and_evaluate(imbalanced, "species", FEATURES, classifier="Random forest", imbalance="Oversampling")
    assert isinstance(oversampled.model["classifier"], T.ResampledClassifier)
    assert T._fitted_estimator(oversampled.model).class_weight is None
    assert oversampled.metadata["imbalance_strategy"] == "Oversampling"
    knn = T.train_and_evaluate(imbalanced, "species", FEATURES, classifier="3-nearest neighbors", imbalance="Oversampling")
    assert knn.model["classifier"].n_resampled_ > len(knn.train_rows)
    assert T.train_and_evaluate(imbalanced, "species", FEATURES, balance=True).metadata["imbalance_strategy"] == "Oversampling"
    assert T.train_and_evaluate(imbalanced, "species", FEATURES).metadata["imbalance_strategy"] == "None"


def test_unsupported_imbalance_choices_are_rejected(imbalanced, iris):
    with pytest.raises(A.DataError, match="3-nearest neighbors does not support class weights"):
        T.train_and_evaluate(imbalanced, "species", FEATURES, classifier="3-nearest neighbors", imbalance="Class weights")
    with pytest.raises(A.DataError, match="not both"):
        T.train_and_evaluate(imbalanced, "species", FEATURES, balance=True, imbalance="Class weights")
    with pytest.raises(A.DataError, match="only for classification"):
        T.train_and_evaluate(iris, "petal_length", ("sepal_length",), classifier="Ridge regression", task="regression", imbalance="Class weights")
    assert not T.supports_class_weights("3-nearest neighbors")
    assert not T.supports_class_weights("Random forest", "regression")


def test_imbalance_strategy_is_recorded_in_comparison_and_history(imbalanced):
    results = T.compare_models(imbalanced, "species", FEATURES, ["Logistic regression", "Random forest"], imbalance="Class weights")
    assert T.comparison_table(results)["imbalance"].tolist() == ["Class weights"] * 2
    assert T.history_table({"run 1": results[0]})["imbalance"].tolist() == ["Class weights"]


# ---------------------------------------------------------------------------
# Fingerprints and progress


def test_fingerprint_is_deterministic_and_sensitive_to_content_and_schema(iris):
    base = T.fingerprint(iris)
    assert base == T.fingerprint(iris.copy()) == T.fingerprint(A.load_csv((ROOT / "iris.csv").read_bytes()).frame)
    changed = iris.copy()
    changed.iloc[7, 2] = "1.41"
    assert T.fingerprint(changed) != base
    assert T.fingerprint(iris.rename(columns={"species": "label"})) != base
    assert T.fingerprint(iris[list(reversed(iris.columns))]) != base
    numbers = pd.DataFrame({"a": [1, 2, 3]})
    assert T.fingerprint(numbers) != T.fingerprint(numbers.astype(float)) != T.fingerprint(numbers.astype(str))
    assert T.fingerprint(iris.iloc[::-1]) != base


def test_compare_models_fingerprints_the_dataset_once(iris, monkeypatch):
    calls = []
    original = T.fingerprint
    monkeypatch.setattr(T, "fingerprint", lambda raw: calls.append(1) or original(raw))
    results = T.compare_models(iris, "species", FEATURES, ["Logistic regression", "Decision tree", "Random forest"])
    assert len(calls) == 1
    assert {r.metadata["dataset_sha256"] for r in results} == {original(iris)}
    calls.clear()
    T.compare_models(iris, "species", FEATURES, ["Logistic regression"], dataset_fingerprint="cached")
    assert not calls


def test_benchmark_script_runs_on_a_small_frame():
    spec = importlib.util.spec_from_file_location("benchmark_fingerprint", ROOT / "scripts" / "benchmark_fingerprint.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    result = benchmark.compare(rows=500, columns=5, models=2, repeats=1)  # no timing assertions: machines differ
    assert result["rows"] == 500 and result["old_seconds"] >= 0 and result["new_seconds"] >= 0
    frame = benchmark.synthetic(50, 4)
    assert T.fingerprint(frame) == T.fingerprint(benchmark.synthetic(50, 4))


def test_progress_reports_each_validation_fit_honestly(iris):
    events = []
    T.train_and_evaluate(iris, "species", FEATURES, cv_folds=3, progress=lambda *event: events.append(event))
    labels = [label for _, _, label in events]
    assert labels == [
        "Cross-validation: 0/3 fits", "Cross-validation: fold 1/3", "Cross-validation: fold 2/3",
        "Refitting the selected model on all training rows", "Predicting held-out rows",
    ]
    assert [done for done, _, _ in events[:-1]] == [0, 1, 2, 3] and {total for _, total, _ in events[:-1]} == {4}


def test_progress_counts_every_hyperparameter_fit(iris):
    events = []
    result = T.train_and_evaluate(iris, "species", FEATURES, classifier="Decision tree", cv_folds=3, tune=True, progress=lambda *event: events.append(event))
    fits = len(result.cv_results) * 3
    search = [label for _, _, label in events if label.startswith("Hyperparameter search")]
    assert len(search) == fits  # the initial 0/N event plus every fit except the last
    assert search[-1] == f"Hyperparameter search: {fits - 1}/{fits} fits ({len(result.cv_results)} configurations × 3 folds)"
    assert events[-2][2] == "Refitting the selected model on all training rows"


def test_compare_models_progress_is_monotonic_and_nested(iris):
    events = []
    T.compare_models(iris, "species", FEATURES, ["Logistic regression", "Decision tree"], cv_folds=2, progress=lambda *event: events.append(event))
    positions = [position for position, _, _ in events]
    assert positions == sorted(positions) and events[-1] == (2, 2, "Complete")
    assert any(label.startswith("Decision tree — Cross-validation: fold") for _, _, label in events)


# ---------------------------------------------------------------------------
# History and explanations


def test_history_table_records_split_cv_and_imbalance_settings(iris):
    iris["site"] = [f"g{i % 10}" for i in range(len(iris))]
    runs = {
        "run 1": T.train_and_evaluate(iris, "species", FEATURES, cv_folds=3, tune=True, seed=1),
        "run 2": T.train_and_evaluate(iris, "species", FEATURES, classifier="Random forest", split_method="Grouped", split_column="site", imbalance="Class weights"),
    }
    table = T.history_table(runs)
    assert table.columns.tolist() == [
        "run", "trained at (UTC)", "task", "model", "target", "test proportion", "split method", "group column",
        "time column", "seed", "CV method", "CV folds", "tuned", "imbalance", "primary metric", "CV score",
        "held-out score", "train rows", "test rows", "features", "split ID",
    ]
    first, second = table.to_dict("records")
    assert first["CV method"] == "StratifiedKFold" and first["CV folds"] == 3 and first["tuned"]
    assert first["primary metric"] == "balanced accuracy" and first["held-out score"] == runs["run 1"].metrics["balanced_accuracy"]
    assert pd.isna(first["group column"]) and first["trained at (UTC)"].endswith("+00:00")
    assert second["split method"] == "Grouped" and second["group column"] == "site" and second["CV method"] == "none"
    assert second["imbalance"] == "Class weights" and second["seed"] == 42
    assert not T.comparable(list(runs.values()))


@pytest.mark.parametrize("name, kind", [
    ("Random forest", "Impurity-based"), ("Gradient boosting", None), ("Decision tree", "Impurity-based"),
    ("Logistic regression", "Mean absolute coefficient"), ("Calibrated logistic regression", "Mean absolute coefficient"),
    ("3-nearest neighbors", None), ("Support vector machine", None),
])
def test_model_importance_or_explicit_fallback(iris, name, kind):
    result = T.train_and_evaluate(iris, "species", FEATURES, classifier=name)
    note, table = T.model_importance(result)
    if kind:
        assert note.startswith(kind) and set(table["feature"]) == set(FEATURES)
    else:
        assert table.empty and "Use permutation importance" in note


@pytest.mark.parametrize("name", ["Random forest", "Support vector machine", "Calibrated logistic regression", "3-nearest neighbors"])
def test_unavailable_row_explanations_say_so(iris, name):
    note, table = T.row_explanation(T.train_and_evaluate(iris, "species", FEATURES, classifier=name), 1)
    assert table.empty and note.startswith(T.LOCAL_UNAVAILABLE)
    assert T.LOCAL_UNAVAILABLE == "Local row explanations are not available for this model. Global feature importance is shown instead."
