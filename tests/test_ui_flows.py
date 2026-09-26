"""Headless AppTest journeys for training-mode screens added after the original heatmap:
regression, model comparison, thresholds, experiment history, leakage, target guidance,
SVM probabilities, multiclass probability mapping and explanation fallbacks."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import analysis as A
import evaluation as E
import training as T

ROOT = Path(__file__).resolve().parents[1]
IRIS = (ROOT / "iris.csv").read_bytes()
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
TRAIN_MODE = "Train a model and evaluate it"


def run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def plain(text: str) -> str:
    return text.replace("\\", "")


def selectbox(at, label):
    return next(s for s in at.selectbox if plain(s.label) == label)


def multiselect(at, label):
    return next(m for m in at.multiselect if plain(m.label) == label)


def checkbox(at, label):
    return next(c for c in at.checkbox if plain(c.label) == label)


def metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def downloads(at) -> list[str]:
    return [d.proto.label for d in at.get("download_button")]


def texts(elements) -> str:
    return "\n".join(plain(e.value) for e in elements)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


@pytest.fixture
def app() -> AppTest:
    return run(AppTest.from_file(str(ROOT / "app.py"), default_timeout=120))


def open_training(at, content=IRIS, name="iris.csv", task="Classification"):
    at.radio(key="workflow").set_value(TRAIN_MODE)
    at.radio(key="prediction_task").set_value(task)
    run(at)
    at.file_uploader(key="train_upload").set_value((name, content, "text/csv"))
    return run(at)


def train(at):
    at.button(key="train_button").click()
    return run(at)


def test_regression_screen_shows_regression_metrics_only(app):
    open_training(app, task="Regression")
    assert any(h.value == "1 · Train a regressor" for h in app.header)
    selectbox(app, "Target column").set_value("petal_length")
    run(app)
    assert not any(s.label == "Class imbalance handling" for s in app.selectbox)
    train(app)
    shown = metrics(app)
    assert {"MAE", "RMSE", "R²", "Evaluated rows"} <= set(shown)
    assert "Accuracy" not in shown and "ROC-AUC" not in shown
    assert not any(s.value == "Confusion matrix" for s in app.subheader)
    assert any(h.value == "2 · Held-out test performance" for h in app.header)
    assert any(e.label == "Regression residuals" for e in app.expander)
    result = app.session_state["training_result"]
    assert result.task == "regression" and result.metadata["cv_metric"] == "MAE"
    assert shown["MAE"] == f"{result.metrics['mae']:.4g}"


def test_model_comparison_uses_identical_rows_and_blocks_uncontrolled_comparisons(app):
    open_training(app)
    selectbox(app, "Classifier").set_value("Logistic regression")
    run(app)
    multiselect(app, "Also compare these models").set_value(["Random forest"])
    run(app)
    train(app)
    assert any(s.value == "Model comparison on the same held-out test set" for s in app.subheader)
    batch = app.session_state["training_batch"]
    assert [r.classifier for r in batch] == ["Logistic regression", "Random forest"]
    assert batch[0].metadata["split_id"] == batch[1].metadata["split_id"]
    assert batch[0].test_rows == batch[1].test_rows and batch[0].train_rows == batch[1].train_rows
    table = next(d.value for d in app.dataframe if "CV mean" in d.value.columns)
    assert table["split"].nunique() == 1 and table["test rows"].tolist() == [30, 30]
    selectbox(app, "Model to inspect").set_value("Random forest")
    run(app)
    assert any("Random forest" in s.value for s in app.success)

    next(n for n in app.number_input if n.label == "Random seed").set_value(7)
    run(app)
    train(app)
    history = list(app.session_state["experiment_history"])
    assert len(history) == 4  # both models again, now with seed 7
    compare = multiselect(app, "Compare saved runs")
    compare.set_value(history[:2])
    run(app)
    assert not any("not a controlled model comparison" in w.value for w in app.warning)
    multiselect(app, "Compare saved runs").set_value([history[0], history[2]])
    run(app)
    assert any("These runs use different test sets or targets" in w.value for w in app.warning)


def test_binary_threshold_changes_metrics_and_keeps_downloads(app):
    iris = A.load_csv(IRIS).frame
    open_training(app, csv_bytes(iris[iris["species"] != "setosa"]), "binary_iris.csv")
    selectbox(app, "Classifier").set_value("Logistic regression")
    run(app)
    train(app)
    assert selectbox(app, "Positive class").value == "virginica"
    default_accuracy = metrics(app)["Accuracy"]
    checkbox(app, "Explore a decision threshold").set_value(True)
    run(app)
    slider = next(s for s in app.slider if s.label == "Positive decision threshold")
    assert slider.value == .5
    assert metrics(app)["Accuracy"] == default_accuracy
    seen = set()
    for threshold in (.05, .95):
        next(s for s in app.slider if s.label == "Positive decision threshold").set_value(threshold)
        run(app)
        seen.add(metrics(app)["Accuracy"])
        assert any(f"P(virginica) ≥ {threshold:.2f}" in plain(i.value) for i in app.info)
    assert seen - {default_accuracy}
    assert {"Download held-out prediction CSV", "Download training metadata", "Download displayed confusion matrix"} <= set(downloads(app))
    app.button(key="prepare_report").click()
    run(app)
    assert {"Download complete evaluation bundle", "Download HTML evaluation report", "Download audit metadata"} <= set(downloads(app))
    audit = json.loads(app.session_state["prepared_report"][1]["audit_metadata.json"])
    assert audit["probability_settings"]["threshold"] == .95
    assert audit["probability_settings"]["positive_class"] == "virginica"


def test_experiment_history_records_each_run(app):
    open_training(app)
    selectbox(app, "Classifier").set_value("Decision tree")
    run(app)
    train(app)
    next(n for n in app.number_input if n.label == "Random seed").set_value(7)
    run(app)
    train(app)
    history = next(e for e in app.expander if e.label == "Session experiment history")
    table = next(d.value for d in history.dataframe if "split method" in d.value.columns)
    assert table["seed"].tolist() == [42, 7]
    assert table["model"].tolist() == ["Decision tree", "Decision tree"]
    assert set(table["split method"]) == {"Random"} and set(table["CV method"]) == {"StratifiedKFold"}
    assert table["trained at (UTC)"].notna().all() and table["split ID"].nunique() == 2
    assert {"Download experiment history", "Download full run metadata (JSON)"} <= set(downloads(app))
    selectbox(app, "Show full metadata for run").set_value(table["run"].iloc[0])
    run(app)
    assert app.get("json")


def test_leakage_warning_names_the_feature_and_reason(app):
    iris = A.load_csv(IRIS).frame
    iris["species_code"] = iris["species"].map({"setosa": "0", "versicolor": "1", "virginica": "2"})
    open_training(app, csv_bytes(iris), "iris_with_code.csv")
    assert any(w.value == "Possible target leakage: review these selected features before training." for w in app.warning)
    markdown = texts(app.markdown)
    assert "**species_code** · Probable target leakage" in markdown
    assert "label-encoded copy of the target" in markdown
    quality = next(e for e in app.expander if e.label.startswith("Dataset quality summary"))
    assert "need attention" in quality.label
    overview = next(d.value for d in quality.dataframe if "check" in d.value.columns).set_index("check")
    assert overview.loc["Potential leakage", "status"] == "Warning"
    train(app)
    review = app.session_state["training_result"].metadata["leakage_review"]
    assert [item["feature"] for item in review] == ["species_code"]


def test_continuous_target_gets_regression_guidance(app):
    open_training(app)
    selectbox(app, "Target column").set_value("sepal_length")
    run(app)
    assert any("appears continuous" in w.value and "Consider switching the task type to Regression" in w.value for w in app.warning)
    assert not any(w.value.startswith("Class imbalance") for w in app.warning)
    treat = checkbox(app, "Treat the values of “sepal_length” as classes anyway")
    treat.set_value(True)
    run(app)
    assert any("only one example" in w.value for w in app.warning)
    train(app)
    assert any("appears continuous" in e.value and "Try Regression" in e.value for e in app.error)


def test_svm_reports_roc_auc_with_averaging_controls(app):
    open_training(app)
    selectbox(app, "Classifier").set_value("Support vector machine")
    run(app)
    train(app)
    assert "ROC-AUC" in metrics(app)
    result = app.session_state["training_result"]
    assert list(result.probability_columns) == ["setosa", "versicolor", "virginica"]
    app.radio(key="auc_average").set_value("weighted")
    app.radio(key="auc_multi_class").set_value("ovo")
    run(app)
    assert "ROC-AUC" in metrics(app)
    assert any("Prevalence-weighted one-vs-one ROC-AUC" in c.value for c in app.caption)


def test_grouped_svm_explains_missing_probabilities(app):
    iris = A.load_csv(IRIS).frame
    iris["site"] = [f"g{i % 10}" for i in range(len(iris))]
    open_training(app, csv_bytes(iris), "iris_sites.csv")
    selectbox(app, "Split method").set_value("Grouped")
    run(app)
    selectbox(app, "Group column").set_value("site")
    selectbox(app, "Classifier").set_value("Support vector machine")
    run(app)
    train(app)
    assert "ROC-AUC" not in metrics(app)
    info = texts(app.info)
    assert "ROC-AUC is unavailable because this model does not provide class probabilities" in info
    assert "Confidence analysis is unavailable" in info


def test_uploaded_multiclass_probabilities_can_be_mapped_for_roc_auc(app):
    trained = T.train_and_evaluate(A.load_csv(IRIS).frame, "species", FEATURES, classifier="Logistic regression")
    at = app
    at.radio(key="source").set_value("Upload a CSV")
    run(at)
    at.file_uploader(key="upload").set_value(("heldout.csv", trained.csv_bytes(), "text/csv"))
    run(at)
    assert "ROC-AUC" not in metrics(at)
    checkbox(at, "Map class probability columns (multiclass ROC-AUC)").set_value(True)
    run(at)
    for label, column in trained.probability_columns.items():
        selectbox(at, f"P({label})").set_value(column)
    run(at)
    assert not at.error
    checkbox(at, "Each column is P(its class) between 0 and 1, from the same model as the predictions").set_value(True)
    run(at)
    trained_rows = A.evaluate(trained.frame, trained.actual, trained.predicted).rows
    expected = pd.DataFrame({label: trained.frame[c] for label, c in trained.probability_columns.items()})
    assert metrics(at)["ROC-AUC"] == f"{E.probability_auc(trained_rows, expected)[0]:.4f}"

    selectbox(at, "P(virginica)").set_value(trained.probability_columns["setosa"])
    run(at)
    assert any("mapped to more than one class" in e.value for e in at.error)
    assert "ROC-AUC" not in metrics(at)


def test_dataset_is_fingerprinted_once_per_upload(app, monkeypatch):
    calls = []
    original = T.fingerprint
    monkeypatch.setattr(T, "fingerprint", lambda raw: calls.append(len(raw)) or original(raw))
    iris = A.load_csv(IRIS).frame.assign(batch=[str(i % 7) for i in range(150)])  # not cached by earlier tests
    open_training(app, csv_bytes(iris), "iris_batch.csv")
    assert calls == [150]
    selectbox(app, "Classifier").set_value("Logistic regression")
    run(app)
    multiselect(app, "Also compare these models").set_value(["Random forest", "Decision tree"])
    run(app)
    train(app)
    next(n for n in app.number_input if n.label == "Random seed").set_value(3)
    run(app)
    train(app)
    assert calls == [150]
    assert len({r.metadata["dataset_sha256"] for r in app.session_state["experiment_history"].values()}) == 1


def test_models_without_local_explanations_say_so_and_fall_back_to_global_importance(app):
    open_training(app)
    train(app)  # the default classifier is 3-nearest neighbors
    explanations = next(e for e in app.expander if e.label == "Feature importance and row explanations")
    assert any("exposes no built-in feature importance" in c.value for c in explanations.caption)
    checkbox(app, "Explain an individual test row").set_value(True)
    run(app)
    assert any(i.value.startswith(T.LOCAL_UNAVAILABLE) for i in app.info)
    importance = next(d.value for d in app.dataframe if "importance" in d.value.columns and "std" in d.value.columns)
    assert set(importance["feature"]) == set(FEATURES)
