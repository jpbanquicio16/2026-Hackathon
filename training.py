"""Leakage-safe training, cross-validation, model comparison and provenance."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, GroupShuffleSplit, KFold, RepeatedKFold,
    RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.metaestimators import available_if

import analysis as A
import evaluation as E

CLASSIFIERS = (
    "3-nearest neighbors", "Logistic regression", "Decision tree",
    "Random forest", "Support vector machine", "Gradient boosting", "Calibrated logistic regression",
)
REGRESSORS = (
    "Ridge regression", "3-nearest neighbors", "Decision tree",
    "Random forest", "Support vector machine", "Gradient boosting",
)
SPLITS = ("Random", "Grouped", "Time ordered")


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Random oversampling happens inside fit, independently in each CV fold."""

    def __init__(self, estimator, random_state=42):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        values, labels = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(labels, return_counts=True)
        rng = np.random.RandomState(self.random_state)
        indices = list(range(len(labels)))
        for label, count in zip(classes, counts):
            if count < counts.max():
                indices.extend(rng.choice(np.flatnonzero(labels == label), int(counts.max() - count), replace=True))
        self.estimator_ = clone(self.estimator).fit(values[indices], labels[indices])
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = values.shape[1]
        self.n_resampled_ = len(indices)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)


@dataclass(frozen=True)
class TrainingResult:
    frame: pd.DataFrame
    actual: str
    predicted: str
    ids: tuple[str, ...]
    train_rows: tuple[int, ...]
    test_rows: tuple[int, ...]
    missing_targets: int
    classifier: str
    seed: int
    model: object
    task: str = "classification"
    features: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    cv_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    training_metrics: dict = field(default_factory=dict)
    probability_columns: dict[str, str] = field(default_factory=dict)
    confidence: str | None = None
    X_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    X_test: pd.DataFrame = field(default_factory=pd.DataFrame)
    y_train: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    y_test: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    notes: tuple[str, ...] = ()

    def csv_bytes(self) -> bytes:
        return self.frame.to_csv(index=False).encode("utf-8")

    def metadata_bytes(self) -> bytes:
        return json.dumps(self.metadata, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(raw: pd.DataFrame) -> str:
    return hashlib.sha256(raw.to_csv(index=True).encode()).hexdigest()


def feature_choices(raw: pd.DataFrame, target: str, split_column: str | None = None) -> A.FeatureOptions:
    ids = A.suggest_id_columns(raw, exclude=(target,))
    reserved = {target: "target", **{column: "identifier" for column in ids}}
    if split_column:
        reserved[split_column] = "split column"
    return A.feature_options(raw, raw.index, reserved)


def _estimator(name: str, task: str, seed: int, balance: bool):
    if task == "classification":
        choices = {
            "3-nearest neighbors": KNeighborsClassifier(n_neighbors=3),
            "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
            "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
            "Random forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
            # Probability calibration is offered separately as a full pipeline.
            "Support vector machine": SVC(C=1, kernel="rbf", random_state=seed),
            "Gradient boosting": HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed),
            "Calibrated logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
        }
    else:
        choices = {
            "Ridge regression": Ridge(alpha=1),
            "3-nearest neighbors": KNeighborsRegressor(n_neighbors=3),
            "Decision tree": DecisionTreeRegressor(max_depth=5, random_state=seed),
            "Random forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
            "Support vector machine": SVR(C=1, kernel="rbf"),
            "Gradient boosting": HistGradientBoostingRegressor(max_iter=100, early_stopping=False, random_state=seed),
        }
    if name not in choices:
        raise A.DataError("Choose a supported model for this task.")
    estimator = choices[name]
    if balance:
        estimator = ResampledClassifier(estimator, random_state=seed)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])
    prefix = "classifier__" + ("estimator__" if balance else "")
    if name == "Calibrated logistic regression":
        # The entire preprocessing pipeline is refitted in each calibration fold.
        pipeline = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=2, ensemble=False, n_jobs=1)
        prefix = "estimator__" + prefix
    return pipeline, prefix


def _grid(name: str, prefix: str, min_train: int) -> dict:
    candidates = {
        "3-nearest neighbors": {"n_neighbors": [n for n in (3, 5, 9) if n <= min_train], "weights": ["uniform", "distance"]},
        "Logistic regression": {"C": [.1, 1, 10]},
        "Calibrated logistic regression": {"C": [.1, 1, 10]},
        "Ridge regression": {"alpha": [.1, 1, 10]},
        "Decision tree": {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]},
        "Random forest": {"max_depth": [8, None], "min_samples_leaf": [1, 3]},
        "Support vector machine": {"C": [.1, 1, 10]},
        "Gradient boosting": {"learning_rate": [.05, .1], "max_leaf_nodes": [15, 31]},
    }
    return {prefix + key: values for key, values in candidates[name].items()}


def _split(raw, labels, test_size, seed, task, method, split_column):
    usable = labels.index
    split_values = None
    if method in ("Grouped", "Time ordered"):
        if split_column not in raw.columns:
            raise A.DataError("Choose a group or time column for this split.")
        if method == "Grouped":
            split_values = A.clean_labels(raw.loc[usable, split_column])
        else:
            split_values = pd.to_datetime(raw.loc[usable, split_column], errors="coerce", format="mixed", utc=True)
        usable = split_values.dropna().index
        split_values = split_values.loc[usable]
    labels = labels.loc[usable]
    if len(labels) < 4:
        raise A.DataError("At least four usable labelled rows are needed for a held-out split.")
    n_test = math.ceil(len(labels) * test_size)
    n_train = len(labels) - n_test
    if method == "Random":
        counts = labels.value_counts()
        if task == "classification":
            if len(counts) < 2:
                raise A.DataError("Classification needs at least two classes with non-missing targets.")
            if counts.min() < 2:
                raise A.DataError("A stratified held-out split needs at least two rows in every class. Some classes have only one row; add examples or choose another target.")
            if min(n_test, n_train) < len(counts):
                raise A.DataError(f"This split gives {n_train} training and {n_test} test rows for {len(counts)} classes. Both sets need at least one row per class; change the proportion or add examples.")
        train, test = train_test_split(usable.to_numpy(), test_size=test_size, random_state=seed, stratify=labels if task == "classification" else None)
        if task == "classification" and (set(labels.loc[train]) != set(labels) or set(labels.loc[test]) != set(labels)):
            raise A.DataError("The stratified split could not put every class in both sets. Change the proportion or add rare-class examples.")
    elif method == "Grouped":
        if split_values.nunique() < 2:
            raise A.DataError("A grouped split needs at least two distinct groups.")
        train_pos, test_pos = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(usable, labels, split_values))
        train, test = usable[train_pos].to_numpy(), usable[test_pos].to_numpy()
    elif method == "Time ordered":
        ordered = split_values.sort_values(kind="stable")
        cutoff = ordered.iloc[n_train]
        train = ordered.index[ordered < cutoff].to_numpy()
        test = ordered.index[ordered >= cutoff].to_numpy()
        if len(train) == 0:
            raise A.DataError("The time boundary leaves no training rows. Choose another proportion or provide more distinct times.")
    else:
        raise A.DataError("Choose a supported split method.")
    if len(train) < 2 or len(test) < 1:
        raise A.DataError("The split needs at least two training rows and one test row.")
    if task == "classification" and labels.loc[train].nunique() < 2:
        raise A.DataError("The training split contains only one class. Change the split or add examples.")
    return labels, train, test, split_values


def _cv_splits(labels, split_values, task, method, folds, repeats, seed):
    notes = []
    requested = folds
    if method == "Random":
        max_folds = int(labels.value_counts().min()) if task == "classification" else len(labels)
        folds = min(folds, max_folds)
        if folds < 2:
            raise A.DataError("Cross-validation needs at least two training examples per class. Disable it or add rare-class examples.")
        if task == "classification":
            cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed) if repeats > 1 else StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed) if repeats > 1 else KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(labels)), labels))
    elif method == "Grouped":
        groups = split_values.loc[labels.index]
        folds = min(folds, groups.nunique())
        if folds < 2:
            raise A.DataError("Grouped cross-validation needs at least two training groups. Disable it or add groups.")
        splits = list(GroupKFold(n_splits=folds).split(np.zeros(len(labels)), labels, groups))
    else:
        times = split_values.loc[labels.index]
        unique_times = np.sort(times.unique())
        folds = min(folds, len(unique_times) - 1)
        if folds < 2:
            raise A.DataError("Time-ordered cross-validation needs at least three distinct training times.")
        splits = []
        for train_times, val_times in TimeSeriesSplit(n_splits=folds).split(unique_times):
            splits.append((np.flatnonzero(times.isin(unique_times[train_times])), np.flatnonzero(times.isin(unique_times[val_times]))))
    if folds != requested:
        notes.append(f"Cross-validation uses {folds} folds rather than {requested} to fit the available training data.")
    for train, validation in splits:
        if len(train) < 2 or (task == "classification" and labels.iloc[train].nunique() < 2):
            raise A.DataError("A validation fold has too few rows or only one training class. Reduce folds, disable cross-validation or add examples.")
    return splits, notes


def _metrics(actual, predicted, task):
    frame = pd.DataFrame({"actual": np.asarray(actual), "predicted": np.asarray(predicted)})
    if task == "classification":
        frame["is_error"] = frame["actual"] != frame["predicted"]
        return E.classification_metrics(frame)[0]
    evaluation = E.regression_evaluation(frame, "actual", "predicted")
    return E.regression_metrics(evaluation.rows)


def _plain(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else str(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    return type(value).__name__


def train_and_evaluate(
    raw: pd.DataFrame, target: str, features: tuple[str, ...],
    test_size: float = .2, seed: int = 42, classifier: str = CLASSIFIERS[0],
    *, task: str = "classification", split_method: str = "Random", split_column: str | None = None,
    cv_folds: int = 0, cv_repeats: int = 1, tune: bool = False, balance: bool = False,
) -> TrainingResult:
    if target not in raw.columns:
        raise A.DataError("Choose a target column.")
    if task not in ("classification", "regression"):
        raise A.DataError("Choose Classification or Regression.")
    if split_column == target:
        raise A.DataError("The target cannot be the split column.")
    allowed = feature_choices(raw, target, split_column).usable
    if not features:
        raise A.DataError("Choose at least one numeric training feature.")
    if len(set(features)) != len(features) or any(c not in allowed for c in features):
        raise A.DataError("Training features must be numeric, distinct, and exclude the target and identifiers or split column.")
    if not 0 < test_size < 1 or not 0 <= seed <= 2**32 - 1:
        raise A.DataError("Use a test proportion between 0 and 1 and a seed between 0 and 4294967295.")
    if balance and task != "classification":
        raise A.DataError("Class balancing is available only for classification.")
    if not 0 <= cv_folds <= 10 or cv_folds == 1 or not 1 <= cv_repeats <= 5:
        raise A.DataError("Use 2–10 validation folds (or disable validation) and 1–5 repeats.")
    if tune and cv_folds < 2:
        raise A.DataError("Hyperparameter selection requires cross-validation on training data.")
    if split_method != "Random" and cv_repeats != 1:
        raise A.DataError("Grouped and time-ordered validation use one pass of folds, not repeated random splits.")
    if classifier == "Calibrated logistic regression" and split_method != "Random":
        raise A.DataError("Calibrated logistic regression currently supports random splits only; use another classifier for grouped or time-ordered data.")

    original_labels = A.clean_labels(raw[target]) if task == "classification" else A.parse_numeric(raw[target]).values
    missing_targets = int(original_labels.isna().sum())
    labels, train_rows, test_rows, split_values = _split(raw, original_labels.dropna(), test_size, seed, task, split_method, split_column)
    values = pd.DataFrame({c: A.parse_numeric(raw[c]).values for c in features})
    X_train, y_train = values.loc[train_rows], labels.loc[train_rows]
    empty_train = X_train.columns[X_train.isna().all()].tolist()
    if empty_train:
        raise A.DataError("No usable training values for: " + ", ".join(empty_train) + ". Remove those features or choose a different split.")
    if classifier == "3-nearest neighbors" and len(train_rows) < 3:
        raise A.DataError("3-nearest neighbors needs at least three training rows; choose another classifier.")
    estimator, prefix = _estimator(classifier, task, seed, balance)
    notes, splits, cv_results = [], [], pd.DataFrame()
    best_params, cv_mean, cv_std = {}, None, None
    metric = "balanced_accuracy" if task == "classification" else "neg_mean_absolute_error"
    if cv_folds:
        splits, cv_notes = _cv_splits(y_train, split_values, task, split_method, cv_folds, cv_repeats, seed)
        notes.extend(cv_notes)
        smallest = min(len(tr) for tr, _ in splits)
        if classifier == "3-nearest neighbors" and smallest < 3:
            raise A.DataError("Nearest neighbors needs at least three rows in every validation training fold. Reduce folds or choose another model.")
        if classifier == "Calibrated logistic regression" and any(y_train.iloc[tr].value_counts().min() < 2 for tr, _ in splits):
            raise A.DataError("Calibration needs two examples per class inside every validation training fold. Add examples or choose another classifier.")
    if classifier == "Calibrated logistic regression" and y_train.value_counts().min() < 2:
        raise A.DataError("Calibration needs at least two training examples in every class.")

    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            if splits:
                grid = _grid(classifier, prefix, min(len(tr) for tr, _ in splits)) if tune else {}
                search = GridSearchCV(estimator, grid, scoring=metric, cv=splits, refit=True, n_jobs=1, error_score="raise")
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_mean = float(search.best_score_) * (1 if task == "classification" else -1)
                cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
                cv_results = pd.DataFrame({
                    "parameters": [json.dumps(_plain(p), sort_keys=True) for p in search.cv_results_["params"]],
                    "mean validation score": np.asarray(search.cv_results_["mean_test_score"]) * (1 if task == "classification" else -1),
                    "standard deviation": search.cv_results_["std_test_score"],
                    "rank": search.cv_results_["rank_test_score"],
                }).sort_values("rank")
            else:
                model = estimator.fit(X_train, y_train)
            notes.extend(dict.fromkeys(str(w.message) for w in recorded if issubclass(w.category, ConvergenceWarning)))
    except ValueError as exc:
        raise A.DataError(f"The model could not fit this data: {exc}") from exc

    test_rows = sorted(int(row) for row in test_rows)
    X_test, y_test = values.loc[test_rows], labels.loc[test_rows]
    predicted = model.predict(X_test)
    original_ids = A.suggest_id_columns(raw, exclude=(target,))
    if split_column and split_column not in original_ids:
        original_ids.append(split_column)
    result = raw.loc[test_rows, [*original_ids, *features]].copy()

    def available_name(base):
        name, suffix = base, 2
        while name in result.columns:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    source_id = available_name("source_row_id")
    result.insert(0, source_id, test_rows)
    actual_col, predicted_col = available_name("actual_label"), available_name("predicted_label")
    result[actual_col] = y_test.to_numpy()
    result[predicted_col] = predicted
    probability_columns, confidence = {}, None
    if task == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        for i, label in enumerate(model.classes_):
            column = available_name(f"probability_{i}")
            probability_columns[str(label)] = column
            result[column] = probabilities[:, i]
        confidence = available_name("predicted_confidence")
        class_index = {label: i for i, label in enumerate(model.classes_)}
        result[confidence] = [probabilities[i, class_index[label]] for i, label in enumerate(predicted)]
    result.index = pd.RangeIndex(1, len(result) + 1, name="row")
    metrics = _metrics(y_test, predicted, task)
    training_metrics = _metrics(y_train, model.predict(X_train), task)
    if task == "classification":
        unseen = set(y_test) - set(y_train)
        if unseen:
            notes.append("Test classes absent from training: " + ", ".join(A.sort_labels(unseen)) + ". They remain in the evaluation.")
    dataset_hash = fingerprint(raw)
    split_identity = {"dataset_sha256": dataset_hash, "task": task, "target": target, "train_rows": sorted(int(v) for v in train_rows), "test_rows": test_rows}
    metadata = {
        "schema_version": 1, "performance_scope": "Held-out test performance", "task": task,
        "target": target, "features": list(features), "model": classifier, "seed": seed,
        "requested_test_proportion": test_size, "split_method": split_method, "split_column": split_column,
        "dataset_sha256": dataset_hash, "split_id": hashlib.sha256(json.dumps(split_identity, sort_keys=True).encode()).hexdigest(),
        "train_row_ids": [int(row) for row in train_rows], "test_row_ids": test_rows,
        "training_rows": len(train_rows), "test_rows": len(test_rows), "missing_targets": missing_targets,
        "missing_split_values": len(original_labels.dropna()) - len(labels),
        "class_balancing": "random oversampling inside each training fit" if balance else "none",
        "preprocessing": "Training-only median imputation and standard scaling; empty feature in a CV training fold uses 0.",
        "tuned": tune, "selected_parameters": _plain(best_params),
        "fitted_parameters": _plain(model.get_params(deep=True)),
        "cv_requested_folds": cv_folds, "cv_repeats": cv_repeats, "cv_splits": len(splits),
        "cv_metric": "balanced accuracy" if task == "classification" else "MAE",
        "cv_mean": cv_mean, "cv_std": cv_std,
        "cv_row_ids": [{"train": y_train.index[tr].tolist(), "validation": y_train.index[va].tolist()} for tr, va in splits],
        "actual_column": actual_col, "predicted_column": predicted_col, "id_columns": [source_id, *original_ids],
        "probability_columns": probability_columns, "confidence_column": confidence,
        "test_metrics": metrics, "training_metrics": training_metrics, "notes": notes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    return TrainingResult(
        result, actual_col, predicted_col, (source_id, *original_ids), tuple(int(row) for row in train_rows),
        tuple(test_rows), missing_targets, classifier, seed, model, task, features, metadata,
        cv_results, metrics, training_metrics, probability_columns, confidence,
        X_train, X_test, y_train, y_test, tuple(notes),
    )


def compare_models(raw, target, features, models, progress: Callable | None = None, **settings):
    results = []
    for i, name in enumerate(models):
        if progress:
            progress(i, len(models), name)
        result = train_and_evaluate(raw, target, features, classifier=name, **settings)
        if results and result.metadata["split_id"] != results[0].metadata["split_id"]:
            raise RuntimeError("Model comparison must use identical training and test rows.")
        results.append(result)
    if progress:
        progress(len(models), len(models), "Complete")
    return results


def comparison_table(results: list[TrainingResult]) -> pd.DataFrame:
    records = []
    for result in results:
        row = {
            "model": result.classifier, "task": result.task, "seed": result.seed,
            "train rows": len(result.train_rows), "test rows": len(result.test_rows),
            "split": result.metadata["split_id"][:12], "features": ", ".join(result.features),
            "CV metric": result.metadata["cv_metric"], "CV mean": result.metadata["cv_mean"], "CV std": result.metadata["cv_std"],
        }
        row.update({f"test {k}": v for k, v in result.metrics.items() if k in ("accuracy", "macro_f1", "balanced_accuracy", "mae", "rmse", "r2")})
        records.append(row)
    return pd.DataFrame(records)


def permutation_scores(result: TrainingResult, repeats: int = 3) -> pd.DataFrame:
    scoring = "balanced_accuracy" if result.task == "classification" else "neg_mean_absolute_error"
    importance = permutation_importance(
        result.model, result.X_test, result.y_test, scoring=scoring, n_repeats=repeats,
        random_state=result.seed, n_jobs=1, max_samples=min(500, len(result.X_test)),
    )
    return pd.DataFrame({"feature": result.features, "importance": importance.importances_mean, "std": importance.importances_std}).sort_values("importance", ascending=False)


def row_explanation(result: TrainingResult, row: int) -> tuple[str, pd.DataFrame]:
    if not isinstance(result.model, Pipeline):
        return "Exact row explanations are available for uncalibrated linear models and individual decision trees. Permutation importance is available above.", pd.DataFrame()
    position = int(row) - 1
    values = result.model[:-1].transform(result.X_test.iloc[[position]])[0]
    estimator = result.model["classifier"]
    if isinstance(estimator, ResampledClassifier):
        estimator = estimator.estimator_
    if hasattr(estimator, "coef_"):
        coefficient = np.asarray(estimator.coef_)
        intercept = np.asarray(estimator.intercept_).reshape(-1)
        if coefficient.ndim == 1:
            coefficient = coefficient[None, :]
        if result.task == "classification":
            predicted = result.frame.loc[row, result.predicted]
            which = list(estimator.classes_).index(predicted)
            sign = 1
            if len(estimator.classes_) == 2:
                sign = 1 if which == 1 else -1
                which = 0
            coefficient = coefficient[which] * sign
            base = float(intercept[which] * sign)
            kind = "predicted-class linear score (binary: log-odds)"
        else:
            coefficient, base, kind = coefficient[0], float(intercept[0]), "predicted value"
        contributions = values * coefficient
        table = pd.DataFrame({"feature": result.features, "transformed value": values, "coefficient": coefficient, "contribution": contributions})
        return f"Training-imputed and scaled values × coefficients, plus intercept {base:.6g}, sum to {kind} {base + contributions.sum():.6g}. These describe the model, not causal effects.", table
    if hasattr(estimator, "tree_"):
        tree = estimator.tree_
        nodes = estimator.decision_path(values.reshape(1, -1)).indices
        records = []
        for node in nodes:
            feature = tree.feature[node]
            if feature < 0:
                continue
            records.append({"feature": result.features[feature], "transformed value": values[feature], "condition": "≤" if values[feature] <= tree.threshold[node] else ">", "threshold": tree.threshold[node]})
        return "Decision path using training-imputed, standardized features. Conditions describe this tree's prediction, not causal effects.", pd.DataFrame(records)
    return "This model has no exact additive row explanation here. Use permutation importance for global behavior.", pd.DataFrame()
