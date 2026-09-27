"""Leakage-safe training, cross-validation, model comparison and provenance."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ParameterGrid,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.metaestimators import available_if

import analysis
import diagnostics
from evaluation import classification_metrics, regression_evaluation, regression_metrics

CLASSIFIERS = (
    "3-nearest neighbors", "Logistic regression", "Decision tree",
    "Random forest", "Support vector machine", "Gradient boosting", "Calibrated logistic regression",
)
REGRESSORS = (
    "Ridge regression", "3-nearest neighbors", "Decision tree",
    "Random forest", "Support vector machine", "Gradient boosting",
)
SPLITS = ("Random", "Grouped", "Time ordered")
IMBALANCE_STRATEGIES = ("None", "Class weights", "Oversampling")
CLASS_WEIGHT_MODELS = (
    "Logistic regression", "Calibrated logistic regression", "Decision tree",
    "Random forest", "Support vector machine", "Gradient boosting",
)
# Calibrated models: probabilities come from sigmoid calibration on training folds.
CALIBRATED = ("Calibrated logistic regression", "Support vector machine")
CALIBRATION_FOLDS = 2
GROUP_SPLIT_CANDIDATES = 100
FINGERPRINT_METHOD = "sha256 of schema + pandas row hashes (v2)"


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
        for label, count in zip(classes, counts, strict=True):
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
    """Deterministic dataset identity from structured row hashes plus the schema, so
    frames with equal values but different columns, order or types never match."""
    schema = {
        "columns": [str(c) for c in raw.columns], "dtypes": [str(t) for t in raw.dtypes],
        "index": [str(raw.index.dtype), raw.index.name], "shape": list(raw.shape),
    }
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(raw, index=True, categorize=False).to_numpy().tobytes())
    return digest.hexdigest()


def supports_class_weights(name: str, task: str = "classification") -> bool:
    return task == "classification" and name in CLASS_WEIGHT_MODELS


def _strategy(balance, imbalance) -> str:
    if imbalance is None:
        return "Oversampling" if balance is True else "None" if balance is False else str(balance)
    if balance is True and imbalance != "Oversampling":
        raise analysis.DataError("Choose one class-imbalance strategy: class weights or oversampling, not both.")
    return imbalance


def feature_choices(raw: pd.DataFrame, target: str, split_column: str | None = None) -> analysis.FeatureOptions:
    ids = analysis.suggest_id_columns(raw, exclude=(target,))
    reserved = {target: "target", **{column: "identifier" for column in ids}}
    if split_column:
        reserved[split_column] = "split column"
    return analysis.feature_options(raw, raw.index, reserved)


def _estimator(name: str, task: str, seed: int, balance: bool | str, probabilities: bool = True):
    """`balance` is an imbalance strategy name (True/False kept for older callers).
    `probabilities=False` fits an SVM without calibration."""
    strategy = _strategy(balance, None)
    if strategy not in IMBALANCE_STRATEGIES:
        raise analysis.DataError("Choose a supported class-imbalance strategy.")
    if task == "classification":
        weight = {"class_weight": "balanced"} if strategy == "Class weights" and supports_class_weights(name) else {}
        choices = {
            "3-nearest neighbors": KNeighborsClassifier(n_neighbors=3),
            "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed, **weight),
            "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=seed, **weight),
            "Random forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1, **weight),
            "Support vector machine": SVC(C=1, kernel="rbf", random_state=seed, **weight),
            "Gradient boosting": HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed, **weight),
            "Calibrated logistic regression": LogisticRegression(max_iter=2000, random_state=seed, **weight),
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
        raise analysis.DataError("Choose a supported model for this task.")
    estimator = choices[name]
    oversample = strategy == "Oversampling" and task == "classification"
    if oversample:
        estimator = ResampledClassifier(estimator, random_state=seed)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])
    prefix = "classifier__" + ("estimator__" if oversample else "")
    if task == "classification" and name in CALIBRATED and (probabilities or name != "Support vector machine"):
        # The entire preprocessing pipeline is refitted in each calibration fold, and
        # predictions are the most probable calibrated class, so confidence matches them.
        pipeline = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=CALIBRATION_FOLDS, ensemble=False, n_jobs=1)
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


def _stratification_error(raw: pd.DataFrame, target: str, counts: pd.Series) -> str:
    singles = analysis.sort_labels(counts.index[counts < 2])
    profile = diagnostics.target_profile(raw, target)
    if profile.kind in ("continuous", "ambiguous"):
        return (
            f"Classification cannot create a stratified train/test split because {len(singles):,} of "
            f"{len(counts):,} target values occur only once (a stratified held-out split needs at least two "
            f"rows in every class). “{target}” appears {'continuous' if profile.kind == 'continuous' else 'not to be categorical'}: "
            f"{profile.reason}. Try Regression, or choose a categorical target."
        )
    shown = ", ".join(f"“{label}”" for label in singles[:5]) + (", …" if len(singles) > 5 else "")
    return (
        "A stratified held-out split needs at least two rows in every class. "
        f"{'This class has' if len(singles) == 1 else 'These classes have'} only one row: {shown}. "
        "Add examples of these classes, remove them from the file, or choose another target."
    )


def _fit_error(exc: Exception) -> str:
    """Actionable text for common scikit-learn failures; the original is chained for debugging."""
    text = str(exc)
    if "least populated class" in text or ("less than" in text and "examples for at least one class" in text):
        return (
            "The selected target cannot be stratified because at least one class has too few rows in a training fit. "
            "Choose a different classification target, reduce validation folds, use an appropriate split strategy, "
            f"or switch to Regression if this is a continuous variable. (Details: {text})"
        )
    if "needs samples of at least 2 classes" in text or "only one class" in text:
        return f"A training fit contains only one class. Reduce validation folds, change the split, or add examples. (Details: {text})"
    return f"The model could not fit this data: {text}"


def _coverage(labels, train, test, split_values, method, seed, candidate) -> tuple[dict, list[str]]:
    """Classes missing from either set make recall, ROC-AUC and the confusion matrix incomplete."""
    classes = analysis.sort_labels(set(labels))
    in_train, in_test = set(labels.loc[train]), set(labels.loc[test])
    info = {
        "missing_from_train": [c for c in classes if c not in in_train],
        "missing_from_test": [c for c in classes if c not in in_test],
        "group_split_candidate": candidate,
    }
    info["representative"] = not (info["missing_from_train"] or info["missing_from_test"])
    notes = []
    if candidate and candidate > 1 and info["representative"]:
        notes.append(
            f"The first random group partition for seed {seed} left a class out of training or test, so this split is "
            f"candidate {candidate} drawn from the same seed: the first in which every class appears in both sets. "
            "Groups never cross between sets."
        )
    if info["missing_from_test"]:
        notes.append(
            "Warning: The held-out test set does not contain all target classes. Missing from test set: "
            + ", ".join(info["missing_from_test"]) + ". Metrics such as recall, ROC-AUC and confusion matrices "
            "may be incomplete or misleading."
        )
    if info["missing_from_train"]:
        notes.append(
            "Warning: The training set does not contain all target classes. Missing from training set: "
            + ", ".join(info["missing_from_train"]) + ". The model cannot predict these classes, so their test rows "
            "are always errors."
        )
    if not info["representative"] and method == "Grouped":
        frame = pd.DataFrame({"label": labels, "group": split_values.loc[labels.index]})
        per_class = frame.groupby("label")["group"].nunique()
        nested = bool((frame.groupby("group")["label"].nunique() == 1).all())
        n_train_groups, n_test_groups = frame.loc[train, "group"].nunique(), frame.loc[test, "group"].nunique()
        single = [c for c in classes if per_class.get(c, 0) < 2]
        if len(single) == len(classes):
            notes.append(
                "Every target class occurs in only one group, so groups align with the classes. Any grouped split must "
                "leave whole classes out of training or test; stratification is impossible without splitting groups. "
                "Use a random split if rows are independent, or collect more groups per class."
            )
        elif single:
            notes.append(
                ", ".join(single) + (" occurs" if len(single) == 1 else " occur") + " in only one group, so a grouped "
                "split cannot place " + ("it" if len(single) == 1 else "them") + " in both training and test sets."
            )
        elif nested and min(n_train_groups, n_test_groups) < len(classes):
            notes.append(
                f"Every group contains a single class, and this split has {n_test_groups} test and {n_train_groups} "
                f"training groups: fewer than the {len(classes)} classes on at least one side, so stratification is "
                "impossible at this test proportion. Increase the test proportion, or use groups that span several classes."
            )
        else:
            notes.append(
                f"None of the {GROUP_SPLIT_CANDIDATES} group partitions drawn from seed {seed} placed every class in both "
                "sets. Try another seed or test proportion."
            )
    elif not info["representative"] and method == "Time ordered":
        notes.append(
            "The time boundary places every row of these classes on one side. A time-ordered split cannot be "
            "rebalanced without training on later rows."
        )
    return info, notes


def _split(raw, labels, test_size, seed, task, method, split_column):
    usable = labels.index
    split_values = None
    candidate = None
    if method in ("Grouped", "Time ordered"):
        if split_column not in raw.columns:
            raise analysis.DataError("Choose a group or time column for this split.")
        if method == "Grouped":
            split_values = analysis.clean_labels(raw.loc[usable, split_column])
        else:
            split_values = pd.to_datetime(raw.loc[usable, split_column], errors="coerce", format="mixed", utc=True)
        usable = split_values.dropna().index
        split_values = split_values.loc[usable]
    labels = labels.loc[usable]
    if len(labels) < 4:
        raise analysis.DataError("At least four usable labelled rows are needed for a held-out split.")
    n_test = math.ceil(len(labels) * test_size)
    n_train = len(labels) - n_test
    if method == "Random":
        counts = labels.value_counts()
        if task == "classification":
            if len(counts) < 2:
                raise analysis.DataError("Classification needs at least two classes with non-missing targets.")
            if counts.min() < 2:
                raise analysis.DataError(_stratification_error(raw, labels.name, counts))
            if min(n_test, n_train) < len(counts):
                raise analysis.DataError(f"This split gives {n_train} training and {n_test} test rows for {len(counts)} classes. Both sets need at least one row per class; change the proportion or add examples.")
        train, test = train_test_split(usable.to_numpy(), test_size=test_size, random_state=seed, stratify=labels if task == "classification" else None)
        if task == "classification" and (set(labels.loc[train]) != set(labels) or set(labels.loc[test]) != set(labels)):
            raise analysis.DataError("The stratified split could not put every class in both sets. Change the proportion or add rare-class examples.")
    elif method == "Grouped":
        if split_values.nunique() < 2:
            raise analysis.DataError("A grouped split needs at least two distinct groups.")
        # Later partitions from the same seed are tried only when earlier ones leave a
        # class out of either set; the first partition is unchanged, groups stay disjoint.
        splitter = GroupShuffleSplit(n_splits=GROUP_SPLIT_CANDIDATES if task == "classification" else 1, test_size=test_size, random_state=seed)
        classes, first = set(labels), None
        for candidate, (train_pos, test_pos) in enumerate(splitter.split(usable, labels, split_values), 1):
            first = first or (candidate, train_pos, test_pos)
            if task != "classification" or (set(labels.iloc[train_pos]) == classes and set(labels.iloc[test_pos]) == classes):
                break
        else:
            candidate, train_pos, test_pos = first
        train, test = usable[train_pos].to_numpy(), usable[test_pos].to_numpy()
    elif method == "Time ordered":
        ordered = split_values.sort_values(kind="stable")
        cutoff = ordered.iloc[n_train]
        train = ordered.index[ordered < cutoff].to_numpy()
        test = ordered.index[ordered >= cutoff].to_numpy()
        if len(train) == 0:
            raise analysis.DataError("The time boundary leaves no training rows. Choose another proportion or provide more distinct times.")
    else:
        raise analysis.DataError("Choose a supported split method.")
    if len(train) < 2 or len(test) < 1:
        raise analysis.DataError("The split needs at least two training rows and one test row.")
    if task == "classification" and labels.loc[train].nunique() < 2:
        raise analysis.DataError("The training split contains only one class. Change the split or add examples.")
    if task == "classification":
        coverage, notes = _coverage(labels, train, test, split_values, method, seed, candidate if method == "Grouped" else None)
    else:
        coverage, notes = {}, []
    return labels, train, test, split_values, coverage, notes


def _cv_splits(labels, split_values, task, method, folds, repeats, seed):
    notes = []
    requested = folds
    if method == "Random":
        max_folds = int(labels.value_counts().min()) if task == "classification" else len(labels)
        folds = min(folds, max_folds)
        if folds < 2:
            raise analysis.DataError("Cross-validation needs at least two training examples per class. Disable it or add rare-class examples.")
        if task == "classification":
            cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed) if repeats > 1 else StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed) if repeats > 1 else KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(labels)), labels))
    elif method == "Grouped":
        groups = split_values.loc[labels.index]
        folds = min(folds, groups.nunique())
        if folds < 2:
            raise analysis.DataError("Grouped cross-validation needs at least two training groups. Disable it or add groups.")
        if task == "classification":
            # Balances class proportions across folds while keeping every group in one fold.
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            cv = GroupKFold(n_splits=folds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            splits = list(cv.split(np.zeros(len(labels)), labels, groups))
    else:
        times = split_values.loc[labels.index]
        unique_times = np.sort(times.unique())
        folds = min(folds, len(unique_times) - 1)
        if folds < 2:
            raise analysis.DataError("Time-ordered cross-validation needs at least three distinct training times.")
        splits = []
        for train_times, val_times in TimeSeriesSplit(n_splits=folds).split(unique_times):
            splits.append((np.flatnonzero(times.isin(unique_times[train_times])), np.flatnonzero(times.isin(unique_times[val_times]))))
    if folds != requested:
        notes.append(f"Cross-validation uses {folds} folds rather than {requested} to fit the available training data.")
    for train, _ in splits:
        if len(train) < 2 or (task == "classification" and labels.iloc[train].nunique() < 2):
            raise analysis.DataError("A validation fold has too few rows or only one training class. Reduce folds, disable cross-validation or add examples.")
    if task == "classification" and method != "Random":
        classes = set(labels)
        partial = sum(set(labels.iloc[validation]) != classes for _, validation in splits)
        if partial:
            notes.append(f"{partial} of {len(splits)} validation folds do not contain every class, so those fold scores cover fewer classes.")
    names = {
        ("Random", "classification"): "RepeatedStratifiedKFold" if repeats > 1 else "StratifiedKFold",
        ("Random", "regression"): "RepeatedKFold" if repeats > 1 else "KFold",
        ("Grouped", "classification"): "StratifiedGroupKFold",
        ("Grouped", "regression"): "GroupKFold",
    }
    method_name = names.get((method, task), "TimeSeriesSplit (expanding window)")
    return splits, notes, method_name


def _metrics(actual, predicted, task):
    frame = pd.DataFrame({"actual": np.asarray(actual), "predicted": np.asarray(predicted)})
    if task == "classification":
        frame["is_error"] = frame["actual"] != frame["predicted"]
        return classification_metrics(frame)[0]
    evaluation = regression_evaluation(frame, "actual", "predicted")
    return regression_metrics(evaluation.rows)


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
    imbalance: str | None = None, progress: Callable[[int, int, str], None] | None = None,
    dataset_fingerprint: str | None = None,
) -> TrainingResult:
    """`imbalance` is one of IMBALANCE_STRATEGIES; `balance=True` means Oversampling.
    `progress(done, total, label)` reports cross-validation fits as they finish."""
    strategy = _strategy(balance, imbalance)
    if target not in raw.columns:
        raise analysis.DataError("Choose a target column.")
    if task not in ("classification", "regression"):
        raise analysis.DataError("Choose Classification or Regression.")
    if split_column == target:
        raise analysis.DataError("The target cannot be the split column.")
    allowed = feature_choices(raw, target, split_column).usable
    if not features:
        raise analysis.DataError("Choose at least one numeric training feature.")
    if len(set(features)) != len(features) or any(c not in allowed for c in features):
        raise analysis.DataError("Training features must be numeric, distinct, and exclude the target and identifiers or split column.")
    if not 0 < test_size < 1 or not 0 <= seed <= 2**32 - 1:
        raise analysis.DataError("Use a test proportion between 0 and 1 and a seed between 0 and 4294967295.")
    if strategy not in IMBALANCE_STRATEGIES:
        raise analysis.DataError("Choose None, Class weights or Oversampling for class imbalance.")
    if strategy != "None" and task != "classification":
        raise analysis.DataError("Class balancing is available only for classification.")
    if strategy == "Class weights" and not supports_class_weights(classifier, task):
        raise analysis.DataError(f"{classifier} does not support class weights. Choose Oversampling or a model that accepts class weights.")
    if not 0 <= cv_folds <= 10 or cv_folds == 1 or not 1 <= cv_repeats <= 5:
        raise analysis.DataError("Use 2–10 validation folds (or disable validation) and 1–5 repeats.")
    if tune and cv_folds < 2:
        raise analysis.DataError("Hyperparameter selection requires cross-validation on training data.")
    if split_method != "Random" and cv_repeats != 1:
        raise analysis.DataError("Grouped and time-ordered validation use one pass of folds, not repeated random splits.")
    if classifier == "Calibrated logistic regression" and split_method != "Random":
        raise analysis.DataError("Calibrated logistic regression currently supports random splits only; use another classifier for grouped or time-ordered data.")

    original_labels = analysis.clean_labels(raw[target]) if task == "classification" else analysis.parse_numeric(raw[target]).values
    missing_targets = int(original_labels.isna().sum())
    labels, train_rows, test_rows, split_values, coverage, split_notes = _split(raw, original_labels.dropna(), test_size, seed, task, split_method, split_column)
    values = pd.DataFrame({c: analysis.parse_numeric(raw[c]).values for c in features})
    X_train, y_train = values.loc[train_rows], labels.loc[train_rows]
    empty_train = X_train.columns[X_train.isna().all()].tolist()
    if empty_train:
        raise analysis.DataError("No usable training values for: " + ", ".join(empty_train) + ". Remove those features or choose a different split.")
    if classifier == "3-nearest neighbors" and len(train_rows) < 3:
        raise analysis.DataError("3-nearest neighbors needs at least three training rows; choose another classifier.")
    notes, splits, cv_results = list(split_notes), [], pd.DataFrame()
    best_params, cv_mean, cv_std, cv_method = {}, None, None, "none"
    metric = "balanced_accuracy" if task == "classification" else "neg_mean_absolute_error"
    if cv_folds:
        splits, cv_notes, cv_method = _cv_splits(y_train, split_values, task, split_method, cv_folds, cv_repeats, seed)
        notes.extend(cv_notes)
        smallest = min(len(tr) for tr, _ in splits)
        if classifier == "3-nearest neighbors" and smallest < 3:
            raise analysis.DataError("Nearest neighbors needs at least three rows in every validation training fold. Reduce folds or choose another model.")
        if classifier == "Calibrated logistic regression" and any(y_train.iloc[tr].value_counts().min() < 2 for tr, _ in splits):
            raise analysis.DataError("Calibration needs two examples per class inside every validation training fold. Add examples or choose another classifier.")
    if classifier == "Calibrated logistic regression" and y_train.value_counts().min() < 2:
        raise analysis.DataError("Calibration needs at least two training examples in every class.")

    probability_status = None
    svm_probabilities = task == "classification" and classifier == "Support vector machine"
    if svm_probabilities and split_method != "Random":
        svm_probabilities = False
        probability_status = (
            "SVM probabilities are unavailable for grouped and time-ordered splits: they come from sigmoid calibration "
            "on internal random training folds, which would mix groups or time periods. The SVM is fitted without "
            "probabilities, so ROC-AUC, confidence analysis and threshold exploration are unavailable for this run."
        )
    elif svm_probabilities and (
        y_train.value_counts().min() < CALIBRATION_FOLDS
        or any(y_train.iloc[tr].value_counts().min() < CALIBRATION_FOLDS for tr, _ in splits)
    ):
        svm_probabilities = False
        probability_status = (
            f"SVM probabilities are unavailable: calibration needs at least {CALIBRATION_FOLDS} training rows of every "
            "class in each fit (including validation folds). The SVM is fitted without probabilities, so ROC-AUC, "
            "confidence analysis and threshold exploration are unavailable for this run."
        )
    if probability_status:
        notes.append(probability_status)
    estimator, prefix = _estimator(classifier, task, seed, strategy, probabilities=svm_probabilities)

    def report(done, total, label):
        if progress:
            progress(done, total, label)

    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            if splits:
                grid = _grid(classifier, prefix, min(len(tr) for tr, _ in splits)) if tune else {}
                n_candidates, n_splits = len(ParameterGrid(grid)), len(splits)
                total_fits, base_scorer, finished = n_candidates * n_splits, get_scorer(metric), [0]

                # GridSearchCV has no callbacks; with n_jobs=1 it scores each (candidate, fold)
                # fit exactly once in this process, so counting scorer calls is exact progress.
                def scorer(fitted, X, y, **kwargs):
                    value = base_scorer(fitted, X, y, **kwargs)
                    finished[0] += 1
                    done = finished[0]
                    if done == total_fits:
                        label = "Refitting the selected model on all training rows"
                    elif tune:
                        label = f"Hyperparameter search: {done}/{total_fits} fits ({n_candidates} configurations × {n_splits} folds)"
                    else:
                        label = f"Cross-validation: fold {done}/{total_fits}"
                    report(done, total_fits + 1, label)
                    return value

                report(0, total_fits + 1, f"{'Hyperparameter search' if tune else 'Cross-validation'}: 0/{total_fits} fits")
                search = GridSearchCV(estimator, grid, scoring=scorer, cv=splits, refit=True, n_jobs=1, error_score="raise")
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
                report(0, 1, "Fitting the model on training rows")
                model = estimator.fit(X_train, y_train)
            notes.extend(dict.fromkeys(str(w.message) for w in recorded if issubclass(w.category, ConvergenceWarning)))
    except ValueError as exc:
        raise analysis.DataError(_fit_error(exc)) from exc
    report(1, 1, "Predicting held-out rows")

    test_rows = sorted(int(row) for row in test_rows)
    X_test, y_test = values.loc[test_rows], labels.loc[test_rows]
    predicted = model.predict(X_test)
    original_ids = analysis.suggest_id_columns(raw, exclude=(target,))
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
    if task == "classification" and probability_status is None:
        if probability_columns:
            probability_status = "available"
        else:
            probability_status = f"{classifier} does not provide class probabilities, so ROC-AUC, confidence analysis and threshold exploration are unavailable."
    probability_method = None
    if probability_columns:
        probability_method = (
            f"sigmoid (Platt) calibration on {CALIBRATION_FOLDS} stratified folds of the training rows; predictions are the most probable class"
            if classifier in CALIBRATED else "the model's predict_proba"
        )
    dataset_hash = dataset_fingerprint or fingerprint(raw)
    split_identity = {"dataset_sha256": dataset_hash, "task": task, "target": target, "train_rows": sorted(int(v) for v in train_rows), "test_rows": test_rows}
    metadata = {
        "schema_version": 1, "performance_scope": "Held-out test performance", "task": task,
        "target": target, "features": list(features), "model": classifier, "seed": seed,
        "requested_test_proportion": test_size, "split_method": split_method, "split_column": split_column,
        "dataset_sha256": dataset_hash, "split_id": hashlib.sha256(json.dumps(split_identity, sort_keys=True).encode()).hexdigest(),
        "train_row_ids": [int(row) for row in train_rows], "test_row_ids": test_rows,
        "training_rows": len(train_rows), "test_rows": len(test_rows), "missing_targets": missing_targets,
        "missing_split_values": len(original_labels.dropna()) - len(labels),
        "imbalance_strategy": strategy,
        "class_balancing": {
            "None": "none", "Oversampling": "random oversampling inside each training fit",
            "Class weights": "class_weight='balanced' inside each training fit",
        }[strategy],
        "split_class_coverage": coverage,
        "preprocessing": "Training-only median imputation and standard scaling; empty feature in a CV training fold uses 0.",
        "tuned": tune, "selected_parameters": _plain(best_params),
        "fitted_parameters": _plain(model.get_params(deep=True)),
        "cv_method": cv_method, "cv_requested_folds": cv_folds, "cv_repeats": cv_repeats, "cv_splits": len(splits),
        "cv_metric": "balanced accuracy" if task == "classification" else "MAE",
        "cv_mean": cv_mean, "cv_std": cv_std,
        "cv_row_ids": [{"train": y_train.index[tr].tolist(), "validation": y_train.index[va].tolist()} for tr, va in splits],
        "actual_column": actual_col, "predicted_column": predicted_col, "id_columns": [source_id, *original_ids],
        "probability_columns": probability_columns, "confidence_column": confidence,
        "probability_status": probability_status, "probability_method": probability_method,
        "test_metrics": metrics, "training_metrics": training_metrics, "notes": notes,
        "fingerprint_method": FINGERPRINT_METHOD,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    return TrainingResult(
        result, actual_col, predicted_col, (source_id, *original_ids), tuple(int(row) for row in train_rows),
        tuple(test_rows), missing_targets, classifier, seed, model, task, features, metadata,
        cv_results, metrics, training_metrics, probability_columns, confidence,
        X_train, X_test, y_train, y_test, tuple(notes),
    )


def compare_models(raw, target, features, models, progress: Callable | None = None, **settings):
    """`progress(position, len(models), label)`; position is fractional while a model's folds run."""
    if not settings.get("dataset_fingerprint"):
        settings["dataset_fingerprint"] = fingerprint(raw)
    results = []
    for i, name in enumerate(models):
        if progress:
            progress(i, len(models), name)

        def inner(done, total, label, i=i, name=name):
            if progress:
                progress(i + done / max(total, 1), len(models), f"{name} — {label}")

        result = train_and_evaluate(raw, target, features, classifier=name, progress=inner, **settings)
        if results and result.metadata["split_id"] != results[0].metadata["split_id"]:
            raise RuntimeError("Model comparison must use identical training and test rows.")
        results.append(result)
    if progress:
        progress(len(models), len(models), "Complete")
    return results


def comparable(results: list[TrainingResult]) -> bool:
    """Runs are a controlled comparison only on identical data, target, task and rows."""
    return len({r.metadata["split_id"] for r in results}) <= 1


def comparison_table(results: list[TrainingResult]) -> pd.DataFrame:
    records = []
    for result in results:
        row = {
            "model": result.classifier, "task": result.task, "seed": result.seed,
            "train rows": len(result.train_rows), "test rows": len(result.test_rows),
            "split": result.metadata["split_id"][:12], "features": ", ".join(result.features),
            "imbalance": result.metadata.get("imbalance_strategy", "None"),
            "CV metric": result.metadata["cv_metric"], "CV mean": result.metadata["cv_mean"], "CV std": result.metadata["cv_std"],
        }
        row.update({f"test {k}": v for k, v in result.metrics.items() if k in ("accuracy", "macro_f1", "balanced_accuracy", "mae", "rmse", "r2")})
        records.append(row)
    return pd.DataFrame(records)


def _cv_description(meta: dict) -> str:
    if not meta.get("cv_splits"):
        return "none"
    repeats = f" × {meta['cv_repeats']} repeats" if meta.get("cv_repeats", 1) > 1 else ""
    return f"{meta.get('cv_method', 'folds')}{repeats}"


def history_table(results: dict[str, TrainingResult] | list[TrainingResult]) -> pd.DataFrame:
    """Readable experiment history; full metadata stays in each run's JSON."""
    items = results.items() if isinstance(results, dict) else ((r.metadata.get("run_id", ""), r) for r in results)
    records = []
    for run, result in items:
        meta = result.metadata
        classification = result.task == "classification"
        column = meta.get("split_column")
        records.append({
            "run": run, "trained at (UTC)": meta.get("trained_at_utc"), "task": result.task, "model": result.classifier,
            "target": meta["target"], "test proportion": meta["requested_test_proportion"],
            "split method": meta["split_method"],
            "group column": column if meta["split_method"] == "Grouped" else None,
            "time column": column if meta["split_method"] == "Time ordered" else None,
            "seed": result.seed, "CV method": _cv_description(meta), "CV folds": meta.get("cv_splits", 0),
            "tuned": bool(meta.get("tuned")), "imbalance": meta.get("imbalance_strategy", "None"),
            "primary metric": "balanced accuracy" if classification else "MAE",
            "CV score": meta.get("cv_mean"),
            "held-out score": result.metrics.get("balanced_accuracy" if classification else "mae"),
            "train rows": len(result.train_rows), "test rows": len(result.test_rows),
            "features": ", ".join(result.features), "split ID": meta["split_id"][:12],
        })
    return pd.DataFrame(records)


def permutation_scores(result: TrainingResult, repeats: int = 3) -> pd.DataFrame:
    scoring = "balanced_accuracy" if result.task == "classification" else "neg_mean_absolute_error"
    importance = permutation_importance(
        result.model, result.X_test, result.y_test, scoring=scoring, n_repeats=repeats,
        random_state=result.seed, n_jobs=1, max_samples=min(500, len(result.X_test)),
    )
    return pd.DataFrame({"feature": result.features, "importance": importance.importances_mean, "std": importance.importances_std}).sort_values("importance", ascending=False)


LOCAL_UNAVAILABLE = "Local row explanations are not available for this model. Global feature importance is shown instead."


def _fitted_estimator(model):
    """The fitted model after preprocessing, unwrapping calibration (ensemble=False keeps
    one base model fitted on all training rows) and oversampling."""
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    estimator = model["classifier"] if isinstance(model, Pipeline) else model
    return estimator.estimator_ if isinstance(estimator, ResampledClassifier) else estimator


def model_importance(result: TrainingResult) -> tuple[str, pd.DataFrame]:
    """Global importance the fitted model exposes itself; empty when it has none."""
    estimator = _fitted_estimator(result.model)
    if hasattr(estimator, "feature_importances_"):
        table = pd.DataFrame({"feature": result.features, "importance": estimator.feature_importances_})
        note = "Impurity-based importance from the fitted trees on training rows. It favours features with many split points and does not show direction."
    elif hasattr(estimator, "coef_"):
        table = pd.DataFrame({"feature": result.features, "importance": np.abs(np.atleast_2d(estimator.coef_)).mean(axis=0)})
        note = "Mean absolute coefficient on standardized features (per class for multiclass). Larger means a larger change in the linear score per standard deviation."
    else:
        return f"{result.classifier} exposes no built-in feature importance. Use permutation importance, which works for any model.", pd.DataFrame()
    return note, table.sort_values("importance", ascending=False)


def row_explanation(result: TrainingResult, row: int) -> tuple[str, pd.DataFrame]:
    if not isinstance(result.model, Pipeline):
        return (
            f"{LOCAL_UNAVAILABLE} Exact row explanations cover uncalibrated linear models and single decision trees; "
            f"{result.classifier} is calibrated, so its probabilities are not an additive function of the features."
        ), pd.DataFrame()
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
    return (
        f"{LOCAL_UNAVAILABLE} {result.classifier} combines many trees, neighbours or kernels, so no exact "
        "additive per-row breakdown exists without a heavy dependency such as SHAP."
    ), pd.DataFrame()
