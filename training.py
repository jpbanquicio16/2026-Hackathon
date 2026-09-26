"""Train classification baselines on one split; export only held-out predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import analysis as A

CLASSIFIERS = ("3-nearest neighbors", "Logistic regression", "Decision tree")


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
    model: Pipeline

    def csv_bytes(self) -> bytes:
        return self.frame.to_csv(index=False).encode("utf-8")


def feature_choices(raw: pd.DataFrame, target: str) -> A.FeatureOptions:
    """Numeric feature candidates; target and detected IDs are always excluded."""
    ids = A.suggest_id_columns(raw, exclude=(target,))
    return A.feature_options(
        raw, raw.index, {target: "target", **{column: "identifier" for column in ids}}
    )


def train_and_evaluate(
    raw: pd.DataFrame,
    target: str,
    features: tuple[str, ...],
    test_size: float = 0.2,
    seed: int = 42,
    classifier: str = CLASSIFIERS[0],
) -> TrainingResult:
    """Split before fitting either the imputer, scaler, or classifier.

    Label spelling is retained using the same missing/whitespace rules as the
    existing-prediction workflow. There is deliberately no training-set fallback.
    """
    if target not in raw.columns:
        raise A.DataError("Choose a target column.")
    allowed = feature_choices(raw, target).usable
    if not features:
        raise A.DataError("Choose at least one numeric training feature.")
    if len(set(features)) != len(features) or any(c not in allowed for c in features):
        raise A.DataError("Training features must be numeric, distinct, and exclude the target and identifiers.")
    if not 0 < test_size < 1 or not 0 <= seed <= 2**32 - 1:
        raise A.DataError("Use a test proportion between 0 and 1 and a seed between 0 and 4294967295.")
    if classifier not in CLASSIFIERS:
        raise A.DataError("Choose one of the supported classifiers.")

    labels = A.clean_labels(raw[target]).dropna()
    counts = labels.value_counts()
    if len(counts) < 2:
        raise A.DataError("Classification needs at least two classes with non-missing targets.")
    if counts.min() < 2:
        raise A.DataError(
            "A stratified held-out split needs at least two rows in every class. "
            "Some classes have only one row; add examples or choose another target."
        )
    n_test = math.ceil(len(labels) * test_size)
    n_train = len(labels) - n_test
    if min(n_test, n_train) < len(counts):
        raise A.DataError(
            f"This split gives {n_train} training and {n_test} test rows for {len(counts)} classes. "
            "Both sets need at least one row per class; change the test proportion or add examples."
        )
    if classifier == CLASSIFIERS[0] and n_train < 3:
        raise A.DataError("3-nearest neighbors needs at least three training rows; choose another classifier.")

    train_rows, test_rows = train_test_split(
        labels.index.to_numpy(), test_size=test_size, random_state=seed, stratify=labels,
    )
    # Extremely imbalanced counts can still omit a rare class despite stratify.
    classes = set(labels)
    if set(labels.loc[train_rows]) != classes or set(labels.loc[test_rows]) != classes:
        raise A.DataError(
            "The stratified split could not put every class in both sets. "
            "Change the test proportion or add examples of rare classes."
        )
    values = pd.DataFrame({c: A.parse_numeric(raw[c]).values for c in features})
    empty_train = values.loc[train_rows].columns[values.loc[train_rows].isna().all()].tolist()
    if empty_train:
        raise A.DataError(
            "No usable training values for: " + ", ".join(empty_train)
            + ". Remove those features or choose a different split."
        )

    classifiers = {
        "3-nearest neighbors": KNeighborsClassifier(n_neighbors=3),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
    }
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifiers[classifier]),
    ])
    try:
        model.fit(values.loc[train_rows], labels.loc[train_rows])
        test_rows = sorted(int(row) for row in test_rows)
        predicted = model.predict(values.loc[test_rows])
    except ValueError as exc:
        raise A.DataError(f"The model could not fit this data: {exc}") from exc

    # Original measurements stay untouched for row inspection and re-upload.
    original_ids = A.suggest_id_columns(raw, exclude=(target,))
    result = raw.loc[test_rows, [*original_ids, *features]].copy()

    def available_name(base: str) -> str:
        name, suffix = base, 2
        while name in result.columns:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    source_id = available_name("source_row_id")
    result.insert(0, source_id, test_rows)
    actual_col = available_name("actual_label")
    result[actual_col] = labels.loc[test_rows].to_numpy()
    predicted_col = available_name("predicted_label")
    result[predicted_col] = predicted
    # The analysis uses positions in the generated CSV; source_row_id retains
    # positions in the uploaded training CSV, even after missing targets drop.
    result.index = pd.RangeIndex(1, len(result) + 1, name="row")
    return TrainingResult(
        result, actual_col, predicted_col, (source_id, *original_ids),
        tuple(int(row) for row in train_rows), tuple(test_rows), len(raw) - len(labels),
        classifier, seed, model,
    )
