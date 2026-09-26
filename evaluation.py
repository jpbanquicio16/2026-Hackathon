"""Classification, regression and probability diagnostics used by UI and exports."""

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import analysis as A

AUC_AVERAGES = ("macro", "weighted")
AUC_STRATEGIES = {"ovr": "one-vs-rest", "ovo": "one-vs-one"}
PROBABILITY_SUM_TOLERANCE = 1e-5  # rounding noise that needs no comment
PROBABILITY_REPAIR_TOLERANCE = .02  # larger deviations mean missing or mis-mapped classes


def regression_evaluation(raw: pd.DataFrame, actual: str, predicted: str, tolerance: float = 0) -> A.Evaluation:
    if actual == predicted:
        raise A.DataError("The actual and predicted values must come from two different columns.")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise A.DataError("The error tolerance must be a finite, non-negative number.")
    truth, pred = A.parse_numeric(raw[actual]), A.parse_numeric(raw[predicted])
    no_actual, no_predicted = ~truth.valid, ~pred.valid
    keep = ~(no_actual | no_predicted)
    rows = pd.DataFrame({"actual": truth.values[keep], "predicted": pred.values[keep]})
    rows["residual"] = rows["predicted"] - rows["actual"]
    rows["absolute_error"] = rows["residual"].abs()
    if not np.isfinite(rows["absolute_error"]).all():
        raise A.DataError("Some residuals overflow numeric precision. Rescale the actual and predicted values.")
    rows["is_error"] = rows["absolute_error"] > tolerance
    return A.Evaluation(
        rows, len(raw), int((no_actual & ~no_predicted).sum()),
        int((no_predicted & ~no_actual).sum()), int((no_actual & no_predicted).sum()),
        task="regression", tolerance=tolerance,
    )


def regression_metrics(rows: pd.DataFrame) -> dict[str, float | int | None]:
    if rows.empty:
        return {"evaluated_rows": 0, "mae": None, "rmse": None, "r2": None}
    actual = rows["actual"].to_numpy(float)
    residual = rows["residual"].to_numpy(float)
    # Scale before squaring to avoid gratuitous overflow on large values.
    scale = max(float(np.max(np.abs(residual))), 1)
    rmse = float(scale * np.sqrt(np.mean((residual / scale) ** 2)))
    target_scale = max(float(np.max(np.abs(actual))), scale, 1)
    denominator = float(np.sum((actual / target_scale - np.mean(actual / target_scale)) ** 2))
    r2 = 1 - float(np.sum((residual / target_scale) ** 2)) / denominator if denominator > 0 and len(rows) >= 2 else None
    return {
        "evaluated_rows": len(rows), "mae": float(np.mean(np.abs(residual))), "rmse": rmse,
        "r2": r2, "mean_residual": float(np.mean(residual)),
        "above_tolerance": int(rows["is_error"].sum()),
    }


def classification_metrics(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if rows.empty:
        return {"evaluated_rows": 0}, pd.DataFrame()
    classes = A.sort_labels(set(rows["actual"]) | set(rows["predicted"]))
    precision, recall, f1, support = precision_recall_fscore_support(
        rows["actual"], rows["predicted"], labels=classes, zero_division=0,
    )
    table = pd.DataFrame({"class": classes, "precision": precision, "recall": recall, "f1": f1, "support": support})
    totals = {
        "evaluated_rows": len(rows), "correct": int((~rows["is_error"]).sum()),
        "errors": int(rows["is_error"].sum()), "accuracy": float((~rows["is_error"]).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
    }
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        totals[f"macro_{name}"] = float(values.mean())
        totals[f"weighted_{name}"] = float(np.average(values, weights=support))
    return totals, table


def normalise_confusion(matrix: pd.DataFrame, mode: str = "Counts") -> pd.DataFrame:
    if mode == "Counts":
        return matrix.copy()
    if mode == "Actual class (%)":
        return matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) * 100
    if mode == "Predicted class (%)":
        return matrix.div(matrix.sum(axis=0).replace(0, np.nan), axis=1) * 100
    raise A.DataError("Choose counts, actual-class percentages or predicted-class percentages.")


def apply_threshold(
    raw: pd.DataFrame, predicted: str, probability: str, positive: str, negative: str, threshold: float,
) -> tuple[pd.DataFrame, int]:
    if positive == negative or not 0 <= threshold <= 1:
        raise A.DataError("Choose different binary classes and a threshold between 0 and 1.")
    parsed = A.parse_numeric(raw[probability]).values
    valid = parsed.notna() & parsed.between(0, 1)
    changed = raw.copy()
    backup = A._unique_name("original_prediction", changed.columns)
    changed[backup] = changed[predicted]
    changed[predicted] = np.where(valid, np.where(parsed >= threshold, positive, negative), "")
    return changed, int((~valid).sum())


def probability_auc(
    rows: pd.DataFrame, probabilities: pd.DataFrame, average: str = "macro", multi_class: str = "ovr",
) -> tuple[float | None, str]:
    """ROC-AUC over actual classes. Weighted averages use actual-class prevalence
    (one-vs-rest) or pair prevalence (one-vs-one, Hand & Till), as scikit-learn does."""
    if average not in AUC_AVERAGES or multi_class not in AUC_STRATEGIES:
        raise A.DataError("Choose macro or weighted averaging and one-vs-rest or one-vs-one.")
    if rows.empty or probabilities.empty:
        return None, "ROC-AUC is unavailable because there are no class probabilities."
    p = probabilities.reindex(rows.index)
    if p.isna().any().any() or not np.isfinite(p.to_numpy(float)).all() or ((p < 0) | (p > 1)).any().any():
        return None, "ROC-AUC unavailable: every evaluated row needs probabilities between 0 and 1."
    actual_classes = set(rows["actual"])
    if len(actual_classes) < 2 or not actual_classes.issubset(p.columns):
        return None, "ROC-AUC needs at least two actual classes and probabilities for each actual class."
    if not np.allclose(p.sum(axis=1), 1, atol=PROBABILITY_SUM_TOLERANCE):
        return None, "ROC-AUC unavailable: class probabilities must sum to 1 on every evaluated row."
    actual, classes = rows["actual"], A.sort_labels(actual_classes)
    if len(classes) == 2:
        auc = roc_auc_score((actual == classes[1]).astype(int), p[classes[1]])
        return float(auc), "Binary ROC-AUC (the same under every averaging mode)."
    if multi_class == "ovr":
        scores = [roc_auc_score((actual == label).astype(int), p[label]) for label in classes]
        weights = [int((actual == label).sum()) for label in classes]
        described = f"one-vs-rest ROC-AUC over {len(classes)} actual classes"
    else:
        scores, weights = [], []
        for a, b in combinations(classes, 2):
            pair = actual.isin([a, b])
            is_a = (actual[pair] == a).astype(int)
            scores.append((roc_auc_score(is_a, p.loc[pair, a]) + roc_auc_score(1 - is_a, p.loc[pair, b])) / 2)
            weights.append(float(pair.mean()))
        described = f"one-vs-one ROC-AUC over {len(scores)} class pairs"
    value = np.average(scores, weights=weights if average == "weighted" else None)
    prefix = "Macro" if average == "macro" else "Prevalence-weighted"
    return float(value), f"{prefix} {described}."


def suggest_probability_columns(columns, classes) -> dict[str, str | None]:
    """Columns whose name contains the class name, e.g. prob_setosa for setosa; never guesses twice."""
    taken, guesses = set(), {}
    for label in classes:
        wanted = A._name_tokens(str(label))
        match = next((
            column for column in columns
            if column not in taken and wanted and all(t in A._name_tokens(column) for t in wanted)
        ), None)
        guesses[label] = match
        if match:
            taken.add(match)
    return guesses


@dataclass(frozen=True)
class ProbabilityMapping:
    probabilities: pd.DataFrame  # index: evaluated rows, columns: classes; empty when unusable
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.errors


def map_class_probabilities(raw: pd.DataFrame, rows: pd.DataFrame, mapping: dict, classes) -> ProbabilityMapping:
    """Validate one probability column per class; never fill in or invent missing scores."""
    errors, warnings = [], []
    missing = [label for label in classes if not mapping.get(label)]
    if missing:
        errors.append("Map a probability column for every class. Missing: " + ", ".join(f"“{c}”" for c in missing) + ".")
    used = [column for column in (mapping.get(label) for label in classes) if column]
    for column in sorted({c for c in used if used.count(c) > 1}):
        errors.append(f"“{column}” is mapped to more than one class; each class needs its own probability column.")
    if errors:
        return ProbabilityMapping(pd.DataFrame(), tuple(errors), ())
    frame = pd.DataFrame(index=rows.index)
    for label in classes:
        column = mapping[label]
        parsed = A.parse_numeric(raw.loc[rows.index, column])
        values = parsed.values
        if parsed.invalid.any():
            errors.append(f"“{column}” (for “{label}”) has {int(parsed.invalid.sum()):,} non-numeric values.")
        elif parsed.missing.any():
            errors.append(f"“{column}” (for “{label}”) is blank on {int(parsed.missing.sum()):,} evaluated rows; ROC-AUC needs every class probability on every row.")
        elif ((values < 0) | (values > 1)).any():
            errors.append(f"“{column}” (for “{label}”) has values outside 0–1 (range {values.min():g} to {values.max():g}), so it is not a probability.")
        frame[label] = values
    if errors:
        return ProbabilityMapping(pd.DataFrame(), tuple(errors), ())
    sums = frame.sum(axis=1)
    deviation = (sums - 1).abs()
    if deviation.max() > PROBABILITY_REPAIR_TOLERANCE:
        errors.append(
            f"Class probabilities must sum to 1 on every row, but {int((deviation > PROBABILITY_REPAIR_TOLERANCE).sum()):,} rows "
            f"deviate by more than {PROBABILITY_REPAIR_TOLERANCE:g} (largest {deviation.max():.3g}). Check that every class has "
            "a column and that the columns are probabilities, not scores."
        )
        return ProbabilityMapping(pd.DataFrame(), tuple(errors), ())
    if deviation.max() > PROBABILITY_SUM_TOLERANCE:
        warnings.append(f"Row sums differ from 1 by up to {deviation.max():.2g}, probably from rounding. Each row was divided by its sum for ROC-AUC.")
        frame = frame.div(sums, axis=0)
    for label in classes:
        tokens = A._name_tokens(mapping[label])
        named = [other for other in classes if other != label and A._name_tokens(str(other)) and all(t in tokens for t in A._name_tokens(str(other)))]
        if named and not all(t in tokens for t in A._name_tokens(str(label))):
            warnings.append(f"“{mapping[label]}” is mapped to “{label}” but its name mentions “{named[0]}”. Check the mapping.")
    agreement = float((frame.idxmax(axis=1) == rows["predicted"]).mean())
    if agreement < .9:
        warnings.append(
            f"The most probable class matches the predicted label on only {agreement:.0%} of rows. "
            "Check that each column is mapped to the right class."
        )
    return ProbabilityMapping(frame, (), tuple(warnings))


def confidence_bins(confidence: pd.Series, is_error: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    values = pd.to_numeric(confidence, errors="coerce")
    valid = values.notna() & values.between(0, 1)
    rows = pd.DataFrame({"confidence": values[valid], "correct": ~is_error.loc[values[valid].index]})
    rows["bin"] = np.minimum((rows["confidence"] * n_bins).astype(int), n_bins - 1)
    table = rows.groupby("bin").agg(examples=("correct", "size"), mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"))
    table = table.reindex(range(n_bins))
    table["examples"] = table["examples"].fillna(0).astype(int)
    table["range"] = [f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)]
    return table.reset_index(drop=True)


def filter_evaluation(evaluation: A.Evaluation, actual: tuple[str, ...], predicted: tuple[str, ...]) -> A.Evaluation:
    rows = evaluation.rows
    keep = pd.Series(True, index=rows.index)
    if actual:
        keep &= rows["actual"].isin(actual)
    if predicted:
        keep &= rows["predicted"].isin(predicted)
    return replace(evaluation, rows=rows.loc[keep])
