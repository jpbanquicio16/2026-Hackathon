"""Classification, regression and probability diagnostics used by UI and exports."""

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import analysis as A


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


def probability_auc(rows: pd.DataFrame, probabilities: pd.DataFrame) -> tuple[float | None, str]:
    if rows.empty or probabilities.empty:
        return None, "ROC-AUC requires class probabilities."
    p = probabilities.reindex(rows.index)
    if p.isna().any().any() or not np.isfinite(p.to_numpy(float)).all() or ((p < 0) | (p > 1)).any().any():
        return None, "ROC-AUC unavailable: every evaluated row needs probabilities between 0 and 1."
    actual_classes = set(rows["actual"])
    if len(actual_classes) < 2 or not actual_classes.issubset(p.columns):
        return None, "ROC-AUC needs at least two actual classes and probabilities for each actual class."
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        return None, "ROC-AUC unavailable: class probabilities must sum to 1 on every evaluated row."
    aucs = [roc_auc_score((rows["actual"] == label).astype(int), p[label]) for label in sorted(actual_classes)]
    return float(np.mean(aucs)), "Macro one-vs-rest ROC-AUC over actual classes with both positive and negative examples."


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
