"""Readable dataset, target and leakage diagnostics; never silently repair labels."""

import re

import numpy as np
import pandas as pd

import analysis as A


def target_suggestion(raw: pd.DataFrame) -> str | None:
    actual, _ = A.suggest_label_columns(raw.columns)
    if actual:
        return actual
    names = {c.strip().lower(): c for c in raw.columns}
    return next((names[c] for c in ("species", "quality", "diagnosis", "outcome", "price") if c in names), None)


def dataset_summary(raw: pd.DataFrame) -> pd.DataFrame:
    ids = set(A.suggest_id_columns(raw))
    records = []
    for column in raw:
        parsed = A.parse_numeric(raw[column])
        clean = A.clean_labels(raw[column])
        n_valid = int(parsed.valid.sum())
        present = int(clean.notna().sum())
        kind = "numeric" if present and n_valid == present else "mixed numeric/text" if n_valid else "text / categorical"
        records.append({
            "column": column, "kind": kind, "missing": int(clean.isna().sum()),
            "unique": int(clean.nunique()), "numeric values": n_valid,
            "non-numeric / infinite": int(parsed.invalid.sum()),
            "minimum": parsed.values.min(), "maximum": parsed.values.max(),
            "possible identifier": column in ids,
        })
    return pd.DataFrame(records)


def duplicate_count(raw: pd.DataFrame) -> int:
    """Count repeated occurrences beyond the first; do not delete observations."""
    return int(raw.duplicated().sum())


def class_counts(labels: pd.Series) -> pd.DataFrame:
    clean = A.clean_labels(labels).dropna()
    counts = clean.value_counts().rename_axis("class").rename("rows").reset_index()
    counts["share"] = counts["rows"] / max(len(clean), 1)
    return counts


def target_warnings(raw: pd.DataFrame, target: str, task: str) -> list[str]:
    if task == "regression":
        parsed = A.parse_numeric(raw[target])
        unusable = int((~parsed.valid).sum())
        return [f"{unusable:,} targets are missing, non-numeric or infinite and will be excluded."] if unusable else []
    labels = A.clean_labels(raw[target]).dropna()
    if labels.empty:
        return ["There are no usable target labels."]
    counts = labels.value_counts()
    warnings = []
    numbers = pd.to_numeric(labels, errors="coerce")
    if np.isfinite(numbers).all() and (len(counts) > 30 or (len(counts) > 10 and len(counts) > len(labels) * .2)):
        warnings.append(
            f"“{target}” has {len(counts):,} distinct numeric values. If these are measurements, "
            "choose Regression. Numeric class codes remain valid classification targets."
        )
    if int(counts.min()) < 2:
        warnings.append("Some target classes have only one example; a stratified split cannot preserve them in both sets.")
    if counts.max() >= counts.min() * 10:
        warnings.append(
            f"Class imbalance: the largest class has {int(counts.max()):,} rows and the smallest "
            f"has {int(counts.min()):,}. Inspect balanced accuracy, macro F1 and per-class recall."
        )
    return warnings


def leakage_warnings(raw: pd.DataFrame, target: str, features: tuple[str, ...]) -> pd.DataFrame:
    """Heuristics flag clues, not proof; availability at prediction time needs user review."""
    target_values = A.clean_labels(raw[target])
    target_numeric = A.parse_numeric(raw[target]).values
    records = []
    for column in features:
        reasons = []
        text = A.clean_labels(raw[column])
        paired = text.notna() & target_values.notna()
        if paired.sum() >= 5 and (text[paired] == target_values[paired]).mean() >= .98:
            reasons.append("at least 98% identical to the target")
        numeric = A.parse_numeric(raw[column]).values
        mask = numeric.notna() & target_numeric.notna()
        if mask.sum() >= 10 and numeric[mask].nunique() > 1 and target_numeric[mask].nunique() > 1:
            corr = float(numeric[mask].corr(target_numeric[mask]))
            if abs(corr) >= .98:
                reasons.append(f"almost perfectly correlated with the target (r={corr:.3f})")
        words = set(re.split(r"[^a-z0-9]+", column.lower()))
        if words & {"predicted", "prediction", "outcome", "result", "future", "after", "post", "target"}:
            reasons.append("name suggests a prediction, outcome or information recorded afterwards")
        if reasons:
            records.append({"feature": column, "review reason": "; ".join(reasons)})
    return pd.DataFrame(records, columns=["feature", "review reason"])
