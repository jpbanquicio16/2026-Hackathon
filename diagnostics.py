"""Readable dataset, target and leakage diagnostics; never silently repair labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

import analysis


def target_suggestion(raw: pd.DataFrame) -> str | None:
    actual, _ = analysis.suggest_label_columns(raw.columns)
    if actual:
        return actual
    names = {c.strip().lower(): c for c in raw.columns}
    return next((names[c] for c in ("species", "quality", "diagnosis", "outcome", "price") if c in names), None)


def dataset_summary(raw: pd.DataFrame) -> pd.DataFrame:
    ids = set(analysis.suggest_id_columns(raw))
    records = []
    for column in raw:
        parsed = analysis.parse_numeric(raw[column])
        clean = analysis.clean_labels(raw[column])
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
    clean = analysis.clean_labels(labels).dropna()
    counts = clean.value_counts().rename_axis("class").rename("rows").reset_index()
    counts["share"] = counts["rows"] / max(len(clean), 1)
    return counts


# ---------------------------------------------------------------------------
# Target type and guidance

TARGET_KINDS = {
    "binary": "binary classification",
    "multiclass": "multiclass classification",
    "continuous": "likely continuous regression target",
    "ambiguous": "ambiguous",
    "constant": "a single value",
    "empty": "no usable values",
}
IMBALANCE_RATIO = 10  # largest ÷ smallest class count that triggers a warning


@dataclass(frozen=True)
class TargetProfile:
    column: str
    kind: str  # a key of TARGET_KINDS
    rows: int  # non-missing target values
    missing: int
    distinct: int
    numeric_share: float
    whole_numbers: bool
    reason: str

    @property
    def class_like(self) -> bool:
        return self.kind in ("binary", "multiclass")

    @property
    def description(self) -> str:
        return TARGET_KINDS[self.kind]


def target_profile(raw: pd.DataFrame, target: str) -> TargetProfile:
    """Likely target type from distinct values, their share of rows, numeric share and sample size.

    Integer values are not automatically classes: a few whole numbers can be class
    codes, many whole numbers (relative to the rows) look like counts or measurements.
    """
    all_labels = analysis.clean_labels(raw[target])
    labels = all_labels.dropna()
    n, missing = len(labels), int(all_labels.isna().sum())
    if n == 0:
        return TargetProfile(target, "empty", 0, missing, 0, 0.0, False, "there are no non-missing target values")
    distinct = int(labels.nunique())
    numbers = analysis.parse_numeric(labels).values.dropna()
    numeric_share = len(numbers) / n
    numeric = numeric_share >= analysis.MIN_NUMERIC_SHARE
    whole = bool(numeric and (numbers == np.floor(numbers)).all())
    ratio = distinct / n

    def profile(kind: str, reason: str) -> TargetProfile:
        return TargetProfile(target, kind, n, missing, distinct, numeric_share, whole, reason)

    if distinct == 1:
        return profile("constant", f"every non-missing value is “{labels.iloc[0]}”")
    if not numeric:
        if distinct == 2:
            return profile("binary", "two distinct text values")
        if distinct > 20 and ratio > .5:
            return profile("ambiguous", f"{distinct:,} distinct text values among {n:,} rows, so most rows have their own value; this looks like an identifier or free text rather than classes")
        return profile("multiclass", f"{distinct:,} distinct text values")
    if distinct == 2:
        return profile("binary", "two distinct numeric codes")
    if whole:
        if distinct <= 10:
            return profile("multiclass", f"{distinct:,} distinct whole numbers, which can be class codes or an ordinal score")
        if distinct > 30 or ratio > .2:
            return profile("continuous", f"{distinct:,} distinct whole numbers ({ratio:.0%} of rows), which looks like a count or measurement")
        return profile("ambiguous", f"{distinct:,} distinct whole numbers, which could be class codes, an ordinal score or a count")
    if distinct <= 10 and ratio <= .2:
        return profile("ambiguous", f"only {distinct:,} distinct decimal values, such as a rating scale")
    return profile("continuous", f"{distinct:,} distinct decimal values ({ratio:.0%} of rows)")


def class_warnings(labels: pd.Series) -> list[str]:
    """Classification-only diagnostics; call only when the values are meant as classes."""
    counts = analysis.clean_labels(labels).dropna().value_counts()
    if counts.empty:
        return []
    warnings = []
    singles = analysis.sort_labels(counts.index[counts < 2])
    if singles:
        shown = ", ".join(f"“{label}”" for label in singles[:5]) + (", …" if len(singles) > 5 else "")
        warnings.append(
            f"{len(singles):,} target {'class has' if len(singles) == 1 else 'classes have'} only one example ({shown}); "
            "a stratified split cannot place them in both training and test sets."
        )
    if counts.max() >= counts.min() * IMBALANCE_RATIO:
        warnings.append(
            f"Class imbalance: the largest class has {int(counts.max()):,} rows and the smallest "
            f"has {int(counts.min()):,}. Inspect balanced accuracy, macro F1 and per-class recall, "
            "and consider class weights or oversampling."
        )
    return warnings


def class_balance(labels: pd.Series) -> dict:
    counts = analysis.clean_labels(labels).dropna().value_counts()
    if counts.empty:
        return {"classes": 0}
    total = int(counts.sum())
    return {
        "classes": len(counts), "largest_class": str(counts.index[0]), "largest": int(counts.iloc[0]),
        "smallest_class": str(counts.index[-1]), "smallest": int(counts.iloc[-1]),
        "largest_share": counts.iloc[0] / total, "smallest_share": counts.iloc[-1] / total,
        "ratio": counts.iloc[0] / counts.iloc[-1], "imbalanced": bool(counts.iloc[0] >= counts.iloc[-1] * IMBALANCE_RATIO),
    }


@dataclass(frozen=True)
class TargetGuidance:
    profile: TargetProfile
    warnings: tuple[str, ...]
    notes: tuple[str, ...]
    show_class_diagnostics: bool  # class counts and imbalance warnings are meaningful


def target_guidance(raw: pd.DataFrame, target: str, task: str, treat_as_classes: bool = False) -> TargetGuidance:
    """Class diagnostics are withheld for continuous or ambiguous targets unless the user
    confirms that the values are classes, so near-unique numbers are never called classes."""
    profile = target_profile(raw, target)
    warnings, notes = [], []
    name = f"“{target}”"
    if task == "regression":
        if profile.kind == "empty":
            warnings.append("There are no usable target values.")
        elif profile.numeric_share < analysis.MIN_NUMERIC_SHARE:
            warnings.append(
                f"{name} is not numeric ({1 - profile.numeric_share:.0%} of its values are text), so Regression "
                "cannot use it. Choose Classification or a numeric target."
            )
        else:
            if profile.kind == "constant":
                warnings.append(f"{name} has a single value, so there is nothing to predict.")
            elif profile.class_like:
                notes.append(
                    f"{name} has only {profile.distinct:,} distinct values. Regression treats them as ordered numbers; "
                    "choose Classification if they are unordered categories."
                )
            unusable = int((~analysis.parse_numeric(raw[target]).valid).sum())
            if unusable:
                warnings.append(f"{unusable:,} targets are missing, non-numeric or infinite and will be excluded.")
        return TargetGuidance(profile, tuple(warnings), tuple(notes), False)

    if profile.kind == "empty":
        warnings.append("There are no usable target labels.")
    elif profile.kind == "constant":
        warnings.append(f"{name} has a single class ({profile.reason}), so there is nothing to classify.")
    elif profile.kind == "continuous":
        warnings.append(
            f"{name} contains {profile.distinct:,} distinct numeric values and appears continuous. "
            "Classification is unlikely to be appropriate for this target. "
            "Consider switching the task type to Regression."
        )
    elif profile.kind == "ambiguous":
        warnings.append(
            f"{name} is ambiguous as a classification target: {profile.reason}. Confirm that its values are "
            "categories before relying on class metrics, or use Regression for a numeric score."
        )
    elif profile.whole_numbers and profile.kind == "multiclass":
        notes.append(
            f"{name} uses {profile.distinct} numeric class codes. They are treated as categories; "
            "if they are an ordinal score, Regression is also possible."
        )
    show = profile.class_like or (treat_as_classes and profile.kind in ("continuous", "ambiguous"))
    if show:
        warnings.extend(class_warnings(raw[target]))
    return TargetGuidance(profile, tuple(warnings), tuple(notes), show)


def target_warnings(raw: pd.DataFrame, target: str, task: str) -> list[str]:
    return list(target_guidance(raw, target, task).warnings)


# ---------------------------------------------------------------------------
# Leakage

_PREDICTION_TERMS = {"pred", "preds", "predict", "predicted", "prediction", "predictions", "yhat", "ypred"}
_PROBABILITY_TERMS = {"prob", "probs", "proba", "probability", "probabilities", "posterior"}
_OUTCOME_TERMS = {"target", "targets", "label", "labels", "outcome", "outcomes"}
_TEMPORAL_TERMS = {"future", "post", "after"}
# Too common in legitimate inputs (credit_score, test_result) to flag on their own.
_CONTEXT_TERMS = {"score", "scores", "result", "results"}
_BIGRAMS = {("y", "pred"): "y_pred", ("y", "hat"): "y_hat", ("y", "prob"): "y_prob", ("y", "score"): "y_score"}
_BENIGN_PAIRS = {("post", "code"), ("post", "office"), ("post", "box"), ("after", "tax"), ("post", "tax")}
_CATEGORY_TEXT = {
    "prediction": "which suggests it may have been produced after the outcome was known",
    "probability": "which suggests a model probability or score rather than an input measurement",
    "outcome": "which suggests it records or is derived from the outcome",
    "temporal": "which suggests information recorded after the outcome",
}


def name_tokens(name: str) -> list[str]:
    """Lowercase words in a column name: snake_case, kebab-case, spaces, camelCase,
    PascalCase, acronyms ("PREDScore" → pred, score) and letter/digit boundaries."""
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", str(name))
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
    spaced = re.sub(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", " ", spaced)
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", spaced) if token]


def _quoted(terms: list[str]) -> str:
    quoted = [f"'{term}'" for term in terms]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _contains(tokens: list[str], part: list[str]) -> bool:
    return bool(part) and any(tokens[i:i + len(part)] == part for i in range(len(tokens) - len(part) + 1))


def _name_reasons(column: str, target: str, class_tokens: dict[str, str]) -> list[tuple[str, str]]:
    tokens = name_tokens(column)
    pairs = list(pairwise(tokens))
    found: dict[str, str] = {}
    for pair, term in _BIGRAMS.items():
        if pair in pairs:
            found[term] = "prediction"
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if token in _PREDICTION_TERMS:
            found.setdefault(token, "prediction")
        elif token in _PROBABILITY_TERMS:
            found.setdefault(token, "probability")
        elif token in _OUTCOME_TERMS:
            found.setdefault(token, "outcome")
        elif token in _TEMPORAL_TERMS and (token, following) not in _BENIGN_PAIRS:
            found.setdefault(token, "temporal")
        elif token in _CONTEXT_TERMS:
            found.setdefault(token, "context")
    target_tokens = name_tokens(target)
    names_target = (
        column != target and _contains(tokens, target_tokens) and max(map(len, target_tokens), default=0) >= 3
    )
    named_class = next((class_tokens[t] for t in tokens if t in class_tokens), None)

    for category in ("prediction", "probability", "outcome", "temporal"):
        if category in found.values():
            terms = list(found)
            return [(f"name: {', '.join(terms)}", f"Column name contains {_quoted(terms)}, {_CATEGORY_TEXT[category]}.")]
    context = [term for term, category in found.items() if category == "context"]
    if context and names_target:
        return [(f"name: {', '.join(context)} + target name", f"Column name combines {_quoted(context)} with the target name “{target}”, which suggests a score or result derived from the outcome.")]
    if context and named_class:
        return [(f"name: {', '.join(context)} + class", f"Column name combines {_quoted(context)} with the target class “{named_class}”, which suggests a per-class model score.")]
    if names_target:
        return [("name: target name", f"Column name includes the target name “{target}”; check that it is not derived from the target.")]
    return []


@dataclass(frozen=True)
class _Values:
    text: pd.Series  # cleaned text; None where missing
    numbers: pd.Series  # float; NaN where missing or not numeric

    @property
    def numeric(self) -> bool:
        present = int(self.text.notna().sum())
        return present > 0 and int(self.numbers.notna().sum()) >= analysis.MIN_NUMERIC_SHARE * present


def _values(series: pd.Series) -> _Values:
    return _Values(analysis.clean_labels(series), analysis.parse_numeric(series).values)


def _linear_description(x: np.ndarray, y: np.ndarray) -> str | None:
    """'the target plus 100' style text if y is exactly a·x + b, else None."""
    slope, intercept = np.polyfit(x, y, 1)
    scale = max(float(np.max(np.abs(y))), 1.0)
    if not np.allclose(y, slope * x + intercept, rtol=1e-9, atol=1e-9 * scale):
        return None
    if abs(slope - 1) <= 1e-9:
        return f"the target plus a constant offset ({intercept:+.6g})"
    if abs(intercept) <= 1e-9 * scale:
        return f"the target multiplied by {slope:.6g}"
    return f"a linear transformation of the target ({slope:.6g} × target {intercept:+.6g})"


def _statistical_reasons(target: _Values, feature: _Values, identifier: bool) -> list[tuple[str, str]]:
    paired = target.text.notna() & feature.text.notna()
    n = int(paired.sum())
    if n >= 5:
        same = float((target.text[paired] == feature.text[paired]).mean())
        if same >= .98:
            return [("copy of the target", f"Column is at least 98% identical to the target ({same:.1%} of {n:,} paired rows match).")]

    reasons: list[tuple[str, str]] = []
    correlation = None
    if target.numeric and feature.numeric:
        both = target.numbers.notna() & feature.numbers.notna()
        x, y = target.numbers[both].to_numpy(), feature.numbers[both].to_numpy()
        if len(x) >= 5 and np.mean(x == y) >= .98:
            return [("numeric copy of the target", f"Column holds the same numbers as the target written differently (for example “1” and “1.0”) on {np.mean(x == y):.1%} of paired rows.")]
        if len(x) >= 10 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            linear = _linear_description(x, y)
            if linear:
                return [("exact linear transform", f"Column is exactly {linear}, so it encodes the target directly.")]
            correlation = float(np.corrcoef(x, y)[0, 1])
            if not identifier and len(np.unique(y)) >= 3:
                ranks = float(np.corrcoef(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])
                if abs(ranks) >= 1 - 1e-9:
                    return [("monotonic transform", f"Column is a strictly monotonic one-to-one transformation of the target (rank correlation {ranks:+.3f}), for example a log, power or rescaled copy.")]

    if n >= 10:
        f_codes, f_values = pd.factorize(feature.text[paired])
        t_codes, t_values = pd.factorize(target.text[paired])
        k_f, k_t = len(f_values), len(t_values)
        # A value seen about once maps "deterministically" by accident, so a mapping
        # direction is judged only when its source values repeat (IDs never qualify).
        ft_ok = 2 <= k_f <= n / 5 and not identifier
        tf_ok = 2 <= k_t <= n / 5 and k_f >= 2
        if ft_ok or tf_ok:
            table = pd.DataFrame({"f": f_codes, "t": t_codes}).value_counts()
            purity_ft = table.groupby(level="f").max().sum() / n if ft_ok else 0
            purity_tf = table.groupby(level="t").max().sum() / n if tf_ok else 0
            baseline_error = 1 - np.bincount(t_codes).max() / n
            unit = "target class" if not target.numeric else "target value"
            if purity_ft == 1 and purity_tf == 1:
                pairs = sorted(table.index, key=lambda p: analysis.label_sort_key(str(f_values[p[0]])))
                shown = ", ".join(f"{f_values[f]} → {t_values[t]}" for f, t in pairs[:3]) + (", …" if len(pairs) > 3 else "")
                reasons.append(("label-encoded copy", f"Column appears to be a label-encoded copy of the target: each of its {k_f:,} values corresponds to exactly one {unit} and vice versa ({shown})."))
            elif purity_ft == 1:
                reasons.append(("deterministic mapping", f"Each value in this feature maps deterministically to exactly one {unit} ({k_f:,} feature values, {k_t:,} target values)."))
            elif purity_tf == 1:
                reasons.append(("determined by the target", f"Every {unit} has exactly one value of this column, so it is fully determined by the target (for example a class-group indicator)."))
            elif ft_ok and 1 - purity_ft <= .02 and 1 - purity_ft <= .1 * baseline_error:
                reasons.append(("near-deterministic mapping", f"Each value in this feature maps to a single {unit} on {purity_ft:.1%} of rows (majority-class baseline {1 - baseline_error:.1%}), a near-deterministic mapping."))
    if correlation is not None and abs(correlation) >= .98:
        reasons.append(("near-perfect correlation", f"Column is almost perfectly correlated with the target (r = {correlation:.3f})."))
    return reasons


@dataclass(frozen=True)
class LeakageFinding:
    feature: str
    risk: str  # "high": the values encode the target; "review": only the name is suspicious
    signals: tuple[str, ...]  # short labels for summaries
    reasons: tuple[str, ...]  # full sentences for the UI


def leakage_findings(raw: pd.DataFrame, target: str, features=None) -> list[LeakageFinding]:
    """Heuristics flag clues, not proof; availability at prediction time needs user review.

    `features` defaults to every column other than the target.
    """
    columns = [c for c in (raw.columns if features is None else features) if c != target and c in raw.columns]
    target_values = _values(raw[target])
    identifiers = set(analysis.suggest_id_columns(raw, exclude=(target,)))
    class_tokens: dict[str, str] = {}
    labels = target_values.text.dropna()
    if not target_values.numeric and labels.nunique() <= 50:
        for label in labels.unique():
            for token in name_tokens(label):
                if len(token) >= 3:
                    class_tokens.setdefault(token, label)
    findings = []
    for column in columns:
        statistical = _statistical_reasons(target_values, _values(raw[column]), column in identifiers)
        named = _name_reasons(column, target, class_tokens)
        if statistical or named:
            everything = statistical + named
            findings.append(LeakageFinding(
                column, "high" if statistical else "review",
                tuple(signal for signal, _ in everything), tuple(reason for _, reason in everything),
            ))
    return sorted(findings, key=lambda finding: finding.risk != "high")


def leakage_warnings(raw: pd.DataFrame, target: str, features: tuple[str, ...]) -> pd.DataFrame:
    records = [
        {"feature": f.feature, "risk": f.risk, "signals": "; ".join(f.signals), "review reason": " ".join(f.reasons)}
        for f in leakage_findings(raw, target, features)
    ]
    return pd.DataFrame(records, columns=["feature", "risk", "signals", "review reason"])


# ---------------------------------------------------------------------------
# Dataset quality overview


def quality_overview(
    raw: pd.DataFrame, target: str | None = None, task: str = "classification", treat_as_classes: bool = False,
    summary: pd.DataFrame | None = None, findings: list[LeakageFinding] | None = None,
) -> pd.DataFrame:
    """One row per question a user should answer before training: is this dataset safe and sensible?"""
    summary = dataset_summary(raw) if summary is None else summary
    rows = []
    missing = int(summary["missing"].sum())
    affected = summary.loc[summary["missing"] > 0, "column"].astype(str).tolist()
    rows.append(("Missing values", "OK" if not missing else "Review",
                 "none" if not missing else f"{missing:,} of {raw.size:,} cells ({missing / max(raw.size, 1):.1%}) in "
                 f"{len(affected)} column{'s' if len(affected) != 1 else ''}: " + ", ".join(affected[:6]) + (", …" if len(affected) > 6 else "")))
    duplicates = duplicate_count(raw)
    rows.append(("Duplicate rows", "OK" if not duplicates else "Review",
                 "none" if not duplicates else f"{duplicates:,} exact repeats; keep repeated entities together with a grouped split"))
    numeric = int((summary["kind"] == "numeric").sum())
    rows.append(("Column types", "Info", f"{numeric} numeric, {len(summary) - numeric} text or mixed (text columns are not training features)"))
    ids = summary.loc[summary["possible identifier"], "column"].astype(str).tolist()
    rows.append(("Possible identifier columns", "Info", ", ".join(ids) + " (excluded from training features)" if ids else "none detected"))
    if target is not None:
        guidance = target_guidance(raw, target, task, treat_as_classes)
        profile = guidance.profile
        fits = profile.class_like if task == "classification" else profile.numeric_share >= analysis.MIN_NUMERIC_SHARE and profile.kind not in ("constant", "empty")
        status = "OK" if fits else "Problem" if profile.kind in ("constant", "empty") else "Warning"
        rows.append(("Target type", status, f"“{target}”: {profile.description} ({profile.reason})"))
        if task == "classification":
            if guidance.show_class_diagnostics:
                balance = class_balance(raw[target])
                rows.append(("Target balance", "Warning" if balance["imbalanced"] or balance["smallest"] < 2 else "OK",
                             f"{balance['classes']} classes; largest “{balance['largest_class']}” {balance['largest']:,} "
                             f"({balance['largest_share']:.1%}), smallest “{balance['smallest_class']}” {balance['smallest']:,} "
                             f"({balance['smallest_share']:.1%}); ratio {balance['ratio']:.1f} : 1"))
            else:
                rows.append(("Target balance", "Not applicable", "not shown because the values do not look like classes"))
        if findings is not None:
            high = [f for f in findings if f.risk == "high"]
            detail = "; ".join(f"{f.feature} ({', '.join(f.signals)})" for f in findings[:6]) + ("; …" if len(findings) > 6 else "")
            rows.append(("Potential leakage", "Warning" if high else "Review" if findings else "OK",
                         f"{len(findings)} suspicious column{'s' if len(findings) != 1 else ''}: {detail}" if findings else "no suspicious columns found"))
    return pd.DataFrame(rows, columns=["check", "status", "detail"])
