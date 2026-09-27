"""Data logic for Model Failure Atlas. No Streamlit imports.

Every number the app shows is computed here, from one set of prepared rows,
so the overview, the failure map and the detail table cannot disagree, and
all of it can be tested without a browser.

Row identity: each data row keeps its 1-based position in the uploaded file
as its index ("row 1" is the first row after the header). Every derived
table uses the same index, so any subset can be traced back to the file.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal
from itertools import pairwise

import numpy as np
import pandas as pd

# Cell values that mean "no value", compared after trimming whitespace.
# Matching is exact and case-sensitive. "None"/"none" are deliberately not
# markers because they are plausible class names (e.g. "no finding").
MISSING_MARKERS = ("", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "<NA>", "#N/A")

DEFAULT_BINS = 4
DEFAULT_SMALL_SAMPLE = 10  # cells with fewer examples than this are flagged
MIN_NUMERIC_SHARE = 0.9  # share of non-missing values that must be numbers


class DataError(ValueError):
    """A problem with the input that the user can fix. The message is shown as-is."""


# ---------------------------------------------------------------------------
# Loading


@dataclass(frozen=True)
class LoadedCSV:
    frame: pd.DataFrame  # every cell as text; index = 1-based source row
    notes: tuple[str, ...] = ()  # non-fatal observations about the file


def load_csv(data: bytes) -> LoadedCSV:
    """Read CSV bytes without type inference.

    All cells stay text exactly as written, so a label such as "01" is never
    turned into the number 1. Numeric parsing happens later, per column.
    """
    if not data.strip():
        raise DataError("The file is empty.")

    notes: list[str] = []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        notes.append(
            "The file is not valid UTF-8, so it was read as Latin-1. "
            "Check that accented characters look right."
        )

    sep = ","
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if "," not in first_line:
        for candidate, name in ((";", "semicolons"), ("\t", "tabs")):
            if candidate in first_line:
                sep = candidate
                notes.append(f"Columns are separated by {name}, so the file was read that way.")
                break

    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise DataError("The file has no header row.") from None
    except pd.errors.ParserError as exc:
        detail = str(exc).strip().removeprefix("Error tokenizing data. C error: ")
        raise DataError(
            f"The file could not be read as a CSV ({detail}). "
            "Check that every row has the same number of columns as the header."
        ) from None

    if frame.shape[1] < 2:
        raise DataError(
            "Only one column was found. The file needs at least separate columns "
            "for the actual label and the predicted label."
        )
    if frame.empty:
        raise DataError("The file has a header row but no data rows.")

    header = next(csv.reader(io.StringIO(text.lstrip()), delimiter=sep), [])
    repeated = sorted({name for name in header if name and header.count(name) > 1})
    if repeated:
        notes.append(
            "Repeated column names were made unique by adding .1, .2, …: "
            + ", ".join(repeated)
        )

    frame = frame.fillna("")  # rows with fewer fields than the header
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="row")
    return LoadedCSV(frame, tuple(notes))


# ---------------------------------------------------------------------------
# Column roles


_ACTUAL_NAMES = (
    "actual", "actual_label", "actual_class", "label", "true_label", "true_class",
    "y_true", "target", "ground_truth", "truth", "gold", "class",
)
_PREDICTED_NAMES = (
    "predicted", "predicted_label", "predicted_class", "prediction", "pred",
    "pred_label", "y_pred", "yhat", "y_hat", "model_prediction", "output",
)
_ACTUAL_TOKENS = {"actual", "true", "truth", "gold", "target", "label"}
_PREDICTED_TOKENS = {"predicted", "prediction", "pred", "yhat"}
_ID_TOKENS = {"id", "ids", "uuid", "guid"}
_CONFIDENCE_TOKENS = {"confidence", "conf"}


def _name_tokens(name: str) -> list[str]:
    """Split a column name into lowercase words: "customerID" -> ["customer", "id"]."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", spaced) if token]


def _normalise_name(name: str) -> str:
    return "_".join(_name_tokens(name))


def suggest_label_columns(columns: Iterable[str]) -> tuple[str | None, str | None]:
    """Guess the actual and predicted label columns from their names.

    Returns None for a role when no name is a convincing match; the app then
    asks the user instead of guessing.
    """
    columns = list(columns)
    normalised = {column: _normalise_name(column) for column in columns}

    def by_name(names: tuple[str, ...], taken: str | None) -> str | None:
        for name in names:
            for column in columns:
                if normalised[column] == name and column != taken:
                    return column
        return None

    def by_token(tokens: set[str], avoid: set[str], taken: str | None) -> str | None:
        for column in columns:
            words = set(_name_tokens(column))
            if column != taken and words & tokens and not words & avoid:
                return column
        return None

    predicted = by_name(_PREDICTED_NAMES, None) or by_token(_PREDICTED_TOKENS, set(), None)
    actual = by_name(_ACTUAL_NAMES, predicted) or by_token(
        _ACTUAL_TOKENS, _PREDICTED_TOKENS, predicted
    )
    return actual, predicted


def suggest_id_columns(raw: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Columns that look like identifiers rather than measurements.

    These are only defaults: the app lets the user add or remove columns.
    A column qualifies if its name contains an ID word ("image_id",
    "customerID", "uuid"), it is a pandas index artifact ("Unnamed: 0"), or
    its values are digit codes with leading zeros or a 1..n row counter.
    """
    skip = set(exclude)
    found: list[str] = []
    for column in raw.columns:
        if column in skip:
            continue
        name = column.strip().lower()
        if (
            set(_name_tokens(column)) & _ID_TOKENS
            or name in {"index", "idx"}
            or column.startswith("Unnamed:")
        ):
            found.append(column)
            continue

        text = raw[column].astype(str).str.strip()
        text = text[~text.isin(MISSING_MARKERS)]
        if text.empty or not text.str.fullmatch(r"\d+").all():
            continue
        if text.str.match(r"0\d").any() or text.str.len().max() > 15:
            found.append(column)  # zero-padded or very long digit codes
            continue
        numbers = text.astype("int64")
        if (
            len(numbers) >= 20
            and numbers.is_unique
            and numbers.max() - numbers.min() + 1 == len(numbers)
        ):
            found.append(column)  # a row counter such as 1..n
    return found


def suggest_confidence_column(columns: Iterable[str]) -> str | None:
    """A column explicitly named as confidence. Names like "probability" or
    "score" are ambiguous (they may be positive-class scores), so they are
    never suggested."""
    for column in columns:
        if set(_name_tokens(column)) & _CONFIDENCE_TOKENS:
            return column
    return None


# ---------------------------------------------------------------------------
# Labels and overall metrics


def clean_labels(values: pd.Series) -> pd.Series:
    """Trim surrounding whitespace and turn missing markers into None.

    Nothing else changes: "Stop" and "stop" stay different classes, and so
    do "1" and "01". Use find_label_variants to warn about such pairs.
    """
    text = values.fillna("").astype(str).str.strip()
    return text.astype(object).where(~text.isin(MISSING_MARKERS), None)


def _finite_number(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def label_sort_key(label: str) -> tuple:
    """Numbers in numeric order first, then text alphabetically."""
    number = _finite_number(label)
    if number is not None:
        return (0, number, label)
    return (1, label.casefold(), label)


def sort_labels(labels: Iterable[str]) -> list[str]:
    return sorted(set(labels), key=label_sort_key)


def find_label_variants(labels: Iterable[str]) -> list[tuple[str, ...]]:
    """Groups of distinct labels that differ only in capitalisation, spacing
    or number formatting ("Stop"/"stop", "1"/"1.0"/"01").

    They are reported, never merged: silently merging could change results.
    """
    groups: dict[tuple, set[str]] = defaultdict(set)
    for label in set(labels):
        number = _finite_number(label)
        key = ("number", number) if number is not None else ("text", " ".join(label.split()).casefold())
        groups[key].add(label)
    variants = [tuple(sort_labels(group)) for group in groups.values() if len(group) > 1]
    return sorted(variants, key=lambda group: label_sort_key(group[0]))


@dataclass(frozen=True)
class Evaluation:
    """Rows with both labels present, plus counts of rows that were excluded."""

    rows: pd.DataFrame  # index: source row; columns: actual, predicted, is_error
    n_uploaded: int
    n_missing_actual: int  # only the actual label is missing
    n_missing_predicted: int  # only the predicted label is missing
    n_missing_both: int
    task: str = "classification"
    tolerance: float = 0.0

    @property
    def n_evaluated(self) -> int:
        return len(self.rows)

    @property
    def n_excluded(self) -> int:
        return self.n_missing_actual + self.n_missing_predicted + self.n_missing_both


def evaluate(raw: pd.DataFrame, actual_col: str, predicted_col: str) -> Evaluation:
    """Mark each row with both labels as correct or an error.

    A row with a missing actual or predicted label cannot be judged, so it
    is excluded from every metric and counted instead.
    """
    if actual_col == predicted_col:
        raise DataError("The actual and predicted labels must come from two different columns.")
    actual = clean_labels(raw[actual_col])
    predicted = clean_labels(raw[predicted_col])
    no_actual, no_predicted = actual.isna(), predicted.isna()
    keep = ~(no_actual | no_predicted)

    rows = pd.DataFrame({"actual": actual[keep], "predicted": predicted[keep]})
    rows["is_error"] = (rows["actual"] != rows["predicted"]).astype(bool)
    return Evaluation(
        rows=rows,
        n_uploaded=len(raw),
        n_missing_actual=int((no_actual & ~no_predicted).sum()),
        n_missing_predicted=int((no_predicted & ~no_actual).sum()),
        n_missing_both=int((no_actual & no_predicted).sum()),
    )


def label_role_warning(raw: pd.DataFrame, actual_col: str, predicted_col: str) -> str | None:
    """Flag likely measurements without rejecting legitimate numeric class codes."""
    clues = []
    kinds = []
    class_sets = []
    for role, column in (("Actual", actual_col), ("Predicted", predicted_col)):
        labels = clean_labels(raw[column]).dropna()
        if labels.empty:
            continue
        class_sets.append(set(labels))
        numeric = pd.to_numeric(labels, errors="coerce")
        numeric_share = float(np.isfinite(numeric).mean())
        is_numeric = numeric_share >= MIN_NUMERIC_SHARE
        kinds.append("numeric" if numeric_share == 1 else "text" if numeric_share == 0 else "mixed numeric/text")
        distinct = labels.nunique()
        if is_numeric and distinct > max(MAX_SUSPECT_CLASSES, int(0.2 * len(labels))):
            clues.append(f"{role.lower()} column “{column}” has {distinct:,} distinct numeric values")
    if len(kinds) == 2 and kinds[0] != kinds[1]:
        clues.append(f"the actual labels are {kinds[0]} and the predicted labels are {kinds[1]}")
    if len(class_sets) == 2 and not class_sets[0].intersection(class_sets[1]):
        clues.append("the actual and predicted class sets do not overlap")
    if not clues:
        return None
    return (
        "Check the label columns: " + "; ".join(clues) + ". These may be measurements "
        "rather than class labels. Actual and predicted must name the same class "
        "outcomes for each row. Numeric class codes are valid when they really are "
        "classes. A model can legitimately predict classes absent from this test set; "
        "these checks are diagnostic, not rejection rules. Labels are not converted or merged. "
        "Confirm the meaning of these columns before interpreting accuracy or errors."
    )


MAX_SUSPECT_CLASSES = 30


@dataclass(frozen=True)
class Overview:
    n_evaluated: int
    n_correct: int
    n_errors: int
    confusion: pd.DataFrame  # index: actual class, columns: predicted class
    predicted_only: tuple[str, ...]  # predicted but never an actual label
    never_predicted: tuple[str, ...]  # actual labels the model never predicted

    @property
    def accuracy(self) -> float | None:
        return self.n_correct / self.n_evaluated if self.n_evaluated else None

    @property
    def error_rate(self) -> float | None:
        return self.n_errors / self.n_evaluated if self.n_evaluated else None


def compute_overview(rows: pd.DataFrame) -> Overview:
    """Accuracy and confusion matrix. The denominator is the evaluated rows."""
    n_errors = int(rows["is_error"].sum())
    actual_classes = set(rows["actual"])
    predicted_classes = set(rows["predicted"])
    classes = sort_labels(actual_classes | predicted_classes)

    confusion = (
        pd.crosstab(rows["actual"], rows["predicted"])
        .reindex(index=classes, columns=classes, fill_value=0)
        .astype(int)
    )
    confusion.index.name = "actual"
    confusion.columns.name = "predicted"
    return Overview(
        n_evaluated=len(rows),
        n_correct=len(rows) - n_errors,
        n_errors=n_errors,
        confusion=confusion,
        predicted_only=tuple(sort_labels(predicted_classes - actual_classes)),
        never_predicted=tuple(sort_labels(actual_classes - predicted_classes)),
    )


def top_confusions(overview: Overview, limit: int = 5) -> pd.DataFrame:
    """The most frequent kinds of mistake: (actual, predicted) pairs by count."""
    pairs = overview.confusion.stack()
    actual = pairs.index.get_level_values(0)
    predicted = pairs.index.get_level_values(1)
    pairs = pairs[(pairs > 0) & (actual != predicted)]
    top = pairs.sort_values(ascending=False, kind="stable").head(limit)
    return pd.DataFrame(
        {
            "actual": top.index.get_level_values(0),
            "predicted": top.index.get_level_values(1),
            "errors": top.to_numpy(dtype=int),
            "share of errors": top.to_numpy(dtype=float) / max(overview.n_errors, 1),
        }
    )


def class_breakdown(rows: pd.DataFrame) -> pd.DataFrame:
    """Examples, errors and error rate per actual class, largest class first.

    A group's error rate can reflect its class mix, so this shows the mix.
    """
    grouped = rows.groupby("actual")["is_error"].agg(examples="size", errors="sum")
    grouped["error rate"] = grouped["errors"] / grouped["examples"]
    grouped = grouped.reindex(sort_labels(grouped.index))
    return (
        grouped.sort_values("examples", ascending=False, kind="stable")
        .rename_axis("actual class")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Numeric features


@dataclass(frozen=True)
class NumericColumn:
    values: pd.Series  # float; NaN where missing or invalid
    missing: pd.Series  # bool: blank or a missing marker
    invalid: pd.Series  # bool: present but not a finite number

    @property
    def valid(self) -> pd.Series:
        return self.values.notna()


def parse_numeric(values: pd.Series) -> NumericColumn:
    """Parse text as numbers, keeping missing and invalid cells apart."""
    text = values.fillna("").astype(str).str.strip()
    missing = pd.Series(text.isin(MISSING_MARKERS).to_numpy(dtype=bool), index=values.index)
    numbers = pd.to_numeric(text.where(~missing), errors="coerce").astype("float64")
    finite = pd.Series(np.isfinite(numbers.to_numpy()), index=values.index)
    return NumericColumn(values=numbers.where(finite), missing=missing, invalid=~missing & ~finite)


@dataclass(frozen=True)
class FeatureOptions:
    usable: tuple[str, ...]  # numeric columns that can be heatmap axes, in file order
    excluded: dict[str, str]  # column -> reason it cannot be an axis


def feature_options(
    raw: pd.DataFrame, index: pd.Index, reserved: Mapping[str, str]
) -> FeatureOptions:
    """Decide which columns can be heatmap axes, judged on the given rows.

    `reserved` maps columns that already have a role (labels, identifiers,
    confidence) to a short description; those are never axes.
    """
    usable: list[str] = []
    excluded: dict[str, str] = {}
    subset = raw.loc[index]
    for column in raw.columns:
        if column in reserved:
            excluded[column] = reserved[column]
            continue
        parsed = parse_numeric(subset[column])
        present = int((~parsed.missing).sum())
        n_valid = int(parsed.valid.sum())
        if present == 0:
            excluded[column] = "no values"
        elif n_valid < MIN_NUMERIC_SHARE * present:
            example = subset[column][parsed.invalid].astype(str).str.strip().iloc[0]
            excluded[column] = f"not numeric (e.g. “{example}”)"
        elif parsed.values.nunique() < 2:
            excluded[column] = f"only one value ({parsed.values.dropna().iloc[0]:g})"
        else:
            usable.append(column)
    return FeatureOptions(tuple(usable), excluded)


@dataclass(frozen=True)
class ConfidenceCheck:
    usable: bool
    message: str


def check_confidence(raw: pd.DataFrame, index: pd.Index, column: str) -> ConfidenceCheck:
    """Confidence must be the predicted-class probability, so 0 to 1."""
    parsed = parse_numeric(raw.loc[index, column])
    values = parsed.values.dropna()
    if values.empty:
        return ConfidenceCheck(False, f"“{column}” has no numeric values, so it can't be a confidence.")
    outside = int(((values < 0) | (values > 1)).sum())
    if outside:
        return ConfidenceCheck(
            False,
            f"{outside:,} values of “{column}” are outside 0–1 (range {values.min():g} to "
            f"{values.max():g}). Confidence must be a probability between 0 and 1, "
            "so it won't be used.",
        )
    gaps = int((~parsed.valid).sum())
    note = f" {gaps:,} evaluated rows have no usable value." if gaps else ""
    return ConfidenceCheck(True, f"“{column}” is between 0 and 1 on every evaluated row that has a value.{note}")


def confidence_by_result(confidence: pd.Series, is_error: pd.Series) -> dict[str, float | None]:
    """Mean confidence of correct and wrong predictions (None if no values)."""
    def mean(values: pd.Series) -> float | None:
        values = values.dropna()
        return float(values.mean()) if len(values) else None

    return {"correct": mean(confidence[~is_error]), "error": mean(confidence[is_error])}


# ---------------------------------------------------------------------------
# Ranges (bins)


@dataclass(frozen=True)
class Bins:
    """Ranges for one feature.

    Decimal features: each range includes its lower edge and excludes its
    upper edge, except the last, which includes the maximum. Whole-number
    features: ranges are inclusive runs of integers. Either way each value
    belongs to exactly one range.
    """

    edges: tuple[float, ...]  # strictly increasing, len(labels) + 1
    labels: tuple[str, ...]
    whole_numbers: bool
    places: int = 0  # decimal places used for the edges
    exact_labels: bool = False

    def __len__(self) -> int:
        return len(self.labels)

    def assign(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Range index (0-based) for each value, which must lie within the edges."""
        positions = np.searchsorted(
            np.asarray(self.edges, dtype=float), np.asarray(values, dtype=float), side="right"
        )
        return np.clip(positions - 1, 0, len(self) - 1)

    def describe(self, i: int, name: str) -> str:
        """The range in words, e.g. "0.26 ≤ brightness < 0.49"."""
        if self.whole_numbers:
            low, high = int(self.edges[i]), int(self.edges[i + 1]) - 1
            return f"{name} = {low:,}" if low == high else f"{low:,} ≤ {name} ≤ {high:,}"
        low, high = ((str(edge) if self.exact_labels else f"{edge:,.{self.places}f}") for edge in self.edges[i : i + 2])
        upper = "≤" if i == len(self) - 1 else "<"
        return f"{low} ≤ {name} {upper} {high}"


_DECIMAL_CONTEXT = Context(prec=1000)


def _round_to(value: float, places: int, rounding: str) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=rounding, context=_DECIMAL_CONTEXT))


def _span_label(low: int, high: int) -> str:
    if low == high:
        return f"{low:,}"
    joiner = " to " if low < 0 or high < 0 else "–"
    return f"{low:,}{joiner}{high:,}"


def make_bins(values: pd.Series | np.ndarray, n_bins: int = DEFAULT_BINS) -> Bins:
    """Split the observed range into `n_bins` equal-width ranges.

    Edges are rounded to a readable precision and the rounded edges are what
    rows are assigned by, so a value printed on a boundary lands in the range
    the labels say. Whole-number features get whole-number edges; if there
    are fewer whole values than ranges, each value gets its own range.
    """
    numbers = np.asarray(values, dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        raise DataError("There are no numeric values to split into ranges.")
    low, high = float(numbers.min()), float(numbers.max())
    if low == high:
        raise DataError(f"Every value is {low:g}, so there is nothing to split into ranges.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    if np.all(numbers == np.floor(numbers)):
        low_int, high_int = int(low), int(high)
        width = high_int - low_int + 1  # number of whole values in the range
        n = min(n_bins, width)
        # Integer arithmetic for round(i * width / n): exact and never repeats an edge.
        edges = [low_int + (2 * i * width + n) // (2 * n) for i in range(n + 1)]
        labels = [_span_label(edges[i], edges[i + 1] - 1) for i in range(n)]
        return Bins(tuple(float(edge) for edge in edges), tuple(labels), whole_numbers=True)

    step = (high - low) / n_bins
    places = max(0, math.ceil(-math.log10(step)) + 1)
    edges = (
        [_round_to(low, places, ROUND_FLOOR)]
        + [_round_to(low + i * step, places, ROUND_HALF_EVEN) for i in range(1, n_bins)]
        + [_round_to(high, places, ROUND_CEILING)]
    )

    def fmt(edge: float) -> str:
        return f"{edge:,.{places}f}"

    labels = [
        f"[{fmt(edges[i])}, {fmt(edges[i + 1])}{']' if i == n_bins - 1 else ')'}"
        for i in range(n_bins)
    ]
    return Bins(tuple(edges), tuple(labels), whole_numbers=False, places=places)


def flexible_bins(values, n_bins: int, method: str = "Equal width", edges: tuple[float, ...] | None = None) -> Bins:
    """Quantiles keep tied values together; custom edges must cover every mapped value."""
    if method == "Equal width":
        return make_bins(values, n_bins)
    numbers = np.asarray(values, dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        raise DataError("There are no numeric values to split into ranges.")
    if method == "Quantiles":
        boundaries = np.unique(np.quantile(numbers, np.linspace(0, 1, n_bins + 1)))
    elif method == "Custom boundaries":
        boundaries = np.asarray(edges if edges is not None else (), dtype=float)
    else:
        raise DataError("Unknown binning method.")
    if len(boundaries) < 2 or len(boundaries) > 21 or not np.isfinite(boundaries).all() or (np.diff(boundaries) <= 0).any():
        raise DataError("Use 2–21 finite, strictly increasing boundaries. Tied quantiles may leave no ranges.")
    if numbers.min() < boundaries[0] or numbers.max() > boundaries[-1]:
        raise DataError(f"Custom boundaries must cover every mapped value ({numbers.min():g} to {numbers.max():g}).")
    boundaries = tuple(float(value) for value in boundaries)
    labels = [f"[{lo}, {hi}{']' if i == len(boundaries) - 2 else ')'}" for i, (lo, hi) in enumerate(pairwise(boundaries))]
    return Bins(boundaries, tuple(labels), False, exact_labels=True)


# ---------------------------------------------------------------------------
# Failure map


@dataclass(frozen=True)
class CellStats:
    total: int
    errors: int
    error_rate: float | None  # None for an empty cell: there is no rate


def small_cells(cells: pd.DataFrame, threshold: int = DEFAULT_SMALL_SAMPLE) -> pd.Series:
    """Non-empty cells with fewer than `threshold` examples: rates are noisy."""
    return (cells["total"] > 0) & (cells["total"] < threshold)


@dataclass(frozen=True)
class FailureMap:
    x_col: str
    y_col: str
    x_bins: Bins
    y_bins: Bins
    # One row per mapped example, index = source row. Bins are assigned once
    # here; the heatmap and the detail table both read them from this frame.
    rows: pd.DataFrame  # x_value, y_value, x_bin, y_bin, is_error
    # One row per grid cell, including empty ones (error_rate NaN, not 0).
    cells: pd.DataFrame  # x_bin, y_bin, total, errors, error_rate
    n_evaluated: int
    n_omitted: int  # evaluated rows missing either feature
    omitted: dict[str, int]  # reason -> rows affected (nonzero only; a row can have two)

    @property
    def n_mapped(self) -> int:
        return len(self.rows)

    def cell(self, x_bin: int, y_bin: int) -> CellStats:
        """The numbers shown on the map for one cell."""
        match = self.cells[(self.cells["x_bin"] == x_bin) & (self.cells["y_bin"] == y_bin)]
        if len(match) != 1:
            raise KeyError(f"No cell ({x_bin}, {y_bin}) in this map")
        total, errors = int(match["total"].iloc[0]), int(match["errors"].iloc[0])
        return CellStats(total, errors, errors / total if total else None)

    def cell_index(self, x_bin: int, y_bin: int, errors_only: bool = False) -> pd.Index:
        """Source rows in one cell, taken from the same assignments as the map."""
        mask = (self.rows["x_bin"] == x_bin) & (self.rows["y_bin"] == y_bin)
        if errors_only:
            mask &= self.rows["is_error"]
        return self.rows.index[mask]


def build_failure_map(
    raw: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int = DEFAULT_BINS,
    method: str = "Equal width",
    x_edges: tuple[float, ...] | None = None,
    y_edges: tuple[float, ...] | None = None,
) -> FailureMap:
    """Group evaluated rows into a grid of feature ranges and count errors.

    Rows missing either feature are left off this map (and counted) but stay
    in the overall metrics. Ranges are computed from the mapped rows.
    """
    if x_col == y_col:
        raise DataError("Choose two different features for the X and Y axes.")

    subset = raw.loc[evaluation_rows.index]
    x = parse_numeric(subset[x_col])
    y = parse_numeric(subset[y_col])
    mappable = x.valid & y.valid

    omitted = {
        f"{x_col} is blank": int(x.missing.sum()),
        f"{x_col} is not a number": int(x.invalid.sum()),
        f"{y_col} is blank": int(y.missing.sum()),
        f"{y_col} is not a number": int(y.invalid.sum()),
    }
    omitted = {reason: count for reason, count in omitted.items() if count}

    rows = pd.DataFrame(
        {
            "x_value": x.values[mappable],
            "y_value": y.values[mappable],
            "is_error": evaluation_rows["is_error"][mappable],
        }
    )
    for column in ("residual", "absolute_error"):
        if column in evaluation_rows:
            rows[column] = evaluation_rows.loc[rows.index, column]
    if rows.empty:
        raise DataError(f"No evaluated row has numeric values for both {x_col} and {y_col}.")

    bins = {}
    for axis, column in (("x", x_col), ("y", y_col)):
        try:
            bins[axis] = flexible_bins(rows[f"{axis}_value"], n_bins, method, x_edges if axis == "x" else y_edges)
        except DataError as exc:
            raise DataError(
                f"{column} can't be split into ranges on the mapped rows: {exc} Choose another feature."
            ) from None
    rows["x_bin"] = bins["x"].assign(rows["x_value"])
    rows["y_bin"] = bins["y"].assign(rows["y_value"])

    grid = pd.MultiIndex.from_product(
        [range(len(bins["x"])), range(len(bins["y"]))], names=["x_bin", "y_bin"]
    )
    cells = (
        rows.groupby(["x_bin", "y_bin"])["is_error"]
        .agg(total="size", errors="sum")
        .reindex(grid, fill_value=0)
        .astype(int)
        .reset_index()
    )
    cells["error_rate"] = cells["errors"] / cells["total"].where(cells["total"] > 0)
    if "absolute_error" in rows:
        grouped = rows.groupby(["x_bin", "y_bin"])
        cells["mae"] = grouped["absolute_error"].mean().reindex(grid).to_numpy()
        cells["mean_residual"] = grouped["residual"].mean().reindex(grid).to_numpy()

    from uncertainty import add_cell_uncertainty

    cells = add_cell_uncertainty(cells)
    return FailureMap(
        x_col=x_col,
        y_col=y_col,
        x_bins=bins["x"],
        y_bins=bins["y"],
        rows=rows,
        cells=cells,
        n_evaluated=len(evaluation_rows),
        n_omitted=int((~mappable).sum()),
        omitted=omitted,
    )


def consistency_checks(overview: Overview, fmap: FailureMap | None = None) -> list[tuple[str, bool]]:
    """The invariants from the build plan, evaluated on real results."""
    checks = [
        (
            "Correct + errors = evaluated rows",
            overview.n_correct + overview.n_errors == overview.n_evaluated,
        ),
        (
            "Confusion matrix total = evaluated rows",
            int(overview.confusion.to_numpy().sum()) == overview.n_evaluated,
        ),
    ]
    if fmap is not None:
        in_range = fmap.rows["x_bin"].between(0, len(fmap.x_bins) - 1) & fmap.rows[
            "y_bin"
        ].between(0, len(fmap.y_bins) - 1)
        checks += [
            ("Every mapped row sits in exactly one cell", bool(in_range.all() and fmap.rows.index.is_unique)),
            ("Cell counts add up to the mapped rows", int(fmap.cells["total"].sum()) == fmap.n_mapped),
            (
                "Cell errors add up to the errors among mapped rows",
                int(fmap.cells["errors"].sum()) == int(fmap.rows["is_error"].sum()),
            ),
            ("Mapped + omitted = evaluated rows in map scope", fmap.n_mapped + fmap.n_omitted == fmap.n_evaluated and fmap.n_evaluated <= overview.n_evaluated),
        ]
    return checks


# ---------------------------------------------------------------------------
# Inspection


def _unique_name(name: str, taken: Iterable[str]) -> str:
    taken = set(taken)
    candidate = name
    while candidate in taken:
        candidate = f"{candidate} (atlas)"
    return candidate


def detail_table(
    raw: pd.DataFrame,
    is_error: pd.Series,
    index: pd.Index,
    lead_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """The original columns for `index`, preceded by row number and result.

    `lead_columns` are moved to the front. `numeric_columns` are shown as
    numbers when every value in these rows parses; otherwise they stay as
    written, so an invalid entry is visible rather than blank.
    """
    table = raw.loc[index].copy()
    for column in numeric_columns:
        parsed = parse_numeric(table[column])
        if not parsed.invalid.any():
            values = parsed.values
            whole = values.dropna()
            table[column] = values.astype("Int64") if (whole == whole.round()).all() else values
    lead = [column for column in dict.fromkeys(lead_columns) if column in table.columns]
    table = table[lead + [column for column in table.columns if column not in lead]]

    result = np.where(is_error.loc[index].to_numpy(dtype=bool), "✗ error", "✓ correct")
    table.insert(0, _unique_name("result", table.columns), result)
    table.insert(0, _unique_name("row", table.columns), table.index.to_numpy())
    return table.reset_index(drop=True)
