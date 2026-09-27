"""Multiclass ROC-AUC averaging modes and validation of uploaded class-probability columns."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import analysis
import evaluation

CLASSES = ["bird", "cat", "dog", "fish"]


def uploaded(n=240, seed=0, weights=(.55, .25, .15, .05)):
    """Probabilities that are informative but imperfect, with imbalanced classes."""
    rng = np.random.RandomState(seed)
    actual = rng.choice(CLASSES, size=n, p=weights)
    logits = rng.normal(size=(n, len(CLASSES))) + 1.5 * (actual[:, None] == np.array(CLASSES))
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    raw = pd.DataFrame({"actual": actual, "predicted": np.array(CLASSES)[probabilities.argmax(axis=1)]})
    for i, label in enumerate(CLASSES):
        raw[f"p_{label}"] = [repr(float(v)) for v in probabilities[:, i]]
    raw.index = pd.RangeIndex(1, n + 1)
    return raw, probabilities


def rows_of(raw):
    return analysis.evaluate(raw, "actual", "predicted").rows


MAPPING = {label: f"p_{label}" for label in CLASSES}


@pytest.mark.parametrize("average", evaluation.AUC_AVERAGES)
@pytest.mark.parametrize("multi_class", list(evaluation.AUC_STRATEGIES))
def test_multiclass_auc_matches_scikit_learn(average, multi_class):
    raw, probabilities = uploaded()
    rows = rows_of(raw)
    mapped = evaluation.map_class_probabilities(raw, rows, MAPPING, CLASSES)
    assert mapped.usable and not mapped.warnings
    auc, note = evaluation.probability_auc(rows, mapped.probabilities, average, multi_class)
    expected = roc_auc_score(raw["actual"], probabilities, multi_class=multi_class, average=average, labels=CLASSES)
    assert auc == pytest.approx(expected)
    assert evaluation.AUC_STRATEGIES[multi_class] in note
    assert note.startswith("Macro" if average == "macro" else "Prevalence-weighted")


def test_averaging_modes_differ_on_imbalanced_classes_and_binary_is_invariant():
    raw, _ = uploaded()
    rows = rows_of(raw)
    probabilities = evaluation.map_class_probabilities(raw, rows, MAPPING, CLASSES).probabilities
    scores = {(a, m): evaluation.probability_auc(rows, probabilities, a, m)[0] for a in evaluation.AUC_AVERAGES for m in evaluation.AUC_STRATEGIES}
    assert len({round(v, 10) for v in scores.values()}) > 1
    binary = raw.loc[raw["actual"].isin(["bird", "cat"])].copy()
    binary["predicted"] = np.where(binary["p_cat"].astype(float) > binary["p_bird"].astype(float), "cat", "bird")
    total = binary["p_bird"].astype(float) + binary["p_cat"].astype(float)
    two = pd.DataFrame({"bird": binary["p_bird"].astype(float) / total, "cat": binary["p_cat"].astype(float) / total})
    values = {evaluation.probability_auc(rows_of(binary), two, a, m)[0] for a in evaluation.AUC_AVERAGES for m in evaluation.AUC_STRATEGIES}
    assert len(values) == 1


def test_invalid_averaging_mode_is_rejected():
    raw, _ = uploaded()
    rows = rows_of(raw)
    with pytest.raises(analysis.DataError):
        evaluation.probability_auc(rows, pd.DataFrame(index=rows.index), "micro")


def test_probability_columns_are_suggested_from_class_names():
    columns = ["id", "prob_setosa", "Prob Versicolor", "probability_virginica", "p_other"]
    assert evaluation.suggest_probability_columns(columns, ["setosa", "versicolor", "virginica", "missing"]) == {
        "setosa": "prob_setosa", "versicolor": "Prob Versicolor", "virginica": "probability_virginica", "missing": None,
    }
    assert evaluation.suggest_probability_columns(["score"], ["a", "b"]) == {"a": None, "b": None}


def test_every_class_needs_its_own_column():
    raw, _ = uploaded()
    rows = rows_of(raw)
    missing = evaluation.map_class_probabilities(raw, rows, {**MAPPING, "fish": None}, CLASSES)
    assert not missing.usable and missing.probabilities.empty
    assert "Missing: “fish”" in missing.errors[0]
    duplicate = evaluation.map_class_probabilities(raw, rows, {**MAPPING, "fish": "p_dog"}, CLASSES)
    assert "“p_dog” is mapped to more than one class" in duplicate.errors[0]


def test_values_must_be_numeric_complete_probabilities():
    raw, _ = uploaded()
    rows = rows_of(raw)
    text = raw.copy()
    text.loc[3, "p_cat"] = "high"
    assert "non-numeric" in evaluation.map_class_probabilities(text, rows, MAPPING, CLASSES).errors[0]
    blank = raw.copy()
    blank.loc[3, "p_cat"] = ""
    assert "blank on 1 evaluated rows" in evaluation.map_class_probabilities(blank, rows, MAPPING, CLASSES).errors[0]
    scores = raw.copy()
    scores["p_cat"] = (scores["p_cat"].astype(float) * 10).astype(str)
    assert "outside 0–1" in evaluation.map_class_probabilities(scores, rows, MAPPING, CLASSES).errors[0]


def test_row_sums_warn_on_rounding_and_fail_when_classes_are_missing():
    raw, _ = uploaded()
    rows = rows_of(raw)
    rounded = raw.copy()
    for column in MAPPING.values():
        rounded[column] = rounded[column].astype(float).round(2).astype(str)
    mapped = evaluation.map_class_probabilities(rounded, rows, MAPPING, CLASSES)
    assert mapped.usable and "probably from rounding" in mapped.warnings[0]
    np.testing.assert_allclose(mapped.probabilities.sum(axis=1), 1)
    assert evaluation.probability_auc(rows, mapped.probabilities)[0] is not None

    partial = raw.drop(columns="p_fish").assign(p_fish=(raw["p_fish"].astype(float) * 0).astype(str))
    failed = evaluation.map_class_probabilities(partial, rows, MAPPING, CLASSES)
    assert not failed.usable and "must sum to 1 on every row" in failed.errors[0]


def test_swapped_columns_are_flagged_but_never_rewritten():
    raw, _ = uploaded()
    raw = raw.rename(columns={"p_bird": "prob_bird", "p_cat": "prob_cat"})
    swapped = {**MAPPING, "bird": "prob_cat", "cat": "prob_bird"}
    mapped = evaluation.map_class_probabilities(raw, rows_of(raw), swapped, CLASSES)
    assert mapped.usable
    assert any("“prob_cat” is mapped to “bird” but its name mentions “cat”" in w for w in mapped.warnings)
    assert any("matches the predicted label on only" in w for w in mapped.warnings)
    np.testing.assert_allclose(mapped.probabilities["bird"], raw["prob_cat"].astype(float))


def test_auc_is_unavailable_without_valid_probabilities():
    raw, _ = uploaded()
    rows = rows_of(raw)
    assert evaluation.probability_auc(rows, pd.DataFrame()) == (None, "ROC-AUC is unavailable because there are no class probabilities.")
    unnormalised = pd.DataFrame({label: np.full(len(rows), .5) for label in CLASSES}, index=rows.index)
    assert "must sum to 1" in evaluation.probability_auc(rows, unnormalised)[1]
