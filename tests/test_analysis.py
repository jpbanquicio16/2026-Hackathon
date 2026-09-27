"""Checks for analysis.py on small, hand-worked inputs.

The main fixture is small enough to verify by hand; the expected numbers in
each test are worked out in the comments next to it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis

ROOT = Path(__file__).resolve().parents[1]

# 13 rows. x spans 0-8 and y spans 0-4, so with 4 ranges per axis the edges
# are x: 0, 2, 4, 6, 8 and y: 0, 1, 2, 3, 4.
#   r9  has no actual label, r10 has "NA" as its prediction -> excluded
#   r11 has padded " b " (still correct) and a blank x -> off the map only
#   r12 has x = "abc" -> off the map only
#   r13 is "B" vs "b": different classes, so an error, and a variant warning
FIXTURE = """id,actual,predicted,x,y
r1,a,a,0.0,0.0
r2,a,b,1.5,0.5
r3,b,b,2.0,0.5
r4,b,a,3.9,1.0
r5,a,a,4.0,1.5
r6,c,a,8.0,4.0
r7,a,a,7.0,3.5
r8,a,d,7.5,3.0
r9,,a,1.0,1.0
r10,b,NA,1.0,1.0
r11, b ,b,,2.0
r12,a,a,abc,2.0
r13,B,b,5.0,2.5
"""


@pytest.fixture
def raw() -> pd.DataFrame:
    return analysis.load_csv(FIXTURE.encode()).frame


@pytest.fixture
def evaluation(raw) -> analysis.Evaluation:
    return analysis.evaluate(raw, "actual", "predicted")


@pytest.fixture
def fmap(raw, evaluation) -> analysis.FailureMap:
    return analysis.build_failure_map(raw, evaluation.rows, "x", "y", n_bins=4)


# ---------------------------------------------------------------------------
# Overview


def test_rows_are_numbered_from_one_and_kept_as_text(raw):
    assert list(raw.index[:3]) == [1, 2, 3]
    assert raw.loc[11, "actual"] == " b "  # untouched until labels are cleaned
    assert raw.loc[1, "x"] == "0.0"


def test_missing_labels_are_excluded_and_counted(evaluation):
    assert evaluation.n_uploaded == 13
    assert (evaluation.n_missing_actual, evaluation.n_missing_predicted, evaluation.n_missing_both) == (1, 1, 0)
    assert evaluation.n_excluded == 2
    assert evaluation.n_evaluated == 11
    assert 9 not in evaluation.rows.index and 10 not in evaluation.rows.index


def test_overview_matches_hand_count(evaluation):
    overview = analysis.compute_overview(evaluation.rows)
    # Errors: r2 a->b, r4 b->a, r6 c->a, r8 a->d, r13 B->b.
    assert sorted(evaluation.rows.index[evaluation.rows["is_error"]]) == [2, 4, 6, 8, 13]
    assert (overview.n_evaluated, overview.n_correct, overview.n_errors) == (11, 6, 5)
    assert overview.accuracy == pytest.approx(6 / 11)
    assert overview.n_correct + overview.n_errors == overview.n_evaluated


def test_confusion_matrix_uses_union_of_classes(evaluation):
    overview = analysis.compute_overview(evaluation.rows)
    confusion = overview.confusion
    # "d" is only ever predicted; "c" and "B" are never predicted.
    assert list(confusion.index) == ["a", "B", "b", "c", "d"]
    assert list(confusion.columns) == ["a", "B", "b", "c", "d"]
    expected = pd.DataFrame(
        [
            [4, 0, 1, 0, 1],  # a: r1 r5 r7 r12 correct, r2 -> b, r8 -> d
            [0, 0, 1, 0, 0],  # B: r13 -> b
            [1, 0, 2, 0, 0],  # b: r3 and r11 correct, r4 -> a
            [1, 0, 0, 0, 0],  # c: r6 -> a
            [0, 0, 0, 0, 0],  # d: never an actual label
        ],
        index=confusion.index,
        columns=confusion.columns,
    )
    pd.testing.assert_frame_equal(confusion, expected)
    assert overview.predicted_only == ("d",)
    assert overview.never_predicted == ("B", "c")
    assert int(np.trace(confusion.to_numpy())) == overview.n_correct


def test_top_confusions_counts_off_diagonal_only(evaluation):
    top = analysis.top_confusions(analysis.compute_overview(evaluation.rows), limit=10)
    assert top["errors"].sum() == 5
    assert set(zip(top["actual"], top["predicted"], strict=True)) =={("a", "b"), ("a", "d"), ("B", "b"), ("b", "a"), ("c", "a")}
    assert top["share of errors"].sum() == pytest.approx(1.0)


def test_labels_are_trimmed_but_never_merged():
    cleaned = analysis.clean_labels(pd.Series([" stop ", "Stop", "01", "1", "", "NA", "None", "n/a", "\tyield\n"]))
    assert cleaned.tolist() == ["stop", "Stop", "01", "1", None, None, "None", None, "yield"]


def test_label_variants_are_reported(evaluation):
    labels = pd.concat([evaluation.rows["actual"], evaluation.rows["predicted"]])
    assert analysis.find_label_variants(labels) == [("B", "b")]
    assert analysis.find_label_variants(["1", "1.0", "01", "2", "stop", "Stop ", "no  entry", "no entry"]) == [
        ("01", "1", "1.0"),
        ("no  entry", "no entry"),
        ("stop", "Stop "),
    ]


# ---------------------------------------------------------------------------
# Failure map


def test_bins_and_assignments_match_hand_count(fmap):
    assert fmap.x_bins.edges == (0.0, 2.0, 4.0, 6.0, 8.0)
    assert fmap.x_bins.labels == ("[0.0, 2.0)", "[2.0, 4.0)", "[4.0, 6.0)", "[6.0, 8.0]")
    assert fmap.y_bins.edges == (0.0, 1.0, 2.0, 3.0, 4.0)
    # Boundary values (x = 2.0, 4.0; y = 1.0, 3.0) land in the upper range;
    # the maximum (x = 8.0, y = 4.0) lands in the last range.
    assigned = {row: (r.x_bin, r.y_bin) for row, r in fmap.rows.iterrows()}
    assert assigned == {
        1: (0, 0), 2: (0, 0), 3: (1, 0), 4: (1, 1), 5: (2, 1),
        6: (3, 3), 7: (3, 3), 8: (3, 3), 13: (2, 2),
    }


def test_missing_features_leave_the_map_but_not_the_metrics(evaluation, fmap):
    assert fmap.n_evaluated == 11
    assert fmap.n_mapped == 9
    assert fmap.n_omitted == 2
    assert fmap.omitted == {"x is blank": 1, "x is not a number": 1}
    assert {11, 12} <= set(evaluation.rows.index)  # still in accuracy
    assert not {11, 12} & set(fmap.rows.index)


def test_cell_counts_and_rates(fmap):
    nonempty = {
        (int(c.x_bin), int(c.y_bin)): (int(c.total), int(c.errors))
        for c in fmap.cells.itertuples()
        if c.total
    }
    assert nonempty == {
        (0, 0): (2, 1),  # r1 correct, r2 error
        (1, 0): (1, 0),  # r3
        (1, 1): (1, 1),  # r4
        (2, 1): (1, 0),  # r5
        (2, 2): (1, 1),  # r13
        (3, 3): (3, 2),  # r6 error, r7 correct, r8 error
    }
    assert fmap.cell(3, 3).error_rate == pytest.approx(2 / 3)
    assert fmap.cell(0, 0).error_rate == pytest.approx(0.5)
    assert fmap.cell(1, 0).error_rate == 0.0  # a real 0%, not an empty cell
    assert len(fmap.cells) == 16


def test_empty_cells_have_no_rate(fmap):
    empty = fmap.cells[fmap.cells["total"] == 0]
    assert len(empty) == 10
    assert empty["error_rate"].isna().all()
    stats = fmap.cell(0, 3)
    assert (stats.total, stats.errors, stats.error_rate) == (0, 0, None)
    assert len(fmap.cell_index(0, 3)) == 0


def test_selected_cell_rows_match_its_numbers(fmap):
    stats = fmap.cell(3, 3)
    assert list(fmap.cell_index(3, 3)) == [6, 7, 8]
    assert list(fmap.cell_index(3, 3, errors_only=True)) == [6, 8]
    assert len(fmap.cell_index(3, 3)) == stats.total
    assert len(fmap.cell_index(3, 3, errors_only=True)) == stats.errors


def test_small_cells_are_flagged_not_hidden(fmap):
    flags = analysis.small_cells(fmap.cells, threshold=3)
    flagged = {(int(c.x_bin), int(c.y_bin)) for c, f in zip(fmap.cells.itertuples(), flags, strict=True) if f}
    assert flagged == {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}  # (3, 3) has 3, empties excluded


def test_consistency_checks_pass(evaluation, fmap):
    checks = analysis.consistency_checks(analysis.compute_overview(evaluation.rows), fmap)
    assert len(checks) == 6
    assert all(ok for _, ok in checks), checks


def test_same_feature_on_both_axes_is_rejected(raw, evaluation):
    with pytest.raises(analysis.DataError, match="two different features"):
        analysis.build_failure_map(raw, evaluation.rows, "x", "x")


def test_non_numeric_feature_is_rejected(raw, evaluation):
    with pytest.raises(analysis.DataError, match="No evaluated row has numeric values"):
        analysis.build_failure_map(raw, evaluation.rows, "id", "y")


def test_feature_constant_on_mapped_rows_is_explained():
    # x varies overall, but the only row with a different x has no y.
    raw = analysis.load_csv(b"actual,predicted,x,y\na,a,1,1\na,b,1,2\nb,b,5,\n").frame
    rows = analysis.evaluate(raw, "actual", "predicted").rows
    with pytest.raises(analysis.DataError, match="x can't be split into ranges"):
        analysis.build_failure_map(raw, rows, "x", "y")


# ---------------------------------------------------------------------------
# Ranges


def test_boundary_values_belong_to_the_upper_range():
    bins = analysis.make_bins([0.0, 1.5, 8.0])  # edges 0, 2, 4, 6, 8
    assert bins.assign([0.0, 1.999, 2.0, 4.0, 6.0, 7.99, 8.0]).tolist() == [0, 0, 1, 2, 3, 3, 3]


def test_printed_edges_are_the_edges_used():
    # Unrounded, the last inner edge is 0.1 + 2 * 0.1 = 0.30000000000000004,
    # which would put a value of exactly 0.3 in the range printed "[0.20, 0.30)".
    assert 0.1 + 2 * ((0.4 - 0.1) / 3) > 0.3
    bins = analysis.make_bins([0.1, 0.2, 0.3, 0.4], 3)
    assert bins.labels == ("[0.10, 0.20)", "[0.20, 0.30)", "[0.30, 0.40]")
    assert bins.assign([0.1, 0.2, 0.3, 0.4]).tolist() == [0, 1, 2, 2]
    assert bins.describe(1, "p") == "0.20 ≤ p < 0.30"
    assert bins.describe(2, "p") == "0.30 ≤ p ≤ 0.40"


def test_repeated_values_are_each_counted_once():
    values = [1.5, 1.5, 1.5, 3.0, 3.0, 4.5]
    bins = analysis.make_bins(values, 3)  # edges 1.5, 2.5, 3.5, 4.5
    assert np.bincount(bins.assign(values), minlength=3).tolist() == [3, 2, 1]


def test_whole_numbers_get_whole_number_ranges():
    bins = analysis.make_bins(np.arange(18, 80), 4)  # 62 ages split as evenly as possible
    assert bins.whole_numbers
    assert bins.labels == ("18–33", "34–48", "49–64", "65–79")
    assert bins.assign([18, 33, 34, 79]).tolist() == [0, 0, 1, 3]
    assert bins.describe(1, "age") == "34 ≤ age ≤ 48"


def test_few_whole_values_get_one_range_each():
    bins = analysis.make_bins([0, 1, 2, 2, 1], 4)
    assert bins.labels == ("0", "1", "2")
    assert bins.assign([0, 1, 2]).tolist() == [0, 1, 2]
    assert bins.describe(1, "calls") == "calls = 1"
    assert analysis.make_bins([0, 1, 1, 0]).labels == ("0", "1")


def test_negative_whole_numbers_are_written_with_words():
    assert analysis.make_bins(np.arange(-10, 10), 2).labels == ("-10 to -1", "0–9")


def test_single_value_cannot_be_split():
    with pytest.raises(analysis.DataError, match="nothing to split"):
        analysis.make_bins([3.0, 3.0, 3.0])


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_bins", [2, 4, 7])
def test_invariants_hold_on_random_data(seed, n_bins):
    rng = np.random.default_rng(seed)
    n = 300
    frame = pd.DataFrame(
        {
            "actual": rng.choice(["a", "b", "c", ""], n, p=[0.3, 0.3, 0.3, 0.1]),
            "predicted": rng.choice(["a", "b", "c"], n),
            "x": rng.normal(size=n).round(2).astype(str),
            "y": rng.integers(0, 50, n).astype(str),
        }
    )
    frame.loc[rng.choice(n, 20, replace=False), "x"] = ""
    raw = analysis.load_csv(frame.to_csv(index=False).encode()).frame
    evaluation = analysis.evaluate(raw, "actual", "predicted")
    overview = analysis.compute_overview(evaluation.rows)
    fmap = analysis.build_failure_map(raw, evaluation.rows, "x", "y", n_bins)

    assert all(ok for _, ok in analysis.consistency_checks(overview, fmap))
    for cell in fmap.cells.itertuples():
        assert len(fmap.cell_index(cell.x_bin, cell.y_bin)) == cell.total
        assert len(fmap.cell_index(cell.x_bin, cell.y_bin, errors_only=True)) == cell.errors
    for axis, bins in (("x", fmap.x_bins), ("y", fmap.y_bins)):
        edges = np.asarray(bins.edges)
        index = fmap.rows[f"{axis}_bin"].to_numpy()
        values = fmap.rows[f"{axis}_value"].to_numpy()
        last = index == len(bins) - 1
        assert (values >= edges[index]).all()
        assert ((values < edges[index + 1]) | (last & (values <= edges[index + 1]))).all()


# ---------------------------------------------------------------------------
# Loading and column roles


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "empty"),
        (b"  \n\n", "empty"),
        (b"actual,predicted\n", "no data rows"),
        (b"only_column\nx\ny\n", "Only one column"),
        (b"a,b\n1,2\n3,4,5\n", "same number of columns"),
    ],
)
def test_unusable_files_are_rejected_with_a_reason(content, message):
    with pytest.raises(analysis.DataError, match=message):
        analysis.load_csv(content)


def test_awkward_but_readable_files_load_with_notes():
    bom = analysis.load_csv(b"\xef\xbb\xbfactual,predicted\na,b\n")
    assert list(bom.frame.columns) == ["actual", "predicted"] and not bom.notes

    semicolons = analysis.load_csv(b"actual;predicted;x\na;a;1\n")
    assert list(semicolons.frame.columns) == ["actual", "predicted", "x"]
    assert "semicolons" in semicolons.notes[0]

    latin = analysis.load_csv("actual,predicted\ncafé,café\n".encode("latin-1"))
    assert latin.frame.loc[1, "actual"] == "café" and "Latin-1" in latin.notes[0]

    short = analysis.load_csv(b"a,b,c\n1,2\n")
    assert short.frame.loc[1, "c"] == ""

    repeated = analysis.load_csv(b"x,x,y\n1,2,3\n")
    assert list(repeated.frame.columns) == ["x", "x.1", "y"] and "Repeated" in repeated.notes[0]


def test_parse_numeric_separates_missing_from_invalid():
    parsed = analysis.parse_numeric(pd.Series(["1", " 2.5 ", "", "NA", "abc", "inf", "1e3", "1,234"]))
    assert parsed.values.tolist()[:2] == [1.0, 2.5] and parsed.values.iloc[6] == 1000.0
    assert parsed.missing.tolist() == [False, False, True, True, False, False, False, False]
    assert parsed.invalid.tolist() == [False, False, False, False, True, True, False, True]


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["image_id", "actual", "predicted", "x"], ("actual", "predicted")),
        (["y_true", "y_pred"], ("y_true", "y_pred")),
        (["label", "prediction", "prob"], ("label", "prediction")),
        (["label", "predicted_label"], ("label", "predicted_label")),
        (["True Species", "Predicted Species"], ("True Species", "Predicted Species")),
        (["a", "b"], (None, None)),
    ],
)
def test_label_columns_are_suggested_by_name(columns, expected):
    assert analysis.suggest_label_columns(columns) == expected


def test_identifier_columns_are_suggested():
    rows = 25
    frame = pd.DataFrame(
        {
            "customerID": [str(i) for i in range(rows)],
            "paid": ["10", "20"] * 12 + ["30"],
            "valid_score": ["1", "2"] * 12 + ["3"],
            "zip": ["02139", "10001"] * 12 + ["94105"],
            "counter": [str(i) for i in range(1, rows + 1)],
            "Unnamed: 0": [str(i) for i in range(rows)],
            "age": [str(20 + i % 7) for i in range(rows)],
        }
    )
    assert analysis.suggest_id_columns(frame) == ["customerID", "zip", "counter", "Unnamed: 0"]


def test_confidence_is_suggested_only_when_named_as_such():
    assert analysis.suggest_confidence_column(["actual", "predicted", "model_confidence"]) == "model_confidence"
    assert analysis.suggest_confidence_column(["actual", "predicted", "churn_probability", "score"]) is None


def test_feature_options_explain_every_exclusion():
    raw = analysis.load_csv(
        b"row_id,actual,predicted,age,constant,city,score\n"
        b"1,a,a,20,5,Paris,0.5\n2,a,b,30,5,Rome,0.7\n3,b,b,40,5,Oslo,0.9\n4,b,a,50,5,Rome,1.2\n"
    ).frame
    reserved = {"actual": "actual label", "predicted": "predicted label", "row_id": "identifier"}
    options = analysis.feature_options(raw, raw.index, reserved)
    assert options.usable == ("age", "score")
    assert options.excluded == {
        "row_id": "identifier",
        "actual": "actual label",
        "predicted": "predicted label",
        "constant": "only one value (5)",
        "city": "not numeric (e.g. “Paris”)",
    }


def test_mostly_numeric_column_stays_usable():
    values = [str(i) for i in range(19)] + ["twelve"]
    raw = pd.DataFrame({"f": values}, index=pd.RangeIndex(1, 21))
    assert analysis.feature_options(raw, raw.index, {}).usable == ("f",)


# ---------------------------------------------------------------------------
# Detail table and confidence


def test_detail_table_shows_original_rows_with_result(raw, evaluation):
    table = analysis.detail_table(
        raw, evaluation.rows["is_error"], pd.Index([6, 7, 8]),
        lead_columns=["id", "actual", "predicted", "x", "y"], numeric_columns=["x", "y"],
    )
    assert list(table.columns) == ["row", "result", "id", "actual", "predicted", "x", "y"]
    assert table["row"].tolist() == [6, 7, 8]
    assert table["result"].tolist() == ["✗ error", "✓ correct", "✗ error"]
    assert table["x"].tolist() == [8.0, 7.0, 7.5]


def test_detail_table_keeps_invalid_values_visible(raw, evaluation):
    table = analysis.detail_table(raw, evaluation.rows["is_error"], pd.Index([12]), numeric_columns=["x"])
    assert table.loc[0, "x"] == "abc"


def test_detail_table_never_overwrites_a_user_column():
    raw = analysis.load_csv(b"actual,predicted,result\na,b,keep me\n").frame
    rows = analysis.evaluate(raw, "actual", "predicted").rows
    table = analysis.detail_table(raw, rows["is_error"], rows.index)
    assert list(table.columns) == ["row", "result (atlas)", "actual", "predicted", "result"]
    assert table.loc[0, "result"] == "keep me"


def test_confidence_must_be_a_probability():
    raw = analysis.load_csv(b"actual,predicted,conf,pct\na,a,0.9,90\na,b,0.4,40\nb,b,,55\n").frame
    assert analysis.check_confidence(raw, raw.index, "conf").usable
    check = analysis.check_confidence(raw, raw.index, "pct")
    assert not check.usable and "outside 0–1" in check.message


def test_confidence_by_result():
    means = analysis.confidence_by_result(pd.Series([0.9, 0.8, 0.4, np.nan]), pd.Series([False, False, True, True]))
    assert means["correct"] == pytest.approx(0.85)
    assert means["error"] == pytest.approx(0.4)


def test_class_breakdown(evaluation):
    breakdown = analysis.class_breakdown(evaluation.rows)
    assert breakdown.iloc[0].tolist() == ["a", 6, 2, pytest.approx(2 / 6)]
    assert breakdown["examples"].sum() == evaluation.n_evaluated


# ---------------------------------------------------------------------------
# The shipped demo data


def test_sample_dataset_supports_the_demo():
    raw = analysis.load_csv((ROOT / "sample_predictions.csv").read_bytes()).frame
    evaluation = analysis.evaluate(raw, "actual", "predicted")
    assert (evaluation.n_uploaded, evaluation.n_excluded) == (1200, 8)
    overview = analysis.compute_overview(evaluation.rows)
    assert 0.8 < overview.accuracy < 0.9
    assert overview.predicted_only == ("speed_80",)

    fmap = analysis.build_failure_map(raw, evaluation.rows, "brightness", "blur_px")
    assert fmap.omitted == {"blur_px is blank": 12}
    worst = fmap.cells.loc[fmap.cells["error_rate"].idxmax()]
    assert (worst["x_bin"], worst["y_bin"]) == (0, 3)  # darkest and blurriest
    assert worst["error_rate"] > 3 * overview.error_rate
    assert all(ok for _, ok in analysis.consistency_checks(overview, fmap))
