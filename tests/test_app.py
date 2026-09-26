"""User-journey checks for app.py, run headlessly with Streamlit's AppTest.

These cover what the analysis tests can't: that the page wires the pieces
together, and that selections never go stale when inputs change.
"""

import csv
import io
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CHURN = (ROOT / "examples" / "churn_predictions.csv").read_bytes()
IRIS_EVAL = (ROOT / "examples" / "iris_heldout_predictions.csv").read_bytes()


def run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture
def app() -> AppTest:
    return run(AppTest.from_file(str(ROOT / "app.py"), default_timeout=60))


def selectbox(at: AppTest, label: str):
    # Labels built from column names are Markdown-escaped ("blur\_px range").
    return next(s for s in at.selectbox if s.label.replace("\\", "") == label)


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def cell_table(at: AppTest):
    """The selected cell's rows: the last table before the class-mix table."""
    return at.dataframe[-2].value


def upload(at: AppTest, name: str, content: bytes) -> AppTest:
    at.radio(key="source").set_value("Upload a CSV")
    run(at)
    at.file_uploader(key="upload").set_value((name, content, "text/csv"))
    return run(at)


def test_sample_opens_with_overview_and_map(app):
    assert [h.value for h in app.header] == ["1 · Data", "2 · Overall performance", "3 · Failure map", "4 · Inspect a cell"]
    assert metric(app, "Accuracy") == "85.0%"
    assert metric(app, "Evaluated rows") == "1,192"
    assert metric(app, "Excluded rows") == "8"
    assert "8 rows excluded" in app.warning[0].value
    assert selectbox(app, "X axis feature").value == "brightness"
    assert selectbox(app, "Y axis feature").value == "blur_px"
    assert len(app.get("vega_lite_chart")) == 1
    assert any("Select a brightness range" in i.value.replace("\\", "") for i in app.info)


def test_selected_cell_lists_exactly_its_rows(app):
    selectbox(app, "brightness range").set_value(0)
    selectbox(app, "blur_px range").set_value(3)
    run(app)
    summary = next(m.value for m in app.markdown if "error rate" in m.value and "out of" in m.value)
    assert "37 errors out of 53 examples" in summary
    table = cell_table(app)
    assert len(table) == 53
    assert (table["result"] == "✗ error").sum() == 37
    assert table["brightness"].between(0.02, 0.26, inclusive="left").all()
    assert table["blur_px"].between(6.0, 8.0).all()
    assert app.expander[-1].label == "Consistency checks: all passed"


def test_errors_only_keeps_the_selected_cell(app):
    selectbox(app, "brightness range").set_value(0)
    selectbox(app, "blur_px range").set_value(3)
    run(app)
    app.toggle(key="errors_only").set_value(True)
    run(app)
    assert (selectbox(app, "brightness range").value, selectbox(app, "blur_px range").value) == (0, 3)
    table = cell_table(app)
    assert len(table) == 37 and set(table["result"]) == {"✗ error"}
    assert any("Showing 37 of the cell's 53 rows (errors only)" in c.value for c in app.caption)
    summary = next(m.value for m in app.markdown if "out of" in m.value)
    assert "37 errors out of 53 examples" in summary  # headline unchanged


def test_changing_an_axis_clears_the_selection(app):
    selectbox(app, "brightness range").set_value(0)
    selectbox(app, "blur_px range").set_value(3)
    run(app)
    selectbox(app, "Y axis feature").set_value("speed_kmh")
    run(app)
    assert selectbox(app, "brightness range").value is None
    assert selectbox(app, "speed_kmh range").value is None
    assert not any("out of" in m.value for m in app.markdown)


def test_changing_the_number_of_ranges_clears_the_selection(app):
    selectbox(app, "brightness range").set_value(1)
    selectbox(app, "blur_px range").set_value(1)
    run(app)
    app.select_slider(key="n_bins").set_value(6)
    run(app)
    x_range = selectbox(app, "brightness range")
    assert x_range.value is None and len(x_range.options) == 6


def test_changing_the_small_sample_threshold_keeps_the_selection(app):
    selectbox(app, "brightness range").set_value(1)
    selectbox(app, "blur_px range").set_value(1)
    run(app)
    app.number_input(key="small_n").set_value(200)
    run(app)
    assert (selectbox(app, "brightness range").value, selectbox(app, "blur_px range").value) == (1, 1)
    assert any("Small sample" in w.value for w in app.warning)


def test_uploading_a_new_file_resets_mapping_and_selection(app):
    selectbox(app, "brightness range").set_value(0)
    selectbox(app, "blur_px range").set_value(3)
    run(app)
    upload(app, "churn.csv", CHURN)
    assert selectbox(app, "Actual label column").value == "label"
    assert selectbox(app, "Predicted label column").value == "prediction"
    assert metric(app, "Excluded rows") == "7"
    assert any("“Stayed” / “stayed”" in w.value for w in app.warning)
    x_col = selectbox(app, "X axis feature").value
    assert x_col == "tenure_months"
    assert selectbox(app, f"{x_col} range").value is None
    assert any("monthly\\_charges is not a number: 4" in c.value for c in app.caption)


def test_switching_back_to_the_sample_starts_clean(app):
    upload(app, "churn.csv", CHURN)
    selectbox(app, "tenure_months range").set_value(0)
    selectbox(app, "monthly_charges range").set_value(3)
    run(app)
    app.radio(key="source").set_value("Sample dataset")
    run(app)
    assert selectbox(app, "brightness range").value is None
    assert metric(app, "Evaluated rows") == "1,192"


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "The file is empty"),
        (b"just,a,header\n", "no data rows"),
        (b"a,b\n1,2\n3,4,5\n", "same number of columns"),
    ],
)
def test_unusable_uploads_show_an_error_instead_of_crashing(app, content, message):
    upload(app, "bad.csv", content)
    assert any(message in e.value for e in app.error)
    assert not app.metric


def test_same_column_for_both_labels_is_rejected(app):
    selectbox(app, "Predicted label column").set_value("actual")
    run(app)
    assert any("two different columns" in e.value for e in app.error)
    assert not app.metric


def test_same_feature_on_both_axes_is_rejected(app):
    selectbox(app, "Y axis feature").set_value("brightness")
    run(app)
    assert any("two different features" in e.value for e in app.error)


def test_file_without_two_numeric_features_still_shows_overview(app):
    upload(app, "labels.csv", b"actual,predicted,note\na,a,x\na,b,y\nb,b,z\n")
    assert metric(app, "Accuracy") == "66.7%"
    assert any("needs two numeric feature columns" in w.value for w in app.warning)


def test_confidence_needs_confirmation(app):
    assert not any(c.value.startswith("Average confidence") for c in app.caption)
    app.checkbox(key="confidence_ok::sample::confidence").set_value(True)
    selectbox(app, "brightness range").set_value(0)
    selectbox(app, "blur_px range").set_value(3)
    run(app)
    assert any(c.value.startswith("Average confidence in this cell") for c in app.caption)


def test_iris_measurement_labels_get_a_clear_warning(app):
    upload(app, "iris.csv", (ROOT / "iris.csv").read_bytes())
    selectbox(app, "Actual label column").set_value("petal_length")
    selectbox(app, "Predicted label column").set_value("sepal_width")
    run(app)
    assert any("measurements rather than class labels" in w.value for w in app.warning)
    assert any("Numeric class codes are valid" in w.value for w in app.warning)


def test_numeric_class_codes_are_not_flagged_as_measurements(app):
    upload(app, "numeric_classes.csv", b"actual,predicted,x,y\n0,0,1,1\n1,0,2,2\n1,1,3,3\n")
    assert metric(app, "Accuracy") == "66.7%"
    assert not any("Check the label columns" in w.value for w in app.warning)


def test_iris_heldout_upload_matches_independent_counts_and_cell_ids(app):
    upload(app, "iris_heldout_predictions.csv", IRIS_EVAL)
    assert not app.warning
    assert {m.label: m.value for m in app.metric} == {
        "Accuracy": "93.3%", "Errors": "2", "Correct": "28",
        "Evaluated rows": "30", "Excluded rows": "0",
    }
    assert selectbox(app, "Actual label column").value == "actual_species"
    assert selectbox(app, "Predicted label column").value == "predicted_species"
    assert selectbox(app, "X axis feature").value == "sepal_length"
    assert selectbox(app, "Y axis feature").value == "sepal_width"
    confusion = next(d.value for d in app.dataframe if d.value.index.name == "actual" and d.value.columns.name == "predicted")
    assert confusion.to_numpy().tolist() == [[10, 0, 0], [0, 10, 0], [0, 2, 8]]
    mapped = next(d.value for d in app.dataframe if "small sample" in d.value.columns)
    expected = {
        (0, 0): (2, 0), (0, 1): (1, 0), (0, 2): (3, 0), (0, 3): (1, 0),
        (1, 0): (1, 0), (1, 1): (1, 0), (1, 2): (2, 0), (1, 3): (3, 0),
        (2, 0): (1, 1), (2, 1): (7, 1), (2, 2): (2, 0), (2, 3): (0, 0),
        (3, 0): (0, 0), (3, 1): (6, 0), (3, 2): (0, 0), (3, 3): (0, 0),
    }
    x_labels = ["[4.40, 5.12)", "[5.12, 5.85)", "[5.85, 6.58)", "[6.58, 7.30]"]
    y_labels = ["[2.30, 2.72)", "[2.72, 3.15)", "[3.15, 3.58)", "[3.58, 4.00]"]
    for (x, y), (total, errors) in expected.items():
        row = mapped[(mapped["sepal_length"] == x_labels[x]) & (mapped["sepal_width"] == y_labels[y])].iloc[0]
        assert (row["total"], row["errors"]) == (total, errors)
        if total:
            assert row["error_rate"] == pytest.approx(100 * errors / total)
        else:
            assert row["error_rate"] != row["error_rate"]  # no rate for empty cells

    selectbox(app, "sepal_length range").set_value(2)
    selectbox(app, "sepal_width range").set_value(1)
    run(app)
    assert cell_table(app)["example_id"].tolist() == [
        "iris_064", "iris_105", "iris_117", "iris_128", "iris_133", "iris_139", "iris_148"
    ]
    app.toggle(key="errors_only").set_value(True)
    run(app)
    assert cell_table(app)["example_id"].tolist() == ["iris_139"]
    selectbox(app, "Y axis feature").set_value("petal_width")
    run(app)
    assert selectbox(app, "sepal_length range").value is None
    assert selectbox(app, "petal_width range").value is None
    selectbox(app, "sepal_length range").set_value(0)
    selectbox(app, "petal_width range").set_value(0)
    run(app)
    upload(app, "iris.csv", (ROOT / "iris.csv").read_bytes())
    assert selectbox(app, "Actual label column").value is None
    assert selectbox(app, "Predicted label column").value is None
    assert not app.metric


@pytest.mark.parametrize("content, warning", [
    (b"actual,predicted,x,y\nsetosa,1.4,1,1\nvirginica,6.8,2,2\n", "actual labels are text and the predicted labels are numeric"),
    (b"actual,predicted,x,y\na,c,1,1\nb,d,2,2\n", "class sets do not overlap"),
])
def test_label_diagnostics_warn_without_rejecting(app, content, warning):
    upload(app, "diagnostic.csv", content)
    assert any(warning in w.value for w in app.warning)
    assert metric(app, "Evaluated rows") == "2"
    assert metric(app, "Accuracy") == "0.0%"


def train_iris(at):
    at.radio(key="workflow").set_value("Train a model and evaluate it")
    run(at)
    at.file_uploader(key="train_upload").set_value(("iris.csv", (ROOT / "iris.csv").read_bytes(), "text/csv"))
    run(at)
    assert selectbox(at, "Target column").value == "species"
    assert "species" not in next(m for m in at.multiselect if m.label == "Training features").options
    at.button(key="train_button").click()
    return run(at)


def test_training_upload_export_and_reupload_match_every_inspected_cell(app):
    train_iris(app)
    assert any(h.value == "2 · Held-out test performance" for h in app.header)
    assert any("120 training rows · 30 held-out test rows" in s.value for s in app.success)
    result = app.session_state["training_result"]
    exported = result.csv_bytes()
    rows = list(csv.DictReader(io.StringIO(exported.decode())))
    n_errors = sum(r["actual_label"] != r["predicted_label"] for r in rows)
    assert len(rows) == 30 and n_errors == 2
    assert metric(app, "Accuracy") == f"{(len(rows) - n_errors) / len(rows):.1%}"
    assert metric(app, "Errors") == str(n_errors)
    x_edges = [4.4, 5.12, 5.85, 6.58, 7.3]
    y_edges = [2.3, 2.72, 3.15, 3.58, 4.0]
    mapped = next(d.value for d in app.dataframe if "small sample" in d.value.columns)
    for x in range(4):
        for y in range(4):
            def in_bin(value, edges, i):
                return edges[i] <= float(value) and (float(value) < edges[i + 1] or i == 3 and float(value) <= edges[i + 1])
            expected = [r for r in rows if in_bin(r["sepal_length"], x_edges, x) and in_bin(r["sepal_width"], y_edges, y)]
            expected_errors = sum(r["actual_label"] != r["predicted_label"] for r in expected)
            # Grid index is x * 4 + y, retained by the sorted display table.
            cell = mapped.loc[x * 4 + y]
            assert (cell["total"], cell["errors"]) == (len(expected), expected_errors)
            if expected:
                assert cell["error_rate"] == pytest.approx(100 * expected_errors / len(expected))
            else:
                assert cell["error_rate"] != cell["error_rate"]
            selectbox(app, "sepal_length range").set_value(x)
            selectbox(app, "sepal_width range").set_value(y)
            run(app)
            if expected:
                table = cell_table(app)
                assert table["source_row_id"].astype(int).tolist() == [int(r["source_row_id"]) for r in expected]
                assert (table["result"] == "✗ error").sum() == expected_errors
            else:
                assert any("No examples in this group" in m.value for m in app.info)
    app.radio(key="workflow").set_value("Analyze an existing prediction CSV")
    run(app)
    upload(app, "heldout_predictions.csv", exported)
    assert selectbox(app, "Actual label column").value == "actual_label"
    assert selectbox(app, "Predicted label column").value == "predicted_label"
    assert metric(app, "Accuracy") == "93.3%"
    assert selectbox(app, "sepal_length range").value is None
    selectbox(app, "sepal_length range").set_value(2)
    selectbox(app, "sepal_width range").set_value(1)
    run(app)
    assert cell_table(app)["source_row_id"].astype(int).tolist() == [64, 105, 117, 128, 133, 139, 148]


def test_training_settings_clear_old_results_and_cell_selection(app):
    train_iris(app)
    selectbox(app, "sepal_length range").set_value(2)
    selectbox(app, "sepal_width range").set_value(1)
    run(app)
    next(n for n in app.number_input if n.label == "Random seed").set_value(7)
    run(app)
    assert not app.metric
    assert not any(e.label == "Held-out predictions" for e in app.expander)
    app.button(key="train_button").click()
    run(app)
    assert selectbox(app, "sepal_length range").value is None
    assert any("random seed 7" in s.value for s in app.success)
    selectbox(app, "Target column").set_value("sepal_length")
    run(app)
    app.button(key="train_button").click()
    run(app)
    assert not app.metric
    assert any("at least two rows in every class" in e.value for e in app.error)
