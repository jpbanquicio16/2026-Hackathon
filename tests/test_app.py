"""User-journey checks for app.py, run headlessly with Streamlit's AppTest.

These cover what the analysis tests can't: that the page wires the pieces
together, and that selections never go stale when inputs change.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CHURN = (ROOT / "examples" / "churn_predictions.csv").read_bytes()


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
