"""Structure of the heatmap's Vega-Lite spec.

A browser is needed to see the chart, but these properties can be checked
from the spec: the fixed colour scale, how empty and small cells are drawn,
and a layering mistake that once stopped the chart rendering at all.
"""

import pytest

import analysis as A
import app
from test_analysis import FIXTURE


@pytest.fixture
def fmap() -> A.FailureMap:
    raw = A.load_csv(FIXTURE.encode()).frame
    rows = A.evaluate(raw, "actual", "predicted").rows
    return A.build_failure_map(raw, rows, "x", "y")  # 6 filled cells, 10 empty


def spec(fmap, selected=(None, None), dark=False, small_n=3) -> dict:
    return app.heatmap(fmap, small_n, selected, dark).to_dict()


def cells_data(chart: dict) -> list[dict]:
    (rows,) = chart["datasets"].values()
    return rows


def test_error_rate_is_the_only_colour_scale_and_is_fixed(fmap):
    chart = spec(fmap)
    scales = [
        layer["encoding"]["color"]["scale"]
        for layer in chart["layer"]
        if "scale" in layer.get("encoding", {}).get("color", {})
    ]
    # A second (null) colour scale in another layer makes Vega-Lite throw
    # "Cannot read properties of null (reading 'type')" in the browser.
    assert len(scales) == 1
    (scale,) = scales
    assert scale["domain"][0] == 0 and scale["domain"][-1] == 1
    assert scale["range"] == app.BLUE_RAMP


def test_dark_theme_flips_the_ramp(fmap):
    (scale,) = [
        layer["encoding"]["color"]["scale"]
        for layer in spec(fmap, dark=True)["layer"]
        if "scale" in layer.get("encoding", {}).get("color", {})
    ]
    assert scale["range"] == app.BLUE_RAMP[::-1]


def test_empty_cells_are_labelled_not_coloured(fmap):
    chart = spec(fmap)
    empty = [row for row in cells_data(chart) if row["total"] == 0]
    assert len(empty) == 10
    assert all(row["error_rate"] is None and row["rate_label"] == "" for row in empty)
    empty_layers = [layer for layer in chart["layer"] if layer.get("transform") == [{"filter": "datum.total == 0"}]]
    assert {layer["mark"]["type"] for layer in empty_layers} == {"rect", "text"}
    text = next(layer for layer in empty_layers if layer["mark"]["type"] == "text")
    assert text["encoding"]["text"] == {"value": "no examples"}


def test_cell_labels_show_rate_counts_and_small_samples(fmap):
    labels = {(row["x_bin"], row["y_bin"]): (row["rate_label"], row["count_label"]) for row in cells_data(spec(fmap))}
    assert labels[(3, 3)] == ("67%", "2 of 3")  # 3 examples: not below the threshold of 3
    assert labels[(0, 0)] == ("50%", "1 of 2 · small")
    assert labels[(1, 0)] == ("0%", "0 of 1 · small")  # a real 0%, unlike the empty cells


def test_selected_cell_gets_an_outline_and_clicks_are_captured(fmap):
    assert not any(layer["mark"].get("filled") is False for layer in spec(fmap)["layer"])
    chart = spec(fmap, selected=(3, 3))
    outline = [layer for layer in chart["layer"] if layer["mark"].get("filled") is False]
    assert len(outline) == 1
    assert outline[0]["transform"] == [{"filter": "datum.x_bin == 3 && datum.y_bin == 3"}]
    (param,) = chart["params"]
    assert param["name"] == "cell" and param["select"]["fields"] == ["x_bin", "y_bin"]
    assert chart["width"] == "container"
