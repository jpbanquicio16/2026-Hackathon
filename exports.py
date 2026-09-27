"""Portable CSV/JSON/HTML reports; no model unpickling or external services."""

import html
import io
import json
import math
import zipfile

import numpy as np
import pandas as pd

import analysis
from evaluation import classification_metrics, normalise_confusion, regression_metrics


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def json_bytes(value) -> bytes:
    return (json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def map_table(fmap: analysis.FailureMap) -> pd.DataFrame:
    cells = fmap.cells.copy()
    cells.insert(2, "x_range", [fmap.x_bins.labels[i] for i in cells["x_bin"]])
    cells.insert(3, "y_range", [fmap.y_bins.labels[i] for i in cells["y_bin"]])
    return cells


def evaluation_files(raw, evaluation, metadata, fmap=None, selected=None) -> dict[str, bytes]:
    """Downloads describe the active predictions, including threshold and filter choices."""
    meta = dict(metadata)
    meta.update({
        "task": evaluation.task, "uploaded_rows": evaluation.n_uploaded,
        "evaluated_rows": evaluation.n_evaluated, "excluded_rows": evaluation.n_excluded,
        "error_tolerance": evaluation.tolerance if evaluation.task == "regression" else None,
    })
    files = {"predictions.csv": raw.loc[evaluation.rows.index].to_csv(index=False).encode()}
    excluded = raw.loc[~raw.index.isin(evaluation.rows.index)]
    if len(excluded):
        files["rows_not_evaluated.csv"] = excluded.to_csv(index=True).encode()
    per_class = pd.DataFrame()
    if evaluation.task == "classification":
        metrics, per_class = classification_metrics(evaluation.rows)
        metrics.update(meta.get("probability_metrics", {}))
        confusion = analysis.compute_overview(evaluation.rows).confusion
        files["confusion_counts.csv"] = confusion.to_csv().encode()
        files["confusion_actual_percent.csv"] = normalise_confusion(confusion, "Actual class (%)").to_csv().encode()
        files["confusion_predicted_percent.csv"] = normalise_confusion(confusion, "Predicted class (%)").to_csv().encode()
        files["per_class_metrics.csv"] = per_class.to_csv(index=False).encode()
    else:
        metrics, confusion = regression_metrics(evaluation.rows), pd.DataFrame()
        files["residuals.csv"] = evaluation.rows.to_csv(index=True).encode()
    meta["active_metrics"] = metrics
    tables = [("Performance", pd.DataFrame([metrics]))]
    if not confusion.empty:
        tables.append(("Confusion matrix — actual rows, predicted columns", confusion))
        tables.append(("Per-class metrics", per_class))
    if fmap is not None:
        table = map_table(fmap)
        files["map_cells.csv"] = table.to_csv(index=False).encode()
        files["map_row_assignments.csv"] = fmap.rows.to_csv(index=True).encode()
        meta["map"] = {
            "x": fmap.x_col, "y": fmap.y_col, "x_edges": fmap.x_bins.edges,
            "y_edges": fmap.y_bins.edges, "mapped_rows": fmap.n_mapped,
            "omitted_rows": fmap.n_omitted, "omission_reasons": fmap.omitted,
            "boundary_rule": "lower-inclusive, upper-exclusive; final bin includes maximum (integer bins use inclusive integer runs)",
            "uncertainty": "95% Wilson intervals for cell error rates; one-sided Fisher cell-vs-rest tests with Holm correction across this map. Independent examples assumed. Repeated map exploration is not corrected; not a causal or confirmatory claim.",
        }
        tables.append(("Map cells (error_rate is a fraction; empty cells have no rate)", table))
        if selected and selected[0] is not None and selected[1] is not None:
            ids = fmap.cell_index(*selected)
            detail = raw.loc[ids]
            files["selected_cell_rows.csv"] = detail.to_csv(index=True).encode()
            meta["selected_cell"] = {"x_bin": selected[0], "y_bin": selected[1], "row_ids": ids.tolist()}
            tables.append(("Selected cell rows", detail))
    meta = json_safe(meta)
    files["audit_metadata.json"] = json_bytes(meta)
    scope = html.escape(meta.get("performance_scope", "Uploaded prediction performance"))
    sections = "".join(f"<section><h2>{html.escape(title)}</h2>{table.to_html(escape=True, na_rep='—', float_format=lambda x: f'{x:.6g}')}</section>" for title, table in tables)
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Model Failure Atlas report</title>"
        "<style>body{font:15px system-ui;margin:32px;color:#182332}h1{margin-bottom:8px}"
        "section{margin:28px 0;overflow:auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd3db;text-align:right}"
        "th{background:#edf3fa}pre{white-space:pre-wrap;word-break:break-word;background:#f3f6fa;padding:16px}</style>"
        f"<h1>Model Failure Atlas</h1><p>{scope}</p>"
        "<p>Rates describe the evaluated examples. Small cells are uncertain; feature associations do not establish causes. "
        "Cross-validation and training scores, if present in metadata, are separate from the performance above.</p>"
        + sections + "<h2>Audit metadata</h2><pre>" + html.escape(json.dumps(meta, indent=2, ensure_ascii=False)) + "</pre></html>"
    )
    files["evaluation_report.html"] = (document + "\n").encode()
    return files


def bundle(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()
