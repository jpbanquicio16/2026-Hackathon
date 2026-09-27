"""Measure CSV load, model fit, map/chart generation and Streamlit server reruns."""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis
import app
import training

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {"load_csv": 3.0, "fit_logistic": 20.0, "map_and_chart": 2.0, "server_rerun": 5.0}


def timed(call, repeats):
    samples, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, {"median_seconds": float(np.median(samples)), "max_seconds": max(samples), "samples_seconds": samples}


def data(n):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 6))
    frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
    frame["actual_label"] = np.where(X[:, 0] + X[:, 1] + rng.normal(size=n) > 0, "positive", "negative")
    frame["predicted_label"] = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
    return frame.to_csv(index=False).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "benchmarks.json")
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 100 for n in args.sizes):
        parser.error("Use at least one repeat and 100 rows per dataset.")
    results = []
    for n in args.sizes:
        content = data(n)
        loaded, load_time = timed(lambda content=content: analysis.load_csv(content), args.repeats)
        raw = loaded.frame
        features = tuple(c for c in raw if c.startswith("feature_"))
        fitted, fit_time = timed(lambda raw=raw, features=features: training.train_and_evaluate(raw, "actual_label", features,
                                 classifier="Logistic regression", cv_folds=0), args.repeats)
        assert len(fitted.frame) == int(.2*n)
        evaluated = analysis.evaluate(raw, "actual_label", "predicted_label")

        def chart(raw=raw, evaluated=evaluated, n=n):
            fmap = analysis.build_failure_map(raw, evaluated.rows, "feature_0", "feature_1", n_bins=8)
            assert int(fmap.cells.total.sum()) == n
            assert all(ok for _, ok in analysis.consistency_checks(analysis.compute_overview(evaluated.rows), fmap))
            return app.heatmap(fmap, 10, (None, None), False).to_json()

        _, map_time = timed(chart, args.repeats)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        at.radio(key="source").set_value("Upload a CSV").run()
        at.file_uploader(key="upload").set_value((f"benchmark_{n}.csv", content, "text/csv"))
        _, initial = timed(at.run, 1)  # upload + full first server render, caches cold for this file
        assert not at.exception
        _, rerun = timed(at.run, args.repeats)
        assert not at.exception
        measurements = {"load_csv": load_time, "fit_logistic": fit_time, "map_and_chart": map_time,
                        "initial_upload_server_render": initial, "server_rerun": rerun}
        results.append({"rows": n, "csv_bytes": len(content), "measurements": measurements,
                        "within_targets": {stage: measurements[stage]["median_seconds"] <= budget for stage, budget in BUDGETS.items()}})
        print({"rows": n, **{s: round(v["median_seconds"], 3) for s, v in measurements.items()}}, flush=True)
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "machine": platform.machine(), "python": platform.python_version(),
              "versions": {p: version(p) for p in ("streamlit", "pandas", "numpy", "scikit-learn", "altair", "scipy")},
              "repeats": args.repeats, "targets_seconds": BUDGETS, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Performance and reliability benchmark", "", f"Measured on {report['platform']}, Python {report['python']}; {args.repeats} repetitions per stage (except first upload).", "",
             "| Rows | CSV load | Logistic fit + test | Map + chart specification | First upload/server render | Warm server rerun |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        m = result["measurements"]
        stages = ("load_csv", "fit_logistic", "map_and_chart", "initial_upload_server_render", "server_rerun")
        lines.append(f"| {result['rows']:,} | " + " | ".join(f"{m[s]['median_seconds']:.3f} s" for s in stages) + " |")
    lines.extend(["", "Numbers are measured local medians, not service-level guarantees. Full samples, package versions and hardware identifiers are in [benchmarks.json](benchmarks.json).", "",
                  "The synthetic binary task has six numeric features. Model timing includes a fresh 80/20 stratified split, preprocessing, a logistic regression fit and predictions, with seed 42 and CV disabled. Map timing includes all rows, an 8×8 grid, uncertainty, consistency checks and Altair JSON generation. The first upload measures a full Streamlit AppTest server render; reruns use warm data/map caches. These timings do not measure network transfer, browser JavaScript painting, concurrent users or cloud cold starts.", "",
                  "Targets on this development machine at up to 50,000 rows: CSV load ≤3 s, logistic training ≤20 s, map/chart generation ≤2 s and warm server rerun ≤5 s. The raw results record whether each target was met. CI runs a small correctness smoke benchmark; machine-dependent latency targets are not hard CI assertions.", "",
                  "```bash", "python scripts/benchmark_atlas.py", "python scripts/benchmark_atlas.py --sizes 1000 --repeats 1 --output /tmp/atlas-benchmark.json", "```", "",
                  "Training runs only after an explicit click; changing a map does not refit a model. Uploads remain capped at 25 MB. Larger workloads, nearest neighbors, SVMs and multi-model searches can exceed these timings; no broad scalability claim is made."])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
