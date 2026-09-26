"""Compare the old CSV-serialisation fingerprint with the current row-hash fingerprint.

Run: python scripts/benchmark_fingerprint.py [rows] [columns] [models]

The old app hashed `raw.to_csv()` on every Streamlit rerun and again inside each
trained model. The current app hashes structured row values once per upload
(cached) and passes the result to every model. Timings vary by machine, so this
is a script, not a test assertion.
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training as T  # noqa: E402


def old_fingerprint(raw: pd.DataFrame) -> str:
    return hashlib.sha256(raw.to_csv(index=True).encode()).hexdigest()


def synthetic(rows: int, columns: int, seed: int = 0) -> pd.DataFrame:
    """Text-valued like an uploaded CSV: numbers, a few categories, some blanks."""
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=rows).round(4).astype(str) for i in range(columns - 2)})
    frame["group"] = rng.choice([f"g{i}" for i in range(50)], size=rows)
    frame["label"] = rng.choice(["a", "b", "c"], size=rows)
    frame.loc[rng.rand(rows) < .01, "x0"] = ""
    frame.index = pd.RangeIndex(1, rows + 1, name="row")
    return frame


def timed(function, raw: pd.DataFrame, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function(raw)
        best = min(best, time.perf_counter() - start)
    return best


def compare(rows: int = 200_000, columns: int = 12, models: int = 3, repeats: int = 3) -> dict:
    raw = synthetic(rows, columns)
    old, new = timed(old_fingerprint, raw, repeats), timed(T.fingerprint, raw, repeats)
    return {
        "rows": rows, "columns": columns, "models": models,
        "old_seconds": old, "new_seconds": new,
        # Old: one hash per rerun in the UI plus one per trained model.
        "old_training_click_seconds": old * (1 + models),
        # New: hashed once per upload and cached; models reuse it.
        "new_training_click_seconds": new,
        "new_cached_rerun_seconds": 0.0,
    }


def main() -> None:
    args = [int(value) for value in sys.argv[1:4]]
    result = compare(*args)
    print(f"Frame: {result['rows']:,} rows × {result['columns']} columns (text values, like an uploaded CSV)")
    print(f"Old fingerprint (to_csv + sha256):          {result['old_seconds']:.3f} s per call")
    print(f"New fingerprint (row hashes + schema):      {result['new_seconds']:.3f} s per call")
    print(f"Speed-up per call:                          {result['old_seconds'] / result['new_seconds']:.1f}×")
    print(f"Training {result['models']} models, old approach:       {result['old_training_click_seconds']:.3f} s of hashing")
    print(f"Training {result['models']} models, new approach:       {result['new_training_click_seconds']:.3f} s (first time; cached reruns skip it)")


if __name__ == "__main__":
    main()
