# Performance and reliability benchmark

Measured on macOS-14.6-arm64-arm-64bit, Python 3.12.13; 3 repetitions per stage (except first upload).

| Rows | CSV load | Logistic fit + test | Map + chart specification | First upload/server render | Warm server rerun |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.002 s | 0.026 s | 0.043 s | 0.120 s | 0.101 s |
| 10,000 | 0.016 s | 0.144 s | 0.078 s | 0.293 s | 0.165 s |
| 50,000 | 0.082 s | 0.617 s | 0.073 s | 0.710 s | 0.396 s |

Numbers are measured local medians, not service-level guarantees. Full samples, package versions and hardware identifiers are in [benchmarks.json](benchmarks.json).

The synthetic binary task has six numeric features. Model timing includes a fresh 80/20 stratified split, preprocessing, a logistic regression fit and predictions, with seed 42 and CV disabled. Map timing includes all rows, an 8×8 grid, uncertainty, consistency checks and Altair JSON generation. The first upload measures a full Streamlit AppTest server render; reruns use warm data/map caches. These timings do not measure network transfer, browser JavaScript painting, concurrent users or cloud cold starts.

Targets on this development machine at up to 50,000 rows: CSV load ≤3 s, logistic training ≤20 s, map/chart generation ≤2 s and warm server rerun ≤5 s. The raw results record whether each target was met. CI runs a small correctness smoke benchmark; machine-dependent latency targets are not hard CI assertions.

```bash
python scripts/benchmark_atlas.py
python scripts/benchmark_atlas.py --sizes 1000 --repeats 1 --output /tmp/atlas-benchmark.json
```

Training runs only after an explicit click; changing a map does not refit a model. Uploads remain capped at 25 MB. Larger workloads, nearest neighbors, SVMs and multi-model searches can exceed these timings; no broad scalability claim is made.
