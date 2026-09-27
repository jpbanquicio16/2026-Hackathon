# Handoff: finish portfolio improvements 1–7

The seven portfolio improvements have been implemented, with public deployment still
waiting for account access. Implementation commit `5884556` is published on `main`;
its Linux/macOS CI jobs and lint have passed. Read `git status --short` and `git diff`
before continuing. The user's pre-existing edit in `app.py` near the consistency
footer (`Join Successful/Join Failed`) was preserved and excluded from the commit.
Only the external deployment steps below remain.

## What is implemented

1. **Real-data case study:** UCI White Wine Quality data is in `data/`, with attribution
   and CC BY 4.0 terms in `data/README.md`. `scripts/make_wine_case_study.py` reproduces
   the deduplication, group-safe split, CV model comparison, one-time held-out scoring,
   independent cell audit and exported reports. Read `docs/CASE_STUDY.md`. Current
   selected run: random forest; 793 held-out rows; 656 correct, 137 errors; 82.72%
   accuracy; 0.6639 balanced accuracy; majority baseline 0.5000 balanced accuracy.
   The highlighted alcohol × volatile-acidity slice has 20 errors among 54 rows (37.0%).
2. **Deployment preparation:** README has a prominent live-demo/deployment section in
   progress; local app works on `http://localhost:8502`. Safari reached Streamlit
   Community Cloud's sign-in page, which includes acceptance of its Terms of Service.
   The user has not signed in yet. Do not accept terms or enter credentials for them;
   ask the user to finish the Streamlit sign-in, then deploy and add the confirmed URL
   to README. Do not invent a live link. Community Cloud setup is documented in
   `docs/DEPLOYING.md`.
3. **Test-set discipline:** `training.py` supports deferred evaluation and
   `ui_training.py` ranks models on training-fold CV, locks one model, then creates
   that model's held-out predictions. A repeat reveal for a dataset/target is marked
   exploratory in the session. Regression and classification use CV on training rows.
4. **Cell uncertainty:** `uncertainty.py` computes Wilson intervals, one-sided
   cell-versus-rest Fisher tests and Holm correction across the cells in one map.
   `analysis.py`, `app.py` and `exports.py` expose counts, intervals and clear
   limitations (independence assumption; axes/filter/threshold exploration is not
   corrected; no causal or confirmatory interpretation).
5. **Mixed features:** `preprocessing.py` and `training.py` use fold-local median
   imputation/scaling for numeric features and most-frequent imputation/one-hot
   encoding for repeated categories. Unknown categories are ignored, numeric category
   spelling is preserved, high-cardinality/free-text/date fields are excluded and
   documented. Map axes remain numeric.
6. **Reproducible runs:** `experiments.py` creates bounded, hashed ZIP archives without
   pickled estimators and verifies/replays a saved run. `scripts/reproduce_run.py`
   provides CLI replay. The archive includes the full uploaded dataset; reports
   restore the original view. Versions/code/splits/prediction bytes are checked.
   Linux/macOS Python 3.12 dependencies are pinned in the two lock files and `.python-version`.
7. **Benchmarks and presentation:** `scripts/benchmark_atlas.py` measures parse, fit,
   map plus chart-spec generation, initial app render and warm server reruns. Raw and
   summarized results are in `docs/benchmarks.json` and `docs/benchmarks.md`. At
   50,000 synthetic rows the measured medians were 0.082 s load, 0.617 s fit,
   0.073 s map/chart and 0.396 s server rerun. These do not include browser painting,
   cloud cold starts or concurrent users. README and methodology/user docs were updated.

## Verification completed in the continuation

Tasks 1 and 3–7 are implemented and verified locally. The full suite now has **244
passing tests**. Ruff, dependency checks, all file hooks and whitespace checks pass.
Dependency lock regeneration is unchanged. The regenerated wine case still has 793
test rows, 656 correct and 137 errors, and CLI replay matches predictions exactly.
Clean hosted Linux and macOS jobs also passed installation, tests and benchmark smoke,
and hosted lint passed: [CI run](https://github.com/jpbanquicio16/2026-Hackathon/actions/runs/36304146796).

Native Safari checks completed the wine existing-predictions flow and a mixed-feature
Iris upload → CV fit → lock → held-out metrics → cell inspection → CSV/archive download
→ archive restore flow. Downloaded predictions and selected row IDs were independently
checked. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for expected/observed values.

Two restore bugs were fixed: malformed archive fields now produce clear errors, and
restoring a run records test exposure for later experiments. Upload instructions and
generated-file formatting were also corrected. The user's pre-existing consistency
footer edit remains intact.

## Remaining external steps

1. Finish task 2 after the user signs in to Streamlit Community Cloud. Safari still
   shows a sign-in page with Terms of Service acceptance. A request for the user to
   complete this step has been sent; no successful authentication is confirmed.
2. Deploy the updated code using `main`, `app.py` and Python 3.12; smoke-test the real
   public app, then add its confirmed URL to README. Do not invent a URL or accept
   legal terms on the user's behalf.

Do not redo completed local checks unless code changes or a failure justifies it.
The verification record distinguishes local results from pending hosted checks.

## Useful paths and facts

- Training/evaluation flow: `training.py`, `ui_training.py`, `preprocessing.py`, `app.py`.
- Cell statistics and uncertainty: `analysis.py`, `uncertainty.py`, `exports.py`.
- Replay: `experiments.py`, `scripts/reproduce_run.py`.
- Case-study builder/data: `scripts/make_wine_case_study.py`, `data/`,
  `examples/wine_case_study/`.
- Performance method/results: `scripts/benchmark_atlas.py`, `docs/benchmarks.md`.
- CI and dependency inputs: `.github/workflows/ci.yml`, `requirements.in`,
  `requirements-lock.txt`, `requirements-dev-lock.txt`, `scripts/lock_dependencies.py`.
- The local Streamlit server was started by the prior agent on port 8502 (session
  27160). Keep it available for browser QA; stop it with Ctrl-C after verification.
- At task start, `app.py` already had a user edit changing consistency-footer wording
  to “Join Successful/Join Failed”. Preserve it; review whether “Join” was intended,
  but do not silently overwrite it.
