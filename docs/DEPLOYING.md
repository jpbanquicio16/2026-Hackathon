# Deploying

The implementation is published on `main`. A public app URL has not yet been verified;
the remaining prerequisite is the owner's Community Cloud sign-in and acceptance of
its Terms of Service. The local verification results are in [VERIFICATION.md](VERIFICATION.md).

The app is a standard Streamlit app with no secrets, databases or system packages,
so it runs on [Streamlit Community Cloud](https://share.streamlit.io) as-is:

1. Push `main` to GitHub (`git push origin main`).
2. Sign in at <https://share.streamlit.io> with your GitHub account and allow it to
   access the repository.
3. Choose **Create app**, then deploy from GitHub with:
   - Repository: `jpbanquicio16/2026-Hackathon`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: any free subdomain, for example `model-failure-atlas`
4. Under **Advanced settings**, choose Python **3.12** (the tested version). No
   secrets are needed.
5. Select **Deploy**. The first build installs `requirements.txt` (not
   `requirements-dev.txt`) and takes a few minutes.
6. Open the public URL in a private window and complete the smoke checks below.
7. After the checks pass, add the confirmed URL beside the case-study link at the top
   of README. Record the deployed commit and URL in `docs/VERIFICATION.md`.

After that, every push to `main` redeploys the app. The sample data ships in the
repository and `.streamlit/config.toml` is read automatically. Community Cloud puts
apps that get no traffic for a while to sleep, so open the link a few minutes before
presenting.

## Public-app smoke checks

- **Existing predictions:** select **White wine case study**. Expect 793 evaluated
  rows, 656 correct, 137 errors and 82.7% accuracy. Set X to `alcohol`, Y to
  `volatile acidity`, four ranges and **Quantiles**. Select alcohol [11.4,14.2]
  and volatile acidity [0.27,0.33): expect 20 errors among 54 rows. Download the rows
  and compare their IDs with `examples/wine_case_study/independent_audit.json`.
- **Training:** upload `iris.csv`, select `species`, retain the four measurement
  features, random split, 20% test, seed 42 and 3-nearest neighbors. Click **Fit and
  compare on training data**: no held-out scores should appear yet. Lock the model;
  expect 120 training rows, 30 test rows, 28 correct and 2 errors. Inspect a cell and
  download the prediction CSV and reproducible archive.
- **Restore:** choose **Restore a saved run**, upload the downloaded archive and
  select **Reproduce saved run**. Expect verified split IDs and exact predictions.
  Upload the prediction CSV in existing-predictions mode and confirm the same counts.
- **State:** change an axis or upload a different dataset after inspecting a cell.
  Old cell selections must clear. Check the page for exceptions and deployment logs
  for installation or runtime failures.

The checked-in wine archive records the environment used to create it. If Cloud uses
different ML versions or calculation code, strict replay should explain the mismatch;
an exploratory replay must be explicitly selected and any prediction differences
reported. A newly downloaded run from the deployed app should replay exactly there.
