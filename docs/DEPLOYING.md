# Deploying

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
6. Open the app URL in a private window and run through the
   [demo walkthrough](../README.md#demo-walkthrough-about-two-minutes) once.

After that, every push to `main` redeploys the app. The sample data ships in the
repository and `.streamlit/config.toml` is read automatically. Community Cloud puts
apps that get no traffic for a while to sleep, so open the link a few minutes before
presenting.
