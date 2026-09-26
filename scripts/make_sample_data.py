"""Regenerate the synthetic demo datasets used by Model Failure Atlas.

    python scripts/make_sample_data.py

Both files are SYNTHETIC. No real model or real images were involved: rows are
drawn from simple distributions and each row's chance of being misclassified
is set by a formula chosen to plant a known failure pattern. The seeds are
fixed, so rerunning the script reproduces the committed CSVs exactly.

sample_predictions.csv (main demo)
    A traffic-sign classifier evaluated on 1,200 images. Errors concentrate
    in dark, blurry images and, to a lesser degree, small (distant) signs.
    Blur partly depends on vehicle speed, so speed also looks risky on the
    map even though the formula never uses it directly. That makes a useful
    reminder that the map shows where errors happen, not why.
    Planted gaps: 8 rows have no actual label (awaiting review) and 12 rows
    have no blur measurement.

examples/churn_predictions.csv (second dataset for testing column mapping)
    A binary churn classifier with different column names, a constant
    column, labels with stray whitespace and one capitalisation variant,
    missing labels, and a few non-numeric values in a numeric column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SIGN_CLASSES = ["stop", "yield", "speed_30", "speed_50", "no_entry"]
SIGN_SHARES = [0.22, 0.18, 0.22, 0.22, 0.16]

# When the classifier is wrong, which class does it pick instead?
# speed_80 exists in the model's label set but never occurs in this
# evaluation batch, so it only ever appears as a prediction.
SIGN_CONFUSIONS = {
    "stop": (["no_entry", "yield", "speed_30"], [0.6, 0.3, 0.1]),
    "yield": (["stop", "no_entry", "speed_50"], [0.5, 0.3, 0.2]),
    "speed_30": (["speed_50", "speed_80", "stop"], [0.75, 0.15, 0.10]),
    "speed_50": (["speed_30", "speed_80", "yield"], [0.70, 0.20, 0.10]),
    "no_entry": (["stop", "yield", "speed_30"], [0.65, 0.25, 0.10]),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_traffic_signs(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    weather = rng.choice(["clear", "rain", "fog", "snow"], size=n, p=[0.6, 0.22, 0.12, 0.06])
    darker = np.isin(weather, ["rain", "fog"])

    brightness = rng.uniform(0.04, 0.96, size=n) - 0.08 * darker
    brightness = np.clip(brightness, 0.02, 0.98)

    speed_kmh = rng.integers(20, 121, size=n)
    weather_blur = np.select([weather == "rain", weather == "fog"], [0.8, 1.2], default=0.0)
    blur_px = 0.05 * (speed_kmh - 20) + weather_blur + rng.exponential(1.4, size=n)
    blur_px = np.clip(blur_px, 0.0, 8.0)

    sign_size_px = rng.integers(16, 129, size=n)

    # Error model: darkness and blur each hurt, and hurt most together.
    darkness = np.clip((0.35 - brightness) / 0.30, 0.0, 1.0)
    small_sign = np.clip((40 - sign_size_px) / 24, 0.0, 1.0)
    logit = -4.5 + 2.3 * darkness + 0.33 * blur_px + 2.8 * darkness * (blur_px / 8) + 0.8 * small_sign
    is_error = rng.random(n) < _sigmoid(logit)

    actual = rng.choice(SIGN_CLASSES, size=n, p=SIGN_SHARES)
    predicted = actual.copy()
    for i in np.flatnonzero(is_error):
        options, weights = SIGN_CONFUSIONS[actual[i]]
        predicted[i] = rng.choice(options, p=weights)

    # Confidence = probability the model gave to the class it predicted.
    correct_conf = 0.55 + 0.44 * rng.beta(5.0, 1.5, size=n) - 0.12 * darkness - 0.015 * blur_px
    wrong_conf = 0.25 + 0.70 * rng.beta(2.2, 2.2, size=n)
    confidence = np.clip(np.where(is_error, wrong_conf, correct_conf), 0.2, 0.999)

    df = pd.DataFrame(
        {
            "image_id": [f"img_{i:05d}" for i in range(1, n + 1)],
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence.round(3),
            "brightness": brightness.round(2),
            "blur_px": blur_px.round(1),
            "speed_kmh": speed_kmh,
            "sign_size_px": sign_size_px,
            "weather": weather,
        }
    )

    # Planted gaps, so the demo shows how exclusions are reported.
    gap_rng = np.random.default_rng(seed + 1)
    unlabelled = gap_rng.choice(n, size=8, replace=False)
    df["actual"] = df["actual"].astype(object)
    df.loc[unlabelled, "actual"] = ""
    no_blur = gap_rng.choice(np.setdiff1d(np.arange(n), unlabelled), size=12, replace=False)
    df["blur_px"] = df["blur_px"].astype(object)
    df.loc[no_blur, "blur_px"] = ""
    return df


def make_churn(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 73, size=n)
    monthly_charges = rng.uniform(18.0, 120.0, size=n).round(2)
    support_calls = rng.poisson(1.3, size=n).clip(0, 7)
    contract = rng.choice(["monthly", "one_year", "two_year"], size=n, p=[0.55, 0.25, 0.20])

    churn_logit = -1.2 - 0.04 * tenure_months + 0.02 * (monthly_charges - 60) + 0.35 * support_calls
    churned = rng.random(n) < _sigmoid(churn_logit)
    churn_probability = np.clip(_sigmoid(churn_logit + rng.normal(0, 0.6, size=n)), 0.01, 0.99)

    # The model struggles with new customers on expensive plans.
    new_and_pricey = (tenure_months < 18) & (monthly_charges > 85)
    err_logit = -2.6 + 2.3 * new_and_pricey + 0.25 * support_calls
    is_error = rng.random(n) < _sigmoid(err_logit)

    label = np.where(churned, "churned", "stayed")
    prediction = np.where(is_error, np.where(churned, "stayed", "churned"), label)

    df = pd.DataFrame(
        {
            "customer_id": [f"{i:06d}" for i in rng.choice(999_999, size=n, replace=False)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "contract": contract,
            "plan_version": 2,
            "churn_probability": churn_probability.round(3),
            "label": label,
            "prediction": prediction,
        }
    ).astype({"label": object, "prediction": object, "monthly_charges": object})

    # Realistic mess: whitespace, one capitalisation variant, missing labels,
    # and a few charges that failed to parse upstream.
    df.loc[[3, 17, 42], "label"] = df.loc[[3, 17, 42], "label"] + " "
    df.loc[[5, 88], "prediction"] = " " + df.loc[[5, 88], "prediction"]
    df.loc[60, "label"] = df.loc[60, "label"].capitalize()
    df.loc[[10, 120, 250, 333, 390], "label"] = ""
    df.loc[[77, 301], "prediction"] = "NA"
    df.loc[[23, 145, 222, 358], "monthly_charges"] = "unknown"
    return df


def main() -> None:
    signs = make_traffic_signs()
    signs.to_csv(ROOT / "sample_predictions.csv", index=False)

    churn = make_churn()
    (ROOT / "examples").mkdir(exist_ok=True)
    churn.to_csv(ROOT / "examples" / "churn_predictions.csv", index=False)

    print(f"Wrote {len(signs):,} rows to sample_predictions.csv")
    print(f"Wrote {len(churn):,} rows to examples/churn_predictions.csv")


if __name__ == "__main__":
    main()
