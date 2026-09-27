"""Rebuild examples/iris_heldout_predictions.csv from the local iris.csv."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SEED = 42


def main() -> None:
    source = pd.read_csv(ROOT / "iris.csv")
    assert len(source) == 150
    assert list(source.columns) == FEATURES + ["species"]
    assert source["species"].value_counts().to_dict() == {
        "setosa": 50, "versicolor": 50, "virginica": 50
    }

    train_idx, test_idx = train_test_split(
        source.index.to_numpy(), test_size=0.20, random_state=SEED,
        stratify=source["species"],
    )
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    model.fit(source.loc[train_idx, FEATURES], source.loc[train_idx, "species"])

    # Keep the source-file row number in the ID and order rows by that number.
    test_idx = sorted(test_idx)
    exported = source.loc[test_idx, FEATURES + ["species"]].copy()
    exported.insert(0, "example_id", [f"iris_{i + 1:03d}" for i in test_idx])
    exported = exported.rename(columns={"species": "actual_species"})
    exported["predicted_species"] = model.predict(source.loc[test_idx, FEATURES])
    destination = ROOT / "examples" / "iris_heldout_predictions.csv"
    exported.to_csv(destination, index=False)
    print(f"Wrote {len(exported)} held-out rows to {destination}")


if __name__ == "__main__":
    main()
