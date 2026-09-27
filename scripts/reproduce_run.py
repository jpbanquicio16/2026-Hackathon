"""Recreate a saved run: python scripts/reproduce_run.py run.atlas.zip --output replay."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments
import exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-environment-change", action="store_true")
    args = parser.parse_args()
    result = experiments.reproduce(args.archive.read_bytes(), allow_environment_change=args.allow_environment_change)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.csv").write_bytes(result.csv_bytes())
    (args.output / "metadata.json").write_bytes(result.metadata_bytes())
    (args.output / "reproduction.json").write_bytes(exports.json_bytes(result.metadata["reproduction"]))
    _, _, files = experiments.read_archive(args.archive.read_bytes())
    for name, content in files.items():
        if name.startswith("report/") and "/" not in name.removeprefix("report/") and "\\" not in name:
            destination = args.output / "report" / Path(name).name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(content)
    print(result.metadata["reproduction"])


if __name__ == "__main__":
    main()
