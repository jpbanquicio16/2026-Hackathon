"""Portable experiment archives: JSON and CSV only, never executable model objects."""

import hashlib
import io
import json
import platform
import zipfile
from importlib.metadata import version
from pathlib import Path

import analysis
import exports
import training

ROOT = Path(__file__).resolve().parent
CODE_FILES = ("analysis.py", "training.py", "preprocessing.py", "diagnostics.py", "evaluation.py", "uncertainty.py")
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn")
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
RECIPE_TYPES = {"target": str, "features": list, "classifier": str, "task": str,
                "test_size": (int, float), "seed": int, "split_method": str,
                "split_column": (str, type(None)), "cv_folds": int, "cv_repeats": int,
                "tune": bool, "imbalance": str, "categorical_features": list}


def code_digest():
    return hashlib.sha256(b"".join(name.encode() + (ROOT / name).read_bytes() for name in CODE_FILES)).hexdigest()


def recipe(result):
    m = result.metadata
    return {"target": m["target"], "features": list(result.features), "classifier": result.classifier,
            "task": result.task, "test_size": m["requested_test_proportion"], "seed": result.seed,
            "split_method": m["split_method"], "split_column": m["split_column"],
            "cv_folds": m["cv_requested_folds"], "cv_repeats": m["cv_repeats"], "tune": m["tuned"],
            "imbalance": m["imbalance_strategy"], "categorical_features": m.get("categorical_features", [])}


def create_archive(result, report_files=None):
    if not result.metadata.get("test_revealed", True):
        raise analysis.DataError("Select a model and reveal its test result before saving an evaluated run.")
    files = {"dataset.csv": result.source.to_csv(index=False).encode(), "predictions.csv": result.csv_bytes(),
             "training_metadata.json": result.metadata_bytes()}
    if report_files is None:
        from evaluation import regression_evaluation
        evaluation = analysis.evaluate(result.frame, result.actual, result.predicted) if result.task == "classification" else regression_evaluation(result.frame, result.actual, result.predicted)
        report_files = exports.evaluation_files(result.frame, evaluation, result.metadata)
    files.update({"report/" + name: value for name, value in report_files.items()})
    manifest = {"format": "model-failure-atlas", "version": 1, "recipe": recipe(result),
                "code_sha256": code_digest(), "python": platform.python_version(),
                "packages": {p: version(p) for p in PACKAGES}, "source_index": result.source.index.tolist(),
                "source_index_name": result.source.index.name,
                "files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}
    files["manifest.json"] = exports.json_bytes(manifest)
    return exports.bundle(files)


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("format") != "model-failure-atlas" or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Unsupported run archive format.")
    for key in ("files", "packages", "recipe"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"Archive {key} must be an object.")
    if set(manifest["packages"]) != set(PACKAGES) or not all(isinstance(v, str) and v for v in manifest["packages"].values()):
        raise ValueError("The archive must record every required ML package version.")
    if not isinstance(manifest.get("python"), str) or not isinstance(manifest.get("code_sha256"), str):
        raise ValueError("Missing Python version or calculation-code fingerprint.")
    config = manifest["recipe"]
    if set(config) != set(RECIPE_TYPES):
        raise ValueError("Run recipe has missing or unexpected settings.")
    for key, expected in RECIPE_TYPES.items():
        types = expected if isinstance(expected, tuple) else (expected,)
        if type(config[key]) not in types:
            raise ValueError(f"Invalid type for recipe setting {key}.")
    if any(not isinstance(c, str) for key in ("features", "categorical_features") for c in config[key]):
        raise ValueError("Feature names must be strings.")
    if not isinstance(manifest.get("source_index"), list) or any(type(i) is not int for i in manifest["source_index"]):
        raise ValueError("Source row IDs must be a list of integers.")
    if manifest.get("source_index_name") is not None and not isinstance(manifest["source_index_name"], str):
        raise ValueError("Invalid source row index name.")


def _read_metadata(files):
    previous = json.loads(files["training_metadata.json"])
    if not isinstance(previous, dict) or not isinstance(previous.get("dataset_sha256"), str):
        raise ValueError("Missing dataset fingerprint in training metadata.")
    for key in ("train_row_ids", "test_row_ids"):
        if not isinstance(previous.get(key), list) or any(type(i) is not int for i in previous[key]):
            raise ValueError(f"Invalid {key} in training metadata.")
    if "run_id" in previous and not isinstance(previous["run_id"], str):
        raise ValueError("Invalid run ID in training metadata.")
    selection = previous.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("Invalid model selection metadata.")
    reveals = selection.get("previous_test_reveals_for_dataset_target", 0)
    if type(reveals) is not int or reveals < 0:
        raise ValueError("Invalid test exposure count in training metadata.")
    return previous


def read_archive(data):
    """Validate bounded ZIP members in memory; do not extract paths or deserialize code."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(data) > MAX_ARCHIVE_BYTES or len(members) > 40 or sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES:
                raise ValueError("Run archive exceeds the 100 MB / 40-file limit.")
            if len({m.filename for m in members}) != len(members):
                raise ValueError("Duplicate archive members.")
            files = {m.filename: archive.read(m) for m in members}
        manifest = json.loads(files["manifest.json"])
        _validate_manifest(manifest)
        for name, digest in manifest["files"].items():
            if hashlib.sha256(files[name]).hexdigest() != digest:
                raise ValueError(f"Integrity check failed for {name}.")
        if not {"dataset.csv", "predictions.csv", "training_metadata.json"} <= set(manifest["files"]):
            raise ValueError("The archive is missing a required file.")
        _read_metadata(files)
        raw = analysis.load_csv(files["dataset.csv"]).frame
        if len(manifest["source_index"]) != len(raw):
            raise ValueError("Source row IDs do not match the dataset.")
        raw.index = manifest["source_index"]
        raw.index.name = manifest["source_index_name"]
        if not raw.index.is_unique or not all(isinstance(i, int) for i in raw.index):
            raise ValueError("Source row IDs must be unique integers.")
        return manifest, raw, files
    except (KeyError, TypeError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise analysis.DataError(f"Cannot restore this run archive: {exc}") from exc


def reproduce(data, *, allow_environment_change=False):
    manifest, raw, files = read_archive(data)
    changes = [p for p, v in manifest["packages"].items() if p not in PACKAGES or version(p) != v]
    if code_digest() != manifest["code_sha256"]:
        changes.append("Atlas calculation code")
    if platform.python_version().split(".")[:2] != manifest["python"].split(".")[:2]:
        changes.append("Python major/minor version")
    if changes and not allow_environment_change:
        raise analysis.DataError("Reproduction environment differs: " + ", ".join(changes) + ". Use the recorded versions, or explicitly allow an exploratory replay.")
    config = dict(manifest["recipe"])
    config["features"], config["categorical_features"] = tuple(config["features"]), tuple(config["categorical_features"])
    previous = _read_metadata(files)
    if training.fingerprint(raw) != previous["dataset_sha256"]:
        raise analysis.DataError("The restored dataset does not match the recorded fingerprint.")
    result = training.train_and_evaluate(raw, **config)
    if list(result.train_rows) != previous["train_row_ids"] or list(result.test_rows) != previous["test_row_ids"]:
        raise analysis.DataError("Reproduction changed the split row IDs; this is not the same experiment.")
    exact = result.csv_bytes() == files["predictions.csv"]
    if not exact and not allow_environment_change:
        raise analysis.DataError("Reproduction did not match the saved predictions byte for byte.")
    result.metadata.update(source_file=previous.get("source_file", "restored dataset"),
                           run_id=previous.get("run_id", result.metadata["split_id"][:16]) + "-replay",
                           reproduction={"exact_predictions": exact, "environment_changes": changes,
                                         "recorded_python": manifest["python"], "runtime_python": platform.python_version()},
                           selection=previous.get("selection", {}), leakage_review=previous.get("leakage_review", []))
    return result
