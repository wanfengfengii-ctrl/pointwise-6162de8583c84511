import argparse
import json
import tempfile
from pathlib import Path

from delivery.app import open_archive
from delivery.auditing import build_review_manifest


def run_daily_workflow(app, request, export_target):
    revisions = app.apply_measurements(request["events"])
    cohort_id = app.freeze_cohort(
        [row["revision_id"] for row in revisions], request["feature_order"])
    fit = app.fit_numeric(cohort_id, request["training_devices"], request["lambda"])
    package = app.evaluate_holdout(fit["fit_id"], request["holdout_devices"])
    predictions = app.predict_numeric(package["package_id"], request["predict"])
    document = app.export_package(package["package_id"], export_target)
    manifest = build_review_manifest(app.read_package(package["package_id"]))
    if document["manifest"] != manifest:
        raise RuntimeError("export manifest differs from stored graph")
    return {"cohort_id": cohort_id, "fit_id": fit["fit_id"],
            "package_id": package["package_id"], "training": fit["diagnostics"],
            "error": package["error"], "predictions": predictions, "manifest": manifest}


def run_sample_workflow(database, export_target, sample_path=None):
    sample = Path(sample_path) if sample_path else Path(__file__).resolve().parents[1] / "examples/day.json"
    with sample.open(encoding="utf-8") as handle:
        request = json.load(handle)
    with open_archive(database) as app:
        result = run_daily_workflow(app, request, export_target)
    with open_archive(database) as app:
        result["reopened_predictions"] = app.predict_numeric(
            result["package_id"], request["predict"])
        if result["predictions"] != result["reopened_predictions"]:
            raise RuntimeError("reopened predictions differ")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the numeric archive daily workflow")
    parser.add_argument("--database")
    parser.add_argument("--export")
    parser.add_argument("--sample")
    args = parser.parse_args(argv)
    if (args.database is None) != (args.export is None):
        parser.error("--database and --export must be supplied together")
    if args.database is None:
        with tempfile.TemporaryDirectory(prefix="numeric-archive-demo-") as directory:
            root = Path(directory)
            result = run_sample_workflow(root / "archive.sqlite", root / "review.json", args.sample)
    else:
        result = run_sample_workflow(args.database, args.export, args.sample)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
