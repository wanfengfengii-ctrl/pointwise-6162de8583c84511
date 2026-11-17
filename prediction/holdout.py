from archive.reading import read_cohort, read_fit, read_package
from archive.publish import publish_artifact
from cohorts.partitions import resolve_holdout_rows
from cohorts.selection import select_numeric_columns
from prediction.errors import squared_errors, summarize_row_error
from prediction.execution import predict_fixed_numeric
from prediction.packages import build_numeric_package


def build_holdout_predictions(fit, rows):
    matrix = select_numeric_columns(rows, fit["feature_order"])
    predictions = predict_fixed_numeric(fit["transform"], fit["coefficients"], matrix)
    targets = [row["target"] for row in rows]
    errors = squared_errors(targets, predictions)
    evidence = [{"revision_id": row["revision_id"], "row_id": row["row_id"],
                 "device_id": row["device_id"], "target": row["target"],
                 "prediction": value, "squared_error": error}
                for row, value, error in zip(rows, predictions, errors)]
    return evidence, summarize_row_error(targets, predictions)


def evaluate_holdout(store, fit_id, holdout_devices, publication_hook=None):
    from archive.reading import read_stored_payload
    stored = read_stored_payload(store, fit_id, "fit")
    if stored.get("model_kind") == "correlated":
        raise ValueError("holdout evaluation does not apply to correlated fits")
    fit = read_fit(store, fit_id)
    cohort = read_cohort(store, fit["cohort_id"])
    rows = resolve_holdout_rows(cohort, fit["scope"]["devices"], holdout_devices)
    evidence, summary = build_holdout_predictions(fit, rows)
    package = build_numeric_package(fit, cohort, evidence, summary)
    identity = publish_artifact(
        store, "package", package.to_payload(),
        {"fit": fit_id, "cohort": cohort["cohort_id"]}, publication_hook)
    return read_package(store, identity)["package"]
