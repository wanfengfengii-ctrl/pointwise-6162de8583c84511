import math
from copy import deepcopy

from archive.schema import ArchiveIntegrityError


def build_training_projection(graph):
    scope = graph["fit"]["scope"]
    selected = set(scope["revision_ids"])
    cohort = graph["cohort"]
    return [{"revision_id": row["revision_id"], "row_id": row["row_id"],
             "device_id": row["device_id"],
             "values": [row["features"][name] for name in cohort["feature_order"]],
             "target": row["target"]}
            for row in cohort["rows"] if row["revision_id"] in selected]


def audit_package_graph(graph):
    from prediction.packages import validate_package_binding
    package, fit, cohort = graph["package"], graph["fit"], graph["cohort"]
    validate_package_binding(package, fit, cohort)
    training = build_training_projection(graph)
    if [row["revision_id"] for row in training] != fit["scope"]["revision_ids"]:
        raise ArchiveIntegrityError("training projection differs")
    errors = []
    for row in package["holdout"]:
        error = row["squared_error"]
        if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
            raise ArchiveIntegrityError("invalid stored squared error")
        errors.append(error)
    summary = package["error"]
    total = math.fsum(errors)
    if (summary["n"] != len(errors) or
            not math.isclose(total, summary["sse"], rel_tol=1e-12, abs_tol=1e-12) or
            not math.isclose(total / len(errors), summary["mse"],
                             rel_tol=1e-12, abs_tol=1e-12)):
        raise ArchiveIntegrityError("stored error summary differs")
    return True


def build_review_manifest(graph):
    audit_package_graph(graph)
    package, fit, cohort = graph["package"], graph["fit"], graph["cohort"]
    return deepcopy({"format": "numeric-review-v1", "package_id": package["package_id"],
                     "fit_id": fit["fit_id"], "cohort_id": cohort["cohort_id"],
                     "scope_id": fit["scope"]["scope_id"],
                     "feature_order": cohort["feature_order"],
                     "cohort_rows": len(cohort["rows"]),
                     "training_rows": len(fit["scope"]["revision_ids"]),
                     "holdout_rows": len(package["holdout"]),
                     "training_devices": fit["scope"]["devices"],
                     "holdout_devices": package["holdout_devices"],
                     "mse": package["error"]["mse"]})
