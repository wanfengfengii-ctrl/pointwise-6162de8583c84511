from copy import deepcopy
from dataclasses import dataclass

from archive.schema import ArchiveIntegrityError, content_identity
from cohorts.partitions import build_training_scope, resolve_holdout_rows
from fitting.features import NumericTransform
from fitting.diagnostics import NumericFit


@dataclass(frozen=True)
class NumericPackage:
    package_id: str
    fit_id: str
    cohort_id: str
    scope: dict
    feature_order: tuple
    transform: dict
    coefficients: tuple
    holdout_devices: tuple
    holdout: tuple
    error: dict

    def to_payload(self):
        return deepcopy({"package_id": self.package_id, "fit_id": self.fit_id,
                         "cohort_id": self.cohort_id, "scope": self.scope,
                         "feature_order": list(self.feature_order),
                         "transform": self.transform, "coefficients": list(self.coefficients),
                         "holdout_devices": list(self.holdout_devices),
                         "holdout": list(self.holdout), "error": self.error})

    @classmethod
    def from_payload(cls, payload):
        p = deepcopy(payload)
        return cls(p["package_id"], p["fit_id"], p["cohort_id"], p["scope"],
                   tuple(p["feature_order"]), p["transform"], tuple(p["coefficients"]),
                   tuple(p["holdout_devices"]), tuple(p["holdout"]), p["error"])


def package_identity(payload):
    return content_identity("package", payload)


def validate_package_binding(package, fit, cohort):
    if isinstance(fit, NumericFit):
        fit = fit.to_payload()
    if package["fit_id"] != fit["fit_id"] or package["cohort_id"] != cohort["cohort_id"]:
        raise ArchiveIntegrityError("package parent identity differs")
    if fit["cohort_id"] != cohort["cohort_id"]:
        raise ArchiveIntegrityError("fit belongs to another cohort")
    for field in ("scope", "feature_order", "transform", "coefficients"):
        if package[field] != fit[field]:
            raise ArchiveIntegrityError("package model binding differs: " + field)
    transform = NumericTransform.from_payload(package["transform"])
    if list(transform.feature_order) != cohort["feature_order"]:
        raise ArchiveIntegrityError("package feature order differs")
    scope = build_training_scope(cohort, fit["scope"]["devices"]).to_payload()
    if scope != fit["scope"]:
        raise ArchiveIntegrityError("package training scope differs")
    rows = resolve_holdout_rows(cohort, scope["devices"], package["holdout_devices"])
    if [r["revision_id"] for r in rows] != [r["revision_id"] for r in package["holdout"]]:
        raise ArchiveIntegrityError("holdout members differ from device range")
    for row, result in zip(rows, package["holdout"]):
        for field in ("revision_id", "row_id", "device_id", "target"):
            if row[field] != result[field]:
                raise ArchiveIntegrityError("holdout evidence identity differs")
    if len(package["coefficients"]) != len(transform.feature_order) + 1:
        raise ArchiveIntegrityError("package coefficient dimensions differ")
    error = package["error"]
    if error["n"] != len(rows) or error["n"] <= 0:
        raise ArchiveIntegrityError("holdout summary count differs")


def build_numeric_package(fit, cohort, holdout, error):
    payload = {"fit_id": fit["fit_id"], "cohort_id": fit["cohort_id"],
               "scope": deepcopy(fit["scope"]), "feature_order": list(fit["feature_order"]),
               "transform": deepcopy(fit["transform"]),
               "coefficients": list(fit["coefficients"]),
               "holdout_devices": sorted({row["device_id"] for row in holdout}),
               "holdout": deepcopy(holdout), "error": error.to_payload()}
    payload["package_id"] = package_identity(payload)
    validate_package_binding(payload, fit, cohort)
    return NumericPackage.from_payload(payload)
