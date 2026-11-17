import math
from copy import deepcopy
from dataclasses import dataclass

from archive.schema import content_identity
from archive.publish import publish_artifact
from archive.reading import read_cohort
from cohorts.partitions import build_training_scope
from cohorts.selection import select_numeric_columns
from fitting.features import fit_numeric_transform, transform_numeric_rows
from fitting.objective import (assemble_normal_equations, mean_squared_loss,
                               ridge_penalty, ridge_objective, validate_lambda)
from fitting.solver import NumericSolveError, solve_linear_system
from measurements.validation import require_finite_number


@dataclass(frozen=True)
class NumericFit:
    fit_id: str
    cohort_id: str
    scope: dict
    feature_order: tuple
    transform: dict
    lambda_: str
    coefficients: tuple
    diagnostics: dict
    training_predictions: tuple

    def to_payload(self):
        return deepcopy({"fit_id": self.fit_id, "cohort_id": self.cohort_id,
                         "scope": self.scope, "feature_order": list(self.feature_order),
                         "transform": self.transform, "lambda": self.lambda_,
                         "coefficients": list(self.coefficients),
                         "diagnostics": self.diagnostics,
                         "training_predictions": list(self.training_predictions)})

    @classmethod
    def from_payload(cls, payload):
        p = deepcopy(payload)
        return cls(p["fit_id"], p["cohort_id"], p["scope"], tuple(p["feature_order"]),
                   p["transform"], p["lambda"], tuple(p["coefficients"]),
                   p["diagnostics"], tuple(p["training_predictions"]))


def fit_identity(payload):
    return content_identity("fit", payload)


def calculate_training_diagnostics(rows, matrix, coefficients, lambda_):
    if len(coefficients) != len(matrix[0]) + 1:
        raise NumericSolveError("coefficient dimensions differ")
    predictions = [coefficients[0] + math.fsum(w * z for w, z in
                   zip(coefficients[1:], row)) for row in matrix]
    targets = [require_finite_number(row["target"]) for row in rows]
    mse = mean_squared_loss(targets, predictions)
    penalty = ridge_penalty(coefficients, lambda_)
    objective = ridge_objective(targets, predictions, coefficients, lambda_)
    evidence = [{"revision_id": row["revision_id"], "row_id": row["row_id"],
                 "device_id": row["device_id"], "target": row["target"],
                 "prediction": prediction, "residual": target - prediction}
                for row, target, prediction in zip(rows, targets, predictions)]
    return {"n": len(rows), "mse": mse, "penalty": penalty,
            "objective": objective}, evidence


def fit_numeric(store, cohort_id, training_devices, lambda_, solver=None,
                publication_hook=None):
    cohort = read_cohort(store, cohort_id)
    scope = build_training_scope(cohort, training_devices)
    parameter, _ = validate_lambda(lambda_)
    selected = set(scope.revision_ids)
    rows = [row for row in cohort["rows"] if row["revision_id"] in selected]
    matrix = select_numeric_columns(rows, cohort["feature_order"])
    transform = fit_numeric_transform(matrix, cohort["feature_order"])
    standardized = transform_numeric_rows(matrix, transform)
    targets = [require_finite_number(row["target"]) for row in rows]
    lhs, rhs = assemble_normal_equations(standardized, targets, parameter)
    implementation = solve_linear_system if solver is None else solver
    raw = implementation([row[:] for row in lhs], rhs[:])
    try:
        coefficients = [require_finite_number(v) for v in raw]
        if len(coefficients) != len(rhs):
            raise NumericSolveError("solver returned wrong dimensions")
        for row, value in zip(lhs, rhs):
            residual = abs(math.fsum(a * b for a, b in zip(row, coefficients)) - value)
            if not math.isfinite(residual) or residual > 1e-8 * max(1.0, abs(value)):
                raise NumericSolveError("solver returned an invalid solution")
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise NumericSolveError("invalid solver result") from exc
    diagnostics, evidence = calculate_training_diagnostics(
        rows, standardized, coefficients, parameter)
    body = {"cohort_id": cohort_id, "scope": scope.to_payload(),
            "feature_order": list(cohort["feature_order"]),
            "transform": transform.to_payload(), "lambda": parameter,
            "coefficients": coefficients, "diagnostics": diagnostics,
            "training_predictions": evidence}
    body["fit_id"] = fit_identity(body)
    fit = NumericFit.from_payload(body)
    return publish_artifact(store, "fit", fit.to_payload(),
                            {"cohort": cohort_id}, publication_hook)
