"""Known-noise correlated fit.

The measured noise covariance is part of the statistical model supplied by
the caller; this module does not estimate it.  Every computation is exact
rational arithmetic and the residual support constraint r in im(C) is a hard
constraint, never a least-squares approximation.
"""
from collections.abc import Mapping
from fractions import Fraction

from archive.publish import publish_artifact
from archive.reading import read_cohort
from archive.schema import ArchiveIntegrityError, content_identity
from cohorts.partitions import build_training_scope
from fitting.exact import (affine_solve, matmul, matvec, nullspace,
                           parse_matrix, pseudoinverse, quadratic_form,
                           require_rational, require_sequence, solve_unique,
                           symmetric_psd_rank, transpose)

MAX_TRAINING_ROWS = 24
MAX_RAW_FEATURES = 4
MIN_TRAINING_ROWS = 1
MIN_RAW_FEATURES = 1


def positive_rational(value, label="lambda"):
    rational = require_rational(value, label)
    if rational <= 0:
        raise ValueError(label + " must be a strictly positive rational")
    return rational


def _training_rows(cohort, scope):
    selected = set(scope.revision_ids)
    rows = [row for row in cohort["rows"] if row["revision_id"] in selected]
    if [row["revision_id"] for row in rows] != list(scope.revision_ids):
        raise ValueError("training rows differ from the fixed scope")
    return rows


def _design(rows, feature_order):
    design = []
    for row in rows:
        try:
            values = [require_rational(row["features"][name], "feature")
                      for name in feature_order]
        except KeyError as exc:
            raise ValueError("member lacks requested features") from exc
        design.append([Fraction(1)] + values)
    return design


def _targets(rows):
    return [require_rational(row["target"], "target") for row in rows]


def _require_training_shape(design, targets):
    size = len(targets)
    if not size or len(design) != size:
        raise ValueError("training design and targets differ")
    width = len(design[0])
    if any(len(row) != width for row in design):
        raise ValueError("ragged training design")
    return size, width


def canonical_noise(noise, revision_ids):
    """Validate caller noise and reorder both axes into training order.

    Returns the canonical revision id list (training order) and the
    covariance matrix in that same order.  Identity and covariance are
    reordered in lockstep, so any axis permutation yields one identity.
    """
    if not isinstance(noise, Mapping):
        raise ValueError("noise must be an object")
    if set(noise) != {"revision_ids", "covariance"}:
        raise ValueError("noise must contain exactly revision_ids and covariance")
    supplied = require_sequence(noise["revision_ids"], "noise revision_ids")
    if any(not isinstance(identity, str) for identity in supplied):
        raise ValueError("noise revision_ids must be strings")
    expected = list(revision_ids)
    if any(not isinstance(identity, str) for identity in expected):
        raise ValueError("training revision identities must be strings")
    if len(supplied) != len(expected) or len(set(supplied)) != len(supplied):
        raise ValueError("noise revision_ids do not match the training revisions")
    if set(supplied) != set(expected):
        raise ValueError("noise revision_ids do not match the training revisions")
    position = {identity: index for index, identity in enumerate(supplied)}
    size = len(expected)
    given = parse_matrix(noise["covariance"], size, size, "noise covariance")
    covariance = [[given[position[left]][position[right]] for right in expected]
                  for left in expected]
    return expected, covariance


def solve_correlated(design, targets, covariance, penalty):
    """Exact unique minimizer of r'C+r/n + penalty*sum(slope^2), r in im(C)."""
    size, width = _require_training_shape(design, targets)
    if len(covariance) != size or any(len(row) != size for row in covariance):
        raise ValueError("noise covariance dimensions differ")
    is_psd, rank = symmetric_psd_rank(covariance)
    if not is_psd:
        raise ValueError("noise covariance must be symmetric positive semidefinite")
    inverse, _ = pseudoinverse(covariance)

    # Hard support constraint r in im(C) <=> N r = 0 for null(C) basis rows N.
    complement = nullspace(covariance)              # k x n
    constraint = matmul(complement, design)         # k x p
    constraint_rhs = matvec(complement, targets)
    particular, freedom = affine_solve(constraint, constraint_rhs, width)
    directions = transpose(freedom)                 # width x d
    generator = matmul(design, directions)          # n x d
    base_residual = [target - sum(particular[j] * row[j] for j in range(width))
                     for target, row in zip(targets, design)]

    weighted = matvec(inverse, base_residual)       # C+ r0
    weighted_generator = matmul(inverse, generator)  # C+ G
    free_gradient = [sum(generator[k][i] * weighted[k] for k in range(size)) / size
                     for i in range(len(freedom))]
    free_hessian = [[sum(generator[k][i] * weighted_generator[k][j]
                         for k in range(size)) / size
                     for j in range(len(freedom))]
                    for i in range(len(freedom))]
    slope_map = directions[1:]                      # (p-1) x d
    slope_targets = matvec(transpose(slope_map), particular[1:])
    for i in range(len(freedom)):
        free_gradient[i] -= penalty * slope_targets[i]
        for j in range(len(freedom)):
            free_hessian[i][j] += penalty * sum(slope_map[row][i] * slope_map[row][j]
                                                for row in range(len(slope_map)))
    steps = solve_unique(free_hessian, free_gradient)
    coefficients = [particular[i] + sum(directions[i][j] * steps[j]
                                        for j in range(len(steps)))
                    for i in range(width)]
    residual = [target - sum(coefficients[j] * row[j] for j in range(width))
                for target, row in zip(targets, design)]
    if any(value != 0 for value in matvec(complement, residual)):
        raise ValueError("training data is infeasible for the noise support")
    loss = quadratic_form(residual, inverse) / size
    slope_penalty = penalty * sum(value * value for value in coefficients[1:])
    return {"coefficients": coefficients, "residuals": residual, "rank": rank,
            "loss": loss, "penalty": slope_penalty, "objective": loss + slope_penalty}


def _evidence(rows, coefficients, residual, feature_order):
    evidence = []
    for row, residual_value in zip(rows, residual):
        prediction = coefficients[0] + sum(
            coefficients[j + 1] * require_rational(row["features"][name])
            for j, name in enumerate(feature_order))
        evidence.append({
            "revision_id": row["revision_id"], "row_id": row["row_id"],
            "device_id": row["device_id"],
            "values": [str(require_rational(row["features"][name]))
                       for name in feature_order],
            "target": str(require_rational(row["target"])),
            "prediction": str(prediction), "residual": str(residual_value)})
    return evidence


def _payload(cohort, scope, order, penalty, noise_ids, covariance, solution, evidence):
    return {
        "model_kind": "correlated", "cohort_id": cohort["cohort_id"],
        "scope": scope.to_payload(), "feature_order": list(order),
        "lambda": str(penalty),
        "noise": {"revision_ids": list(noise_ids),
                  "covariance": [[str(value) for value in row] for row in covariance]},
        "coefficients": [str(value) for value in solution["coefficients"]],
        "residuals": [str(value) for value in solution["residuals"]],
        "diagnostics": {"n": len(scope.revision_ids),
                        "noise_rank": solution["rank"],
                        "loss": str(solution["loss"]),
                        "penalty": str(solution["penalty"]),
                        "objective": str(solution["objective"])},
        "training_predictions": evidence,
    }


def fit_correlated(store, cohort_id, training_devices, lambda_, noise,
                   publication_hook=None):
    cohort = read_cohort(store, cohort_id)
    order = tuple(cohort["feature_order"])
    if not MIN_RAW_FEATURES <= len(order) <= MAX_RAW_FEATURES:
        raise ValueError("correlated fitting supports one to four raw features")
    scope = build_training_scope(cohort, training_devices)
    count = len(scope.revision_ids)
    if not MIN_TRAINING_ROWS <= count <= MAX_TRAINING_ROWS:
        raise ValueError("correlated fitting supports one to 24 training rows")
    penalty = positive_rational(lambda_)
    rows = _training_rows(cohort, scope)
    design = _design(rows, order)
    targets = _targets(rows)
    noise_ids, covariance = canonical_noise(noise, scope.revision_ids)
    solution = solve_correlated(design, targets, covariance, penalty)
    evidence = _evidence(rows, solution["coefficients"], solution["residuals"], order)
    body = _payload(cohort, scope, order, penalty, noise_ids, covariance,
                    solution, evidence)
    body["fit_id"] = content_identity("fit", body)
    return publish_artifact(store, "fit", body, {"cohort": cohort_id},
                            publication_hook)


def validate_correlated_payload(store, payload):
    """Return a fresh exact correlated fit dictionary for a stored payload."""
    try:
        if payload.get("model_kind") != "correlated":
            raise ArchiveIntegrityError("fit is not a correlated fit")
        try:
            cohort = read_cohort(store, payload["cohort_id"])
        except ValueError as exc:
            raise ArchiveIntegrityError("missing or invalid fit parent cohort") from exc
        order = tuple(payload["feature_order"])
        if list(order) != cohort["feature_order"]:
            raise ArchiveIntegrityError("fit feature order differs from cohort")
        if not MIN_RAW_FEATURES <= len(order) <= MAX_RAW_FEATURES:
            raise ArchiveIntegrityError("correlated fit feature count differs")
        try:
            scope = build_training_scope(
                cohort, payload["scope"]["devices"]).to_payload()
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveIntegrityError("fit training scope is invalid") from exc
        if scope != payload["scope"]:
            raise ArchiveIntegrityError("fit training binding differs")
        revision_ids = list(scope["revision_ids"])
        count = len(revision_ids)
        if not MIN_TRAINING_ROWS <= count <= MAX_TRAINING_ROWS:
            raise ArchiveIntegrityError("correlated fit training count differs")
        penalty = positive_rational(payload["lambda"])
        noise = payload["noise"]
        if set(noise) != {"revision_ids", "covariance"}:
            raise ArchiveIntegrityError("correlated fit noise fields differ")
        if list(noise["revision_ids"]) != revision_ids:
            raise ArchiveIntegrityError("fit noise axis order differs")
        covariance = parse_matrix(noise["covariance"], count, count,
                                  "fit noise covariance")
        is_psd, rank = symmetric_psd_rank(covariance)
        if not is_psd:
            raise ArchiveIntegrityError(
                "fit noise covariance is not positive semidefinite")
        coefficients = [require_rational(value) for value in payload["coefficients"]]
        if len(coefficients) != len(order) + 1:
            raise ArchiveIntegrityError("correlated fit coefficient dimensions differ")
        residuals = [require_rational(value) for value in payload["residuals"]]
        if len(residuals) != count:
            raise ArchiveIntegrityError("correlated fit residual count differs")
        selected = set(revision_ids)
        rows = [row for row in cohort["rows"] if row["revision_id"] in selected]
        if [row["revision_id"] for row in rows] != revision_ids:
            raise ArchiveIntegrityError("fit training rows differ from the cohort")
        design = _design(rows, order)
        targets = _targets(rows)
        recomputed = [target - sum(coefficients[j] * row[j] for j in range(len(row)))
                      for target, row in zip(targets, design)]
        if recomputed != residuals:
            raise ArchiveIntegrityError("stored residuals differ from fixed rows")
        complement = nullspace(covariance)
        if any(value != 0 for value in matvec(complement, residuals)):
            raise ArchiveIntegrityError("stored residuals leave the noise support")
        inverse, _ = pseudoinverse(covariance)
        loss = quadratic_form(residuals, inverse) / count
        penalty_value = penalty * sum(value * value for value in coefficients[1:])
        diagnostics = payload["diagnostics"]
        if set(diagnostics) != {"n", "noise_rank", "loss", "penalty", "objective"}:
            raise ArchiveIntegrityError("correlated fit diagnostics differ")
        if type(diagnostics["n"]) is not int or diagnostics["n"] != count:
            raise ArchiveIntegrityError("correlated fit diagnostic count differs")
        if (type(diagnostics["noise_rank"]) is not int or
                diagnostics["noise_rank"] != rank):
            raise ArchiveIntegrityError("correlated fit diagnostic rank differs")
        if (require_rational(diagnostics["loss"]) != loss or
                require_rational(diagnostics["penalty"]) != penalty_value or
                require_rational(diagnostics["objective"]) != loss + penalty_value):
            raise ArchiveIntegrityError("correlated fit diagnostic values differ")
        evidence = payload["training_predictions"]
        if not isinstance(evidence, list) or \
                [item.get("revision_id") for item in evidence] != revision_ids:
            raise ArchiveIntegrityError("fit training evidence differs")
        for item, residual_value, row, target, design_row in zip(
                evidence, residuals, rows, targets, design):
            if require_rational(item["residual"]) != residual_value:
                raise ArchiveIntegrityError("fit training evidence residual differs")
            if item["row_id"] != row["row_id"] or item["device_id"] != row["device_id"]:
                raise ArchiveIntegrityError("fit training evidence identity differs")
            if [require_rational(value) for value in item["values"]] != design_row[1:]:
                raise ArchiveIntegrityError("fit training evidence values differ")
            if require_rational(item["target"]) != target:
                raise ArchiveIntegrityError("fit training evidence target differs")
            prediction = sum(coefficients[j] * design_row[j]
                             for j in range(len(design_row)))
            if require_rational(item["prediction"]) != prediction:
                raise ArchiveIntegrityError("fit training evidence prediction differs")
        return {"model_kind": "correlated", "fit_id": payload["fit_id"],
                "cohort_id": payload["cohort_id"], "scope": scope,
                "feature_order": list(order), "lambda": penalty,
                "noise": {"revision_ids": revision_ids, "covariance": covariance},
                "coefficients": coefficients, "residuals": residuals,
                "diagnostics": {"n": count, "noise_rank": rank, "loss": loss,
                                "penalty": penalty_value,
                                "objective": loss + penalty_value}}
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ArchiveIntegrityError("malformed correlated fit") from exc
    except ArchiveIntegrityError:
        raise
    except (ValueError, ArithmeticError) as exc:
        raise ArchiveIntegrityError("malformed correlated fit values") from exc


def read_correlated_fit(store, fit_id):
    """Read a fit that must be a correlated fit; V1 fits raise ValueError."""
    from archive.reading import read_stored_payload
    record = store.read_artifact(fit_id)
    if record["kind"] != "fit":
        raise ValueError("artifact is not a fit: " + str(fit_id))
    payload = read_stored_payload(store, fit_id, "fit")
    if payload.get("model_kind") != "correlated":
        raise ValueError("fit is not a correlated fit: " + str(fit_id))
    return validate_correlated_payload(store, payload)
