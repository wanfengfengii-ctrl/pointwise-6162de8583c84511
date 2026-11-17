import math

from measurements.validation import require_finite_number, require_fraction


def validate_lambda(value):
    rational = require_fraction(value)
    number = require_finite_number(rational)
    if rational <= 0 or number <= 0:
        raise ValueError("lambda must be a positive representable rational")
    return str(rational), number


def _training_inputs(matrix, targets):
    rows = [[require_finite_number(x) for x in row] for row in matrix]
    y = [require_finite_number(value) for value in targets]
    if not rows or len(rows) != len(y) or not rows[0]:
        raise ValueError("empty or incompatible training data")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged training data")
    return rows, y


def assemble_normal_equations(matrix, targets, lambda_):
    rows, targets = _training_inputs(matrix, targets)
    _, penalty = validate_lambda(lambda_)
    design = [[1.0] + row for row in rows]
    n, width = len(rows), len(design[0])
    try:
        lhs = [[math.fsum(row[i] * row[j] for row in design) / n
                + (penalty if i == j and i > 0 else 0.0)
                for j in range(width)] for i in range(width)]
        rhs = [math.fsum(row[j] * y for row, y in zip(design, targets)) / n
               for j in range(width)]
    except ArithmeticError as exc:
        raise ValueError("normal equation overflow") from exc
    if any(not math.isfinite(v) for row in lhs for v in row):
        raise ValueError("non-finite normal matrix")
    if any(not math.isfinite(v) for v in rhs):
        raise ValueError("non-finite normal target")
    return lhs, rhs


def mean_squared_loss(targets, predictions):
    y = [require_finite_number(v) for v in targets]
    p = [require_finite_number(v) for v in predictions]
    if not y or len(y) != len(p):
        raise ValueError("loss needs nonempty paired values")
    try:
        value = math.fsum((a - b) ** 2 for a, b in zip(y, p)) / len(y)
    except ArithmeticError as exc:
        raise ValueError("loss overflow") from exc
    if not math.isfinite(value):
        raise ValueError("loss must be finite")
    return value


def ridge_penalty(coefficients, lambda_):
    _, penalty = validate_lambda(lambda_)
    coefficients = [require_finite_number(v) for v in coefficients]
    if not coefficients:
        raise ValueError("intercept is required")
    try:
        value = penalty * math.fsum(w * w for w in coefficients[1:])
    except ArithmeticError as exc:
        raise ValueError("penalty overflow") from exc
    if not math.isfinite(value):
        raise ValueError("penalty must be finite")
    return value


def ridge_objective(targets, predictions, coefficients, lambda_):
    result = mean_squared_loss(targets, predictions) + ridge_penalty(coefficients, lambda_)
    if not math.isfinite(result):
        raise ValueError("objective must be finite")
    return result
