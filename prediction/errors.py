import math
from dataclasses import dataclass

from measurements.validation import require_finite_number


@dataclass(frozen=True)
class ErrorSummary:
    sse: float
    n: int
    mse: float

    def to_payload(self):
        return {"sse": self.sse, "n": self.n, "mse": self.mse}


def squared_errors(targets, predictions):
    y = [require_finite_number(v) for v in targets]
    p = [require_finite_number(v) for v in predictions]
    if len(y) != len(p):
        raise ValueError("error vectors differ in length")
    try:
        errors = [(a - b) ** 2 for a, b in zip(y, p)]
    except ArithmeticError as exc:
        raise ValueError("squared error overflow") from exc
    if any(not math.isfinite(value) for value in errors):
        raise ValueError("squared errors must be finite")
    return errors


def summarize_row_error(targets, predictions):
    errors = squared_errors(targets, predictions)
    if not errors:
        raise ValueError("holdout errors cannot be empty")
    try:
        total = math.fsum(errors)
    except ArithmeticError as exc:
        raise ValueError("total squared error overflow") from exc
    if not math.isfinite(total):
        raise ValueError("total squared error must be finite")
    return ErrorSummary(total, len(errors), total / len(errors))
