import math
from dataclasses import dataclass

from measurements.validation import require_finite_number
from cohorts.selection import validate_feature_order


@dataclass(frozen=True)
class NumericTransform:
    feature_order: tuple
    means: tuple
    scales: tuple
    constants: tuple

    def to_payload(self):
        return {"feature_order": list(self.feature_order), "means": list(self.means),
                "scales": list(self.scales), "constants": list(self.constants)}

    @classmethod
    def from_payload(cls, payload):
        order = validate_feature_order(payload["feature_order"])
        means = tuple(require_finite_number(x) for x in payload["means"])
        scales = tuple(require_finite_number(x) for x in payload["scales"])
        constants = tuple(payload["constants"])
        if not (len(order) == len(means) == len(scales) == len(constants)):
            raise ValueError("transform dimensions differ")
        if any(type(flag) is not bool for flag in constants):
            raise ValueError("invalid constant flags")
        if any(scale <= 0 for scale in scales):
            raise ValueError("transform scales must be positive")
        if any(flag and scale != 1 for flag, scale in zip(constants, scales)):
            raise ValueError("constant columns need effective scale one")
        return cls(order, means, scales, constants)


def _matrix(rows, width):
    result = []
    for row in rows:
        if isinstance(row, (str, bytes, dict)):
            raise ValueError("numeric row must be a positional sequence")
        try:
            values = [require_finite_number(value) for value in row]
        except TypeError as exc:
            raise ValueError("numeric row must be iterable") from exc
        if len(values) != width:
            raise ValueError("numeric row dimension differs")
        result.append(values)
    return result


def fit_numeric_transform(rows, feature_order):
    order = validate_feature_order(feature_order)
    matrix = _matrix(rows, len(order))
    if not matrix:
        raise ValueError("training matrix is empty")
    means, scales, constants = [], [], []
    try:
        for j in range(len(order)):
            column = [row[j] for row in matrix]
            mean = math.fsum(column) / len(column)
            constant = all(value == column[0] for value in column)
            variance = math.fsum((value - mean) ** 2 for value in column) / len(column)
            scale = 1.0 if constant else math.sqrt(variance)
            if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("non-finite or degenerate numeric transform")
            means.append(mean)
            scales.append(scale)
            constants.append(constant)
    except (ArithmeticError, OverflowError) as exc:
        raise ValueError("numeric standardization failed") from exc
    return NumericTransform(order, tuple(means), tuple(scales), tuple(constants))


def transform_numeric_rows(rows, transform):
    if isinstance(transform, dict):
        transform = NumericTransform.from_payload(transform)
    matrix = _matrix(rows, len(transform.feature_order))
    output = []
    for row in matrix:
        values = [0.0 if constant else (value - mean) / scale
                  for value, mean, scale, constant in
                  zip(row, transform.means, transform.scales, transform.constants)]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("numeric transformation overflow")
        output.append(values)
    return output
