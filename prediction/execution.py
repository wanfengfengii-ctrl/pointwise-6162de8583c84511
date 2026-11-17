import math

from fitting.features import transform_numeric_rows
from measurements.validation import require_finite_number


def predict_fixed_numeric(transform, coefficients, rows):
    weights = [require_finite_number(value) for value in coefficients]
    matrix = transform_numeric_rows(rows, transform)
    if len(weights) != len(transform["feature_order"]) + 1:
        raise ValueError("prediction coefficient dimensions differ")
    output = []
    try:
        for row in matrix:
            value = weights[0] + math.fsum(w * z for w, z in zip(weights[1:], row))
            if not math.isfinite(value):
                raise ValueError("prediction must be finite")
            output.append(value)
    except ArithmeticError as exc:
        raise ValueError("prediction overflow") from exc
    return output


def predict_numeric(store, package_id, rows):
    from archive.reading import read_package
    package = read_package(store, package_id)["package"]
    return predict_fixed_numeric(package["transform"], package["coefficients"], rows)
