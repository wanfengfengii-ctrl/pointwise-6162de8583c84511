"""Conditional joint prediction consuming a fixed correlated fit.

The fit's coefficients are treated as fixed: no parameter-estimation
uncertainty is added.  The joint matrix [[C, B'],[B, D]] is verified in full,
exactly, before anything is returned.
"""
from fractions import Fraction

from fitting.correlated import read_correlated_fit
from fitting.exact import (matmul, parse_matrix, pseudoinverse,
                           require_rational, require_sequence,
                           symmetric_psd_rank, transpose)

MAX_NEW_ROWS = 8


def _new_rows(rows, width):
    outer = require_sequence(rows, "rows")
    if len(outer) > MAX_NEW_ROWS:
        raise ValueError("at most eight new rows are supported")
    matrix = []
    for entry in outer:
        row = require_sequence(entry, "rows")
        if len(row) != width:
            raise ValueError("new row feature count differs")
        matrix.append([require_rational(value, "feature") for value in row])
    return matrix


def _check_joint(covariance, cross, new_noise):
    joint = [[covariance[i][j] for j in range(len(covariance))] +
             [cross[j][i] for j in range(len(cross))]
             for i in range(len(covariance))]
    joint += [list(cross[i]) + list(new_noise[i]) for i in range(len(cross))]
    is_psd, _ = symmetric_psd_rank(joint)
    if not is_psd:
        raise ValueError("joint measurement covariance is not positive semidefinite")
    return joint


def predict_correlated(store, fit_id, rows, cross, new_noise):
    fit = read_correlated_fit(store, fit_id)
    width = len(fit["feature_order"])
    count = len(fit["scope"]["revision_ids"])
    design = _new_rows(rows, width)
    size = len(design)
    if size == 0:
        if require_sequence(cross, "new-training covariance") or \
                require_sequence(new_noise, "new noise covariance"):
            raise ValueError("empty rows require empty cross and noise matrices")
        return {"mean": [], "covariance": []}
    cross_matrix = parse_matrix(cross, size, count, "new-training covariance")
    noise_matrix = parse_matrix(new_noise, size, size, "new noise covariance")
    covariance = fit["noise"]["covariance"]
    _check_joint(covariance, cross_matrix, noise_matrix)
    inverse, _ = pseudoinverse(covariance)
    residual = fit["residuals"]
    coefficients = fit["coefficients"]
    solve = matmul(cross_matrix, inverse)
    mean = [coefficients[0] + sum(coefficients[j + 1] * row[j]
                                  for j in range(width)) +
            sum(solve[i][k] * residual[k] for k in range(count))
            for i, row in enumerate(design)]
    correction = matmul(solve, transpose(cross_matrix))
    result_covariance = [[noise_matrix[i][j] - correction[i][j]
                          for j in range(size)] for i in range(size)]
    return {"mean": [Fraction(value) for value in mean],
            "covariance": [[Fraction(value) for value in row]
                           for row in result_covariance]}
