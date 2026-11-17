import math

from measurements.validation import require_finite_number


class NumericSolveError(ValueError):
    pass


def solve_linear_system(matrix, rhs):
    try:
        size = len(rhs)
        if not size or len(matrix) != size or any(len(row) != size for row in matrix):
            raise NumericSolveError("linear system dimensions differ")
        original = [[require_finite_number(value) for value in row] for row in matrix]
        target = [require_finite_number(value) for value in rhs]
        augmented = [row[:] + [value] for row, value in zip(original, target)]
        for column in range(size):
            pivot = max(range(column, size), key=lambda i: abs(augmented[i][column]))
            if augmented[pivot][column] == 0:
                raise NumericSolveError("singular linear system")
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            for i in range(column + 1, size):
                factor = augmented[i][column] / augmented[column][column]
                augmented[i][column] = 0.0
                for j in range(column + 1, size + 1):
                    augmented[i][j] -= factor * augmented[column][j]
        solution = [0.0] * size
        for i in range(size - 1, -1, -1):
            solution[i] = (augmented[i][-1] - math.fsum(
                augmented[i][j] * solution[j] for j in range(i + 1, size))) / augmented[i][i]
        if any(not math.isfinite(value) for value in solution):
            raise NumericSolveError("non-finite linear solution")
        for row, value in zip(original, target):
            residual = abs(math.fsum(a * b for a, b in zip(row, solution)) - value)
            scale = max(1.0, abs(value), math.fsum(
                abs(a * b) for a, b in zip(row, solution)))
            if not math.isfinite(residual) or residual > 1e-10 * scale:
                raise NumericSolveError("linear residual exceeds tolerance")
        return solution
    except (ArithmeticError, TypeError, ValueError) as exc:
        if isinstance(exc, NumericSolveError):
            raise
        raise NumericSolveError("linear solve failed") from exc
