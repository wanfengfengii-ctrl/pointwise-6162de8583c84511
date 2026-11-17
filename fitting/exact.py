"""Exact rational linear algebra used by correlated fitting.

No tolerances, no floating point and no eigenvalue truncation: every
decision here is made on Fraction values.
"""
from collections.abc import Mapping
from fractions import Fraction


def require_rational(value, label="value"):
    """Accept only non-bool int, Fraction or an exactly rational string."""
    if isinstance(value, bool):
        raise ValueError(label + " must be a rational number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            raise ValueError(label + " must be a rational number") from exc
    raise ValueError(label + " must be a rational number")


def require_sequence(value, label="value"):
    if isinstance(value, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(label + " must be an ordered sequence")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(label + " must be an ordered sequence") from exc


def parse_matrix(value, rows, columns, label="matrix"):
    outer = require_sequence(value, label)
    if len(outer) != rows:
        raise ValueError(label + " row count differs")
    matrix = []
    for entry in outer:
        row = require_sequence(entry, label + " row")
        if len(row) != columns:
            raise ValueError(label + " column count differs")
        matrix.append([require_rational(item, label) for item in row])
    return matrix


def zeros(rows, columns):
    return [[Fraction(0)] * columns for _ in range(rows)]


def transpose(matrix):
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    if not left or not right:
        return []
    columns = len(right[0])
    return [[sum(left_row[k] * right[k][j] for k in range(len(right)))
             for j in range(columns)] for left_row in left]


def matvec(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector)))
            for i in range(len(matrix))]


def quadratic_form(vector, matrix):
    size = len(vector)
    return sum(vector[i] * matrix[i][j] * vector[j]
               for i in range(size) for j in range(size))


def rref(matrix):
    """Reduced row echelon form over Fraction; returns (rows, pivot columns)."""
    if not matrix:
        return [], []
    work = [list(row) for row in matrix]
    row_count = len(work)
    column_count = len(work[0])
    pivots = []
    current = 0
    for column in range(column_count):
        pivot = None
        for index in range(current, row_count):
            if work[index][column] != 0:
                pivot = index
                break
        if pivot is None:
            continue
        work[current], work[pivot] = work[pivot], work[current]
        value = work[current][column]
        work[current] = [item / value for item in work[current]]
        for index in range(row_count):
            if index != current and work[index][column] != 0:
                factor = work[index][column]
                work[index] = [item - factor * base
                               for item, base in zip(work[index], work[current])]
        pivots.append(column)
        current += 1
        if current == row_count:
            break
    return work, pivots


def nullspace(matrix, columns=None):
    if matrix:
        reduced, pivots = rref(matrix)
        if columns is None:
            columns = len(reduced[0]) if reduced else 0
    else:
        reduced, pivots = [], []
        if columns is None:
            columns = 0
    pivot_set = set(pivots)
    basis = []
    for free in range(columns):
        if free in pivot_set:
            continue
        vector = [Fraction(0)] * columns
        vector[free] = Fraction(1)
        for row_index, pivot_column in enumerate(pivots):
            vector[pivot_column] = -reduced[row_index][free]
        basis.append(vector)
    return basis


def solve_unique(matrix, rhs):
    """Solve a square nonsingular system exactly, or raise ValueError."""
    size = len(rhs)
    if size == 0:
        if matrix:
            raise ValueError("linear system dimensions differ")
        return []
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("linear system dimensions differ")
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    reduced, pivots = rref(augmented)
    if any(pivot == size for pivot in pivots) or len(pivots) < size:
        raise ValueError("linear system has no unique exact solution")
    solution = [Fraction(0)] * size
    for row_index, pivot_column in enumerate(pivots):
        solution[pivot_column] = reduced[row_index][size]
    return solution


def inverse(matrix):
    size = len(matrix)
    if not size or any(len(row) != size for row in matrix):
        raise ValueError("inverse dimensions differ")
    augmented = [list(row) + [Fraction(1 if i == j else 0) for j in range(size)]
                 for i, row in enumerate(matrix)]
    reduced, pivots = rref(augmented)
    if pivots[:size] != list(range(size)):
        raise ValueError("matrix is singular")
    return [row[size:] for row in reduced]


def affine_solve(matrix, rhs, columns):
    """Solve matrix * x = rhs over Fraction.

    Returns (particular, nullspace basis).  Raises ValueError when the
    system is inconsistent; free variables of the particular solution are 0.
    """
    if len(matrix) != len(rhs):
        raise ValueError("constraint dimensions differ")
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    if augmented:
        reduced, pivots = rref(augmented)
        if any(pivot == columns for pivot in pivots):
            raise ValueError("training data is infeasible for the noise support")
    else:
        reduced, pivots = [], []
    particular = [Fraction(0)] * columns
    for row_index, pivot_column in enumerate(pivots):
        if pivot_column < columns:
            particular[pivot_column] = reduced[row_index][columns]
    return particular, nullspace(matrix, columns)


def symmetric_psd_rank(matrix):
    """Exact PSD test by symmetric Schur elimination; returns (is_psd, rank).

    A symmetric matrix is PSD iff a symmetric elimination on positive pivots
    leaves a zero block, and every pivot/diagonal met on the way is
    nonnegative.  No tolerance or eigenvalue truncation is involved.
    """
    size = len(matrix)
    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i][j] != matrix[j][i]:
                return False, None
    work = [list(row) for row in matrix]
    active = list(range(size))
    rank = 0
    while active:
        pivot = None
        for index in active:
            if work[index][index] < 0:
                return False, None
            if work[index][index] > 0 and pivot is None:
                pivot = index
        if pivot is None:
            for i in active:
                for j in active:
                    if work[i][j] != 0:
                        return False, None
            return True, rank
        rank += 1
        denominator = work[pivot][pivot]
        rest = [index for index in active if index != pivot]
        for i in rest:
            for j in rest:
                work[i][j] -= work[i][pivot] * work[pivot][j] / denominator
        active = rest
    return True, rank


def pseudoinverse(matrix):
    """Moore-Penrose inverse of a rational symmetric PSD matrix."""
    size = len(matrix)
    _, pivots = rref(matrix)
    rank = len(pivots)
    if rank == 0:
        return zeros(size, size), 0
    basis = [[matrix[i][column] for column in pivots] for i in range(size)]
    gram = matmul(transpose(basis), matmul(matrix, basis))
    return matmul(basis, matmul(inverse(gram), transpose(basis))), rank
