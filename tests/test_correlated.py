"""Independent accounting for correlated fits and conditional predictions.

Every expectation below is either a hand calculation recorded in
CORRELATED.md or an independent exact-rational recomputation built in this
file; the tests never reuse the production solver as their oracle.
"""
import copy
import itertools
import json
import shutil
import tempfile
from fractions import Fraction as F
from pathlib import Path

from archive.publish import PublicationFailure
from archive.schema import ArchiveIntegrityError
from delivery.app import open_archive
from tests.fixtures import ArchiveCase, daily_events


def fraction_matrix(matrix):
    return [[F(value) for value in row] for row in matrix]


def determinant(matrix):
    size = len(matrix)
    work = [list(row) for row in matrix]
    value = F(1)
    for column in range(size):
        pivot = None
        for index in range(column, size):
            if work[index][column] != 0:
                pivot = index
                break
        if pivot is None:
            return F(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            value = -value
        value *= work[column][column]
        for index in range(column + 1, size):
            factor = work[index][column] / work[column][column]
            for j in range(column, size):
                work[index][j] -= factor * work[column][j]
    return value


def is_psd_by_minors(matrix):
    size = len(matrix)
    for order in range(1, size + 1):
        for rows in itertools.combinations(range(size), order):
            block = [[matrix[i][j] for j in rows] for i in rows]
            if determinant(block) < 0:
                return False
    return True


def inverse(matrix):
    size = len(matrix)
    augmented = [list(row) + [F(1 if i == j else 0) for j in range(size)]
                 for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = next(i for i in range(column, size)
                     if augmented[i][column] != 0)
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for index in range(size):
            if index != column and augmented[index][column] != 0:
                factor = augmented[index][column]
                augmented[index] = [a - factor * b
                                    for a, b in zip(augmented[index],
                                                    augmented[column])]
    return [row[size:] for row in augmented]


def independent_pseudoinverse(matrix):
    """C+ from a range basis: B (B' C B)^-1 B', computed independently."""
    size = len(matrix)
    basis = []

    def gram_of(vectors):
        width = len(vectors)
        return [[sum(vectors[a][i] * matrix[i][j] * vectors[b][j]
                     for i in range(size) for j in range(size))
                 for b in range(width)] for a in range(width)]

    for j in range(size):
        candidate = [matrix[i][j] for i in range(size)]
        trial = basis + [candidate]
        if determinant(gram_of(trial)) != 0:
            basis = trial
    rank = len(basis)
    if rank == 0:
        return [[F(0)] * size for _ in range(size)], 0
    gram_inverse = inverse(gram_of(basis))
    return [[sum(basis[a][i] * gram_inverse[a][b] * basis[b][j]
                 for a in range(rank) for b in range(rank))
             for j in range(size)] for i in range(size)], rank


def independent_fit(design, targets, covariance, penalty):
    """Minimize over the feasible affine set using pivoted RREF, no pseudoinverse.

    Returns the coefficient vector, or None when no residual lies in im(C).
    """
    rows, width = len(targets), len(design[0])
    null = independent_nullspace(covariance)
    constraint = [[sum(null[r][k] * design[k][j] for k in range(rows))
                   for j in range(width)] for r in range(len(null))]
    rhs = [sum(null[r][k] * targets[k] for k in range(rows))
           for r in range(len(null))]
    reduced, pivots = rref(constraint, rhs)
    if any(pivot == width for pivot in pivots):
        return None
    particular = [F(0)] * width
    for index, column in enumerate(pivots):
        particular[column] = reduced[index][width]
    pivot_set = set(pivots)
    free = [j for j in range(width) if j not in pivot_set]
    directions = [[F(0)] * len(free) for _ in range(width)]
    for slot, column in enumerate(free):
        directions[column][slot] = F(1)
        for index, pivot in enumerate(pivots):
            directions[pivot][slot] = -reduced[index][column]
    size, d = len(targets), len(free)
    basis_residual = [targets[i] - sum(particular[j] * design[i][j]
                                       for j in range(width))
                      for i in range(size)]
    generator = [[sum(design[i][j] * directions[j][k] for j in range(width))
                  for k in range(d)] for i in range(size)]
    inverse_covariance, _ = independent_pseudoinverse(covariance)
    weighted = [sum(inverse_covariance[i][j] * basis_residual[j]
                    for j in range(size)) for i in range(size)]
    gradient = [sum(generator[k][i] * weighted[k] for k in range(size)) / size
                for i in range(d)]
    hessian = [[sum(generator[k][i] * sum(inverse_covariance[k][l]
                                          * generator[l][j]
                                          for l in range(size))
                    for k in range(size)) / size for j in range(d)]
               for i in range(d)]
    slope = directions[1:]
    slope_offset = [particular[j] for j in range(1, width)]
    for i in range(d):
        gradient[i] -= penalty * sum(slope[r][i] * slope_offset[r]
                                     for r in range(len(slope)))
        for j in range(d):
            hessian[i][j] += penalty * sum(slope[r][i] * slope[r][j]
                                           for r in range(len(slope)))
    if d:
        augmented = [hessian[i] + [gradient[i]] for i in range(d)]
        solved, solution_pivots = rref(augmented)
        if len(solution_pivots) != d or solution_pivots != list(range(d)):
            raise AssertionError("independent oracle expects a unique optimum")
        steps = [solved[i][d] for i in range(d)]
    else:
        steps = []
    return [particular[j] + sum(directions[j][k] * steps[k] for k in range(d))
            for j in range(width)]


def rref(matrix, extra=None):
    work = [list(row) + ([] if extra is None else [extra[i]])
            for i, row in enumerate(matrix)]
    if not work:
        return [], []
    row_count, column_count = len(work), len(work[0])
    pivots, current = [], 0
    for column in range(column_count):
        pivot = None
        for index in range(current, row_count):
            if work[index][column] != 0:
                pivot = index
                break
        if pivot is None:
            continue
        work[current], work[pivot] = work[pivot], work[current]
        scale = work[current][column]
        work[current] = [value / scale for value in work[current]]
        for index in range(row_count):
            if index != current and work[index][column] != 0:
                factor = work[index][column]
                work[index] = [a - factor * b
                               for a, b in zip(work[index], work[current])]
        pivots.append(column)
        current += 1
        if current == row_count:
            break
    return work, pivots


def independent_nullspace(matrix):
    size = len(matrix)
    reduced, pivots = rref(matrix)
    pivot_set = set(pivots)
    basis = []
    for free in range(size):
        if free in pivot_set:
            continue
        vector = [F(0)] * size
        vector[free] = F(1)
        for index, pivot in enumerate(pivots):
            vector[pivot] = -reduced[index][free]
        basis.append(vector)
    return basis


def events(prefix, xs, ys, device=None):
    device = prefix if device is None else device
    return [{"row_id": prefix + str(i), "device_id": device, "version": 1,
             "features": {"x": x}, "target": y}
            for i, (x, y) in enumerate(zip(xs, ys))]


def reject_without_change(case, operation):
    before = case.app.store.inventory()
    with case.assertRaises(ValueError):
        operation()
    case.assertEqual(case.app.store.inventory(), before)


class TestCorrelatedFit(ArchiveCase):
    def prepare_single(self, xs, ys, device="a", order=("x",)):
        revisions = self.app.apply_measurements(events("r", xs, ys, device))
        cohort = self.app.freeze_cohort(
            [revision["revision_id"] for revision in revisions], order)
        return cohort, [revision["revision_id"] for revision in revisions]

    def test_hand_calculated_shared_and_zero_noise(self):
        cohort, ids = self.prepare_single([-1, 1], [0, 2])
        fit = self.app.fit_correlated(
            cohort, ["a"], 1,
            {"revision_ids": ids, "covariance": [[1, 1], [1, 1]]})
        self.assertEqual(fit["model_kind"], "correlated")
        self.assertEqual(fit["coefficients"], [F(1), F(1)])
        self.assertEqual(fit["residuals"], [F(0), F(0)])
        self.assertEqual(fit["lambda"], F(1))
        self.assertEqual(fit["feature_order"], ["x"])
        self.assertEqual(fit["noise"]["revision_ids"], ids)
        self.assertEqual(fit["scope"]["devices"], ["a"])
        self.assertEqual(fit["scope"]["revision_ids"], ids)
        self.assertEqual(fit["diagnostics"],
                         {"n": 2, "noise_rank": 1, "loss": F(0),
                          "penalty": F(1), "objective": F(1)})
        for value in fit["coefficients"] + fit["residuals"]:
            self.assertIsInstance(value, F)
        for row in fit["noise"]["covariance"]:
            for value in row:
                self.assertIsInstance(value, F)

        zero = self.app.fit_correlated(
            cohort, ["a"], 1,
            {"revision_ids": ids, "covariance": [[0, 0], [0, 0]]})
        self.assertEqual(zero["coefficients"], [F(1), F(1)])
        self.assertEqual(zero["residuals"], [F(0), F(0)])
        self.assertEqual(zero["diagnostics"]["noise_rank"], 0)

    def test_partial_noise_hand_check(self):
        revisions = self.app.apply_measurements(
            events("part", [0, 1, 2], [0, 1, 10], device="part"))
        ids = [revision["revision_id"] for revision in revisions]
        cohort = self.app.freeze_cohort(ids, ["x"])
        fit = self.app.fit_correlated(
            cohort, ["part"], 1,
            {"revision_ids": ids,
             "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 0]]})
        # Support forces r3=0, so b0+2*b1=10; minimizing the two noisy
        # residuals plus the ridge gives b1=29/8, b0=11/4.
        self.assertEqual(fit["coefficients"], [F(11, 4), F(29, 8)])
        self.assertEqual(fit["residuals"], [F(-11, 4), F(-43, 8), F(0)])
        self.assertEqual(fit["diagnostics"],
                         {"n": 3, "noise_rank": 2, "loss": F(2333, 192),
                          "penalty": F(841, 64), "objective": F(607, 24)})
        prediction = self.app.predict_correlated(
            fit["fit_id"], [[5]], [[1, 0, 0]], [[2]])
        self.assertEqual(prediction,
                         {"mean": [F(145, 8)], "covariance": [[F(1)]]})

    def test_diagonal_noise_with_conditioned_covariance(self):
        cohort, ids = self.prepare_single([7, 7], [0, 2], device="d")
        fit = self.app.fit_correlated(
            cohort, ["d"], "1",
            {"revision_ids": ids, "covariance": [[1, 0], [0, 4]]})
        self.assertEqual(fit["coefficients"], [F(2, 5), F(0)])
        self.assertEqual(fit["residuals"], [F(-2, 5), F(8, 5)])
        self.assertEqual(fit["diagnostics"],
                         {"n": 2, "noise_rank": 2, "loss": F(2, 5),
                          "penalty": F(0), "objective": F(2, 5)})
        rows = [[7], [99]]
        cross = [[1, 0], [1, 0]]
        noise = [[1, 1], [1, 1]]
        prediction = self.app.predict_correlated(
            fit["fit_id"], rows, cross, noise)
        self.assertEqual(prediction, {
            "mean": [F(0), F(0)],
            "covariance": [[F(0), F(0)], [F(0), F(0)]]})
        independent = self.app.predict_correlated(
            fit["fit_id"], rows, [[0, 0], [0, 0]], noise)
        self.assertEqual(independent, {
            "mean": [F(2, 5), F(2, 5)],
            "covariance": [[F(1), F(1)], [F(1), F(1)]]})
        for value in prediction["mean"] + sum(prediction["covariance"], []):
            self.assertIsInstance(value, F)
        self.assertEqual(self.app.predict_correlated(fit["fit_id"], [], [], []),
                         {"mean": [], "covariance": []})

    def test_infeasible_support_is_rejected(self):
        cohort, ids = self.prepare_single([7, 7], [0, 2], device="c")
        before = self.app.store.inventory()
        with self.assertRaises(ValueError):
            self.app.fit_correlated(
                cohort, ["c"], 1,
                {"revision_ids": ids, "covariance": [[1, 1], [1, 1]]})
        self.assertEqual(self.app.store.inventory(), before)
        self.assertEqual(self.app.read_cohort(cohort)["rows"][0]["target"], "0")

    def test_noise_validation_and_exact_types(self):
        cohort, ids = self.prepare_single([-1, 1], [0, 2], device="n")
        good = {"revision_ids": ids, "covariance": [[2, 1], [1, 1]]}
        fit = self.app.fit_correlated(cohort, ["n"], "1/3", good)
        self.assertEqual(fit["lambda"], F(1, 3))
        for bad in (
                {"revision_ids": ids[:1], "covariance": [[1]]},
                {"revision_ids": [ids[0], ids[0]],
                 "covariance": [[1, 1], [1, 1]]},
                {"revision_ids": ids + ["foreign"],
                 "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                {"revision_ids": ids, "covariance": [[1, 2], [2, 1]]},
                {"revision_ids": ids, "covariance": [[0, 1], [1, 0]]},
                {"revision_ids": ids, "covariance": [[1, 0], [1, 1]]},
                {"revision_ids": ids, "covariance": [[1.0, 0], [0, 1]]},
                {"revision_ids": ids, "covariance": [[True, 0], [0, 1]]},
                {"revision_ids": ids, "covariance": [[1, 0], [0, 1]],
                 "extra": 1},
                {"covariance": [[1, 0], [0, 1]]}):
            reject_without_change(
                self, lambda bad=bad: self.app.fit_correlated(
                    cohort, ["n"], 1, bad))
        for bad_penalty in (0, -1, 1.5, True, "banana"):
            reject_without_change(
                self, lambda value=bad_penalty: self.app.fit_correlated(
                    cohort, ["n"], value, good))
        for bad_devices in ([], ["unknown"]):
            reject_without_change(
                self, lambda value=bad_devices: self.app.fit_correlated(
                    cohort, value, 1, good))

    def test_axis_reordering_and_replay_share_one_identity(self):
        cohort, ids = self.prepare_single([1, 2, 4], [1, 2, 3], device="s")
        covariance = [[2, 0, 1], [0, 3, 0], [1, 0, 2]]
        fit = self.app.fit_correlated(
            cohort, ["s"], 2,
            {"revision_ids": ids, "covariance": covariance})
        before = self.app.store.inventory()
        order = [2, 0, 1]
        permuted = [[covariance[i][j] for j in order] for i in order]
        reordered = self.app.fit_correlated(
            cohort, ["s"], F(2),
            {"revision_ids": [ids[i] for i in order],
             "covariance": permuted})
        self.assertEqual(reordered["fit_id"], fit["fit_id"])
        self.assertEqual(reordered["noise"]["revision_ids"], ids)
        self.assertEqual(reordered["noise"]["covariance"],
                         fraction_matrix(covariance))
        self.assertEqual(self.app.store.inventory(), before)
        self.assertEqual(self.app.fit_correlated(
            cohort, ["s"], 2,
            {"revision_ids": ids, "covariance": covariance}), fit)
        self.assertEqual(self.app.store.inventory(), before)

    def test_row_and_feature_limits(self):
        xs = [i % 5 for i in range(24)]
        ys = [i for i in range(24)]
        cohort, ids = self.prepare_single(xs, ys, device="big")
        covariance = [[1 if i == j else 0 for j in range(24)] for i in range(24)]
        fit = self.app.fit_correlated(
            cohort, ["big"], 1,
            {"revision_ids": ids, "covariance": covariance})
        self.assertEqual(fit["diagnostics"]["n"], 24)
        self.assertEqual(fit["diagnostics"]["noise_rank"], 24)
        revisions = self.app.apply_measurements(
            events("too", [i % 5 for i in range(25)],
                   [i for i in range(25)], device="too"))
        too_cohort = self.app.freeze_cohort(
            [revision["revision_id"] for revision in revisions], ["x"])
        too_ids = [revision["revision_id"] for revision in revisions]
        reject_without_change(
            self, lambda: self.app.fit_correlated(
                too_cohort, ["too"], 1,
                {"revision_ids": too_ids,
                 "covariance": [[1 if i == j else 0 for j in range(25)]
                                for i in range(25)]}))
        features = self.app.apply_measurements([
            {"row_id": "f" + str(i), "device_id": "f", "version": 1,
             "features": {"a": i, "b": i, "c": i, "d": i, "e": i},
             "target": i} for i in range(2)])
        feature_cohort = self.app.freeze_cohort(
            [row["revision_id"] for row in features],
            ["a", "b", "c", "d", "e"])
        reject_without_change(
            self, lambda: self.app.fit_correlated(
                feature_cohort, ["f"], 1,
                {"revision_ids": [row["revision_id"] for row in features],
                 "covariance": [[1, 0], [0, 1]]}))

    def test_device_subset_counts_only_selected_rows(self):
        batch = (events("left", [i for i in range(24)], [i for i in range(24)],
                        device="L") +
                 events("right", [i for i in range(10)], [i for i in range(10)],
                        device="R"))
        revisions = self.app.apply_measurements(batch)
        cohort = self.app.freeze_cohort(
            [revision["revision_id"] for revision in revisions], ["x"])
        selected = [revision for revision in revisions
                    if revision["device_id"] == "L"]
        identity_list = [revision["revision_id"] for revision in selected]
        fit = self.app.fit_correlated(
            cohort, ["L"], 1,
            {"revision_ids": identity_list,
             "covariance": [[1 if i == j else 0 for j in range(24)]
                            for i in range(24)]})
        self.assertEqual(fit["diagnostics"]["n"], 24)
        self.assertEqual(fit["scope"]["devices"], ["L"])
        self.assertEqual(fit["scope"]["revision_ids"], identity_list)

    def test_solver_matches_independent_oracle(self):
        designs = [
            ([[F(1), F(0)], [F(1), F(2)]], [F(3), F(7)],
             [[F(1), F(0)], [F(0), F(1)]], F(1)),
            ([[F(1), F(1)], [F(1), F(3)], [F(1), F(-2)]],
             [F(2), F(0), F(5)],
             [[F(2), F(1), F(0)], [F(1), F(2), F(0)], [F(0), F(0), F(0)]],
             F(1, 2)),
            ([[F(1), F(1)], [F(1), F(1)], [F(1), F(3)]],
             [F(4), F(6), F(2)],
             [[F(1), F(1), F(0)], [F(1), F(1), F(0)], [F(0), F(0), F(1)]],
             F(2)),
        ]
        for design, targets, covariance, penalty in designs:
            expected = independent_fit(design, targets, covariance, penalty)
            with open_archive() as app:
                revisions = app.apply_measurements([
                    {"row_id": "oracle" + str(i), "device_id": "o", "version": 1,
                     "features": {"x": design[i][1]}, "target": targets[i]}
                    for i in range(len(targets))])
                cohort = app.freeze_cohort(
                    [revision["revision_id"] for revision in revisions], ["x"])
                try:
                    fit = app.fit_correlated(
                        cohort, ["o"], penalty,
                        {"revision_ids": [revision["revision_id"]
                                          for revision in revisions],
                         "covariance": covariance})
                except ValueError:
                    self.assertIsNone(expected)
                    continue
                self.assertEqual(fit["coefficients"], expected)

    def test_correlated_and_v1_coexist(self):
        revisions = self.app.apply_measurements(events("mix", [-1, 0, 1], [0, 1, 2]))
        ids = [revision["revision_id"] for revision in revisions]
        cohort = self.app.freeze_cohort(ids, ["x"])
        v1 = self.app.fit_numeric(cohort, ["mix"], 1)
        correlated = self.app.fit_correlated(
            cohort, ["mix"], 1,
            {"revision_ids": ids,
             "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        before = self.app.store.inventory()
        self.assertEqual(self.app.read_fit(v1["fit_id"])["fit_id"],
                         v1["fit_id"])
        self.assertEqual(self.app.read_fit(correlated["fit_id"])
                         ["model_kind"], "correlated")
        self.assertEqual(self.app.predict_correlated(
            correlated["fit_id"], [[2]], [[0, 0, 0]], [[1]])["mean"],
            [F(9, 5)])
        with self.assertRaises(ValueError):
            self.app.evaluate_holdout(correlated["fit_id"], ["mix"])
        self.assertEqual(self.app.store.inventory(), before)
        reloaded = self.app.fit_numeric(cohort, ["mix"], 1)
        self.assertEqual(reloaded, v1)

    def test_cross_kind_rejections(self):
        revisions = self.app.apply_measurements(events("v", [-1, 1], [0, 2]))
        cohort = self.app.freeze_cohort(
            [revision["revision_id"] for revision in revisions], ["x"])
        v1_fit = self.app.fit_numeric(cohort, ["v"], 1)
        correlated = self.app.fit_correlated(
            cohort, ["v"], 1,
            {"revision_ids": [revision["revision_id"] for revision in revisions],
             "covariance": [[1, 1], [1, 1]]})
        reject_without_change(
            self, lambda: self.app.predict_correlated(
                v1_fit["fit_id"], [[0]], [[1, 0]], [[1]]))
        reject_without_change(
            self, lambda: self.app.export_correlated_fit(
                v1_fit["fit_id"], "/tmp/unused-correlated.json"))
        reject_without_change(
            self, lambda: self.app.evaluate_holdout(correlated["fit_id"], ["v"]))
        reject_without_change(
            self, lambda: self.app.fit_correlated(
                cohort, ["v"], 1,
                {"revision_ids": ["unknown", "other"],
                 "covariance": [[1, 0], [0, 1]]}))
        self.assertEqual(self.app.read_fit(v1_fit["fit_id"])["fit_id"],
                         v1_fit["fit_id"])

    def test_prediction_verifies_full_joint_covariance(self):
        cohort, ids = self.prepare_single([-1, 1], [0, 2], device="j")
        fit = self.app.fit_correlated(
            cohort, ["j"], 1,
            {"revision_ids": ids, "covariance": [[1, 1], [1, 1]]})
        before = self.app.store.inventory()
        # Formal Schur complement is 3/4 > 0, but the joint matrix is not PSD.
        covariance = fraction_matrix([[1, 1], [1, 1]])
        cross = fraction_matrix([[1, 0]])
        new_noise = fraction_matrix([[1]])
        joint = [[covariance[i][j] for j in range(2)] +
                 [cross[j][i] for j in range(1)] for i in range(2)]
        joint += [list(cross[0]) + list(new_noise[0])]
        self.assertFalse(is_psd_by_minors(joint))
        inverse_covariance, _ = independent_pseudoinverse(covariance)
        schur = new_noise[0][0] - sum(
            cross[0][k] * inverse_covariance[k][l] * cross[0][l]
            for k in range(2) for l in range(2))
        self.assertEqual(schur, F(3, 4))
        with self.assertRaises(ValueError):
            self.app.predict_correlated(
                fit["fit_id"], [[0]], [[1, 0]], [[1]])
        self.assertEqual(self.app.store.inventory(), before)
        for bad_cross, bad_noise in (
                ([[1, 0]], [[0]]),
                ([[1, 0]], [[1, 1]]),
                ([[1]], [[1]]),
                ([[1, 0], [0, 1]], [[1, 0]]),
                ([[1, 0], [0, 1]], [[1, 2], [2, 1]])):
            with self.assertRaises(ValueError):
                self.app.predict_correlated(
                    fit["fit_id"], [[0]] * len(bad_cross), bad_cross, bad_noise)
        with self.assertRaises(ValueError):
            self.app.predict_correlated(fit["fit_id"], [[0]] * 9,
                                        [[0, 0]] * 9,
                                        [[1 if i == j else 0 for j in range(9)]
                                         for i in range(9)])
        with self.assertRaises(ValueError):
            self.app.predict_correlated(fit["fit_id"], [[1.0]], [[1, 0]], [[1]])
        with self.assertRaises(ValueError):
            self.app.predict_correlated("unknown", [], [], [])
        with self.assertRaises(ValueError):
            self.app.predict_correlated(fit["fit_id"], [], [[0, 0]], [])

    def test_seeded_fuzz_against_independent_oracle(self):
        import random
        generator = random.Random(20261117)
        compared = 0
        for _ in range(40):
            count = generator.randint(1, 5)
            design = [[F(1), F(generator.randint(-2, 2)),
                       F(generator.randint(-1, 2))] for _ in range(count)]
            targets = [F(generator.randint(-3, 3), generator.randint(1, 2))
                       for _ in range(count)]
            penalty = F(generator.randint(1, 3), generator.randint(1, 2))
            rank = generator.randint(0, count)
            factor = [[F(generator.randint(-2, 2)) for _ in range(rank)]
                      for _ in range(count)]
            covariance = [[sum(factor[i][k] * factor[j][k]
                               for k in range(rank))
                           for j in range(count)] for i in range(count)]
            expected = independent_fit(design, targets, covariance, penalty)
            with open_archive() as app:
                revisions = app.apply_measurements([
                    {"row_id": "fuzz" + str(i), "device_id": "z", "version": 1,
                     "features": {"x": design[i][1], "z": design[i][2]},
                     "target": targets[i]} for i in range(count)])
                cohort = app.freeze_cohort(
                    [revision["revision_id"] for revision in revisions],
                    ["x", "z"])
                arguments = {"revision_ids": [revision["revision_id"]
                                              for revision in revisions],
                             "covariance": covariance}
                if expected is None:
                    with self.assertRaises(ValueError):
                        app.fit_correlated(cohort, ["z"], penalty, arguments)
                    continue
                fit = app.fit_correlated(cohort, ["z"], penalty, arguments)
                self.assertEqual(fit["coefficients"], expected)
                residual = [targets[i] - sum(expected[j] * design[i][j]
                                             for j in range(3))
                            for i in range(count)]
                self.assertEqual(fit["residuals"], residual)
                self.assertEqual(fit["diagnostics"]["noise_rank"],
                                 count - len(independent_nullspace(
                                     covariance)))
                compared += 1
        self.assertGreater(compared, 20)

    def test_seeded_joint_covariance_acceptance(self):
        import random
        generator = random.Random(424242)
        cohort, ids = self.prepare_single([-1, 0, 1], [-1, 0, 1], device="g")
        fit = self.app.fit_correlated(
            cohort, ["g"], 1,
            {"revision_ids": ids,
             "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        accepted = 0
        for _ in range(30):
            size = generator.randint(0, 3)
            rows = [[F(generator.randint(-2, 2))] for _ in range(size)]
            cross = [[F(generator.randint(-1, 1)) for _ in range(3)]
                     for _ in range(size)]
            noise = [[F(generator.randint(-1, 2)) for _ in range(size)]
                     for _ in range(size)]
            for i in range(size):
                for j in range(i + 1, size):
                    noise[j][i] = noise[i][j]
            joint = [[F(1 if i == j else 0) for j in range(3)] +
                     [cross[j][i] for j in range(size)] for i in range(3)]
            joint += [list(cross[i]) + list(noise[i]) for i in range(size)]
            expected_ok = is_psd_by_minors(joint)
            try:
                result = self.app.predict_correlated(
                    fit["fit_id"], rows, cross, noise)
            except ValueError:
                self.assertFalse(expected_ok, joint)
                continue
            self.assertTrue(expected_ok, joint)
            self.assertEqual(len(result["mean"]), size)
            accepted += 1
        self.assertGreater(accepted, 5)

    def test_conditional_covariance_keeps_cross_terms(self):
        cohort, ids = self.prepare_single([-1, 1], [0, 2], device="k")
        fit = self.app.fit_correlated(
            cohort, ["k"], 1,
            {"revision_ids": ids, "covariance": [[1, 0], [0, 1]]})
        # C is the identity, so B = I lies in the noise range; picking
        # D = [[2,1],[1,2]] makes the full joint matrix PSD.
        self.assertEqual(fit["coefficients"], [F(1), F(1, 2)])
        self.assertEqual(fit["residuals"], [F(-1, 2), F(1, 2)])
        prediction = self.app.predict_correlated(
            fit["fit_id"], [[-1], [3]], [[1, 0], [0, 1]],
            [[2, 1], [1, 2]])
        self.assertEqual(prediction["mean"], [F(0), F(3)])
        self.assertEqual(prediction["covariance"], [[F(1), F(1)],
                                                   [F(1), F(1)]])
        self.assertEqual(prediction["covariance"][0][1], F(1))
        # Prediction consumes the fixed fit and creates no new artifact.
        before = self.app.store.inventory()
        repeated = self.app.predict_correlated(
            fit["fit_id"], [[-1], [3]], [[1, 0], [0, 1]], [[2, 1], [1, 2]])
        self.assertEqual(repeated, prediction)
        self.assertEqual(self.app.store.inventory(), before)
        self.assertEqual(self.app.read_fit(fit["fit_id"])["noise"]["covariance"],
                         [[F(1), F(0)], [F(0), F(1)]])

    def test_export_read_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, target = root / "archive.sqlite", root / "review.json"
            with open_archive(database) as app:
                revisions = app.apply_measurements(events("e", [7, 7], [0, 2],
                                                          device="e"))
                ids = [revision["revision_id"] for revision in revisions]
                cohort = app.freeze_cohort(ids, ["x"])
                noise = {"revision_ids": ids,
                         "covariance": [[1, 0], [0, 4]]}
                fit = app.fit_correlated(cohort, ["e"], 1, noise)
                identity = fit["fit_id"]
                frozen = copy.deepcopy(app.read_fit(identity))
                exported = app.export_correlated_fit(identity, target)
                self.assertEqual(exported["format"], "correlated-review-v1")
                self.assertEqual(exported["fit"]["coefficients"], ["2/5", "0"])
                self.assertEqual(exported["fit"]["residuals"],
                                 ["-2/5", "8/5"])
                self.assertEqual(exported["fit"]["lambda"], "1")
                self.assertEqual(exported["fit"]["diagnostics"]["n"], 2)
                self.assertEqual(exported["fit"]["diagnostics"]["loss"], "2/5")
                self.assertEqual(exported["cohort"],
                                 app.read_cohort(cohort))
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                                 exported)
                stored_rows = {row["revision_id"]: row
                               for row in exported["cohort"]["rows"]}
                order = exported["fit"]["scope"]["revision_ids"]
                self.assertEqual(
                    [stored_rows[i]["features"]["x"] for i in order],
                    ["7", "7"])
                self.assertEqual([stored_rows[i]["target"] for i in order],
                                 ["0", "2"])
                # Caller edits and later source revisions must not leak in.
                noise["covariance"][0][0] = 999
                fit["coefficients"][0] = -999
                fit["noise"]["covariance"].clear()
                app.apply_measurements([
                    {"row_id": "e0", "device_id": "e", "version": 2,
                     "features": {"x": 700}, "target": 900}])
                app.apply_measurements([
                    {"row_id": "e1", "version": 2, "op": "withdraw"}])
                self.assertEqual(app.read_fit(identity), frozen)
                prediction = app.predict_correlated(
                    identity, [[7]], [[1, 0]], [[1]])
                self.assertEqual(prediction["mean"], [F(0)])
                self.assertEqual(prediction["covariance"], [[F(0)]])
                prediction["mean"][0] = 999
            with open_archive(database) as app:
                self.assertEqual(app.read_fit(identity), frozen)
                self.assertEqual(app.export_correlated_fit(identity, target),
                                 exported)
                self.assertEqual(app.predict_correlated(
                    identity, [[7]], [[1, 0]], [[1]])["mean"], [F(0)])

    def test_export_failure_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            healthy = root / "healthy.sqlite"
            with open_archive(healthy) as app:
                revisions = app.apply_measurements(events("h", [-1, 1], [0, 2]))
                cohort = app.freeze_cohort(
                    [revision["revision_id"] for revision in revisions], ["x"])
                fit = app.fit_correlated(
                    cohort, ["h"], 1,
                    {"revision_ids": [revision["revision_id"]
                                      for revision in revisions],
                     "covariance": [[1, 1], [1, 1]]})
                target = root / "existing_directory"
                target.mkdir()
                before = app.store.inventory()
                with self.assertRaises(OSError):
                    app.export_correlated_fit(fit["fit_id"], target)
                self.assertEqual(app.store.inventory(), before)
                self.assertEqual(list(root.glob(".existing_directory.*.tmp")),
                                 [])
            for mode in ("missing_parent", "wrong_type", "missing_child"):
                broken = root / (mode + ".sqlite")
                shutil.copyfile(healthy, broken)
                with open_archive(broken) as app:
                    connection = app.store.connection
                    if mode == "missing_parent":
                        connection.execute("PRAGMA foreign_keys=OFF")
                        connection.execute(
                            "DELETE FROM artifacts WHERE artifact_id=?",
                            (cohort,))
                    elif mode == "wrong_type":
                        connection.execute(
                            "UPDATE artifact_links SET parent_id=? "
                            "WHERE child_id=? AND role='cohort'",
                            (fit["fit_id"], fit["fit_id"]))
                    else:
                        connection.execute(
                            "DELETE FROM artifact_children WHERE artifact_id=? "
                            "AND ordinal=0", (fit["fit_id"],))
                    with self.assertRaises(ArchiveIntegrityError):
                        app.read_fit(fit["fit_id"])
                    output = root / (mode + ".json")
                    with self.assertRaises(ArchiveIntegrityError):
                        app.export_correlated_fit(fit["fit_id"], output)
                    self.assertFalse(output.exists())

    def test_publication_rollback_and_retry(self):
        for stage in ("after_header", "after_children", "before_commit"):
            with self.subTest(stage=stage), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "archive.sqlite"
                with open_archive(path) as app:
                    revisions = app.apply_measurements(
                        events("p", [7, 7], [0, 2], device="p"))
                    ids = [revision["revision_id"] for revision in revisions]
                    cohort = app.freeze_cohort(ids, ["x"])
                    noise = {"revision_ids": ids,
                             "covariance": [[1, 0], [0, 4]]}

                    def fail(actual_stage, artifact_id, kind, stage=stage):
                        if kind == "fit" and actual_stage == stage:
                            raise PublicationFailure(stage)

                    before = app.store.inventory()
                    app.publication_hook = fail
                    with self.assertRaises(PublicationFailure):
                        app.fit_correlated(cohort, ["p"], 2, noise)
                    self.assertEqual(app.store.inventory(), before)
                with open_archive(path) as app:
                    self.assertEqual(app.store.inventory(), before)
                    fit = app.fit_correlated(cohort, ["p"], 2, noise)
                    success = app.store.inventory()
                    self.assertEqual(app.fit_correlated(cohort, ["p"], 2, noise),
                                     fit)
                    self.assertEqual(app.store.inventory(), success)
                    self.assertEqual(app.store.connection.execute(
                        "PRAGMA foreign_key_check").fetchall(), [])


class TestVBoundaries(ArchiveCase):
    def test_daily_v1_still_works(self):
        revisions = self.app.apply_measurements(daily_events())
        cohort = self.app.freeze_cohort(
            [revision["revision_id"] for revision in revisions],
            ["x", "reference"])
        fit = self.app.fit_numeric(cohort, {"A", "B"}, 1)
        self.assertNotIn("model_kind", fit)
        package = self.app.evaluate_holdout(fit["fit_id"], {"C", "D"})
        values = self.app.predict_numeric(package["package_id"], [[8, 7]])
        self.assertNear(values[0], 14)
        self.assertEqual(package["error"]["sse"], 31.0)
