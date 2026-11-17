import math

from fitting.solver import NumericSolveError, solve_linear_system
from fitting.features import transform_numeric_rows
from tests.fixtures import ArchiveCase, prepare


def failed_solver(matrix, rhs):
    raise NumericSolveError("deterministic solver failure fixture")


class TestFitting(ArchiveCase):
    def test_constant_feature_and_solver_failure(self):
        _, cohort, fit = prepare(self.app)
        transform = fit["transform"]
        self.assertEqual(transform["constants"], [False, True])
        self.assertEqual(transform["scales"][1], 1)
        self.assertNear(fit["coefficients"][0], 10)
        self.assertNear(fit["coefficients"][1], math.sqrt(2.5) / 2)
        self.assertNear(fit["coefficients"][2], 0)
        self.assertEqual(transform_numeric_rows([[0, 999999]], transform), [[0, 0]])
        for field, expected in [("mse", 0.625), ("penalty", 0.625), ("objective", 1.25)]:
            self.assertNear(fit["diagnostics"][field], expected)
        before = self.state()
        with self.assertRaises(NumericSolveError):
            self.app.fit_numeric(cohort, ["A", "B"], 1, solver=failed_solver)
        self.assertEqual(before, self.state())
        self.assertEqual(self.app.read_fit(fit["fit_id"]), fit)

    def test_numeric_overflow_and_real_singular_system(self):
        with self.assertRaises(NumericSolveError):
            solve_linear_system([[1, 1], [1, 1]], [1, 1])
        self.assertEqual(solve_linear_system([[0, 1], [2, 1]], [3, 5]), [1, 3])
        revisions = self.app.apply_measurements([
            {"row_id": "huge" + str(i), "device_id": str(i), "version": 1,
             "features": {"x": value}, "target": 1}
            for i, value in enumerate([-1e308, 1e308])])
        cohort = self.app.freeze_cohort([r["revision_id"] for r in revisions], ["x"])
        before = self.state()
        with self.assertRaises(ValueError):
            self.app.fit_numeric(cohort, ["0", "1"], 1)
        self.assertEqual(before, self.state())
        with self.assertRaises(ValueError):
            self.app.apply_measurements([
                {"row_id": "unrepresentable", "version": 1, "device_id": "X",
                 "features": {"x": "1e10000"}, "target": 1}])
        self.assertEqual(before, self.state())
