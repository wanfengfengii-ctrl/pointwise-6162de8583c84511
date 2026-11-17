import copy
import json
import math

from tests.fixtures import ArchiveCase, square_events, daily_events, prepare


class TestDailyFit(ArchiveCase):
    def test_numeric_fit_handcheck(self):
        events = square_events() + [
            {"row_id": "hold", "device_id": "C", "version": 1,
             "features": {"u": 2, "v": 2}, "target": 16}]
        original = copy.deepcopy(events)
        revisions = self.app.apply_measurements(events)
        cohort = self.app.freeze_cohort([r["revision_id"] for r in revisions], ["u", "v"])
        fit = self.app.fit_numeric(cohort, ["A", "B"], 1)
        self.assertEqual(events, original)
        self.assertEqual(fit["transform"]["means"], [0, 0])
        self.assertEqual(fit["transform"]["scales"], [1, 1])
        for actual, expected in zip(fit["coefficients"], [10, 1, 0.5]):
            self.assertNear(actual, expected)
        for field, expected in [("mse", 1.25), ("penalty", 1.25), ("objective", 2.5)]:
            self.assertNear(fit["diagnostics"][field], expected)
        for row, expected in zip(fit["training_predictions"], [8.5, 9.5, 10.5, 11.5]):
            self.assertNear(row["prediction"], expected)
        package = self.app.evaluate_holdout(fit["fit_id"], ["C"])
        self.assertNear(package["holdout"][0]["prediction"], 13)
        predicted = self.app.predict_numeric(package["package_id"], [[3, 4]])[0]
        self.assertNear(predicted, 15)
        print("POINTWISE_SCENARIO " + json.dumps({
            "case_id": "v1", "input": events, "expected": [10, 1, 0.5, 2.5],
            "actual": fit["coefficients"] + [fit["diagnostics"]["objective"]],
            "downstream_expected": 15, "downstream_actual": predicted,
            "code_paths": ["fitting/features.py", "fitting/objective.py",
                           "fitting/diagnostics.py", "prediction/execution.py"],
            "oracle": "Orthogonal columns: E[u*y]=2,E[v*y]=1; divide by 1+lambda."}))

    def test_training_scope_isolated(self):
        _, cohort, baseline = prepare(self.app)
        changed = daily_events()
        for event in changed[4:]:
            event["version"] = 2
            event["features"]["x"] *= 10000
            event["target"] = -99999
        revisions = self.app.apply_measurements(changed)
        other = self.app.freeze_cohort([r["revision_id"] for r in revisions], ["x", "reference"])
        fit = self.app.fit_numeric(other, ["B", "A"], "1/1")
        for field in ("transform", "coefficients", "diagnostics", "training_predictions"):
            self.assertEqual(baseline[field], fit[field])
        self.assertNear(fit["transform"]["scales"][0], math.sqrt(2.5))
        before = self.state()
        for devices, parameter in [([], 1), (["unknown"], 1), (["A"], 0),
                                   (["A"], -1), (["A"], True), (["A"], "NaN")]:
            with self.assertRaises(ValueError):
                self.app.fit_numeric(cohort, devices, parameter)
            self.assertEqual(before, self.state())
