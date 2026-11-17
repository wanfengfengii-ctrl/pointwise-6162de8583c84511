import json

from tests.fixtures import ArchiveCase, prepare


class TestHoldout(ArchiveCase):
    def test_repeated_device_overlap(self):
        _, _, fit = prepare(self.app)
        self.assertEqual(len(fit["scope"]["revision_ids"]), 4)
        before = self.state()
        for devices in (["A"], ["B", "C"], ["A", "B"], [], ["unknown"]):
            with self.assertRaises(ValueError):
                self.app.evaluate_holdout(fit["fit_id"], devices)
            self.assertEqual(before, self.state())
        package = self.app.evaluate_holdout(fit["fit_id"], ["D"])
        self.assertEqual([r["row_id"] for r in package["holdout"]], ["d0", "d1", "d2"])
        self.assertEqual(package["error"]["n"], 3)

    def test_row_weighted_error(self):
        _, _, fit = prepare(self.app)
        package = self.app.evaluate_holdout(fit["fit_id"], ["D", "C"])
        expected_predictions = [12, 13, 13, 13]
        expected_errors = [4, 9, 9, 9]
        for row, prediction, error in zip(package["holdout"], expected_predictions, expected_errors):
            self.assertNear(row["prediction"], prediction)
            self.assertNear(row["squared_error"], error)
        for field, expected in [("sse", 31), ("n", 4), ("mse", 7.75)]:
            self.assertNear(package["error"][field], expected)
        self.assertNotAlmostEqual(package["error"]["mse"], 6.5)
        self.assertEqual(self.app.read_package(package["package_id"])["package"], package)
        output = self.app.predict_numeric(package["package_id"], [[8, 7]])[0]
        self.assertNear(output, 14)
        print("POINTWISE_SCENARIO " + json.dumps({
            "case_id": "v6", "input": {"training_x": [-2, -1, 1, 2],
                                        "holdout_x": [4, 6, 6, 6], "lambda": 1},
            "expected": {"sse": 31, "n": 4, "mse": 7.75}, "actual": package["error"],
            "downstream_expected": 14, "downstream_actual": output,
            "code_paths": ["prediction/holdout.py", "prediction/errors.py",
                           "prediction/packages.py", "prediction/execution.py"],
            "oracle": "Ridge slope=1/2, intercept=10; SSE=2^2+3*3^2=31."}))
