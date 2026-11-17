import copy
import unittest

from delivery.app import open_archive


def daily_events():
    specifications = [
        ("a0", "A", -2, 8), ("a1", "A", -1, 9),
        ("b0", "B", 1, 11), ("b1", "B", 2, 12),
        ("c0", "C", 4, 14), ("d0", "D", 6, 16),
        ("d1", "D", 6, 16), ("d2", "D", 6, 16),
    ]
    return [{"row_id": row, "device_id": device, "version": 1,
             "features": {"x": x, "reference": 7}, "target": target}
            for row, device, x, target in specifications]


def square_events():
    return [{"row_id": "s" + str(i), "device_id": "A" if i < 2 else "B",
             "version": 1, "features": {"u": u, "v": v}, "target": y}
            for i, (u, v, y) in enumerate(
                [(-1, -1, 7), (-1, 1, 9), (1, -1, 11), (1, 1, 13)])]


def prepare(app, order=("x", "reference")):
    revisions = app.apply_measurements(daily_events())
    cohort = app.freeze_cohort([r["revision_id"] for r in revisions], order)
    fit = app.fit_numeric(cohort, {"A", "B"}, 1)
    return revisions, cohort, fit


class ArchiveCase(unittest.TestCase):
    def setUp(self):
        self.app = open_archive()

    def tearDown(self):
        self.app.close()

    def assertNear(self, actual, expected):
        self.assertAlmostEqual(actual, expected, delta=1e-8)

    def state(self):
        return copy.deepcopy((self.app.store.inventory(),
                              self.app.store.load_revisions(), self.app.store.load_heads()))
