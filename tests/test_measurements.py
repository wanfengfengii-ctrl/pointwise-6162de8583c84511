import copy
from fractions import Fraction

from tests.fixtures import ArchiveCase, daily_events


class TestMeasurements(ArchiveCase):
    def test_conflicting_batch_rollback(self):
        events = daily_events()
        self.app.apply_measurements(events)
        before = self.state()
        correction = copy.deepcopy(events[0])
        correction.update(version=2, target=100)
        conflict = copy.deepcopy(events[1])
        conflict["target"] = 101
        with self.assertRaises(ValueError):
            self.app.apply_measurements([correction, conflict])
        self.assertEqual(before, self.state())
        valid = self.app.apply_measurements([correction])
        self.assertEqual(valid[0]["version"], 2)

    def test_historical_replay_and_exact_content(self):
        event = daily_events()[0]
        first = self.app.apply_measurements([event])[0]
        second_event = dict(event, version=3, target=100)
        second = self.app.apply_measurements([second_event])[0]
        count = self.app.store.inventory()
        self.assertEqual(self.app.apply_measurements([event])[0], first)
        self.assertEqual(count, self.app.store.inventory())
        self.assertEqual(self.app.store.load_heads()[event["row_id"]], second["revision_id"])
        with self.assertRaises(ValueError):
            self.app.apply_measurements([dict(event, version=2)])
        for field, value in [("target", 9), ("device_id", "another")]:
            with self.assertRaises(ValueError):
                self.app.apply_measurements([dict(event, **{field: value})])
        rational = dict(event, row_id="rational", target=Fraction(1, 10))
        self.app.apply_measurements([rational])
        equivalent = dict(rational, target=0.1)
        self.app.apply_measurements([equivalent])
        with self.assertRaises(ValueError):
            self.app.apply_measurements([dict(rational, target=Fraction(1, 10) + Fraction(1, 10**20))])

    def test_invalid_batch_and_withdrawal(self):
        events = daily_events()
        self.app.apply_measurements(events)
        before = self.state()
        correction = dict(events[0], version=2)
        invalids = [dict(events[1], version=True), dict(events[1], target=True),
                    dict(events[1], target=float("nan")),
                    dict(events[1], features={"x": float("inf")}),
                    {"row_id": "missing", "version": 1, "op": "withdraw"}]
        missing = dict(events[1])
        del missing["target"]
        invalids.append(missing)
        for invalid in invalids:
            with self.assertRaises(ValueError):
                self.app.apply_measurements([correction, invalid])
            self.assertEqual(before, self.state())
        withdrawn = self.app.apply_measurements(
            [{"row_id": "a0", "version": 2, "op": "withdraw"}])[0]
        self.assertTrue(withdrawn["withdrawn"])
        self.assertEqual(withdrawn["features"], {"reference": "7", "x": "-2"})
        self.assertEqual(withdrawn["target"], "8")
        self.app.apply_measurements([dict(events[0], version=3, target=80)])
        replay = self.app.apply_measurements(
            [{"row_id": "a0", "version": 2, "op": "withdraw"}])[0]
        self.assertEqual(replay, withdrawn)
        self.assertEqual(self.app.store.load_revisions()[
            self.app.store.load_heads()["a0"]]["version"], 3)
        original = copy.deepcopy(events)
        self.app.apply_measurements(events)
        self.assertEqual(events, original)
