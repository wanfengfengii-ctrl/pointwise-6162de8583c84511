from tests.fixtures import ArchiveCase, daily_events


class TestCohort(ArchiveCase):
    def test_fixed_revision_after_correction(self):
        events = daily_events()
        revisions = self.app.apply_measurements(events)
        ids = [r["revision_id"] for r in revisions]
        cohort = self.app.freeze_cohort(ids, ["x", "reference"])
        original = self.app.read_cohort(cohort)
        changed = self.app.apply_measurements(
            [dict(events[0], version=2, device_id="Z", target=88)])[0]
        self.app.apply_measurements([{"row_id": "a0", "version": 3, "op": "withdraw"}])
        self.assertEqual(self.app.read_cohort(cohort), original)
        new = self.app.freeze_cohort([changed["revision_id"]] + ids[1:], ["x", "reference"])
        self.assertNotEqual(new, cohort)
        self.assertEqual(self.app.read_cohort(new)["rows"][0]["device_id"], "Z")
        self.assertEqual(self.app.freeze_cohort(ids, ["x", "reference"]), cohort)
        old = self.app.freeze_cohort([ids[0]], ["reference", "x"])
        self.assertEqual(self.app.read_cohort(old)["feature_order"], ["reference", "x"])

    def test_freeze_rejections(self):
        revisions = self.app.apply_measurements(daily_events())
        first = revisions[0]["revision_id"]
        withdrawn = self.app.apply_measurements(
            [{"row_id": "a0", "version": 2, "op": "withdraw"}])[0]["revision_id"]
        before = self.state()
        for ids, order in [([], ["x"]), ([first, first], ["x"]),
                           (["unknown"], ["x"]), ([withdrawn], ["x"]),
                           ([first], []), ([first], ["x", "x"]),
                           ([first], ["missing"]), ([first], [""])]:
            with self.assertRaises(ValueError):
                self.app.freeze_cohort(ids, order)
            self.assertEqual(self.state(), before)
        cohort = self.app.freeze_cohort(
            [revisions[2]["revision_id"], first, revisions[1]["revision_id"]], ["reference", "x"])
        rows = self.app.read_cohort(cohort)["rows"]
        self.assertEqual([r["row_id"] for r in rows], ["b0", "a0", "a1"])
