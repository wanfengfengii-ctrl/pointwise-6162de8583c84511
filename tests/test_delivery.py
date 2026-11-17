import copy
import json
import shutil
import tempfile
from pathlib import Path

from archive.schema import ArchiveIntegrityError
from delivery.app import open_archive
from delivery.workflows import run_sample_workflow
from tests.fixtures import ArchiveCase, prepare, daily_events


class TestDelivery(ArchiveCase):
    def test_reopen_predict_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, export = root / "archive.sqlite", root / "review.json"
            result = run_sample_workflow(database, export)
            identity = result["package_id"]
            self.assertNear(result["predictions"][0], 14)
            self.assertEqual(result["predictions"], result["reopened_predictions"])
            with open_archive(database) as app:
                original = app.read_package(identity)
                before_export = json.loads(export.read_text(encoding="utf-8"))
                app.apply_measurements([dict(daily_events()[0], version=2, target=800)])
                app.apply_measurements([{"row_id": "b0", "version": 2, "op": "withdraw"}])
                changed = app.read_package(identity)
                changed["package"]["coefficients"][0] = -900
                changed["cohort"]["rows"][0]["features"]["x"] = "999"
                changed["package"]["holdout"].clear()
                changed["fit"]["scope"]["devices"].append("foreign")
                self.assertEqual(app.read_package(identity), original)
                self.assertNear(app.predict_numeric(identity, [[8, 7]])[0], 14)
                self.assertEqual(app.predict_numeric(identity, []), [])
            with open_archive(database) as app:
                document = app.export_package(identity, export)
                self.assertEqual(document, before_export)
                self.assertEqual(json.loads(export.read_text(encoding="utf-8")), document)
                self.assertEqual(document["manifest"]["training_rows"], 4)
                self.assertEqual(document["manifest"]["holdout_rows"], 4)
                self.assertEqual(document["training_projection"][0]["values"], ["-2", "7"])
                self.assertEqual(app.read_package(identity), original)
            repeated = run_sample_workflow(database, export)
            self.assertEqual(repeated["package_id"], identity)
            self.assertEqual(repeated["predictions"], [14])

    def test_feature_order_and_invalid_inputs(self):
        for order, inputs in [(["x", "reference"], [[8, 7], [10, -20]]),
                              (["reference", "x"], [[7, 8], [-20, 10]])]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "archive.sqlite"
                with open_archive(path) as app:
                    _, _, fit = prepare(app, order)
                    package = app.evaluate_holdout(fit["fit_id"], ["C", "D"])
                    identity = package["package_id"]
                    original_inputs = copy.deepcopy(inputs)
                    values = app.predict_numeric(identity, inputs)
                    self.assertNear(values[0], 14)
                    self.assertNear(values[1], 15)
                    self.assertEqual(inputs, original_inputs)
                    for invalid in ([[1]], [[8, 7], [float("nan"), 2]], [[True, 2]], [None]):
                        with self.assertRaises(ValueError):
                            app.predict_numeric(identity, invalid)
                    with self.assertRaises(ValueError):
                        app.predict_numeric("unknown", [])
                with open_archive(path) as app:
                    self.assertEqual(app.predict_numeric(identity, inputs), values)
                    graph = app.read_package(identity)
                    self.assertEqual(graph["package"]["feature_order"], order)
                    document = app.export_package(identity, Path(directory) / "review.json")
                    self.assertEqual(document["manifest"]["feature_order"], order)

    def test_parent_integrity_and_export_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            healthy = root / "healthy.sqlite"
            with open_archive(healthy) as app:
                _, cohort, fit = prepare(app)
                package = app.evaluate_holdout(fit["fit_id"], ["C", "D"])
                identity = package["package_id"]
                before = app.store.inventory()
                blocked_target = root / "existing_directory"
                blocked_target.mkdir()
                with self.assertRaises(OSError):
                    app.export_package(identity, blocked_target)
                self.assertEqual(before, app.store.inventory())
                self.assertEqual(list(root.glob(".existing_directory.*.tmp")), [])
                self.assertTrue(blocked_target.is_dir())
            for mode in ("missing_parent", "wrong_type", "missing_child"):
                broken = root / (mode + ".sqlite")
                shutil.copyfile(healthy, broken)
                with open_archive(broken) as app:
                    connection = app.store.connection
                    if mode == "missing_parent":
                        connection.execute("PRAGMA foreign_keys=OFF")
                        connection.execute("DELETE FROM artifacts WHERE artifact_id=?", (fit["fit_id"],))
                    elif mode == "wrong_type":
                        connection.execute("UPDATE artifact_links SET parent_id=? "
                                           "WHERE child_id=? AND role='fit'", (cohort, identity))
                    else:
                        connection.execute("DELETE FROM artifact_children WHERE artifact_id=? AND ordinal=0",
                                           (identity,))
                    with self.assertRaises(ArchiveIntegrityError):
                        app.read_package(identity)
                    output = root / (mode + ".json")
                    with self.assertRaises(ArchiveIntegrityError):
                        app.export_package(identity, output)
                    self.assertFalse(output.exists())
            with open_archive(healthy) as app:
                self.assertNear(app.predict_numeric(identity, [[8, 7]])[0], 14)
