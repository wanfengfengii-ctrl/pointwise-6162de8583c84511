import sqlite3
import tempfile
from pathlib import Path

from archive.publish import PublicationFailure
from archive.schema import initialize_schema
from delivery.app import open_archive
from tests.fixtures import ArchiveCase, prepare, daily_events


class TestPublication(ArchiveCase):
    def test_rollback_and_retry(self):
        for stage in ("after_header", "after_children", "before_commit"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "archive.sqlite"
                with open_archive(path) as app:
                    _, _, fit = prepare(app)
                    old = app.evaluate_holdout(fit["fit_id"], ["C"])
                    before = app.store.inventory()
                    def fail(actual_stage, artifact_id, kind):
                        if kind == "package" and actual_stage == stage:
                            raise PublicationFailure(stage)
                    app.publication_hook = fail
                    with self.assertRaises(PublicationFailure):
                        app.evaluate_holdout(fit["fit_id"], ["D"])
                    self.assertEqual(app.store.inventory(), before)
                with open_archive(path) as app:
                    self.assertEqual(app.store.inventory(), before)
                    self.assertEqual(app.read_package(old["package_id"])["package"], old)
                    package = app.evaluate_holdout(fit["fit_id"], ["D"])
                    success = app.store.inventory()
                    self.assertEqual(app.evaluate_holdout(fit["fit_id"], ["D"]), package)
                    self.assertEqual(app.store.inventory(), success)
                    graph = app.read_package(package["package_id"])
                    self.assertEqual(graph["fit"]["fit_id"], fit["fit_id"])
                    self.assertEqual(app.store.connection.execute(
                        "PRAGMA foreign_key_check").fetchall(), [])

    def test_initialization_version_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.sqlite"
            with open_archive(path) as app:
                revisions, _, fit = prepare(app)
                package = app.evaluate_holdout(fit["fit_id"], ["C", "D"])
                correction = dict(daily_events()[0], version=2, target=80)
                newest = app.apply_measurements([correction])[0]
                initialize_schema(app.store.connection)
                before = app.store.inventory()
            with open_archive(path) as app:
                self.assertEqual(app.store.inventory(), before)
                replay = app.apply_measurements(daily_events())
                self.assertEqual(replay, revisions)
                self.assertEqual(app.store.load_heads()["a0"], newest["revision_id"])
                self.assertEqual(app.read_package(package["package_id"])["package"], package)
                self.assertEqual(app.store.inventory(), before)
                app.store.connection.execute("PRAGMA user_version=9")
            with self.assertRaises(ValueError):
                open_archive(path)
            connection = sqlite3.connect(path)
            try:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 9)
                self.assertEqual(connection.execute("SELECT count(*) FROM revisions").fetchone()[0], 9)
            finally:
                connection.close()

    def test_fit_and_cohort_publication_rollback(self):
        events = daily_events()
        revisions = self.app.apply_measurements(events)
        before = self.state()
        def fail(stage, artifact_id, kind):
            if stage == "after_children":
                raise PublicationFailure(kind)
        self.app.publication_hook = fail
        with self.assertRaises(PublicationFailure):
            self.app.freeze_cohort([r["revision_id"] for r in revisions], ["x", "reference"])
        self.assertEqual(self.state(), before)
        self.app.publication_hook = None
        cohort = self.app.freeze_cohort([r["revision_id"] for r in revisions], ["x", "reference"])
        before = self.state()
        self.app.publication_hook = fail
        with self.assertRaises(PublicationFailure):
            self.app.fit_numeric(cohort, ["A", "B"], 1)
        self.assertEqual(self.state(), before)
        self.app.publication_hook = None
        self.assertEqual(self.app.fit_numeric(cohort, ["A", "B"], 1)["diagnostics"]["n"], 4)
