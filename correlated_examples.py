"""Public inputs and hand checks; contains no correlated solver."""
import argparse
import copy
import json
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path

from archive.publish import PublicationFailure
from delivery.app import open_archive


def rational_json(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: rational_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rational_json(v) for v in value]
    return value


def prepare(app, prefix, xs):
    revisions = app.apply_measurements([
        {"row_id": prefix + str(i), "device_id": prefix, "version": 1,
         "features": {"x": x}, "target": y}
        for i, (x, y) in enumerate(zip(xs, [0, 2]))])
    ids = [r["revision_id"] for r in revisions]
    cohort = app.freeze_cohort(ids, ["x"])
    return cohort, ids


def rejected(app, operation):
    before = app.store.inventory()
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert app.store.inventory() == before


def assert_fractions(values):
    assert all(isinstance(v, F) for v in values), values


def scenario(root, extended):
    database = root / "archive.sqlite"
    target = root / "correlated.json"
    with open_archive(database) as app:
        cohort, ids = prepare(app, "shared", [-1, 1])
        noise = {"revision_ids": ids[:], "covariance": [[1, 1], [1, 1]]}
        fit = app.fit_correlated(cohort, ["shared"], 1, noise)
        assert fit["coefficients"] == [1, 1]
        assert_fractions(fit["coefficients"] + fit["residuals"])
        assert fit["residuals"] == [0, 0]
        assert fit["diagnostics"] == {
            "n": 2, "noise_rank": 1, "loss": F(0),
            "penalty": F(1), "objective": F(1)}
        assert_fractions([fit["diagnostics"][k]
                          for k in ("loss", "penalty", "objective")])
        assert fit["noise"]["revision_ids"] == fit["scope"]["revision_ids"] == ids
        rejected(app, lambda: app.predict_correlated(
            fit["fit_id"], [[0]], [[1, 0]], [[1]]))
        if extended:
            zero = app.fit_correlated(cohort, ["shared"], 1,
                {"revision_ids": ids, "covariance": [[0, 0], [0, 0]]})
            assert zero["coefficients"] == [1, 1]
            assert zero["diagnostics"]["noise_rank"] == 0
            for bad in (
                {"revision_ids": ids[:1], "covariance": [[1]]},
                {"revision_ids": [ids[0], ids[0]], "covariance": [[1, 1], [1, 1]]},
                {"revision_ids": ids, "covariance": [[1, 2], [2, 1]]},
                {"revision_ids": ids, "covariance": [[0, 1], [1, 0]]},
                {"revision_ids": ids, "covariance": [[1, 0], [1, 1]]},
                {"revision_ids": ids, "covariance": [[1.0, 0], [0, 1]]}):
                rejected(app, lambda bad=bad: app.fit_correlated(
                    cohort, ["shared"], 1, bad))
            rejected(app, lambda: app.evaluate_holdout(fit["fit_id"], ["shared"]))

        diagonal_cohort, diagonal_ids = prepare(app, "diagonal", [7, 7])
        diagonal_noise = {"revision_ids": diagonal_ids[:],
                          "covariance": [[1, 0], [0, 4]]}
        diagonal = app.fit_correlated(
            diagonal_cohort, ["diagonal"], "1", diagonal_noise)
        identity = diagonal["fit_id"]
        assert diagonal["coefficients"] == [F(2, 5), F(0)]
        assert diagonal["residuals"] == [F(-2, 5), F(8, 5)]
        assert diagonal["diagnostics"] == {
            "n": 2, "noise_rank": 2, "loss": F(2, 5),
            "penalty": F(0), "objective": F(2, 5)}
        assert_fractions(diagonal["coefficients"] + diagonal["residuals"])
        before_replay = app.store.inventory()
        reversed_fit = app.fit_correlated(diagonal_cohort, ["diagonal"], F(1),
            {"revision_ids": diagonal_ids[::-1], "covariance": [[4, 0], [0, 1]]})
        assert reversed_fit["fit_id"] == identity
        assert app.store.inventory() == before_replay
        rows = [[7], [99]]
        cross = [[1, 0], [1, 0]]
        new_noise = [[1, 1], [1, 1]]
        prediction = app.predict_correlated(identity, rows, cross, new_noise)
        assert prediction == {"mean": [F(0), F(0)],
                              "covariance": [[F(0), F(0)], [F(0), F(0)]]}
        assert_fractions(prediction["mean"] + sum(prediction["covariance"], []))
        independent = app.predict_correlated(
            identity, rows, [[0, 0], [0, 0]], new_noise)
        assert independent == {"mean": [F(2, 5), F(2, 5)],
                               "covariance": [[F(1), F(1)], [F(1), F(1)]]}
        assert app.predict_correlated(identity, [], [], []) == {
            "mean": [], "covariance": []}
        rejected(app, lambda: app.predict_correlated("unknown", [], [], []))
        if extended:
            rejected(app, lambda: app.fit_correlated(
                diagonal_cohort, ["diagonal"], 1,
                {"revision_ids": diagonal_ids, "covariance": [[1, 1], [1, 1]]}))
            for stage in ("after_header", "after_children", "before_commit"):
                def fail(actual, artifact_id, kind, stage=stage):
                    if actual == stage and kind == "fit":
                        raise PublicationFailure(stage)
                before = app.store.inventory()
                app.publication_hook = fail
                try:
                    app.fit_correlated(diagonal_cohort, ["diagonal"], 2, diagonal_noise)
                except PublicationFailure:
                    pass
                else:
                    raise AssertionError("publication hook was bypassed")
                assert app.store.inventory() == before
                app.publication_hook = None
            retried = app.fit_correlated(
                diagonal_cohort, ["diagonal"], 2, diagonal_noise)
            before = app.store.inventory()
            assert app.fit_correlated(
                diagonal_cohort, ["diagonal"], 2, diagonal_noise) == retried
            assert app.store.inventory() == before

        frozen = copy.deepcopy(app.read_fit(identity))
        exported = app.export_correlated_fit(identity, target)
        assert exported["format"] == "correlated-review-v1"
        assert exported["fit"] == rational_json(frozen)
        assert exported["cohort"] == app.read_cohort(diagonal_cohort)
        assert json.loads(target.read_text(encoding="utf-8")) == exported
        # Caller changes and later source revisions must not alter this fit.
        diagonal_noise["covariance"][0][0] = 999
        diagonal_noise["revision_ids"].reverse()
        diagonal["coefficients"][0] = -999
        diagonal["noise"]["covariance"].clear()
        app.apply_measurements([
            {"row_id": "diagonal0", "device_id": "diagonal", "version": 2,
             "features": {"x": 700}, "target": 900}])
        app.apply_measurements([
            {"row_id": "diagonal1", "version": 2, "op": "withdraw"}])
        assert app.read_fit(identity) == frozen
        assert app.predict_correlated(identity, rows, cross, new_noise) == prediction
        prediction["mean"][0] = 999
    with open_archive(database) as app:
        reopened = app.predict_correlated(identity, rows, cross, new_noise)
        assert reopened == {"mean": [F(0), F(0)],
                            "covariance": [[F(0), F(0)], [F(0), F(0)]]}
        assert app.read_fit(identity) == frozen
        assert app.export_correlated_fit(identity, target) == exported
    return {"fit_id": identity, "coefficients": frozen["coefficients"],
            "residuals": frozen["residuals"], "conditional": reopened,
            "independent_training_noise": independent,
            "reopened_export_equal": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "demo"))
    args = parser.parse_args()
    if args.mode == "verify":
        suite = unittest.defaultTestLoader.discover("tests")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        if not result.wasSuccessful():
            return 1
    with tempfile.TemporaryDirectory(prefix="correlated-review-") as directory:
        result = scenario(Path(directory), args.mode == "verify")
    print(json.dumps(rational_json(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
