from archive.store import ArchiveStore
from archive.reading import read_cohort, read_fit, read_package
from measurements.batches import apply_measurements
from cohorts.snapshot import freeze_cohort
from fitting.diagnostics import fit_numeric
from fitting.correlated import fit_correlated
from prediction.holdout import evaluate_holdout
from prediction.execution import predict_numeric
from prediction.correlated import predict_correlated
from delivery.exporting import export_correlated_fit, export_package


class NumericArchiveApp:
    def __init__(self, path=":memory:", *, solver=None, publication_hook=None):
        self.store = ArchiveStore(path)
        self.solver = solver
        self.publication_hook = publication_hook
        self._closed = False

    def apply_measurements(self, events):
        return apply_measurements(self.store, events)

    def freeze_cohort(self, revision_ids, feature_order):
        return freeze_cohort(self.store, revision_ids, feature_order, self.publication_hook)

    def fit_numeric(self, cohort_id, training_devices, lambda_, *, solver=None):
        identity = fit_numeric(self.store, cohort_id, training_devices, lambda_,
                               self.solver if solver is None else solver,
                               self.publication_hook)
        return read_fit(self.store, identity)

    def fit_correlated(self, cohort_id, devices, lambda_, noise):
        identity = fit_correlated(self.store, cohort_id, devices,
                                  lambda_, noise, self.publication_hook)
        return read_fit(self.store, identity)

    def predict_correlated(self, fit_id, rows, B, D):
        return predict_correlated(self.store, fit_id, rows, B, D)

    def export_correlated_fit(self, fit_id, target):
        return export_correlated_fit(self.store, fit_id, target)

    def evaluate_holdout(self, fit_id, holdout_devices):
        return evaluate_holdout(self.store, fit_id, holdout_devices, self.publication_hook)

    def predict_numeric(self, package_id, rows):
        return predict_numeric(self.store, package_id, rows)

    def read_cohort(self, cohort_id):
        return read_cohort(self.store, cohort_id)

    def read_fit(self, fit_id):
        return read_fit(self.store, fit_id)

    def read_package(self, package_id):
        return read_package(self.store, package_id)

    def export_package(self, package_id, target):
        return export_package(self.store, package_id, target)

    def close(self):
        if not self._closed:
            self.store.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open_archive(path=":memory:", **kwargs):
    return NumericArchiveApp(path, **kwargs)
