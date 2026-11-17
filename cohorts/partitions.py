from dataclasses import dataclass

from cohorts.identity import training_scope_identity
from measurements.validation import require_identifier


@dataclass(frozen=True)
class TrainingScope:
    scope_id: str
    cohort_id: str
    devices: tuple
    revision_ids: tuple

    def to_payload(self):
        return {"scope_id": self.scope_id, "cohort_id": self.cohort_id,
                "devices": list(self.devices), "revision_ids": list(self.revision_ids)}

    @classmethod
    def from_payload(cls, payload):
        return cls(payload["scope_id"], payload["cohort_id"],
                   tuple(payload["devices"]), tuple(payload["revision_ids"]))


def _devices(cohort, devices):
    if isinstance(devices, (str, bytes)):
        raise ValueError("devices must be an iterable of identities")
    try:
        selected = {require_identifier(device, "device_id") for device in devices}
    except TypeError as exc:
        raise ValueError("devices must be iterable") from exc
    known = {row["device_id"] for row in cohort["rows"]}
    if not selected or not selected <= known:
        raise ValueError("device range is empty or contains unknown devices")
    return selected


def build_training_scope(cohort, devices):
    chosen = _devices(cohort, devices)
    identities = tuple(row["revision_id"] for row in cohort["rows"]
                       if row["device_id"] in chosen)
    scope_id = training_scope_identity(cohort["cohort_id"], chosen, identities)
    return TrainingScope(scope_id, cohort["cohort_id"], tuple(sorted(chosen)), identities)


def validate_device_partition(cohort, training_devices, holdout_devices):
    training = _devices(cohort, training_devices)
    holdout = _devices(cohort, holdout_devices)
    if training & holdout:
        raise ValueError("training and holdout devices overlap")
    return training, holdout


def resolve_holdout_rows(cohort, training_devices, holdout_devices):
    _, selected = validate_device_partition(cohort, training_devices, holdout_devices)
    return [row for row in cohort["rows"] if row["device_id"] in selected]
