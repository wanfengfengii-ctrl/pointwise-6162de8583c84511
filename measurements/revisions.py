from dataclasses import dataclass

from archive.schema import content_identity, encode_payload
from measurements.records import MeasurementRevision


@dataclass(frozen=True)
class RevisionBatchPlan:
    new_revisions: tuple
    heads: dict
    accepted: tuple


def revision_identity(payload):
    return content_identity("revision", payload)


def same_revision_content(left, right):
    return encode_payload(left) == encode_payload(right)


def plan_revisions(events, revisions, heads):
    history = dict(revisions)
    current = dict(heads)
    by_version = {(p["row_id"], p["version"]): p for p in history.values()}
    created, accepted = [], []
    for event in events:
        existing = by_version.get((event.row_id, event.version))
        head = history.get(current.get(event.row_id))
        if event.op == "withdraw" and head is None:
            raise ValueError("cannot withdraw an unknown measurement")
        if event.device_id is None:
            base = existing if existing is not None else head
            device, features, target = (base["device_id"],
                                        base["features"], base["target"])
        else:
            device, features, target = (event.device_id,
                                        dict(event.features), event.target)
        payload = {"row_id": event.row_id, "device_id": device,
                   "version": event.version, "features": dict(features),
                   "target": target, "withdrawn": event.op == "withdraw"}
        payload["revision_id"] = revision_identity(payload)
        if existing is not None:
            if not same_revision_content(existing, payload):
                raise ValueError("conflicting content for known source version")
            accepted.append(MeasurementRevision.from_payload(existing))
            continue
        if head is not None and event.version <= head["version"]:
            raise ValueError("unknown out-of-order source version")
        history[payload["revision_id"]] = payload
        by_version[(event.row_id, event.version)] = payload
        current[event.row_id] = payload["revision_id"]
        revision = MeasurementRevision.from_payload(payload)
        created.append(revision)
        accepted.append(revision)
    return RevisionBatchPlan(tuple(created), current, tuple(accepted))
