from archive.schema import content_identity, encode_payload
import hashlib


def cohort_identity(payload):
    return content_identity("cohort", payload)


def training_scope_identity(cohort_id, devices, revision_ids):
    body = {"version": 1, "cohort_id": cohort_id,
            "devices": sorted(devices), "revision_ids": list(revision_ids)}
    return "scope_" + hashlib.sha256(encode_payload(body).encode("utf-8")).hexdigest()
