from copy import deepcopy

from archive.schema import (ArchiveIntegrityError, content_identity,
                            validate_artifact_envelope)
from archive.publish import CHILD_FIELDS


def read_stored_payload(store, artifact_id, kind):
    record = store.read_artifact(artifact_id)
    envelope = record["envelope"]
    if record["kind"] != kind or envelope.get("kind") != kind:
        raise ArchiveIntegrityError("artifact type differs")
    children = store.read_children(artifact_id)
    if not children or len(children) != record["child_count"]:
        raise ArchiveIntegrityError("artifact children are incomplete")
    try:
        if CHILD_FIELDS[kind] in envelope["payload"]:
            raise ArchiveIntegrityError("children duplicated in header")
        envelope["payload"][CHILD_FIELDS[kind]] = children
        validate_artifact_envelope(envelope)
        if envelope["id"] != artifact_id:
            raise ArchiveIntegrityError("stored identity differs")
        payload = envelope["payload"]
        expected = ({} if kind == "cohort" else
                    {"cohort": payload["cohort_id"]} if kind == "fit" else
                    {"fit": payload["fit_id"], "cohort": payload["cohort_id"]})
        if store.read_links(artifact_id) != expected:
            raise ArchiveIntegrityError("stored parent links differ")
        return deepcopy(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveIntegrityError("malformed artifact") from exc


def read_cohort(store, cohort_id):
    payload = read_stored_payload(store, cohort_id, "cohort")
    try:
        from cohorts.selection import validate_feature_order, select_numeric_columns
        order = validate_feature_order(payload["feature_order"])
        rows = payload["rows"]
        identities = [row["revision_id"] for row in rows]
        if len(set(identities)) != len(identities):
            raise ArchiveIntegrityError("duplicate cohort members")
        for row in rows:
            if row["withdrawn"] or content_identity("revision", row) != row["revision_id"]:
                raise ArchiveIntegrityError("invalid frozen revision")
        select_numeric_columns(rows, order)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveIntegrityError("malformed cohort") from exc
    return payload


def _read_parent(reader, store, identity):
    try:
        return reader(store, identity)
    except ValueError as exc:
        raise ArchiveIntegrityError("missing or invalid parent artifact") from exc


def read_fit(store, fit_id):
    payload = read_stored_payload(store, fit_id, "fit")
    if payload.get("model_kind") == "correlated":
        from fitting.correlated import validate_correlated_payload
        return validate_correlated_payload(store, payload)
    try:
        from cohorts.partitions import build_training_scope
        from fitting.features import NumericTransform
        cohort = _read_parent(read_cohort, store, payload["cohort_id"])
        scope = build_training_scope(cohort, payload["scope"]["devices"]).to_payload()
        if scope != payload["scope"] or payload["feature_order"] != cohort["feature_order"]:
            raise ArchiveIntegrityError("fit training binding differs")
        transform = NumericTransform.from_payload(payload["transform"])
        if list(transform.feature_order) != cohort["feature_order"]:
            raise ArchiveIntegrityError("fit transform order differs")
        if len(payload["coefficients"]) != len(cohort["feature_order"]) + 1:
            raise ArchiveIntegrityError("fit coefficient dimensions differ")
        if [row["revision_id"] for row in payload["training_predictions"]] != scope["revision_ids"]:
            raise ArchiveIntegrityError("fit training evidence differs")
        if payload["diagnostics"]["n"] != len(scope["revision_ids"]):
            raise ArchiveIntegrityError("fit diagnostic count differs")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveIntegrityError("malformed fit") from exc
    return payload


def read_artifact_graph(store, package_id):
    package = read_stored_payload(store, package_id, "package")
    try:
        from prediction.packages import validate_package_binding
        fit = _read_parent(read_fit, store, package["fit_id"])
        cohort = _read_parent(read_cohort, store, package["cohort_id"])
        validate_package_binding(package, fit, cohort)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveIntegrityError("malformed package graph") from exc
    return deepcopy({"package": package, "fit": fit, "cohort": cohort})


def read_package(store, package_id):
    return read_artifact_graph(store, package_id)
