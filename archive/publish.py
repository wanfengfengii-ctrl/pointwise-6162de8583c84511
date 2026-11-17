from copy import deepcopy

from archive.schema import (ArchiveIntegrityError, encode_payload,
                            validate_artifact_envelope)

CHILD_FIELDS = {"cohort": "rows", "fit": "training_predictions",
                "package": "holdout"}


class PublicationFailure(RuntimeError):
    pass


def publication_checkpoint(hook, stage, artifact_id, kind):
    if hook is not None:
        hook(stage, artifact_id, kind)


def publish_artifact(store, kind, payload, parents=None, publication_hook=None):
    parents = {} if parents is None else dict(parents)
    artifact_id = payload[kind + "_id"]
    full = {"version": 1, "id": artifact_id, "kind": kind,
            "payload": deepcopy(payload)}
    validate_artifact_envelope(full)
    expected = ({} if kind == "cohort" else
                {"cohort": payload["cohort_id"]} if kind == "fit" else
                {"fit": payload["fit_id"], "cohort": payload["cohort_id"]})
    if parents != expected:
        raise ArchiveIntegrityError("publication parents differ from payload")
    with store.transaction():
        for role, parent in parents.items():
            try:
                record = store.read_artifact(parent)
            except ValueError as exc:
                raise ArchiveIntegrityError("missing publication parent") from exc
            if record["kind"] != role:
                raise ArchiveIntegrityError("wrong publication parent type")
        exists = store.connection.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if exists:
            from archive.reading import read_stored_payload
            old = read_stored_payload(store, artifact_id, kind)
            if encode_payload(old) != encode_payload(payload):
                raise ArchiveIntegrityError("artifact identity collision")
            if store.read_links(artifact_id) != parents:
                raise ArchiveIntegrityError("existing artifact links differ")
            return artifact_id
        header = deepcopy(full)
        children = header["payload"].pop(CHILD_FIELDS[kind])
        if not isinstance(children, list) or not children:
            raise ArchiveIntegrityError("artifact needs ordered child records")
        store.write_artifact(artifact_id, kind, header, len(children))
        publication_checkpoint(publication_hook, "after_header", artifact_id, kind)
        store.write_children(artifact_id, children)
        store.write_links(artifact_id, parents)
        publication_checkpoint(publication_hook, "after_children", artifact_id, kind)
        publication_checkpoint(publication_hook, "before_commit", artifact_id, kind)
    return artifact_id
