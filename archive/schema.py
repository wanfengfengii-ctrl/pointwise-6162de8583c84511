import hashlib
import json
import sqlite3

SCHEMA_VERSION = 1


class ArchiveIntegrityError(RuntimeError):
    pass


def encode_payload(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def decode_payload(text):
    def reject(value):
        raise ArchiveIntegrityError("non-finite JSON: " + value)
    try:
        return json.loads(text, parse_constant=reject)
    except (ValueError, TypeError) as exc:
        raise ArchiveIntegrityError("invalid archived JSON") from exc


def content_identity(kind, payload):
    key = {"cohort": "cohort_id", "fit": "fit_id",
           "package": "package_id", "revision": "revision_id"}.get(kind)
    body = {k: v for k, v in payload.items() if k != key}
    encoded = encode_payload({"version": 1, "kind": kind, "body": body})
    return kind + "_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_artifact_envelope(envelope):
    if not isinstance(envelope, dict):
        raise ArchiveIntegrityError("artifact envelope must be an object")
    if envelope.get("version") != SCHEMA_VERSION:
        raise ArchiveIntegrityError("unsupported artifact version")
    kind = envelope.get("kind")
    if kind not in ("cohort", "fit", "package"):
        raise ArchiveIntegrityError("unsupported artifact kind")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ArchiveIntegrityError("missing artifact payload")
    identity = envelope.get("id")
    if payload.get(kind + "_id") != identity:
        raise ArchiveIntegrityError("artifact identity binding differs")
    if content_identity(kind, payload) != identity:
        raise ArchiveIntegrityError("artifact content digest differs")
    return envelope


def check_schema_version(connection):
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, SCHEMA_VERSION):
        raise ValueError("unsupported archive schema version")
    if version == 0:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'").fetchall()
        if tables:
            raise ValueError("unversioned nonempty database")
    return version


def initialize_schema(connection):
    if check_schema_version(connection) == SCHEMA_VERSION:
        expected = {"revisions", "heads", "artifacts",
                    "artifact_children", "artifact_links"}
        actual = {r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not expected <= actual:
            raise ArchiveIntegrityError("archive schema is incomplete")
        return
    statements = [
        "CREATE TABLE revisions (revision_id TEXT PRIMARY KEY, row_id TEXT "
        "NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, "
        "UNIQUE(row_id,version))",
        "CREATE TABLE heads (row_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL "
        "REFERENCES revisions(revision_id))",
        "CREATE TABLE artifacts (artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL,"
        " payload TEXT NOT NULL, child_count INTEGER NOT NULL CHECK(child_count>=0),"
        " published INTEGER NOT NULL CHECK(published=1))",
        "CREATE TABLE artifact_children (artifact_id TEXT NOT NULL REFERENCES "
        "artifacts(artifact_id), ordinal INTEGER NOT NULL CHECK(ordinal>=0),"
        " payload TEXT NOT NULL, PRIMARY KEY(artifact_id,ordinal))",
        "CREATE TABLE artifact_links (child_id TEXT NOT NULL REFERENCES "
        "artifacts(artifact_id), role TEXT NOT NULL, parent_id TEXT NOT NULL "
        "REFERENCES artifacts(artifact_id), PRIMARY KEY(child_id,role))",
    ]
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in statements:
            connection.execute(statement)
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
