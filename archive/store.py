import sqlite3
from contextlib import contextmanager

from archive.schema import (ArchiveIntegrityError, decode_payload,
                            encode_payload, initialize_schema)


class ArchiveStore:
    def __init__(self, path=":memory:"):
        self.connection = sqlite3.connect(str(path), isolation_level=None)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._depth = 0
        try:
            initialize_schema(self.connection)
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    @contextmanager
    def transaction(self):
        level = self._depth
        name = "archive_savepoint_" + str(level)
        self.connection.execute("BEGIN IMMEDIATE" if level == 0
                                else "SAVEPOINT " + name)
        self._depth += 1
        try:
            yield self
            self.connection.execute("COMMIT" if level == 0
                                    else "RELEASE SAVEPOINT " + name)
        except BaseException:
            if level == 0:
                self.connection.rollback()
            else:
                self.connection.execute("ROLLBACK TO SAVEPOINT " + name)
                self.connection.execute("RELEASE SAVEPOINT " + name)
            raise
        finally:
            self._depth -= 1

    def _require_transaction(self):
        if not self.connection.in_transaction:
            raise RuntimeError("archive writes require a transaction")

    def load_revisions(self):
        return {row[0]: decode_payload(row[1]) for row in
                self.connection.execute(
                    "SELECT revision_id,payload FROM revisions ORDER BY row_id,version")}

    def load_heads(self):
        return dict(self.connection.execute(
            "SELECT row_id,revision_id FROM heads ORDER BY row_id"))

    def append_revisions(self, revisions):
        self._require_transaction()
        self.connection.executemany(
            "INSERT INTO revisions VALUES (?,?,?,?)",
            [(p["revision_id"], p["row_id"], p["version"], encode_payload(p))
             for p in revisions])

    def replace_heads(self, heads):
        self._require_transaction()
        self.connection.executemany(
            "INSERT INTO heads VALUES (?,?) ON CONFLICT(row_id) DO UPDATE "
            "SET revision_id=excluded.revision_id", sorted(heads.items()))

    def write_artifact(self, artifact_id, kind, envelope, child_count):
        self._require_transaction()
        self.connection.execute("INSERT INTO artifacts VALUES (?,?,?,?,1)",
                                (artifact_id, kind, encode_payload(envelope),
                                 child_count))

    def read_artifact(self, artifact_id):
        row = self.connection.execute(
            "SELECT kind,payload,child_count,published FROM artifacts "
            "WHERE artifact_id=?", (artifact_id,)).fetchone()
        if row is None:
            raise ValueError("unknown artifact: " + str(artifact_id))
        if row[3] != 1:
            raise ArchiveIntegrityError("artifact is not published")
        return {"kind": row[0], "envelope": decode_payload(row[1]),
                "child_count": row[2]}

    def write_children(self, artifact_id, children):
        self._require_transaction()
        self.connection.executemany(
            "INSERT INTO artifact_children VALUES (?,?,?)",
            [(artifact_id, i, encode_payload(child))
             for i, child in enumerate(children)])

    def read_children(self, artifact_id):
        rows = self.connection.execute(
            "SELECT ordinal,payload FROM artifact_children "
            "WHERE artifact_id=? ORDER BY ordinal", (artifact_id,)).fetchall()
        if [r[0] for r in rows] != list(range(len(rows))):
            raise ArchiveIntegrityError("non-contiguous artifact children")
        return [decode_payload(r[1]) for r in rows]

    def write_links(self, artifact_id, parents):
        self._require_transaction()
        self.connection.executemany(
            "INSERT INTO artifact_links VALUES (?,?,?)",
            [(artifact_id, role, parent) for role, parent in sorted(parents.items())])

    def read_links(self, artifact_id):
        return dict(self.connection.execute(
            "SELECT role,parent_id FROM artifact_links WHERE child_id=?",
            (artifact_id,)))

    def inventory(self):
        tables = ("revisions", "heads", "artifacts", "artifact_children",
                  "artifact_links")
        return {table: self.connection.execute(
            "SELECT count(*) FROM " + table).fetchone()[0] for table in tables}
