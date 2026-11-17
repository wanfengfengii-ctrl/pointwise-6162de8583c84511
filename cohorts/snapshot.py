from copy import deepcopy
from dataclasses import dataclass

from archive.publish import publish_artifact
from cohorts.identity import cohort_identity
from cohorts.selection import resolve_revision_selection


@dataclass(frozen=True)
class FixedCohort:
    cohort_id: str
    feature_order: tuple
    rows: tuple

    def to_payload(self):
        return {"cohort_id": self.cohort_id,
                "feature_order": list(self.feature_order),
                "rows": deepcopy(list(self.rows))}

    @classmethod
    def from_payload(cls, payload):
        return cls(payload["cohort_id"], tuple(payload["feature_order"]),
                   tuple(deepcopy(payload["rows"])))


def freeze_cohort(store, revision_ids, feature_order, publication_hook=None):
    with store.transaction():
        order, rows = resolve_revision_selection(store, revision_ids, feature_order)
        body = {"feature_order": list(order), "rows": rows}
        cohort = FixedCohort(cohort_identity(body), order, tuple(rows))
        return publish_artifact(store, "cohort", cohort.to_payload(),
                                publication_hook=publication_hook)
