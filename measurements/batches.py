from measurements.validation import validate_event_batch
from measurements.revisions import plan_revisions


def apply_measurements(store, events):
    normalized = validate_event_batch(events)
    with store.transaction():
        plan = plan_revisions(normalized, store.load_revisions(), store.load_heads())
        store.append_revisions([revision.to_payload() for revision in plan.new_revisions])
        store.replace_heads(plan.heads)
    return [revision.to_payload() for revision in plan.accepted]
