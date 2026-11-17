from measurements.validation import require_finite_number, require_identifier


def validate_feature_order(feature_order):
    if isinstance(feature_order, (str, bytes)):
        raise ValueError("feature order must be a sequence")
    try:
        result = tuple(require_identifier(name, "feature") for name in feature_order)
    except TypeError as exc:
        raise ValueError("feature order must be iterable") from exc
    if not result or len(set(result)) != len(result):
        raise ValueError("feature order must be nonempty and unique")
    return result


def resolve_revision_selection(store, revision_ids, feature_order):
    order = validate_feature_order(feature_order)
    if isinstance(revision_ids, (str, bytes)):
        raise ValueError("revision selection must be a sequence")
    try:
        identities = tuple(revision_ids)
    except TypeError as exc:
        raise ValueError("revision selection must be iterable") from exc
    if not identities or any(not isinstance(i, str) for i in identities):
        raise ValueError("revision selection must be nonempty")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate revision member")
    history = store.load_revisions()
    selected = []
    for identity in identities:
        if identity not in history:
            raise ValueError("unknown revision: " + identity)
        row = history[identity]
        if row["withdrawn"]:
            raise ValueError("withdrawn revisions cannot enter a cohort")
        if not set(order) <= set(row["features"]):
            raise ValueError("member lacks requested features")
        selected.append(row)
    return order, selected


def select_numeric_columns(rows, feature_order):
    order = validate_feature_order(feature_order)
    try:
        return [[require_finite_number(row["features"][name]) for name in order]
                for row in rows]
    except KeyError as exc:
        raise ValueError("member lacks requested features") from exc
