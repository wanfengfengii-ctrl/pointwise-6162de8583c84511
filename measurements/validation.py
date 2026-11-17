import math
from collections.abc import Mapping
from fractions import Fraction

from measurements.records import MeasurementEvent


def require_fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        raise ValueError("expected a finite rational number")
    try:
        result = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("expected a finite rational number") from exc
    return result


def require_finite_number(value):
    try:
        number = float(require_fraction(value))
    except (OverflowError, ValueError) as exc:
        raise ValueError("number cannot be represented finitely") from exc
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def canonical_number(value):
    number = require_fraction(value)
    require_finite_number(number)
    return str(number)


def require_identifier(value, label="identity"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")
    return value


def normalize_event(raw):
    if not isinstance(raw, Mapping):
        raise ValueError("measurement event must be an object")
    allowed = {"row_id", "version", "op", "device_id", "features", "target"}
    if set(raw) - allowed or not {"row_id", "version"} <= set(raw):
        raise ValueError("invalid event fields")
    row_id = require_identifier(raw["row_id"], "row_id")
    version = raw["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("source version must be a positive integer")
    op = raw.get("op", "put")
    if op not in ("put", "withdraw"):
        raise ValueError("unknown measurement operation")
    supplied = {"device_id", "features", "target"} & set(raw)
    if op == "withdraw" and not supplied:
        return MeasurementEvent(row_id, version, op)
    if supplied != {"device_id", "features", "target"}:
        raise ValueError("device, features and target are required together")
    device = require_identifier(raw["device_id"], "device_id")
    features = raw["features"]
    if not isinstance(features, Mapping) or not features:
        raise ValueError("features must be a nonempty numeric mapping")
    normalized = []
    for name, value in features.items():
        normalized.append((require_identifier(name, "feature"), canonical_number(value)))
    return MeasurementEvent(row_id, version, op, device,
                            tuple(sorted(normalized)), canonical_number(raw["target"]))


def validate_event_batch(events):
    if isinstance(events, (str, bytes, Mapping)):
        raise ValueError("batch must be an iterable of events")
    try:
        return tuple(normalize_event(event) for event in events)
    except TypeError as exc:
        raise ValueError("batch must be iterable") from exc
