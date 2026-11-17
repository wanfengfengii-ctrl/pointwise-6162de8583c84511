import os
import tempfile
from fractions import Fraction
from pathlib import Path

from archive.reading import read_artifact_graph, read_cohort
from archive.schema import encode_payload
from delivery.auditing import build_review_manifest, build_training_projection


def rational_document(value):
    """Deep independent document with Fractions as reduced rational strings."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: rational_document(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rational_document(item) for item in value]
    return value


def export_correlated_fit(store, fit_id, target):
    from fitting.correlated import read_correlated_fit
    fit = read_correlated_fit(store, fit_id)
    cohort = read_cohort(store, fit["cohort_id"])
    document = {"format": "correlated-review-v1",
                "fit": rational_document(fit),
                "cohort": cohort}
    write_export_atomically(document, target)
    return document


def build_export_document(graph):
    manifest = build_review_manifest(graph)
    return {"format_version": 1, "package": graph["package"], "fit": graph["fit"],
            "cohort": graph["cohort"], "training_projection": build_training_projection(graph),
            "holdout_results": graph["package"]["holdout"], "manifest": manifest}


def write_export_atomically(document, target):
    destination = Path(target)
    text = encode_payload(document) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(destination.parent),
                prefix="." + destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(destination)


def export_package(store, package_id, target):
    graph = read_artifact_graph(store, package_id)
    document = build_export_document(graph)
    write_export_atomically(document, target)
    return document
