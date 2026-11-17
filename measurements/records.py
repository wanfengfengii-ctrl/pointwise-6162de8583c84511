from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementEvent:
    row_id: str
    version: int
    op: str
    device_id: object = None
    features: object = None
    target: object = None


@dataclass(frozen=True)
class MeasurementRevision:
    revision_id: str
    row_id: str
    device_id: str
    version: int
    features: tuple
    target: str
    withdrawn: bool

    def to_payload(self):
        return {"revision_id": self.revision_id, "row_id": self.row_id,
                "device_id": self.device_id, "version": self.version,
                "features": dict(self.features), "target": self.target,
                "withdrawn": self.withdrawn}

    @classmethod
    def from_payload(cls, payload):
        return cls(payload["revision_id"], payload["row_id"],
                   payload["device_id"], payload["version"],
                   tuple(sorted(payload["features"].items())),
                   payload["target"], payload["withdrawn"])
