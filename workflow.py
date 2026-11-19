from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FeedbackRecord:
    item_id: str
    domain: str
    text: str
    primary_category: str
    memberships: Dict[str, float]
    actionability_score: float
    conflict_resolution: str
    recommended_action: str
    status: str = "PENDING_REVIEW"


class FeedbackWorkflow:
    """Routes characterized items into prioritized queues and tracks resolution status."""

    def __init__(self) -> None:
        self.records: List[FeedbackRecord] = []

    def ingest(self, item_id: str, domain: str, text: str, analysis: Dict[str, object]) -> FeedbackRecord:
        record = FeedbackRecord(
            item_id=item_id,
            domain=domain,
            text=text,
            primary_category=str(analysis["primary_category"]),
            memberships=dict(analysis["memberships"]),  # type: ignore[arg-type]
            actionability_score=float(analysis["actionability_score"]),  # type: ignore[arg-type]
            conflict_resolution=str(analysis["conflict_resolution"]),
            recommended_action=str(analysis["recommended_action"]),
        )
        self.records.append(record)
        return record

    def prioritized_queue(self, exclude_noise: bool = True) -> List[FeedbackRecord]:
        items = [
            r for r in self.records
            if not (exclude_noise and r.primary_category == "noise" and r.actionability_score < 25.0)
        ]
        return sorted(items, key=lambda r: r.actionability_score, reverse=True)

    def mark_resolved(self, item_id: str) -> bool:
        for r in self.records:
            if r.item_id == item_id:
                r.status = "RESOLVED"
                return True
        return False