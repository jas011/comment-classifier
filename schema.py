from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DomainConfig:
    domain: str
    conflict_strategy: str = "severity_then_domain_weight"
    action_priority_order: List[str] = field(
        default_factory=lambda: ["bad_experience", "real_query", "appreciation", "noise"]
    )
    category_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "real_query": 1.0,
            "appreciation": 0.7,
            "bad_experience": 1.2,
            "noise": 0.1,
        }
    )
    playbooks: Dict[str, str] = field(
        default_factory=lambda: {
            "bad_experience": "Investigate issue and respond with a fix or resolution.",
            "real_query": "Provide a clear answer and add to knowledge base/content queue.",
            "appreciation": "Acknowledge and highlight positive feedback.",
            "noise": "Filter out of active queue.",
        }
    )


DEFAULT_CONFIGS: Dict[str, DomainConfig] = {
    "creator": DomainConfig(
        domain="creator",
        category_weights={
            "real_query": 1.15,
            "appreciation": 0.75,
            "bad_experience": 1.10,
            "noise": 0.10,
        },
        playbooks={
            "bad_experience": "Pin clarification/errata or fix pacing/audio in next upload.",
            "real_query": "Reply with details or turn recurring questions into a follow-up video/Short.",
            "appreciation": "Heart/pin standout comments and share in community updates.",
            "noise": "Hide/filter spam and low-signal chatter.",
        },
    ),
    "product": DomainConfig(
        domain="product",
        category_weights={
            "real_query": 1.10,
            "appreciation": 0.65,
            "bad_experience": 1.30,
            "noise": 0.10,
        },
        playbooks={
            "bad_experience": "Trigger defect/support triage, offer replacement/refund, and log QA item.",
            "real_query": "Clarify specs/compatibility/usage and update product listing or docs.",
            "appreciation": "Tag for social proof/reviews and follow up for repeat purchase.",
            "noise": "Suppress spam/irrelevant reviews from support queue.",
        },
    ),
    "service": DomainConfig(
        domain="service",
        category_weights={
            "real_query": 1.05,
            "appreciation": 0.65,
            "bad_experience": 1.40,
            "noise": 0.10,
        },
        playbooks={
            "bad_experience": "Escalate for immediate service recovery and resolve SLA/experience gap.",
            "real_query": "Confirm scope/pricing/schedule and add to service FAQ.",
            "appreciation": "Thank customer, recognize staff/delivery, and request referral/review.",
            "noise": "Archive off-topic or spam entries.",
        },
    ),
}