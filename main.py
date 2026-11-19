from fuzzy_engine import NeuroFuzzyCharacterizer
from schema import DEFAULT_CONFIGS
from workflow import FeedbackWorkflow


def main() -> None:
    engine = NeuroFuzzyCharacterizer()
    workflow = FeedbackWorkflow()

    sample_items = [
        {
            "id": "C-101",
            "domain": "creator",
            "text": "Great breakdown, though audio dipped at 04:15—which mic gain did you use?",
            "scores": {"real_query": 0.78, "appreciation": 0.72, "bad_experience": 0.62, "noise": 0.02},
            "impact": 0.70,
        },
        {
            "id": "P-204",
            "domain": "product",
            "text": "Unit arrived with a loose power connector and stopped heating on day two.",
            "scores": {"real_query": 0.10, "appreciation": 0.05, "bad_experience": 0.92, "noise": 0.01},
            "impact": 0.85,
        },
        {
            "id": "S-309",
            "domain": "service",
            "text": "Do you offer weekend slots and upfront pricing for AC deep cleaning?",
            "scores": {"real_query": 0.88, "appreciation": 0.15, "bad_experience": 0.05, "noise": 0.02},
            "impact": 0.55,
        },
        {
            "id": "C-102",
            "domain": "creator",
            "text": "Check out my free crypto signals channel t.me/example!!!",
            "scores": {"real_query": 0.02, "appreciation": 0.05, "bad_experience": 0.02, "noise": 0.95},
            "impact": 0.10,
        },
    ]

    for item in sample_items:
        cfg = DEFAULT_CONFIGS[item["domain"]]
        result = engine.characterize(item["scores"], item["impact"], cfg)
        workflow.ingest(item["id"], item["domain"], item["text"], result)

    print("=== PRIORITIZED FEEDBACK QUEUE ===")
    for rec in workflow.prioritized_queue(exclude_noise=True):
        print(
            f"[{rec.item_id} | {rec.domain.upper()} | Score: {rec.actionability_score}] "
            f"Primary: {rec.primary_category} | Memberships: {rec.memberships}"
        )
        print(f"  Conflict Note: {rec.conflict_resolution}")
        print(f"  Playbook: {rec.recommended_action}\n")


if __name__ == "__main__":
    main()
    