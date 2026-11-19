from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import numpy as np
import skfuzzy as fuzz
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TARGET_COLS = ["real_query", "appreciation", "bad_experience", "noise"]


@dataclass
class DomainConfig:
    domain: str
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


DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
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


class DistilBERTCommentScorer:
    """Loads a fine-tuned DistilBERT model and predicts soft [0, 1] intent scores."""

    def __init__(self, model_dir: str = "./distilbert_comment_model", max_length: int = 192) -> None:
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Trained model directory not found: {path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()

    def predict_scores(self, text: str) -> Dict[str, float]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).squeeze(0).cpu().numpy()

        return {
            col: round(float(np.clip(probs[i], 0.0, 1.0)), 4)
            for i, col in enumerate(TARGET_COLS)
        }


class NeuroFuzzyCharacterizer:
    def __init__(self) -> None:
        self.x_01 = np.linspace(0.0, 1.0, 101)
        self.x_score = np.linspace(0.0, 100.0, 101)

        self.low_mf = fuzz.trimf(self.x_01, [0.0, 0.0, 0.45])
        self.med_mf = fuzz.trimf(self.x_01, [0.25, 0.50, 0.75])
        self.high_mf = fuzz.trimf(self.x_01, [0.55, 1.0, 1.0])

        self.act_low = fuzz.trimf(self.x_score, [0.0, 15.0, 35.0])
        self.act_med = fuzz.trimf(self.x_score, [25.0, 50.0, 75.0])
        self.act_high = fuzz.trimf(self.x_score, [60.0, 80.0, 95.0])
        self.act_crit = fuzz.trimf(self.x_score, [75.0, 92.0, 100.0])

    def _fuzzify_01(self, value: float) -> Dict[str, float]:
        v = float(np.clip(value, 0.0, 1.0))
        return {
            "low": float(fuzz.interp_membership(self.x_01, self.low_mf, v)),
            "medium": float(fuzz.interp_membership(self.x_01, self.med_mf, v)),
            "high": float(fuzz.interp_membership(self.x_01, self.high_mf, v)),
        }

    def characterize(self, raw_scores: Dict[str, float], impact: float, config: DomainConfig) -> Dict[str, object]:
        memberships = {
            col: round(min(1.0, float(raw_scores.get(col, 0.0)) * config.category_weights.get(col, 1.0)), 3)
            for col in TARGET_COLS
        }

        q_f = self._fuzzify_01(memberships["real_query"])
        a_f = self._fuzzify_01(memberships["appreciation"])
        b_f = self._fuzzify_01(memberships["bad_experience"])
        n_f = self._fuzzify_01(memberships["noise"])
        imp_f = self._fuzzify_01(impact)

        r_critical_issue = min(max(b_f["high"], b_f["medium"]), max(imp_f["medium"], imp_f["high"]))
        r_issue = b_f["high"]
        r_mixed_conflict = min(a_f["high"], max(b_f["medium"], b_f["high"]))
        r_query = min(q_f["high"], max(1.0 - b_f["high"], 0.4))
        r_appreciation = min(a_f["high"], b_f["low"])
        r_noise = n_f["high"]

        issue_boost = config.category_weights.get("bad_experience", 1.0)
        query_boost = config.category_weights.get("real_query", 1.0)

        agg = np.zeros_like(self.x_score)
        agg = np.fmax(agg, np.fmin(min(1.0, r_critical_issue * issue_boost), self.act_crit))
        agg = np.fmax(agg, np.fmin(min(1.0, r_issue * issue_boost), self.act_high))
        agg = np.fmax(agg, np.fmin(min(1.0, r_mixed_conflict * issue_boost), self.act_high))
        agg = np.fmax(agg, np.fmin(min(1.0, r_query * query_boost), self.act_high))
        agg = np.fmax(agg, np.fmin(r_appreciation, self.act_low))
        agg = np.fmax(agg, np.fmin(r_noise, self.act_low))

        if np.sum(agg) == 0:
            agg = np.fmax(agg, np.fmin(0.3, self.act_med))

        score = round(float(fuzz.defuzz(self.x_score, agg, "centroid")), 1)
        primary_category, conflict_note = self._resolve_primary(memberships, config)

        return {
            "raw_scores": raw_scores,
            "memberships": memberships,
            "primary_category": primary_category,
            "conflict_resolution": conflict_note,
            "actionability_score": score,
            "recommended_action": config.playbooks[primary_category],
        }

    def _resolve_primary(self, memberships: Dict[str, float], config: DomainConfig):
        ranked = sorted(memberships.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_val = ranked[0]
        second_cat, second_val = ranked[1]

        if second_val >= 0.45 and (top_val - second_val) <= 0.20:
            for p_cat in config.action_priority_order:
                if p_cat in (top_cat, second_cat) and memberships[p_cat] >= 0.45:
                    return (
                        p_cat,
                        f"Overlap between '{top_cat}' ({top_val}) and '{second_cat}' ({second_val}); "
                        f"prioritized '{p_cat}' via {config.domain} policy.",
                    )
        return top_cat, f"Clear lead: '{top_cat}' ({top_val})."


class FeedbackService:
    """Combines DistilBERT inference, fuzzy characterization, and queue state."""

    def __init__(self, model_dir: str = "./distilbert_comment_model") -> None:
        self.scorer = DistilBERTCommentScorer(model_dir=model_dir)
        self.fuzzy = NeuroFuzzyCharacterizer()
        self.queue: Dict[str, Dict] = {}

    def process_item(
        self,
        text: str,
        domain: str = "creator",
        impact: float = 0.5,
        item_id: Optional[str] = None,
    ) -> Dict:
        cfg = DOMAIN_CONFIGS.get(domain.lower(), DOMAIN_CONFIGS["creator"])
        raw_scores = self.scorer.predict_scores(text)
        analysis = self.fuzzy.characterize(raw_scores, impact=impact, config=cfg)

        rid = item_id or f"item_{uuid.uuid4().hex[:8]}"
        record = {
            "item_id": rid,
            "domain": cfg.domain,
            "text": text,
            "impact": round(float(np.clip(impact, 0.0, 1.0)), 3),
            "status": "PENDING_REVIEW",
            **analysis,
        }
        self.queue[rid] = record
        return record

    def get_queue(self, exclude_noise: bool = True) -> List[Dict]:
        items = list(self.queue.values())
        if exclude_noise:
            items = [
                r for r in items
                if not (r["primary_category"] == "noise" and r["actionability_score"] < 25.0)
            ]
        return sorted(items, key=lambda r: r["actionability_score"], reverse=True)

    def resolve_item(self, item_id: str) -> Optional[Dict]:
        if item_id not in self.queue:
            return None
        self.queue[item_id]["status"] = "RESOLVED"
        return self.queue[item_id]