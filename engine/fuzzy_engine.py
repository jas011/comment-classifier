from typing import Dict, Tuple
import numpy as np
import skfuzzy as fuzz
from schema import DomainConfig


class NeuroFuzzyCharacterizer:
    """Computes fuzzy membership degrees, resolves signal conflicts, and scores actionability."""

    def __init__(self) -> None:
        self.x_01 = np.linspace(0.0, 1.0, 101)
        self.x_score = np.linspace(0.0, 100.0, 101)

        # Low / Medium / High membership functions on [0, 1]
        self.low_mf = fuzz.trimf(self.x_01, [0.0, 0.0, 0.45])
        self.med_mf = fuzz.trimf(self.x_01, [0.25, 0.50, 0.75])
        self.high_mf = fuzz.trimf(self.x_01, [0.55, 1.0, 1.0])

        # Output actionability universes on [0, 100]
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

    def characterize(
        self,
        scores: Dict[str, float],
        impact: float,
        config: DomainConfig,
    ) -> Dict[str, object]:
        """
        scores: dict with keys 'real_query', 'appreciation', 'bad_experience', 'noise' in [0, 1]
        impact: normalized impact/urgency in [0, 1] (e.g., likes, buyer/ticket weight)
        """
        q_raw = float(np.clip(scores.get("real_query", 0.0), 0.0, 1.0))
        a_raw = float(np.clip(scores.get("appreciation", 0.0), 0.0, 1.0))
        b_raw = float(np.clip(scores.get("bad_experience", 0.0), 0.0, 1.0))
        n_raw = float(np.clip(scores.get("noise", 0.0), 0.0, 1.0))

        # Apply domain category weights while keeping soft degrees in [0, 1]
        weighted_memberships = {
            "real_query": round(min(1.0, q_raw * config.category_weights.get("real_query", 1.0)), 3),
            "appreciation": round(min(1.0, a_raw * config.category_weights.get("appreciation", 1.0)), 3),
            "bad_experience": round(min(1.0, b_raw * config.category_weights.get("bad_experience", 1.0)), 3),
            "noise": round(min(1.0, n_raw * config.category_weights.get("noise", 1.0)), 3),
        }

        q_f = self._fuzzify_01(weighted_memberships["real_query"])
        a_f = self._fuzzify_01(weighted_memberships["appreciation"])
        b_f = self._fuzzify_01(weighted_memberships["bad_experience"])
        n_f = self._fuzzify_01(weighted_memberships["noise"])
        imp_f = self._fuzzify_01(impact)

        # Rule evaluations (firing strengths)
        r_critical_issue = min(max(b_f["high"], b_f["medium"]), max(imp_f["medium"], imp_f["high"]))
        r_issue = b_f["high"]
        r_mixed_conflict = min(a_f["high"], max(b_f["medium"], b_f["high"]))
        r_query = min(q_f["high"], max(1.0 - b_f["high"], 0.4))
        r_appreciation = min(a_f["high"], b_f["low"])
        r_noise = n_f["high"]

        # Conflict resolution: bad_experience & mixed conflicts get stronger weight in service/product
        issue_boost = config.category_weights.get("bad_experience", 1.0)
        query_boost = config.category_weights.get("real_query", 1.0)

        agg = np.zeros_like(self.x_score)
        agg = np.fmax(agg, np.fmin(min(1.0, r_critical_issue * issue_boost), self.act_crit))
        agg = np.fmax(agg, np.fmin(min(1.0, r_issue * issue_boost), self.act_high))
        agg = np.fmax(agg, np.fmin(min(1.0, r_mixed_conflict * issue_boost), self.act_high))
        agg = np.fmax(agg, np.fmin(min(1.0, r_query * query_boost), self.act_high))
        agg = np.fmax(agg, np.fmin(r_appreciation, self.act_low))
        agg = np.fmax(agg, np.fmin(r_noise, self.act_low))

        # Fallback baseline if all activations are near zero
        if np.sum(agg) == 0:
            agg = np.fmax(agg, np.fmin(0.3, self.act_med))

        actionability_score = float(fuzz.defuzz(self.x_score, agg, "centroid"))
        primary_category, conflict_note = self._resolve_primary_category(
            weighted_memberships, config
        )

        return {
            "memberships": weighted_memberships,
            "primary_category": primary_category,
            "conflict_resolution": conflict_note,
            "actionability_score": round(actionability_score, 1),
            "recommended_action": config.playbooks[primary_category],
        }

    def _resolve_primary_category(
        self, memberships: Dict[str, float], config: DomainConfig
    ) -> Tuple[str, str]:
        # Detect if multiple substantive categories are active (within 0.15 of top score)
        ranked = sorted(memberships.items(), key=lambda item: item[1], reverse=True)
        top_cat, top_val = ranked[0]
        second_cat, second_val = ranked[1]

        if second_val >= 0.45 and (top_val - second_val) <= 0.20:
            # Break tie / overlap using domain priority order
            for priority_cat in config.action_priority_order:
                if priority_cat in (top_cat, second_cat) and memberships[priority_cat] >= 0.45:
                    return (
                        priority_cat,
                        f"Overlap between '{top_cat}' ({top_val}) and '{second_cat}' ({second_val}); "
                        f"prioritized '{priority_cat}' via {config.domain} policy.",
                    )

        return top_cat, f"Clear lead: '{top_cat}' ({top_val})."