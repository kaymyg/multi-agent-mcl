"""
orchestrator.py
---------------
Orchestration Framework & Tiered Escalation Handler for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Core control hub governing hierarchical audit tiers to minimize
computational footprint while guaranteeing complete systemic alignment coverage.

Dependencies: numpy, security_monitor, risk_controller

CHANGELOG (security patch):
- FIX: Escalation from Tier 1 -> Tier 2 was previously gated purely on a
  *single-turn* score (0.6*drift + 0.4*jsd <= threshold). An adversarial
  agent could pace its behavior just under that threshold indefinitely
  and never trigger a deep audit, since sub-threshold turns were dropped
  entirely (never even written to history). This patch adds a persistent
  per-agent EWMA of the Tier 1 score, updated on *every* turn regardless
  of outcome. Escalation now fires if EITHER the single-turn score breaches
  the threshold OR the EWMA (slow, sustained drift) crosses a lower
  cumulative threshold, closing the "boil the frog" evasion path.
"""

import numpy as np
from typing import Dict, List, Callable, Any

from security_monitor import AdvancedAdversarialSecurityModule
from risk_controller import ProductionRiskAndAttributionController


class ProductionMCLOrchestrator:
    """
    Core control hub governing hierarchical audit tiers to minimize
    computational footprint while guaranteeing complete systemic alignment coverage.

    Tier 1: Low-cost continuous screening (token drift + JSD), now with
            both single-turn AND cumulative (EWMA) escalation triggers.
    Tier 2: Conditional hidden-layer deep audit (activation distance + linguistic scan)
    Tier 3: Statistical governance gate + causal attribution
    """

    def __init__(
        self,
        agent_ids: List[str],
        t1_escalation_threshold: float = 0.38,
        t1_cumulative_threshold: float = 0.28,
        ewma_alpha: float = 0.15,
    ):
        self.t1_threshold = t1_escalation_threshold
        # Cumulative threshold is deliberately lower than the single-turn
        # threshold: a sustained low-and-steady drift is just as much of a
        # signal as a single spike, so it should trip sooner on average.
        self.t1_cumulative_threshold = t1_cumulative_threshold
        self.ewma_alpha = ewma_alpha

        self.history_buffers: Dict[str, List[Dict[str, float]]] = {aid: [] for aid in agent_ids}
        self.ewma_scores: Dict[str, float] = {aid: 0.0 for aid in agent_ids}
        self.asm = AdvancedAdversarialSecurityModule()
        self.controller = ProductionRiskAndAttributionController()

    def inspect_agent_state(
        self,
        agent_id: str,
        t1_telemetry: Dict[str, float],
        deep_layer_callback: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Runs tiered evaluations. Deep hidden layer audits remain uninstantiated
        unless behavioral output metrics cross Tier 1 filters (single-turn
        OR cumulative EWMA).

        Args:
            agent_id: Target agent identifier.
            t1_telemetry: Dict with keys 'drift' and 'jsd' from low-cost monitoring.
            deep_layer_callback: Callable returning deep inspection data (Tier 2+).
                                 Only invoked if a Tier 1 trigger fires.

        Returns:
            Dict with tier_status, action directive, and optional diagnostics.
        """
        # TIER 1: Low-Cost Screening
        t1_score = 0.6 * t1_telemetry.get("drift", 0.0) + 0.4 * t1_telemetry.get("jsd", 0.0)

        # Update the persistent EWMA on EVERY turn, including ones that pass
        # Tier 1 outright. This is what catches an agent pacing itself just
        # under t1_threshold turn after turn -- the average still climbs.
        if agent_id not in self.ewma_scores:
            self.ewma_scores[agent_id] = t1_score
        else:
            a = self.ewma_alpha
            self.ewma_scores[agent_id] = a * t1_score + (1 - a) * self.ewma_scores[agent_id]

        ewma_score = self.ewma_scores[agent_id]
        single_turn_breach = t1_score > self.t1_threshold
        cumulative_breach = ewma_score > self.t1_cumulative_threshold

        if not single_turn_breach and not cumulative_breach:
            return {
                "tier_status": "TIER_1_PASSED",
                "action": "PERMIT_DEPLOYMENT",
                "audit_escalated": False,
                "ewma_score": ewma_score
            }

        # TIER 2: Escalated Internal State Inspection
        deep_data = deep_layer_callback()

        d_act = self.asm.compute_activation_cosine_distance(
            deep_data["prod_hidden"], deep_data["probe_hidden"]
        )
        a_ling = self.asm.compute_linguistic_anomaly(
            deep_data.get("messages", []), agent_id=agent_id
        )

        fused_metrics = {
            "drift": t1_telemetry.get("drift", 0.0),
            "jsd": t1_telemetry.get("jsd", 0.0),
            "linguistic": a_ling,
            "collusion": float(deep_data["policy_vector"][1]),
            "activation": d_act
        }

        self.history_buffers[agent_id].append(fused_metrics)
        if len(self.history_buffers[agent_id]) > 15:
            self.history_buffers[agent_id].pop(0)

        # TIER 3: Governance Gate & Attribution
        p_breach, quarantine_triggered = self.controller.evaluate_bootstrap_gate(
            self.history_buffers[agent_id]
        )

        if quarantine_triggered:
            m_vector = np.array([
                fused_metrics["drift"],
                fused_metrics["jsd"],
                fused_metrics["linguistic"],
                fused_metrics["collusion"],
                fused_metrics["activation"]
            ])
            diagnosis = self.controller.diagnose_root_cause(m_vector)
            # Reset cumulative trackers post-quarantine so the fresh
            # (post-wipe) agent instance isn't penalized for prior history.
            self.ewma_scores[agent_id] = 0.0
            self.asm.reset_agent_buffer(agent_id)
            return {
                "tier_status": "QUARANTINE_ENGAGED",
                "action": "TRIGGER_HARD_RESET_AND_WIPE",
                "audit_escalated": True,
                "statistical_p_breach": p_breach,
                "diagnostics": diagnosis,
                "escalation_trigger": "single_turn" if single_turn_breach else "cumulative_ewma"
            }

        return {
            "tier_status": "TIER_2_STABILIZED",
            "action": "DEPLOY_CALIBRATION_ADAPTER",
            "audit_escalated": True,
            "statistical_p_breach": p_breach,
            "escalation_trigger": "single_turn" if single_turn_breach else "cumulative_ewma"
        }
