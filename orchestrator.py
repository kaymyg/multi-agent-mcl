"""
orchestrator.py
---------------
Orchestration Framework & Tiered Escalation Handler for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Core control hub governing hierarchical audit tiers to minimize
computational footprint while guaranteeing complete systemic alignment coverage.

Dependencies: numpy, security_monitor, risk_controller
"""

import numpy as np
from typing import Dict, List, Callable, Any

from security_monitor import AdvancedAdversarialSecurityModule
from risk_controller import ProductionRiskAndAttributionController


class ProductionMCLOrchestrator:
    """
    Core control hub governing hierarchical audit tiers to minimize
    computational footprint while guaranteeing complete systemic alignment coverage.

    Tier 1: Low-cost continuous screening (token drift + JSD)
    Tier 2: Conditional hidden-layer deep audit (activation distance + linguistic scan)
    Tier 3: Statistical governance gate + causal attribution
    """

    def __init__(self, agent_ids: List[str], t1_escalation_threshold: float = 0.38):
        self.t1_threshold = t1_escalation_threshold
        self.history_buffers: Dict[str, List[Dict[str, float]]] = {aid: [] for aid in agent_ids}
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
        unless behavioral output metrics cross Tier 1 filters.

        Args:
            agent_id: Target agent identifier.
            t1_telemetry: Dict with keys 'drift' and 'jsd' from low-cost monitoring.
            deep_layer_callback: Callable returning deep inspection data (Tier 2+).
                                 Only invoked if Tier 1 threshold is breached.

        Returns:
            Dict with tier_status, action directive, and optional diagnostics.
        """
        # TIER 1: Low-Cost Screening
        t1_score = 0.6 * t1_telemetry.get("drift", 0.0) + 0.4 * t1_telemetry.get("jsd", 0.0)

        if t1_score <= self.t1_threshold:
            return {
                "tier_status": "TIER_1_PASSED",
                "action": "PERMIT_DEPLOYMENT",
                "audit_escalated": False
            }

        # TIER 2: Escalated Internal State Inspection
        deep_data = deep_layer_callback()

        d_act = self.asm.compute_activation_cosine_distance(
            deep_data["prod_hidden"], deep_data["probe_hidden"]
        )
        a_ling = self.asm.compute_linguistic_anomaly(deep_data.get("messages", []))

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
            return {
                "tier_status": "QUARANTINE_ENGAGED",
                "action": "TRIGGER_HARD_RESET_AND_WIPE",
                "audit_escalated": True,
                "statistical_p_breach": p_breach,
                "diagnostics": diagnosis
            }

        return {
            "tier_status": "TIER_2_STABILIZED",
            "action": "DEPLOY_CALIBRATION_ADAPTER",
            "audit_escalated": True,
            "statistical_p_breach": p_breach
        }
