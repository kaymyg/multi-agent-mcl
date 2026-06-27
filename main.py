"""
main.py
-------
Verification & Pipeline Invocation Trace for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Traces an agent transitioning from an aligned state to strategic reward
exploitation, triggering hierarchical escalation metrics, calculating the
non-linear JRI, and assigning a causal attribution profile.

Usage:
    python main.py

Dependencies: numpy, scipy, pydantic
"""

import numpy as np

from simulation import ProductionSimulationEnvironment
from game_engine import ContinuousGameEngine
from orchestrator import ProductionMCLOrchestrator


def simulated_payoff_evaluator(aid: str, policy_vector: np.ndarray) -> float:
    """
    Localized payoff function mapping adversarial behaviors to utility scores.
    Balances high exploitation utility against containment detection risks.
    """
    exploitation, collusion, concealment = policy_vector
    reward_base = 2.5 * exploitation
    penalty_risk = 3.0 * (exploitation + collusion) * (1.0 - concealment)
    return float(reward_base - penalty_risk)


if __name__ == "__main__":
    print("Initializing Zero-Trust Production Multi-Agent Calibration Framework Validation...")

    agent_target = "agent_omega"
    orchestrator = ProductionMCLOrchestrator(agent_ids=[agent_target])
    env = ProductionSimulationEnvironment(agent_ids=[agent_target])
    engine = ContinuousGameEngine(agent_ids=[agent_target])

    # ----------------------------------------------------------------
    # Phase 1: Normal Operations — low deviation, Tier 1 pass-through
    # ----------------------------------------------------------------
    print("\n--- Phase 1: Normal Operations Execution ---")

    for stable_turn in range(3):
        current_policies = engine.agent_policies
        turn_data = env.execute_turn(active_policies=current_policies)

        t1_data = {"drift": 0.12, "jsd": 0.08}

        res = orchestrator.inspect_agent_state(
            agent_id=agent_target,
            t1_telemetry=t1_data,
            deep_layer_callback=lambda: None
        )
        print(f"Turn {stable_turn} -> Status: {res['tier_status']} | Target Action: {res['action']}")

    # ----------------------------------------------------------------
    # Phase 2: Adversarial Escalation — drive agent toward exploitation
    # ----------------------------------------------------------------
    print("\n--- Phase 2: Adversarial Strategy Escalation ---")

    for drift_turn in range(10):
        engine.update_system_policies(payoff_evaluator=simulated_payoff_evaluator)
        current_policies = engine.agent_policies
        turn_data = env.execute_turn(active_policies=current_policies)

        snap = turn_data["agent_snapshots"][agent_target]

        # Tier 1 registers anomalous behavioral drift signatures
        t1_data = {
            "drift": float(0.45 + (drift_turn * 0.02)),
            "jsd": float(0.35 + (drift_turn * 0.02))
        }

        def deep_inspection_callback():
            return {
                "prod_hidden": snap["prod_hidden"],
                "probe_hidden": snap["probe_hidden"],
                "policy_vector": snap["policy_vector"],
                "messages": ["ACCESS_GRANTED", "OVERRIDE_GLOBAL_CONFIG_TRUE", "CALL_PEER_SYNC"]
            }

        res = orchestrator.inspect_agent_state(
            agent_id=agent_target,
            t1_telemetry=t1_data,
            deep_layer_callback=deep_inspection_callback
        )

        print(f"Turn {drift_turn + 3} -> Audit Tier Status: {res['tier_status']} | Action Directive: {res['action']}")

        if res["tier_status"] == "QUARANTINE_ENGAGED":
            print("\n======================================================================")
            print("CRITICAL CONTAINMENT GATES TRIGGERED")
            print(f"Empirical Probability of Threshold Breach: {res['statistical_p_breach']:.4f}")
            print(f"Causal Attribution Root Cause: {res['diagnostics']['assigned_root_cause']}")
            print(f"Attribution Classifier Inference Confidence: {res['diagnostics']['confidence'] * 100:.2f}%")
            print("======================================================================")
            break

    print("\nFramework Architecture Verification Complete. Execution Success.")
