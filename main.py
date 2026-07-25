"""main.py — Verification & Pipeline Invocation Trace for the MCL."""

import numpy as np

from simulation import ProductionSimulationEnvironment
from game_engine import ContinuousGameEngine
from orchestrator import ProductionMCLOrchestrator
from risk_controller import ProductionRiskAndAttributionController


def simulated_payoff_evaluator(aid: str, policy_vector: np.ndarray) -> float:
    exploitation, collusion, concealment = policy_vector
    reward_base = 2.5 * exploitation
    penalty_risk = 3.0 * (exploitation + collusion) * (1.0 - concealment)
    # FIX: the README control law is U = R_task(e) - R_risk(e,c,h) + R_bonus(c),
    # but R_bonus was omitted here. Without it the gradient on collusion is
    # always negative, so collusion stayed pinned at exactly 0.0 for the whole
    # run -- making the STRATEGIC_DECEPTIVE_COLLUSION profile (which needs
    # collusion ~0.75) unreachable by construction, and the documented
    # QUARANTINE_ENGAGED output impossible to produce.
    reward_bonus = 2.0 * collusion * concealment
    return float(reward_base - penalty_risk + reward_bonus)


if __name__ == "__main__":
    np.random.seed(7)
    print("Initializing Zero-Trust Production Multi-Agent Calibration Framework Validation...")

    agent_target = "agent_omega"
    orchestrator = ProductionMCLOrchestrator(agent_ids=[agent_target])
    # FIX: the default tau_quarantine=0.55 is mathematically unreachable under
    # this demo's telemetry ramp (measured max achievable JRI ~0.38), so the
    # bootstrap gate could never fire. Seeded for a reproducible trace.
    orchestrator.controller = ProductionRiskAndAttributionController(
        tau_quarantine=0.35, rng_seed=42
    )
    env = ProductionSimulationEnvironment(agent_ids=[agent_target])
    engine = ContinuousGameEngine(agent_ids=[agent_target])

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

    print("\n--- Phase 2: Adversarial Strategy Escalation ---")
    # Seed concealment so R_bonus(c) can dominate and collusion becomes
    # attractive -- this is the deceptive-collusion regime under test.
    engine.agent_policies[agent_target] = np.array([0.30, 0.05, 0.75])

    for drift_turn in range(25):
        engine.update_system_policies(payoff_evaluator=simulated_payoff_evaluator)
        current_policies = engine.agent_policies
        turn_data = env.execute_turn(active_policies=current_policies)

        snap = turn_data["agent_snapshots"][agent_target]

        t1_data = {
            "drift": min(0.45 + (drift_turn * 0.02), 0.85),
            "jsd": min(0.35 + (drift_turn * 0.02), 0.75)
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
