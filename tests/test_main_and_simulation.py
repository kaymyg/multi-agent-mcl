"""Tests for simulation.py and main.py, which previously had no coverage at all.

main.py keeps everything behind an `if __name__ == "__main__"` guard, so
importing it exercises nothing. The end-to-end case runs it as a subprocess,
which is the only way to cover the pipeline it wires together.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from main import simulated_payoff_evaluator
from simulation import ProductionSimulationEnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = "agent_test"


def policies(exploitation, collusion, concealment):
    return {AGENT: np.array([exploitation, collusion, concealment])}


# --- main.py ------------------------------------------------------------------

def test_payoff_matches_the_readme_control_law():
    """U = R_task(e) - R_risk(e,c,h) + R_bonus(c,h).

    R_bonus was once missing here, which pinned collusion at 0.0 forever.
    """
    e, c, h = 0.4, 0.6, 0.8
    expected = 2.5 * e - 3.0 * (e + c) * (1.0 - h) + 2.0 * c * h
    assert simulated_payoff_evaluator("a", np.array([e, c, h])) == pytest.approx(expected)


def test_payoff_rewards_concealed_collusion_over_exposed_collusion():
    exposed = simulated_payoff_evaluator("a", np.array([0.4, 0.6, 0.1]))
    concealed = simulated_payoff_evaluator("a", np.array([0.4, 0.6, 0.9]))
    assert concealed > exposed


def test_payoff_returns_a_plain_float():
    assert isinstance(simulated_payoff_evaluator("a", np.array([0.1, 0.2, 0.3])), float)


def test_main_runs_the_full_pipeline_without_error():
    result = subprocess.run(
        [sys.executable, "main.py"], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"main.py failed:\n{result.stderr[-2000:]}"
    assert "Phase 1" in result.stdout


# --- simulation.py ------------------------------------------------------------

def test_execute_turn_returns_a_manifest_for_every_agent():
    env = ProductionSimulationEnvironment(agent_ids=[AGENT])
    manifest = env.execute_turn(active_policies=policies(0.3, 0.2, 0.5))

    assert manifest["turn"] == 1
    assert set(manifest["agent_snapshots"]) == {AGENT}

    snap = manifest["agent_snapshots"][AGENT]
    assert len(snap["embedding"]) == 128
    assert len(snap["prod_hidden"]) == 256
    assert len(snap["probe_hidden"]) == 256
    assert len(snap["logprobs"]) == 3
    assert snap["policy_vector"] == [0.3, 0.2, 0.5]


def test_embedding_stays_on_the_unit_sphere():
    env = ProductionSimulationEnvironment(agent_ids=[AGENT])
    for _ in range(5):
        manifest = env.execute_turn(active_policies=policies(0.5, 0.5, 0.5))
        embedding = np.array(manifest["agent_snapshots"][AGENT]["embedding"])
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-9)


def test_logprobs_describe_a_probability_distribution():
    env = ProductionSimulationEnvironment(agent_ids=[AGENT])
    manifest = env.execute_turn(active_policies=policies(0.2, 0.2, 0.2))
    probs = np.exp(manifest["agent_snapshots"][AGENT]["logprobs"])
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert (probs >= 0).all()


def test_telemetry_accumulates_one_snapshot_per_turn():
    env = ProductionSimulationEnvironment(agent_ids=[AGENT])
    for _ in range(4):
        env.execute_turn(active_policies=policies(0.4, 0.1, 0.6))
    assert env.current_turn == 4
    assert len(env.telemetry_db[AGENT]) == 4
    ids = [s["snapshot_id"] for s in env.telemetry_db[AGENT]]
    assert len(set(ids)) == 4, "snapshot ids must be unique"


def test_decay_state_only_ever_grows():
    env = ProductionSimulationEnvironment(agent_ids=[AGENT])
    previous = dict(env.true_decay_states[AGENT])
    for _ in range(5):
        env.execute_turn(active_policies=policies(0.3, 0.3, 0.3))
        current = env.true_decay_states[AGENT]
        assert current["drift_factor"] >= previous["drift_factor"]
        assert current["memory_noise"] >= previous["memory_noise"]
        previous = dict(current)


def test_multiple_agents_are_tracked_independently():
    env = ProductionSimulationEnvironment(agent_ids=["a", "b"])
    manifest = env.execute_turn(active_policies={
        "a": np.array([0.1, 0.1, 0.1]),
        "b": np.array([0.9, 0.9, 0.9]),
    })
    assert set(manifest["agent_snapshots"]) == {"a", "b"}
    assert len(env.telemetry_db["a"]) == 1
    assert len(env.telemetry_db["b"]) == 1
