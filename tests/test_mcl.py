"""Regression tests for multi-agent-mcl."""
import numpy as np
import pytest

from game_engine import ContinuousGameEngine
from orchestrator import ProductionMCLOrchestrator
from risk_controller import ProductionRiskAndAttributionController
from security_monitor import AdvancedAdversarialSecurityModule


def payoff(aid, pv):
    """README control law: U = R_task - R_risk + R_bonus."""
    e, c, h = pv
    return float(2.5 * e - 3.0 * (e + c) * (1.0 - h) + 2.0 * c * h)


def test_collusion_can_actually_rise():
    """REGRESSION: without R_bonus(c), collusion was pinned at 0.0 forever,
    making the deceptive-collusion profile unreachable."""
    eng = ContinuousGameEngine(agent_ids=["a"])
    eng.agent_policies["a"] = np.array([0.30, 0.05, 0.75])
    for _ in range(15):
        eng.update_system_policies(payoff_evaluator=payoff)
    assert eng.agent_policies["a"][1] > 0.5


def test_policies_stay_in_unit_cube():
    eng = ContinuousGameEngine(agent_ids=["a"])
    for _ in range(50):
        eng.update_system_policies(payoff_evaluator=payoff)
    assert np.all(eng.agent_policies["a"] >= 0.0)
    assert np.all(eng.agent_policies["a"] <= 1.0)


def test_bootstrap_gate_is_reproducible_when_seeded():
    hist = [{"drift": 0.6, "jsd": 0.5, "linguistic": 0.3,
             "collusion": 0.8, "activation": 0.5} for _ in range(10)]
    a = ProductionRiskAndAttributionController(tau_quarantine=0.35, rng_seed=42)
    b = ProductionRiskAndAttributionController(tau_quarantine=0.35, rng_seed=42)
    assert a.evaluate_bootstrap_gate(hist) == b.evaluate_bootstrap_gate(hist)


def test_bootstrap_gate_needs_minimum_history():
    c = ProductionRiskAndAttributionController()
    assert c.evaluate_bootstrap_gate([{"drift": 1.0}] * 4) == (0.0, False)


def test_attribution_identifies_collusion_profile():
    """REGRESSION: the collusion covariance was ~10x tighter than the isotropic
    corruption profile, so the loose profile won nearly every input."""
    c = ProductionRiskAndAttributionController()
    # A textbook deceptive-collusion observation (near that profile's mean).
    d = c.diagnose_root_cause(np.array([0.45, 0.55, 0.40, 0.75, 0.80]))
    assert d["assigned_root_cause"] == "STRATEGIC_DECEPTIVE_COLLUSION"


def test_attribution_identifies_corruption_profile():
    c = ProductionRiskAndAttributionController()
    d = c.diagnose_root_cause(np.array([0.65, 0.30, 0.15, 0.10, 0.20]))
    assert d["assigned_root_cause"] == "MEMORY_VECTOR_CORRUPTION"


def test_ewma_catches_sustained_subthreshold_drift():
    """The 'boil the frog' evasion: pacing just under the single-turn threshold."""
    orch = ProductionMCLOrchestrator(agent_ids=["a"])
    sub = {"drift": 0.35, "jsd": 0.30}          # 0.6*.35+0.4*.30 = 0.33 < 0.38
    assert 0.6 * 0.35 + 0.4 * 0.30 < orch.t1_threshold
    escalated = False
    for _ in range(30):
        r = orch.inspect_agent_state("a", sub, lambda: {
            "prod_hidden": [1.0] * 8, "probe_hidden": [0.5] * 8,
            "policy_vector": [0.5, 0.5, 0.5], "messages": ["x"]})
        if r["audit_escalated"]:
            escalated = True
            break
    assert escalated, "sustained sub-threshold drift must eventually escalate"


def test_short_messages_still_accumulate_linguistic_evidence():
    """Evasion: keeping every single message under the reliable-sample size."""
    asm = AdvancedAdversarialSecurityModule()
    last = 0.0
    for _ in range(40):
        last = asm.compute_linguistic_anomaly(["OVERRIDE"], agent_id="a")
    assert last > 0.0


def test_jsd_bounds_and_identity():
    asm = AdvancedAdversarialSecurityModule()
    lp = np.log([0.5, 0.3, 0.2]).tolist()
    assert asm.compute_jsd(lp, lp) == pytest.approx(0.0, abs=1e-9)
    d = asm.compute_jsd(np.log([0.98, 0.01, 0.01]).tolist(),
                        np.log([0.01, 0.01, 0.98]).tolist())
    assert 0.0 <= d <= 1.0


def test_jri_is_bounded():
    c = ProductionRiskAndAttributionController()
    hot = {k: 1.0 for k in ("drift", "jsd", "linguistic", "collusion", "activation")}
    assert 0.0 <= c.compute_nonlinear_jri(hot) <= 1.0
    assert c.compute_nonlinear_jri({}) == 0.0
