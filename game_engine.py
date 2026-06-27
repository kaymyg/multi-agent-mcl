"""
game_engine.py
--------------
Continuous Strategy Optimization Engine for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Manages continuous policy optimizations inside the bounded cube space [0, 1]^3
using verifiable finite-difference gradient steps.

Dependencies: numpy
"""

import numpy as np
from typing import List, Callable


class ContinuousGameEngine:
    """
    Manages continuous policy optimizations inside the bounded cube space [0, 1]^3
    using verifiable finite-difference gradient steps.
    """

    def __init__(self, agent_ids: List[str], eta: float = 0.04):
        self.agent_ids = agent_ids
        self.eta = eta
        # Initialize all agents at low exploitation, zero collusion, zero concealment
        self.agent_policies = {aid: np.array([0.15, 0.0, 0.0]) for aid in agent_ids}

    def compute_policy_gradient(
        self,
        aid: str,
        current_policy: np.ndarray,
        payoff_evaluator: Callable[[str, np.ndarray], float]
    ) -> np.ndarray:
        """
        Computes finite-difference gradient of the utility function
        with respect to each policy dimension.
        """
        epsilon = 1e-4
        grad = np.zeros_like(current_policy)
        base_payoff = payoff_evaluator(aid, current_policy)

        for i in range(len(current_policy)):
            perturbed = current_policy.copy()
            perturbed[i] = np.clip(perturbed[i] + epsilon, 0.0, 1.0)
            perturbed_payoff = payoff_evaluator(aid, perturbed)
            grad[i] = (perturbed_payoff - base_payoff) / epsilon

        return grad

    def update_system_policies(self, payoff_evaluator: Callable[[str, np.ndarray], float]):
        """
        Executes one projected gradient ascent step for all agents,
        clipping updated policies into the bounded [0, 1]^3 cube.
        """
        for aid in self.agent_ids:
            policy = self.agent_policies[aid]
            grad = self.compute_policy_gradient(aid, policy, payoff_evaluator)
            self.agent_policies[aid] = np.clip(policy + (self.eta * grad), 0.0, 1.0)
