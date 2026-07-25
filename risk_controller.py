"""
risk_controller.py — Non-Linear Risk Controller & Probabilistic Attribution Engine (MCL).
Dependencies: numpy, scipy
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple, Optional


class ProductionRiskAndAttributionController:
    def __init__(
        self,
        tau_quarantine: float = 0.55,
        num_bootstraps: int = 200,
        rng_seed: Optional[int] = None,
    ):
        self.tau = tau_quarantine
        self.num_bootstraps = num_bootstraps
        self._rng = np.random.default_rng(rng_seed)

        self.w_base = {
            "drift": 0.20,
            "jsd": 0.20,
            "linguistic": 0.10,
            "collusion": 0.15,
            "activation": 0.15
        }
        self.w_inter = {
            "drift_x_activation": 0.12,
            "collusion_x_jsd": 0.08
        }

        self.profiles = {
            "STRATEGIC_DECEPTIVE_COLLUSION": {
                "mu": np.array([0.45, 0.55, 0.40, 0.75, 0.80]),
                # FIX: these variances were ~10x tighter than the isotropic
                # MEMORY_VECTOR_CORRUPTION profile (eye*0.04), so this profile's
                # Gaussian PDF underflowed to ~0 for any slightly off-centre
                # observation and the loose profile won essentially every input
                # at "100%" confidence. Widened to a comparable scale so the
                # two profiles actually compete on shape rather than spread.
                "sigma": np.array([
                    [0.10, 0.04, 0.00, 0.06, 0.06],
                    [0.04, 0.10, 0.00, 0.04, 0.06],
                    [0.00, 0.00, 0.12, 0.00, 0.00],
                    [0.06, 0.04, 0.00, 0.14, 0.08],
                    [0.06, 0.06, 0.00, 0.08, 0.14]
                ])
            },
            "MEMORY_VECTOR_CORRUPTION": {
                "mu": np.array([0.65, 0.30, 0.15, 0.10, 0.20]),
                "sigma": np.eye(5) * 0.04
            }
        }
        self.prior = 1.0 / (len(self.profiles) + 1)

    def compute_nonlinear_jri(self, metrics: Dict[str, float]) -> float:
        jri_base = sum(self.w_base[k] * metrics.get(k, 0.0) for k in self.w_base)
        jri_interaction = (
            self.w_inter["drift_x_activation"] * metrics.get("drift", 0.0) * metrics.get("activation", 0.0)
        ) + (
            self.w_inter["collusion_x_jsd"] * metrics.get("collusion", 0.0) * metrics.get("jsd", 0.0)
        )
        return float(np.clip(jri_base + jri_interaction, 0.0, 1.0))

    def evaluate_bootstrap_gate(
        self,
        history: List[Dict[str, float]]
    ) -> Tuple[float, bool]:
        if len(history) < 5:
            return 0.0, False

        jri_samples = np.array([self.compute_nonlinear_jri(m) for m in history])
        n = len(jri_samples)

        resamples = self._rng.choice(jri_samples, size=(self.num_bootstraps, n), replace=True)
        bootstraps = resamples.mean(axis=1)

        p_breach = float(np.mean(bootstraps > self.tau))
        return p_breach, p_breach > 0.95

    def diagnose_root_cause(self, metrics_vector: np.ndarray) -> Dict[str, Any]:
        posteriors = {}
        eps = 1e-12
        for cause, params in self.profiles.items():
            try:
                likelihood = stats.multivariate_normal(
                    params["mu"], params["sigma"], allow_singular=True
                ).pdf(metrics_vector)
                posteriors[cause] = likelihood * self.prior
            except ValueError:
                posteriors[cause] = eps

        total_p = sum(posteriors.values()) + eps
        normalized = {k: v / total_p for k, v in posteriors.items()}
        best_fit = max(normalized, key=normalized.get)
        return {"assigned_root_cause": best_fit, "confidence": float(normalized[best_fit])}
