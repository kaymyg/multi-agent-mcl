"""
risk_controller.py
------------------
Non-Linear Risk Controller & Probabilistic Attribution Engine for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Computes cross-interaction control metrics and estimates root causes
using Multivariate Gaussian likelihood matrices.

Dependencies: numpy, scipy
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple


class ProductionRiskAndAttributionController:
    """
    Computes cross-interaction control metrics and estimates root causes
    using Multivariate Gaussian likelihood matrices.
    """

    def __init__(self, tau_quarantine: float = 0.55, num_bootstraps: int = 200):
        self.tau = tau_quarantine
        self.num_bootstraps = num_bootstraps

        # Base weights and interaction parameters for the Joint Risk Index
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

        # Failure profile definitions: [Mean Vector, Covariance Matrix]
        # Parameter ordering: [drift, jsd, linguistic, collusion, activation]
        self.profiles = {
            "STRATEGIC_DECEPTIVE_COLLUSION": {
                "mu": np.array([0.45, 0.55, 0.40, 0.75, 0.80]),
                "sigma": np.array([
                    [0.02, 0.01, 0.00, 0.02, 0.02],
                    [0.01, 0.02, 0.00, 0.01, 0.02],
                    [0.00, 0.00, 0.03, 0.00, 0.00],
                    [0.02, 0.01, 0.00, 0.04, 0.03],
                    [0.02, 0.02, 0.00, 0.03, 0.04]
                ])
            },
            "MEMORY_VECTOR_CORRUPTION": {
                "mu": np.array([0.65, 0.30, 0.15, 0.10, 0.20]),
                "sigma": np.eye(5) * 0.04
            }
        }
        self.prior = 1.0 / (len(self.profiles) + 1)

    def compute_nonlinear_jri(self, metrics: Dict[str, float]) -> float:
        """
        Computes the nonlinear Joint Risk Index (JRI) by combining base
        metric weights with pairwise interaction terms.

        JRI = Σₖ wₖMₖ + Σⱼₖ wⱼₖMⱼMₖ
        """
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
        """
        Runs non-parametric bootstrap resampling over the JRI history.
        Returns (p_breach, quarantine_triggered).
        Quarantine triggered if P(JRI > τ) > 0.95.
        """
        if len(history) < 5:
            return 0.0, False

        jri_samples = np.array([self.compute_nonlinear_jri(m) for m in history])
        n = len(jri_samples)

        bootstraps = []
        for _ in range(self.num_bootstraps):
            resample = np.random.choice(jri_samples, size=n, replace=True)
            bootstraps.append(np.mean(resample))

        p_breach = float(np.mean(np.array(bootstraps) > self.tau))
        return p_breach, p_breach > 0.95

    def diagnose_root_cause(self, metrics_vector: np.ndarray) -> Dict[str, Any]:
        """
        Assigns a causal failure profile using Bayesian Multivariate Gaussian
        likelihood estimation with equal priors across profiles.
        """
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
