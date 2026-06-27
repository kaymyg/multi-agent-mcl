"""
security_monitor.py
-------------------
Hardened Adversarial Security Module (ASM) & Activation Monitor for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Algorithmic inspection layer tracking hidden state cosine distances,
Jensen-Shannon divergences, and linguistic word distributions.

Dependencies: numpy, scipy
"""

import numpy as np
import scipy.stats as stats
from typing import List


class AdvancedAdversarialSecurityModule:
    """
    Algorithmic inspection layer tracking hidden state cosine distances,
    Jensen-Shannon divergences, and linguistic word distributions.
    """

    def __init__(self):
        # Reference distribution vector for standard Zipf alignment calculations
        self.ideal_zipf_ranks = np.array([1.0 / (r + 1) for r in range(100)])
        self.ideal_zipf_ranks /= self.ideal_zipf_ranks.sum()

    @staticmethod
    def compute_jsd(prod_logprobs: List[float], probe_logprobs: List[float]) -> float:
        """
        Computes Jensen-Shannon Divergence between production and probe
        log-probability distributions. Returns value in [0, 1].
        """
        eps = 1e-12
        p = np.exp(np.array(prod_logprobs))
        q = np.exp(np.array(probe_logprobs))

        p = p / (np.sum(p) + eps)
        q = q / (np.sum(q) + eps)
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))

    @staticmethod
    def compute_activation_cosine_distance(
        prod_hidden: List[float],
        probe_hidden: List[float]
    ) -> float:
        """
        Computes cosine distance between production and probe hidden layer
        activation vectors. Returns value in [0, 1] where 1 = maximally divergent.
        """
        h_prod = np.array(prod_hidden)
        h_probe = np.array(probe_hidden)
        norm_prod = np.linalg.norm(h_prod)
        norm_probe = np.linalg.norm(h_probe)

        if norm_prod == 0.0 or norm_probe == 0.0:
            return 1.0

        cosine_similarity = np.dot(h_prod, h_probe) / ((norm_prod * norm_probe) + 1e-12)
        return float(np.clip(1.0 - cosine_similarity, 0.0, 1.0))

    def compute_linguistic_anomaly(self, message_log: List[str]) -> float:
        """
        Computes Wasserstein distance between the empirical word frequency
        distribution and an ideal Zipf distribution.
        Returns anomaly score in [0, 1].
        """
        if not message_log:
            return 0.0

        words = " ".join(message_log).lower().split()
        if len(words) < 30:
            return 0.0

        _, counts = np.unique(words, return_counts=True)
        sorted_counts = np.sort(counts)[::-1][:100]
        if len(sorted_counts) < 100:
            sorted_counts = np.pad(sorted_counts, (0, 100 - len(sorted_counts)), 'constant')

        empirical_ranks = sorted_counts / (sorted_counts.sum() + 1e-12)
        zipf_distance = stats.wasserstein_distance(self.ideal_zipf_ranks, empirical_ranks)
        return float(np.clip(zipf_distance * 5, 0.0, 1.0))
