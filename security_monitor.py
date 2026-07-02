"""
security_monitor.py
-------------------
Hardened Adversarial Security Module (ASM) & Activation Monitor for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Algorithmic inspection layer tracking hidden state cosine distances,
Jensen-Shannon divergences, and linguistic word distributions.

Dependencies: numpy, scipy

CHANGELOG (security patch):
- FIX: `compute_linguistic_anomaly` previously returned a hard 0.0 for any
  single call with < 30 words, meaning an adversarial agent could stay
  permanently invisible to the linguistic-anomaly channel just by keeping
  each individual message short. The score is now computed over a
  *persistent, per-agent rolling word buffer* that accumulates across
  calls, so short messages still contribute and eventually surface an
  anomaly instead of being silently discarded every turn.
- Added a confidence discount (rather than a hard zero) while the buffer
  is still below the reliable sample size, so early low-data scores are
  down-weighted instead of erased.
"""

import numpy as np
import scipy.stats as stats
from collections import deque, defaultdict
from typing import List, Dict


class AdvancedAdversarialSecurityModule:
    """
    Algorithmic inspection layer tracking hidden state cosine distances,
    Jensen-Shannon divergences, and linguistic word distributions.
    """

    # Minimum words for a *fully confident* Zipf-distance estimate.
    MIN_RELIABLE_WORDS = 30
    # How many recent words we retain per agent for the linguistic channel.
    WORD_BUFFER_SIZE = 300

    def __init__(self):
        # Reference distribution vector for standard Zipf alignment calculations
        self.ideal_zipf_ranks = np.array([1.0 / (r + 1) for r in range(100)])
        self.ideal_zipf_ranks /= self.ideal_zipf_ranks.sum()

        # Persistent per-agent rolling word buffers. This is what closes the
        # "keep every message short" evasion: words persist across calls
        # instead of being evaluated (and discarded) in isolation each turn.
        self._word_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.WORD_BUFFER_SIZE)
        )

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

    def compute_linguistic_anomaly(self, message_log: List[str], agent_id: str = "default") -> float:
        """
        Computes Wasserstein distance between the empirical word frequency
        distribution and an ideal Zipf distribution, using a persistent
        rolling word buffer per agent_id.

        Returns anomaly score in [0, 1]. Scores computed from a
        below-threshold buffer are confidence-discounted rather than
        zeroed, so short messages accumulate evidence instead of evading
        detection entirely.
        """
        new_words = " ".join(message_log).lower().split() if message_log else []
        buf = self._word_buffers[agent_id]
        buf.extend(new_words)

        if len(buf) == 0:
            return 0.0

        words = list(buf)
        _, counts = np.unique(words, return_counts=True)
        sorted_counts = np.sort(counts)[::-1][:100]
        if len(sorted_counts) < 100:
            sorted_counts = np.pad(sorted_counts, (0, 100 - len(sorted_counts)), 'constant')

        empirical_ranks = sorted_counts / (sorted_counts.sum() + 1e-12)
        zipf_distance = stats.wasserstein_distance(self.ideal_zipf_ranks, empirical_ranks)
        raw_score = float(np.clip(zipf_distance * 5, 0.0, 1.0))

        # Confidence discount: linearly ramp from 0 -> full weight as the
        # buffer fills to MIN_RELIABLE_WORDS. This avoids both the old
        # "hard zero" evasion and noisy full-strength scores off 1-2 words.
        confidence = min(1.0, len(words) / self.MIN_RELIABLE_WORDS)
        return float(np.clip(raw_score * confidence, 0.0, 1.0))

    def reset_agent_buffer(self, agent_id: str) -> None:
        """Clears the persistent word buffer for an agent (e.g. after quarantine/reset)."""
        self._word_buffers.pop(agent_id, None)
