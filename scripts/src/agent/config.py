from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SuitConfig:
    # Pass 1: which items are allowed to seed suits
    seed_psignal_min: float = 0.35
    seed_max_items: int = 6000

    # Clustering threshold (cosine on TF-IDF). Higher => fewer, purer suits.
    sim_threshold: float = 0.27

    # Pass 2: expansion threshold; lower => more recall.
    expand_sim_threshold: float = 0.18

    # How many suits + evidence to show
    max_suits: int = 8
    top_queries_per_suit: int = 10
    top_domains_per_suit: int = 8

    # Session expansion
    top_sessions_per_suit: int = 10
    session_expand_items: int = 20

    # Precision gates to prevent session leakage
    session_gate_sim: float = 0.24  # stricter than expand_sim_threshold
    domain_max_session_frac: float = 0.18  # drop domains that appear in too many sessions