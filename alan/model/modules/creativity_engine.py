"""
ALAN v4 — Creativity Engine Module
ECONX GROUP (PTY) LTD

Cross-domain attention heads that connect patterns across different knowledge areas.
Generates novel connections, metaphors, analogies, and lateral thinking solutions.

Key mechanisms (all LEARNED from training data):
- Cross-domain retrieval: pull patterns from unrelated domains
- Analogical mapping: "X is to Y as A is to B" reasoning
- Divergent generation: produce multiple candidate ideas before selecting
- Metaphor construction: map abstract concepts to concrete imagery
- Constraint relaxation: temporarily ignore assumptions to find novel paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class CreativityEngine(nn.Module):
    """
    Creativity module — activates cross-domain attention patterns within
    the transformer. Triggered by the TaskRouter for creative/open-ended tasks.

    Training signals:
    - Reward novel but valid cross-domain connections
    - Reward multiple divergent approaches before selection
    - Penalize superficial or trivial connections
    """

    def __init__(self, hidden_dim: int, num_candidates: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates

        # Cross-domain attention: learns to attend across knowledge domains
        self.cross_domain_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Divergent candidate generator: produces multiple reasoning paths
        self.candidate_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_candidates)
        ])

        # Candidate selector: picks the best divergent path
        self.candidate_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Novelty gate: how much creative divergence to apply
        self.novelty_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply creativity processing to hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states
            activation_weight: (B,) how much to activate this module

        Returns:
            x: processed hidden states
            metadata: creativity metadata
        """
        B, T, D = x.shape

        # Cross-domain projection
        cross_domain = self.cross_domain_projector(x)

        # Generate multiple candidate paths (divergent thinking)
        candidates = []
        for gen in self.candidate_generators:
            candidate = gen(cross_domain)
            candidates.append(candidate)

        # Score each candidate
        candidate_scores = []
        for cand in candidates:
            score = self.candidate_scorer(cand.mean(dim=1))  # (B, 1)
            candidate_scores.append(score)

        scores = torch.cat(candidate_scores, dim=-1)  # (B, num_candidates)
        weights = F.softmax(scores, dim=-1)  # (B, num_candidates)

        # Weighted combination of candidates
        stacked = torch.stack(candidates, dim=1)  # (B, num_candidates, T, D)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B, num_candidates, 1, 1)
        creative_output = (stacked * weights_expanded).sum(dim=1)  # (B, T, D)

        # Novelty gate: blend creative output with original
        gate_input = torch.cat([x.mean(dim=1), creative_output.mean(dim=1)], dim=-1)
        gate = self.novelty_gate(gate_input).unsqueeze(1)  # (B, 1, D)

        if activation_weight is not None:
            gate = gate * activation_weight.view(-1, 1, 1)

        x = x + gate * (creative_output - x)
        x = self.norm(x)

        best_candidate = weights.argmax(dim=-1)[0].item()

        return x, {
            "num_candidates": self.num_candidates,
            "best_candidate": best_candidate,
            "candidate_scores": scores[0].detach().tolist(),
            "novelty_gate_mean": gate.mean().item(),
        }
