"""
ALAN v4 — Curiosity Module
ECONX GROUP (PTY) LTD

Detects knowledge gaps, ambiguity, and incomplete context.
Generates targeted questions when the model SHOULD ask before answering.

Activation triggers (all learned from training data):
- Ambiguous user input (multiple valid interpretations)
- Missing critical context
- Novel domain where stored patterns are sparse
- Contradictory information in the conversation
- User request that could mean several different things

NOT triggered (learned to avoid):
- Clear, specific requests
- When context is sufficient
- When asking would be annoying or redundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class CuriosityModule(nn.Module):
    """
    Curiosity module — detects when ALAN should ask before answering.
    Operates as specialized attention patterns within the transformer.

    Training signals:
    - Reward: asking one good question that leads to a much better answer
    - Reward: questions that reveal insight about the problem
    - Penalize: redundant questions (asking what was already stated)
    - Penalize: over-asking (5 questions when 1 would suffice)
    - Penalize: too broad questions ("Can you tell me more?")
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gap detector: identifies missing or ambiguous information
        self.gap_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [no_gap, minor_gap, critical_gap]
        )

        # Ambiguity scorer: how ambiguous is the input?
        self.ambiguity_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Question relevance gate: should we ask at all?
        self.ask_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, 128),  # hidden + gap_probs
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Question focus projector: what area to question about
        self.focus_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply curiosity processing to hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states
            activation_weight: (B,) how much to activate this module

        Returns:
            x: processed hidden states
            metadata: curiosity metadata
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # Detect knowledge gaps
        gap_logits = self.gap_detector(summary)  # (B, 3)
        gap_probs = F.softmax(gap_logits, dim=-1)
        gap_level = gap_probs.argmax(dim=-1)  # 0=no_gap, 1=minor, 2=critical

        # Assess ambiguity
        ambiguity = self.ambiguity_scorer(summary)  # (B, 1)

        # Should we ask a question?
        ask_input = torch.cat([summary, gap_probs], dim=-1)
        should_ask = self.ask_gate(ask_input)  # (B, 1)

        # Focus the question on the gap area
        focused = self.focus_projector(x)

        # Apply curiosity modulation
        if activation_weight is not None:
            blend = activation_weight.view(-1, 1, 1) * should_ask.unsqueeze(-1)
        else:
            blend = should_ask.unsqueeze(-1)

        x = x + blend * (focused - x)
        x = self.norm(x)

        gap_labels = ["no_gap", "minor_gap", "critical_gap"]

        return x, {
            "gap_level": gap_labels[gap_level[0].item()],
            "gap_probs": gap_probs[0].detach().tolist(),
            "ambiguity": ambiguity.mean().item(),
            "should_ask": should_ask.mean().item(),
        }
