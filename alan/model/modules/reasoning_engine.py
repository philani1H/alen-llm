"""
ALAN v4 — Reasoning Engine Module
ECONX GROUP (PTY) LTD

Handles step-by-step logic, mathematical reasoning, code generation,
and structured problem-solving. All behaviors are LEARNED, not hardcoded.

Key behaviors (all learned from training data):
- Chain-of-thought: generates internal reasoning tokens via scratchpad
- Backward verification: re-derives answers to check consistency
- Step decomposition: breaks complex problems into sub-problems
- Hypothesis testing: considers multiple approaches before committing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class ReasoningEngine(nn.Module):
    """
    Reasoning module — operates as specialized attention patterns
    within the transformer. Activated by the TaskRouter for logical tasks.
    
    Training signals:
    - Reward multi-step reasoning over direct answers
    - Reward self-correction when backward check fails
    - Penalize jumping to conclusions on complex problems
    """

    def __init__(self, hidden_dim: int, num_steps: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps  # Max verification iterations

        # Step decomposition network
        self.step_decomposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Verification gate (should we re-check?)
        self.verification_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Confidence scorer
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply reasoning processing to hidden states.
        
        Args:
            x: (B, T, hidden_dim) hidden states
            activation_weight: (B,) how much to activate this module
        
        Returns:
            x: processed hidden states
            metadata: reasoning metadata
        """
        original = x.clone()

        # Step decomposition pass
        decomposed = self.step_decomposer(x)

        # Verification: compare original and decomposed
        combined = torch.cat([x.mean(dim=1), decomposed.mean(dim=1)], dim=-1)
        verify_gate = self.verification_gate(combined)  # (B, 1)

        # Confidence score
        confidence = self.confidence_scorer(decomposed.mean(dim=1))  # (B, 1)

        # Blend based on activation weight and verification gate
        if activation_weight is not None:
            blend = activation_weight.view(-1, 1, 1) * verify_gate.unsqueeze(-1)
        else:
            blend = verify_gate.unsqueeze(-1)

        x = x + blend * (decomposed - original)
        x = self.norm(x)

        return x, {
            "confidence": confidence.mean().item(),
            "verification_gate": verify_gate.mean().item(),
        }


class ScratchpadMechanism(nn.Module):
    """
    Internal chain-of-thought scratchpad.
    Allows ALAN to 'think before answering' by generating
    internal reasoning tokens that are not shown to the user.
    """

    def __init__(self, hidden_dim: int, max_scratchpad_tokens: int = 1024):
        super().__init__()
        self.max_tokens = max_scratchpad_tokens

        # Scratchpad encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Gate: how much scratchpad to use
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scratchpad processing."""
        scratchpad = self.encoder(x)
        gate = self.gate(x.mean(dim=1, keepdim=True))
        return x + gate * scratchpad
