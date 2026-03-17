"""
ALAN v4 — Feedback Integration Module
ECONX GROUP (PTY) LTD

Processes corrections, updates patterns, and adjusts behavior in real-time.

When a user corrects ALAN or provides new information:
1. DETECT: Identify what was wrong and what the correction is
2. DIFF: Compute the delta between old understanding and correction
3. UPDATE: Modify relevant memory patterns immediately
4. PROPAGATE: Check if this correction affects other stored knowledge
5. REINFORCE: Increase confidence in the corrected pattern
6. ANTI-REINFORCE: Decrease confidence in the incorrect pattern

This is NOT retraining weights in real-time.
This updates external memory and adjusts retrieval priority.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class FeedbackIntegration(nn.Module):
    """
    Feedback integration module — processes corrections and new
    information to update ALAN's understanding in real-time.

    Training signals:
    - Reward natural acknowledgment of corrections
    - Reward correct restatement of corrected knowledge
    - Reward connecting corrections to related knowledge
    - Penalize defensive or dismissive responses to feedback
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Correction detector: is the user correcting something?
        self.correction_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 3),  # [no_correction, gentle_correction, explicit_correction]
        )

        # Delta computer: what changed between old and new understanding
        self.delta_computer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Integration gate: how much to update based on correction
        self.integration_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.Sigmoid(),
        )

        # Rehearsal trigger: should we practice the corrected knowledge?
        self.rehearsal_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Confidence adjuster: update confidence after correction
        self.confidence_adjuster = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply feedback integration to hidden states.

        Args:
            x: (B, T, hidden_dim) current hidden states
            previous_output: (B, T, hidden_dim) previous response states (if available)
            activation_weight: (B,) how much to activate this module

        Returns:
            x: processed hidden states
            metadata: feedback integration metadata
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # If we have previous output, detect corrections
        if previous_output is not None:
            prev_summary = previous_output.mean(dim=1)
            combined = torch.cat([summary, prev_summary], dim=-1)
        else:
            combined = torch.cat([summary, summary], dim=-1)

        # Detect if correction is happening
        correction_logits = self.correction_detector(combined)  # (B, 3)
        correction_probs = F.softmax(correction_logits, dim=-1)
        correction_type = correction_probs.argmax(dim=-1)

        # Compute the delta (what changed)
        delta = self.delta_computer(combined)  # (B, hidden_dim)

        # Integration gate
        gate_input = torch.cat([summary, correction_probs], dim=-1)
        gate = self.integration_gate(gate_input).unsqueeze(1)  # (B, 1, hidden_dim)

        # Should we rehearse?
        rehearsal_score = self.rehearsal_scorer(delta)  # (B, 1)

        # Adjusted confidence
        new_confidence = self.confidence_adjuster(delta)  # (B, 1)

        # Apply integration
        if activation_weight is not None:
            gate = gate * activation_weight.view(-1, 1, 1)

        x = x + gate * delta.unsqueeze(1)
        x = self.norm(x)

        correction_labels = ["no_correction", "gentle_correction", "explicit_correction"]

        return x, {
            "correction_type": correction_labels[correction_type[0].item()],
            "correction_probs": correction_probs[0].detach().tolist(),
            "rehearsal_score": rehearsal_score.mean().item(),
            "new_confidence": new_confidence.mean().item(),
        }
