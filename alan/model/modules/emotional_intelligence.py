"""
ALAN v4 — Emotional Intelligence Module
ECONX GROUP (PTY) LTD

Handles tone detection, empathy calibration, and engagement modulation.
All behaviors are LEARNED from training data, not scripted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


EMOTIONAL_TONES = ["neutral", "frustrated", "excited", "confused", "sad", "happy", "anxious"]


class EmotionalIntelligence(nn.Module):
    """
    Detects user emotional tone and calibrates response accordingly.
    Learns empathy, encouragement, and directness from training examples.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Tone classifier (learned from training data)
        self.tone_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(EMOTIONAL_TONES)),
        )

        # Empathy modulation: adjusts hidden states based on detected tone
        self.empathy_modulator = nn.Sequential(
            nn.Linear(hidden_dim + len(EMOTIONAL_TONES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Engagement dimensions
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # [empathy, directness, encouragement]
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # Detect tone
        tone_logits = self.tone_classifier(summary)
        tone_probs = F.softmax(tone_logits, dim=-1)
        detected_tone = tone_probs.argmax(dim=-1)

        # Modulate hidden states with emotional context
        tone_context = tone_probs.unsqueeze(1).expand(-1, x.shape[1], -1)
        x_with_tone = torch.cat([x, tone_context], dim=-1)
        modulated = self.empathy_modulator(x_with_tone)

        # Engagement dimensions
        engagement = self.engagement_head(summary)

        # Apply based on activation weight
        if activation_weight is not None:
            blend = activation_weight.view(-1, 1, 1)
        else:
            blend = torch.ones(x.shape[0], 1, 1, device=x.device)

        x = x + blend * (modulated - x)
        x = self.norm(x)

        return x, {
            "detected_tone": EMOTIONAL_TONES[detected_tone[0].item()],
            "tone_confidence": tone_probs.max().item(),
            "empathy": engagement[0, 0].item(),
            "directness": engagement[0, 1].item(),
            "encouragement": engagement[0, 2].item(),
        }


class MetaReasoning(nn.Module):
    """
    Self-check module: verifies logical consistency, detects contradictions,
    and scores confidence. This is ALAN checking ITSELF, not an external filter.
    """

    def __init__(self, hidden_dim: int, max_iterations: int = 3):
        super().__init__()
        self.max_iterations = max_iterations

        # Contradiction detector
        self.contradiction_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Completeness checker
        self.completeness_checker = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Confidence scorer
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Refinement network (used when check fails)
        self.refinement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        input_repr: Optional[torch.Tensor] = None,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run meta-reasoning check. If contradiction detected, refine.
        Max iterations: 3 (as per spec).
        """
        summary = x.mean(dim=1)

        contradiction_score = 0.0
        completeness_score = 1.0
        confidence = 0.5

        for iteration in range(self.max_iterations):
            # Check for contradictions with input
            if input_repr is not None:
                input_summary = input_repr.mean(dim=1)
                combined = torch.cat([summary, input_summary], dim=-1)
                contradiction_score = self.contradiction_detector(combined).mean().item()
            else:
                contradiction_score = 0.0

            completeness_score = self.completeness_checker(summary).mean().item()
            confidence = self.confidence_scorer(summary).mean().item()

            # If contradiction detected or incomplete, refine
            if contradiction_score > 0.5 or completeness_score < 0.5:
                x = x + self.refinement(x)
                x = self.norm(x)
                summary = x.mean(dim=1)
            else:
                break  # Check passed

        return x, {
            "contradiction_score": contradiction_score,
            "completeness_score": completeness_score,
            "confidence": confidence,
            "iterations": iteration + 1,
        }
