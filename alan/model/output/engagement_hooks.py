"""
ALAN v4 — Engagement Hook System
ECONX GROUP (PTY) LTD

ALAN doesn't just answer — it makes users WANT to keep talking.
Like a great teacher or conversation partner, it hooks attention
and builds momentum.

CRITICAL: Hooks are NATURAL, not manipulative. They emerge from
genuine engagement with the topic, not from psychological tricks.

Hook calibration:
- Simple questions: NO hook (just answer)
- Medium complexity: LIGHT hook (brief connection or follow-up)
- Deep/complex topics: NATURAL hook (insight, challenge, or expansion)
- Ongoing dialogue: PERSONALIZED hook (reference past discussion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


HOOK_TYPES = [
    "none",                 # No hook needed (simple answer)
    "insight_hook",         # End with a surprising connection or insight
    "challenge_hook",       # Pose a thought-provoking follow-up question
    "pattern_reveal",       # Point out a pattern the user might not have noticed
    "expansion_hook",       # Suggest related areas worth exploring
    "personalization_hook", # Reference user's past interests or approaches
    "incomplete_revelation",# Share part of an interesting insight naturally
]


class EngagementHookSystem(nn.Module):
    """
    Learns to produce natural engagement hooks based on context.
    All hook behavior is trained, not scripted.

    Training signals:
    - Reward: natural hooks that deepen conversation productively
    - Reward: hooks arising from genuine insight about the topic
    - Penalize: forced/generic hooks ("Isn't that interesting?")
    - Penalize: over-hooking on simple questions
    - Penalize: hooks that feel manipulative
    """

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.num_hook_types = len(HOOK_TYPES)

        # Hook type classifier
        self.hook_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_hook_types),
        )

        # Hook intensity scorer (how strong should the hook be?)
        self.intensity_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0=no hook, 1=strong hook
        )

        # Complexity detector (determines if hook is appropriate)
        self.complexity_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # [simple, medium, complex]
        )

        # Hook embedding generator (conditions output generation)
        self.hook_embedder = nn.Sequential(
            nn.Linear(hidden_dim + self.num_hook_types, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        engagement_level: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply engagement hook conditioning to hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states
            engagement_level: 0-1 from output controller

        Returns:
            x: hook-conditioned hidden states
            metadata: hook selection metadata
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # Classify complexity
        complexity_logits = self.complexity_detector(summary)
        complexity_probs = F.softmax(complexity_logits, dim=-1)
        complexity_level = complexity_probs.argmax(dim=-1)

        # Determine hook type
        hook_logits = self.hook_classifier(summary)
        hook_probs = F.softmax(hook_logits, dim=-1)
        hook_type_idx = hook_probs.argmax(dim=-1)

        # Hook intensity
        intensity = self.intensity_scorer(summary)

        # Suppress hooks for simple questions
        complexity_labels = ["simple", "medium", "complex"]
        if complexity_level[0].item() == 0:  # simple
            intensity = intensity * 0.1  # Almost no hook
        elif complexity_level[0].item() == 1:  # medium
            intensity = intensity * 0.5

        # Scale by engagement level
        intensity = intensity * engagement_level

        # Generate hook embedding
        hook_input = torch.cat([summary, hook_probs], dim=-1)
        hook_emb = self.hook_embedder(hook_input)  # (B, hidden_dim)

        # Blend into hidden states
        x = x + intensity.unsqueeze(-1) * hook_emb.unsqueeze(1)
        x = self.norm(x)

        return x, {
            "hook_type": HOOK_TYPES[hook_type_idx[0].item()],
            "hook_probs": {
                HOOK_TYPES[i]: hook_probs[0, i].item()
                for i in range(self.num_hook_types)
            },
            "intensity": intensity.mean().item(),
            "complexity": complexity_labels[complexity_level[0].item()],
        }
