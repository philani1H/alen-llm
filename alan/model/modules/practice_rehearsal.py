"""
ALAN v4 — Practice & Rehearsal System
ECONX GROUP (PTY) LTD

Implements the learning reinforcement loop from model.md Section 9.
Ensures the model UNDERSTANDS knowledge, not just memorizes it.

Rehearsal stages (each increases retention multiplier):
- RESTATE  (1.3x): reformulate learned knowledge to confirm understanding
- APPLY    (1.6x): try using knowledge in new context
- CONNECT  (2.0x): link to existing knowledge (dot-connection)
- QUESTION (2.2x): generate test questions about what was learned
- TEACH    (2.5x): explain it back (Feynman technique)

Base level:
- HEARD_ONCE (1.0x): information received but not rehearsed

The system detects which rehearsal stage the model is operating at,
and modulates hidden states to encourage deeper processing at higher stages.

All behavior is LEARNED from training data — the architecture provides
the capacity, but actual rehearsal patterns emerge from training signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# Rehearsal stages with their reinforcement multipliers
REHEARSAL_STAGES = {
    "heard_once": 0,
    "restated": 1,
    "applied": 2,
    "connected": 3,
    "questioned": 4,
    "taught_back": 5,
}

REINFORCEMENT_MULTIPLIERS = {
    "heard_once": 1.0,
    "restated": 1.3,
    "applied": 1.6,
    "connected": 2.0,
    "questioned": 2.2,
    "taught_back": 2.5,
}

NUM_STAGES = len(REHEARSAL_STAGES)


class PracticeRehearsal(nn.Module):
    """
    Practice & Rehearsal system — detects and encourages deeper knowledge
    processing through progressive rehearsal stages.

    The module:
    1. Detects which rehearsal stage the model is currently operating at
    2. Computes a reinforcement multiplier based on the stage
    3. Applies stage-specific transformations to encourage deeper processing
    4. Tracks verification quality (is the restatement/application/teaching accurate?)

    Training signals:
    - Reward: accurate restatement of learned knowledge
    - Reward: successful application to new contexts
    - Reward: meaningful connections to existing knowledge
    - Reward: generating insightful test questions
    - Reward: clear teaching explanations (Feynman technique)
    - Penalize: shallow repetition disguised as restatement
    - Penalize: incorrect application of knowledge
    - Penalize: superficial connections
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Stage classifier: detect which rehearsal stage is active
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, NUM_STAGES),
        )

        # Per-stage processing networks
        # RESTATE: reformulation transform
        self.restate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # APPLY: context-transfer transform
        self.apply_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CONNECT: cross-domain linking transform
        self.connect_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # QUESTION: question generation transform
        self.question_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # TEACH: teaching/explanation transform
        self.teach_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Verification scorer: how well did the rehearsal go?
        self.verification_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Reinforcement multiplier predictor (learned, not hardcoded)
        self.multiplier_predictor = nn.Sequential(
            nn.Linear(NUM_STAGES + hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        # Output blend gate
        self.blend_gate = nn.Sequential(
            nn.Linear(hidden_dim + NUM_STAGES, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply practice & rehearsal processing to hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states
            activation_weight: (B,) optional gating weight

        Returns:
            x: processed hidden states with rehearsal modulation
            metadata: dict with stage classification, multiplier, verification
        """
        B, T, D = x.shape
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # 1. Classify rehearsal stage
        stage_logits = self.stage_classifier(summary)  # (B, NUM_STAGES)
        stage_probs = F.softmax(stage_logits, dim=-1)  # (B, NUM_STAGES)
        stage_idx = stage_probs.argmax(dim=-1)  # (B,)

        # 2. Apply stage-specific transforms (weighted by stage probabilities)
        # Each stage transform operates on the full sequence
        stage_outputs = torch.zeros_like(x)  # (B, T, D)

        # heard_once: identity (no transform)
        # restated:
        restated = self.restate_net(x)
        stage_outputs = stage_outputs + stage_probs[:, 1].view(B, 1, 1) * restated

        # applied:
        applied = self.apply_net(x)
        stage_outputs = stage_outputs + stage_probs[:, 2].view(B, 1, 1) * applied

        # connected:
        connected = self.connect_net(x)
        stage_outputs = stage_outputs + stage_probs[:, 3].view(B, 1, 1) * connected

        # questioned:
        questioned = self.question_net(x)
        stage_outputs = stage_outputs + stage_probs[:, 4].view(B, 1, 1) * questioned

        # taught_back:
        taught = self.teach_net(x)
        stage_outputs = stage_outputs + stage_probs[:, 5].view(B, 1, 1) * taught

        # heard_once contributes the original (identity)
        stage_outputs = stage_outputs + stage_probs[:, 0].view(B, 1, 1) * x

        # 3. Verification: compare original and rehearsed representations
        combined = torch.cat([summary, stage_outputs.mean(dim=1)], dim=-1)
        verification = self.verification_scorer(combined)  # (B, 1)

        # 4. Compute reinforcement multiplier
        mult_input = torch.cat([stage_probs, summary], dim=-1)
        multiplier = self.multiplier_predictor(mult_input)  # (B, 1)
        # Clamp to the valid range of multipliers (1.0 to 2.5)
        multiplier = torch.clamp(multiplier, min=1.0, max=2.5)

        # 5. Blend gate: how much rehearsal processing to apply
        blend_input = torch.cat([summary, stage_probs], dim=-1)
        blend = self.blend_gate(blend_input).unsqueeze(1)  # (B, 1, D)

        if activation_weight is not None:
            blend = blend * activation_weight.view(-1, 1, 1)

        x = x + blend * (stage_outputs - x)
        x = self.norm(x)

        # Map stage index to name
        stage_names = list(REHEARSAL_STAGES.keys())

        return x, {
            "stage": stage_names[stage_idx[0].item()],
            "stage_probs": {
                name: stage_probs[0, i].item()
                for i, name in enumerate(stage_names)
            },
            "verification_score": verification.mean().item(),
            "reinforcement_multiplier": multiplier.mean().item(),
            # Tensor outputs for training loss computation
            "stage_logits_tensor": stage_logits,  # (B, NUM_STAGES)
            "stage_probs_tensor": stage_probs,  # (B, NUM_STAGES)
            "verification_tensor": verification.squeeze(-1),  # (B,)
            "multiplier_tensor": multiplier.squeeze(-1),  # (B,)
        }
