"""
ALAN v4 — Dopamine Reward System
ECONX GROUP (PTY) LTD

Neural reward system that provides internal "dopamine signals" when the model
learns something new, makes connections, or solves problems. This creates a
natural drive toward curiosity, learning, and creative insight.

Reward signals:
- learning_new_pattern: encountering genuinely novel information
- making_connections: linking disparate concepts together
- solving_problems: successfully completing a reasoning chain
- curiosity_satisfied: resolving a knowledge gap
- teaching_successfully: explaining something clearly (Feynman technique)
- creative_insight: generating novel combinations or analogies

All behavior is LEARNED from training data — the architecture provides
the capacity for reward modulation, but the actual reward patterns emerge
from training signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# Reward signal types and their base weights (learned scaling on top)
REWARD_SIGNALS = [
    "learning_new_pattern",
    "making_connections",
    "solving_problems",
    "curiosity_satisfied",
    "teaching_successfully",
    "creative_insight",
]


class NoveltyDetector(nn.Module):
    """
    Detects whether the current input pattern is genuinely novel vs repetition.
    Higher novelty -> higher dopamine reward.
    """

    def __init__(self, hidden_dim: int, memory_size: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project hidden states to a compact novelty signature
        self.signature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Novelty scorer: compare current signature against running memory
        self.novelty_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 4 * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Running memory of recent pattern signatures (not learned, just a buffer)
        self.register_buffer(
            "pattern_memory",
            torch.zeros(memory_size, hidden_dim // 4),
        )
        self.register_buffer("memory_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("memory_count", torch.tensor(0, dtype=torch.long))
        self.memory_size = memory_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute novelty score for input hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states

        Returns:
            novelty_score: (B, 1) how novel this pattern is (0=repetition, 1=novel)
            signature: (B, hidden_dim//4) compact representation of current pattern
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)
        signature = self.signature_proj(summary)  # (B, hidden_dim//4)

        # Compare against stored patterns
        if self.memory_count.item() > 0:
            count = min(self.memory_count.item(), self.memory_size)
            stored = self.pattern_memory[:count]  # (count, hidden_dim//4)
            # Find most similar stored pattern
            # (B, hidden_dim//4) x (hidden_dim//4, count) -> (B, count)
            similarities = torch.matmul(
                F.normalize(signature, dim=-1),
                F.normalize(stored, dim=-1).T,
            )
            most_similar, _ = similarities.max(dim=-1, keepdim=True)  # (B, 1)
            # Combine signature with its closest match for novelty scoring
            closest_idx = similarities.argmax(dim=-1)  # (B,)
            closest_pattern = stored[closest_idx]  # (B, hidden_dim//4)
            combined = torch.cat([signature, closest_pattern], dim=-1)
            novelty = self.novelty_scorer(combined)  # (B, 1)
        else:
            # No stored patterns yet — everything is novel
            combined = torch.cat([signature, torch.zeros_like(signature)], dim=-1)
            novelty = self.novelty_scorer(combined)

        # Update pattern memory (only during training)
        if self.training:
            with torch.no_grad():
                # Store the mean signature of this batch
                mean_sig = signature.mean(dim=0)  # (hidden_dim//4,)
                ptr = self.memory_ptr.item()
                self.pattern_memory[ptr] = mean_sig
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
                self.memory_count = torch.clamp(
                    self.memory_count + 1, max=self.memory_size
                )

        return novelty, signature


class DopamineSystem(nn.Module):
    """
    Neural reward system that modulates hidden states based on internal
    reward signals. Creates a natural drive toward learning, connection-making,
    and problem-solving.

    The system:
    1. Detects what type of cognitive activity is occurring
    2. Computes reward signals for each activity type
    3. Assesses novelty (higher reward for genuinely new patterns)
    4. Modulates hidden states via excitement gating
    5. Outputs a reinforcement multiplier that grows with successful learning

    Training signals:
    - Reward: high dopamine when model makes correct novel connections
    - Reward: increasing engagement on interesting problems
    - Penalize: low dopamine on repetitive/memorized outputs
    - Penalize: false excitement on trivial tasks
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_signals = len(REWARD_SIGNALS)

        # Reward signal detectors — one per reward type
        self.signal_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 256),
            nn.GELU(),
            nn.Linear(256, self.num_signals),
            nn.Sigmoid(),  # Each signal is 0-1 activation
        )

        # Novelty detector
        self.novelty_detector = NoveltyDetector(hidden_dim)

        # Reinforcement multiplier — scales reward based on learning history
        # Starts at 1.0, increases with successful learning (like addiction to thinking)
        self.reinforcement_scaler = nn.Sequential(
            nn.Linear(hidden_dim + self.num_signals, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # Always positive, unbounded above
        )

        # Excitement modulation — how much to boost hidden state engagement
        self.excitement_gate = nn.Sequential(
            nn.Linear(hidden_dim + self.num_signals + 1, hidden_dim),
            nn.Sigmoid(),
        )

        # Excitement transform — what the excited state looks like
        self.excitement_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combined reward output (scalar summary for training signal)
        self.reward_combiner = nn.Sequential(
            nn.Linear(self.num_signals + 1, 64),  # signals + novelty
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute dopamine reward signals and modulate hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states
            activation_weight: (B,) optional gating weight (not typically used;
                               dopamine is always active but self-regulating)

        Returns:
            x: modulated hidden states with excitement gating
            metadata: dict with reward signals, novelty, reinforcement multiplier
        """
        B, T, D = x.shape
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # 1. Detect reward signals
        signals = self.signal_detector(summary)  # (B, num_signals)

        # 2. Assess novelty
        novelty, _ = self.novelty_detector(x)  # (B, 1)

        # 3. Compute reinforcement multiplier
        reinf_input = torch.cat([summary, signals], dim=-1)
        reinforcement = self.reinforcement_scaler(reinf_input)  # (B, 1)
        # Clamp to reasonable range (1.0 to 5.0)
        reinforcement = torch.clamp(reinforcement, min=1.0, max=5.0)

        # 4. Compute combined reward (novelty-weighted)
        reward_input = torch.cat([signals * novelty, novelty], dim=-1)
        combined_reward = self.reward_combiner(reward_input)  # (B, 1)
        # Scale by reinforcement multiplier
        combined_reward = combined_reward * reinforcement

        # 5. Excitement modulation of hidden states
        excite_input = torch.cat([summary, signals, novelty], dim=-1)
        excitement = self.excitement_gate(excite_input)  # (B, hidden_dim)
        excited_state = self.excitement_transform(x)  # (B, T, hidden_dim)

        # Apply excitement gating
        excitement_expanded = excitement.unsqueeze(1)  # (B, 1, hidden_dim)

        # Optional external activation weight
        if activation_weight is not None:
            excitement_expanded = excitement_expanded * activation_weight.view(-1, 1, 1)

        x = x + excitement_expanded * (excited_state - x)
        x = self.norm(x)

        # Build per-signal metadata
        signal_dict = {
            name: signals[:, i].mean().item()
            for i, name in enumerate(REWARD_SIGNALS)
        }

        return x, {
            "reward_signals": signal_dict,
            "novelty": novelty.mean().item(),
            "reinforcement_multiplier": reinforcement.mean().item(),
            "combined_reward": combined_reward.mean().item(),
            # Tensor outputs for training loss computation
            "combined_reward_tensor": combined_reward.squeeze(-1),  # (B,)
            "signals_tensor": signals,  # (B, num_signals)
            "novelty_tensor": novelty.squeeze(-1),  # (B,)
            "reinforcement_tensor": reinforcement.squeeze(-1),  # (B,)
        }
