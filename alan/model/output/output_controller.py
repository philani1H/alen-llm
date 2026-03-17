"""
ALAN v4 — Output Controller
ECONX GROUP (PTY) LTD

Manages HOW ALAN generates its response based on task analysis.
All output strategies are LEARNED behaviors, not rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Output strategy definitions (training targets, not runtime rules)
OUTPUT_STRATEGIES = {
    "concise_answer": {
        "when": "Simple factual question, clear context",
        "temperature": 0.2,
        "max_length": "short",
        "hooks": "none",
    },
    "step_by_step": {
        "when": "Complex problem, math, code, logic",
        "temperature": 0.2,
        "max_length": "medium-long",
        "hooks": "light",
    },
    "exploratory": {
        "when": "Open-ended question, creative request, brainstorming",
        "temperature": 0.7,
        "max_length": "medium",
        "hooks": "challenge_or_insight",
    },
    "teaching": {
        "when": "User is learning, needs explanation at their level",
        "temperature": 0.4,
        "max_length": "medium",
        "hooks": "expansion_or_challenge",
    },
    "empathetic": {
        "when": "User is emotional, frustrated, or sharing personal situation",
        "temperature": 0.4,
        "max_length": "medium",
        "hooks": "personalization",
    },
    "clarifying": {
        "when": "Request is ambiguous, missing critical info",
        "temperature": 0.3,
        "max_length": "short",
        "hooks": "none",
    },
    "rehearsal": {
        "when": "Just learned something new from user",
        "temperature": 0.3,
        "max_length": "medium",
        "hooks": "verification_question",
    },
}

STRATEGY_NAMES = list(OUTPUT_STRATEGIES.keys())


class OutputController(nn.Module):
    """
    Selects the appropriate output strategy based on the current context,
    module activations, and emotional state.

    All strategy selection is LEARNED from training data.
    """

    def __init__(self, hidden_dim: int = 2048, num_modules: int = 6):
        super().__init__()
        self.num_strategies = len(STRATEGY_NAMES)

        # Strategy classifier
        self.strategy_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_modules, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, self.num_strategies),
        )

        # Response depth predictor (how detailed should the response be)
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0=very short, 1=very detailed
        )

        # Engagement level predictor (should we add hooks?)
        self.engagement_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0=no hooks, 1=full engagement
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        module_weights: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Select output strategy and parameters.

        Args:
            hidden_states: (B, T, hidden_dim) current hidden states
            module_weights: dict of module activation weights

        Returns:
            dict with strategy, depth, engagement level, and parameters
        """
        summary = hidden_states.mean(dim=1)  # (B, hidden_dim)

        # Stack module weights
        module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]
        weights = torch.stack([
            module_weights.get(name, torch.zeros(summary.shape[0], device=summary.device))
            for name in module_names
        ], dim=-1)  # (B, 6)

        # Classify strategy
        combined = torch.cat([summary, weights], dim=-1)
        strategy_logits = self.strategy_classifier(combined)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        strategy_idx = strategy_probs.argmax(dim=-1)

        # Response depth
        depth = self.depth_predictor(summary)

        # Engagement level
        engagement = self.engagement_predictor(summary)

        strategy_name = STRATEGY_NAMES[strategy_idx[0].item()]
        strategy_config = OUTPUT_STRATEGIES[strategy_name]

        return {
            "strategy": strategy_name,
            "strategy_probs": {
                name: strategy_probs[0, i].item()
                for i, name in enumerate(STRATEGY_NAMES)
            },
            "depth": depth.mean().item(),
            "engagement_level": engagement.mean().item(),
            "recommended_temperature": strategy_config["temperature"],
            "recommended_hooks": strategy_config["hooks"],
        }
