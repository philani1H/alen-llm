"""
ALAN v4 — Dynamic Temperature Controller
ECONX GROUP (PTY) LTD

Temperature is dynamically controlled per-module:
- Reasoning Engine: low temperature (0.1-0.3) for precision
- Creativity Engine: high temperature (0.7-1.0) for novelty
- The ROUTING LAYER decides which temperature profile to use
  based on the classified task type.

All temperature selection is LEARNED, not hardcoded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Temperature profiles (targets for learning, not hardcoded rules)
TEMPERATURE_PROFILES = {
    "reasoning_mode": {"temperature": 0.2, "top_p": 0.9},
    "balanced_mode": {"temperature": 0.5, "top_p": 0.95},
    "creative_mode": {"temperature": 0.8, "top_p": 0.98},
    "exploration_mode": {"temperature": 1.0, "top_p": 1.0},
}


class DynamicTemperatureController(nn.Module):
    """
    Learns to select the right temperature profile based on task type
    and module activation weights.

    The controller outputs a continuous temperature value and top_p value
    that modulate generation behavior.
    """

    def __init__(self, hidden_dim: int = 2048, num_modules: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Task-to-temperature mapping (learned)
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim + num_modules, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # [temperature, top_p]
        )

        # Module-specific temperature biases (learned)
        self.module_temp_bias = nn.Parameter(torch.tensor([
            0.2,   # reasoning: low
            0.8,   # creativity: high
            0.4,   # curiosity: medium
            0.4,   # emotional: medium
            0.3,   # memory: low-medium
            0.3,   # meta: low-medium
        ]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        module_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute dynamic temperature and top_p based on current context.

        Args:
            hidden_states: (B, T, hidden_dim) current hidden states
            module_weights: dict of module activation weights

        Returns:
            dict with 'temperature' and 'top_p' values
        """
        summary = hidden_states.mean(dim=1)  # (B, hidden_dim)

        # Stack module weights
        module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]
        weights = torch.stack([
            module_weights.get(name, torch.zeros(summary.shape[0], device=summary.device))
            for name in module_names
        ], dim=-1)  # (B, 6)

        # Compute weighted temperature from module biases
        weighted_temp = (weights * self.module_temp_bias.to(weights.device)).sum(dim=-1)
        weight_sum = weights.sum(dim=-1).clamp(min=1e-8)
        base_temp = weighted_temp / weight_sum  # (B,)

        # Neural adjustment based on hidden states
        combined = torch.cat([summary, weights], dim=-1)
        adjustments = self.temperature_head(combined)  # (B, 2)

        # Temperature: sigmoid to keep in [0.05, 1.5] range
        temperature = 0.05 + 1.45 * torch.sigmoid(adjustments[:, 0] + base_temp)

        # Top-p: sigmoid to keep in [0.8, 1.0] range
        top_p = 0.8 + 0.2 * torch.sigmoid(adjustments[:, 1])

        return {
            "temperature": temperature.mean().item(),
            "top_p": top_p.mean().item(),
            "base_temp": base_temp.mean().item(),
        }
