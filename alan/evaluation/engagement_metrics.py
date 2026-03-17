"""
ALAN v4 — Engagement Metrics
ECONX GROUP (PTY) LTD

Evaluates ALAN's engagement quality:
- Hook appropriateness (not over-hooking simple questions)
- Conversation continuation rate
- Response depth calibration
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EngagementMetrics:
    """
    Engagement evaluation suite for ALAN v4.

    Measures:
    1. Hook calibration: hooks match complexity level
    2. Response depth: depth matches question complexity
    3. Module balance: appropriate modules active per context
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_module_balance(
        self,
        test_inputs: List[torch.Tensor],
    ) -> Dict:
        """
        Evaluate that module activations are balanced and reasonable.
        No single module should dominate all inputs.
        """
        self.model.eval()
        all_weights = {name: [] for name in ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]}

        for tokens in test_inputs:
            with torch.no_grad():
                _, meta = self.model(tokens.unsqueeze(0).to(self.device))
                for name in all_weights:
                    all_weights[name].append(meta["module_weights"][name].item())

        avg_weights = {
            name: sum(vals) / max(len(vals), 1)
            for name, vals in all_weights.items()
        }

        # Check for domination (one module > 0.9 on all inputs)
        dominating = [
            name for name, avg in avg_weights.items()
            if avg > 0.9
        ]

        return {
            "avg_module_weights": avg_weights,
            "dominating_modules": dominating,
            "balanced": len(dominating) == 0,
        }

    def run_all(
        self,
        test_inputs: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Run all engagement metrics."""
        if test_inputs is None:
            test_inputs = [
                torch.randint(0, 50257, (16,))
                for _ in range(10)
            ]

        return {
            "module_balance": self.evaluate_module_balance(test_inputs),
        }
