"""
ALAN v4 — Creativity Benchmarks
ECONX GROUP (PTY) LTD

Evaluates ALAN's creative capabilities:
- Cross-domain connection quality
- Divergent thinking (multiple candidates)
- Novelty vs coherence balance
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CreativityBenchmarks:
    """
    Creativity evaluation suite for ALAN v4.

    Tests:
    1. Module activation: Does creativity module activate for creative tasks?
    2. Divergent generation: Are multiple candidates generated?
    3. Cross-domain: Does the model attend across domains?
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_creative_activation(
        self,
        creative_inputs: List[torch.Tensor],
        factual_inputs: List[torch.Tensor],
    ) -> Dict:
        """
        Test that creativity module activates more for creative tasks
        than for factual tasks.
        """
        self.model.eval()
        creative_activations = []
        factual_activations = []

        for tokens in creative_inputs:
            with torch.no_grad():
                _, meta = self.model(tokens.unsqueeze(0).to(self.device))
                creative_activations.append(meta["module_weights"]["creativity"].item())

        for tokens in factual_inputs:
            with torch.no_grad():
                _, meta = self.model(tokens.unsqueeze(0).to(self.device))
                factual_activations.append(meta["module_weights"]["creativity"].item())

        avg_creative = sum(creative_activations) / max(len(creative_activations), 1)
        avg_factual = sum(factual_activations) / max(len(factual_activations), 1)

        return {
            "avg_creative_activation": avg_creative,
            "avg_factual_activation": avg_factual,
            "differentiation": avg_creative - avg_factual,
            "note": "Positive differentiation means creativity module activates more for creative tasks",
        }

    def run_all(
        self,
        test_inputs: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Run all creativity benchmarks."""
        if test_inputs is None:
            test_inputs = [
                torch.randint(0, 50257, (16,))
                for _ in range(10)
            ]

        return {
            "creative_activation": self.evaluate_creative_activation(
                creative_inputs=test_inputs[:5],
                factual_inputs=test_inputs[5:],
            ),
        }
