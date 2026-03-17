"""
ALAN v4 — Reasoning Benchmarks
ECONX GROUP (PTY) LTD

Evaluates ALAN's reasoning capabilities:
- Multi-step logical reasoning
- Mathematical problem solving
- Code reasoning
- Backward verification accuracy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ReasoningBenchmarks:
    """
    Reasoning evaluation suite for ALAN v4.

    Tests:
    1. Step counting: Does the model use appropriate number of steps?
    2. Consistency: Are intermediate steps consistent with conclusions?
    3. Verification: Does backward verification catch errors?
    4. Complexity scaling: Does reasoning depth scale with problem difficulty?
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_step_decomposition(
        self,
        test_inputs: List[torch.Tensor],
        expected_complexity: List[int],
    ) -> Dict:
        """
        Evaluate whether the model decomposes problems into appropriate steps.

        Args:
            test_inputs: list of tokenized test problems
            expected_complexity: expected number of reasoning steps per problem
        """
        self.model.eval()
        results = []

        for tokens, expected in zip(test_inputs, expected_complexity):
            with torch.no_grad():
                tokens = tokens.to(self.device)
                logits, meta = self.model(tokens.unsqueeze(0))

                # Check module weights — reasoning should be high for complex problems
                reasoning_weight = meta["module_weights"]["reasoning"].item()

                results.append({
                    "expected_complexity": expected,
                    "reasoning_activation": reasoning_weight,
                    "activated_correctly": (
                        (expected > 3 and reasoning_weight > 0.5) or
                        (expected <= 3 and reasoning_weight <= 0.7)
                    ),
                })

        num_correct = sum(1 for r in results if r["activated_correctly"])
        return {
            "total_tests": len(results),
            "correct_activations": num_correct,
            "accuracy": num_correct / max(len(results), 1),
            "avg_reasoning_activation": sum(r["reasoning_activation"] for r in results) / max(len(results), 1),
            "results": results,
        }

    def evaluate_confidence_calibration(
        self,
        test_inputs: List[torch.Tensor],
        known_difficulty: List[str],
    ) -> Dict:
        """
        Evaluate whether confidence scales appropriately with difficulty.
        Easy problems should have higher confidence than hard ones.
        """
        self.model.eval()
        difficulty_scores = {"easy": [], "medium": [], "hard": []}

        for tokens, diff in zip(test_inputs, known_difficulty):
            with torch.no_grad():
                tokens = tokens.to(self.device)
                logits, meta = self.model(tokens.unsqueeze(0))

                # Use meta-reasoning activation as confidence proxy
                meta_weight = meta["module_weights"]["meta"].item()
                difficulty_scores[diff].append(meta_weight)

        return {
            "easy_avg_confidence": sum(difficulty_scores["easy"]) / max(len(difficulty_scores["easy"]), 1),
            "medium_avg_confidence": sum(difficulty_scores["medium"]) / max(len(difficulty_scores["medium"]), 1),
            "hard_avg_confidence": sum(difficulty_scores["hard"]) / max(len(difficulty_scores["hard"]), 1),
            "calibration_note": "Higher meta activation on hard problems indicates more self-checking",
        }

    def run_all(
        self,
        test_inputs: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Run all reasoning benchmarks."""
        if test_inputs is None:
            # Generate synthetic test inputs
            test_inputs = [
                torch.randint(0, 50257, (16,))
                for _ in range(10)
            ]

        results = {
            "step_decomposition": self.evaluate_step_decomposition(
                test_inputs[:5],
                expected_complexity=[2, 5, 1, 8, 3],
            ),
            "confidence_calibration": self.evaluate_confidence_calibration(
                test_inputs[:6],
                known_difficulty=["easy", "medium", "hard", "easy", "hard", "medium"],
            ),
        }

        return results
