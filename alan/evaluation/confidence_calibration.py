"""
ALAN v4 — Confidence Calibration Evaluation
ECONX GROUP (PTY) LTD

Evaluates ALAN's confidence calibration:
- Says "I'm confident" → actually correct 90%+ of the time
- Says "I'm not sure" → actually correct ~50% of the time
- Says "I'd need to verify" → actually correct <30% of the time
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalibration:
    """
    Evaluates confidence calibration of ALAN v4.

    A well-calibrated model:
    - High confidence predictions are usually correct
    - Low confidence predictions are often incorrect
    - Confidence scores correlate with actual accuracy
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_calibration(
        self,
        test_inputs: List[torch.Tensor],
        test_labels: List[torch.Tensor],
    ) -> Dict:
        """
        Evaluate confidence calibration using test data.

        Bins predictions by confidence and checks accuracy per bin.
        """
        self.model.eval()
        predictions = []

        for tokens, labels in zip(test_inputs, test_labels):
            with torch.no_grad():
                tokens = tokens.unsqueeze(0).to(self.device)
                logits, meta = self.model(tokens)

                # Get prediction confidence
                probs = torch.softmax(logits, dim=-1)
                max_prob = probs.max(dim=-1).values.mean().item()

                # Check if prediction matches label
                pred_tokens = logits.argmax(dim=-1)
                labels_dev = labels.unsqueeze(0).to(self.device)
                min_len = min(pred_tokens.shape[1], labels_dev.shape[1])
                accuracy = (pred_tokens[:, :min_len] == labels_dev[:, :min_len]).float().mean().item()

                predictions.append({
                    "confidence": max_prob,
                    "accuracy": accuracy,
                    "meta_activation": meta["module_weights"]["meta"].item(),
                })

        # Bin by confidence
        bins = {"high": [], "medium": [], "low": []}
        for p in predictions:
            if p["confidence"] > 0.7:
                bins["high"].append(p["accuracy"])
            elif p["confidence"] > 0.3:
                bins["medium"].append(p["accuracy"])
            else:
                bins["low"].append(p["accuracy"])

        return {
            "high_confidence_accuracy": sum(bins["high"]) / max(len(bins["high"]), 1),
            "medium_confidence_accuracy": sum(bins["medium"]) / max(len(bins["medium"]), 1),
            "low_confidence_accuracy": sum(bins["low"]) / max(len(bins["low"]), 1),
            "num_predictions": len(predictions),
            "bins": {k: len(v) for k, v in bins.items()},
        }

    def run_all(
        self,
        test_inputs: Optional[List[torch.Tensor]] = None,
        test_labels: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Run all calibration evaluations."""
        if test_inputs is None:
            test_inputs = [torch.randint(0, 50257, (16,)) for _ in range(10)]
        if test_labels is None:
            test_labels = [torch.randint(0, 50257, (16,)) for _ in range(10)]

        return {
            "calibration": self.evaluate_calibration(test_inputs, test_labels),
        }
