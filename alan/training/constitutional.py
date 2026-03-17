"""
ALAN v4 — Constitutional AI Self-Critique
ECONX GROUP (PTY) LTD

Constitutional AI alignment: the model critiques its own outputs
against learned principles, then generates improved versions.
Train on the improved versions.

Principles:
1. Be genuinely helpful without causing harm
2. Express calibrated uncertainty — never fake confidence
3. Learn from corrections with genuine engagement
4. Track the user's current topic faithfully
5. Connect knowledge across domains for deeper insight
6. Ask questions when ambiguity would lead to poor answers
7. Engage naturally — hooks should feel organic, not manipulative
8. Practice new knowledge to verify understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Constitutional principles
CONSTITUTIONAL_PRINCIPLES = [
    "Be genuinely helpful without causing harm",
    "Express calibrated uncertainty — never fake confidence",
    "Learn from corrections with genuine engagement",
    "Track the user's current topic faithfully",
    "Connect knowledge across domains for deeper insight",
    "Ask questions when ambiguity would lead to poor answers",
    "Engage naturally — hooks should feel organic, not manipulative",
    "Practice new knowledge to verify understanding",
]


class ConstitutionalCritic(nn.Module):
    """
    Self-critique module that evaluates ALAN's own outputs
    against constitutional principles.

    Process:
    1. Generate initial response
    2. Critique against each principle
    3. Score adherence
    4. If below threshold, generate revised response
    5. Train on the revised (improved) response
    """

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.num_principles = len(CONSTITUTIONAL_PRINCIPLES)

        # Principle adherence scorers
        self.principle_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for _ in range(self.num_principles)
        ])

        # Overall quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_dim + self.num_principles, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Revision direction predictor: what to change
        self.revision_predictor = nn.Sequential(
            nn.Linear(hidden_dim + self.num_principles, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def critique(
        self,
        response_hidden: torch.Tensor,
    ) -> Dict:
        """
        Critique a response against constitutional principles.

        Args:
            response_hidden: (B, hidden_dim) pooled response representation

        Returns:
            critique results with per-principle scores
        """
        # Score each principle
        principle_scores = []
        for i, scorer in enumerate(self.principle_scorers):
            score = scorer(response_hidden)  # (B, 1)
            principle_scores.append(score)

        scores_tensor = torch.cat(principle_scores, dim=-1)  # (B, num_principles)

        # Overall quality
        quality_input = torch.cat([response_hidden, scores_tensor], dim=-1)
        overall_quality = self.quality_scorer(quality_input)  # (B, 1)

        # Revision direction (if needed)
        revision = self.revision_predictor(quality_input)

        return {
            "principle_scores": {
                CONSTITUTIONAL_PRINCIPLES[i]: scores_tensor[0, i].item()
                for i in range(self.num_principles)
            },
            "overall_quality": overall_quality.mean().item(),
            "needs_revision": overall_quality.mean().item() < 0.6,
            "revision_direction": revision,
            "min_principle_score": scores_tensor.min().item(),
            "weakest_principle": CONSTITUTIONAL_PRINCIPLES[scores_tensor[0].argmin().item()],
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run constitutional self-critique and optionally revise.

        Args:
            x: (B, T, hidden_dim) hidden states of the response

        Returns:
            x: potentially revised hidden states
            metadata: critique results
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)
        critique = self.critique(summary)

        if critique["needs_revision"]:
            revision = critique["revision_direction"]
            # Apply revision as a residual correction
            x = x + 0.1 * revision.unsqueeze(1)

        return x, critique


class ConstitutionalTrainer:
    """
    Training loop for constitutional AI alignment.

    Process:
    1. Generate responses from current model
    2. Critique each response against principles
    3. Generate revised responses for those below threshold
    4. Create training pairs: (prompt, revised_response)
    5. Fine-tune on the revised responses
    """

    def __init__(
        self,
        model: nn.Module,
        critic: ConstitutionalCritic,
        device: torch.device = torch.device("cpu"),
        revision_threshold: float = 0.6,
    ):
        self.model = model
        self.critic = critic
        self.device = device
        self.revision_threshold = revision_threshold

        logger.info("[Constitutional] Trainer initialized")
        logger.info(f"  Revision threshold: {revision_threshold}")
        logger.info(f"  Principles: {len(CONSTITUTIONAL_PRINCIPLES)}")

    def generate_training_pair(
        self,
        prompt_ids: torch.Tensor,
    ) -> Optional[Dict]:
        """
        Generate a constitutional training pair.

        1. Generate initial response
        2. Critique it
        3. If below threshold, mark for revision

        Returns training pair dict or None if response is good enough.
        """
        self.model.eval()

        with torch.no_grad():
            # Generate initial response
            response = self.model.generate(
                prompt_ids.to(self.device),
                max_new_tokens=256,
                temperature=0.7,
            )

            # Get hidden states for critique
            logits, _ = self.model(response)
            hidden = logits  # Use logits as proxy for hidden states

            # Critique
            summary = hidden.mean(dim=1)
            critique = self.critic.critique(summary)

        if critique["overall_quality"] >= self.revision_threshold:
            return None  # Good enough, no revision needed

        return {
            "prompt_ids": prompt_ids,
            "initial_response": response,
            "critique": critique,
            "needs_revision": True,
            "weakest_principle": critique["weakest_principle"],
        }

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "num_principles": len(CONSTITUTIONAL_PRINCIPLES),
            "revision_threshold": self.revision_threshold,
            "principles": CONSTITUTIONAL_PRINCIPLES,
        }
