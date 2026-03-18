"""
ALAN v4 — Knowledge Awareness System
ECONX GROUP (PTY) LTD

Enables ALAN to know what it knows and what it doesn't know.
This is the foundation of honest, non-hallucinating AI.

Core capabilities:
- Epistemic confidence: assess own knowledge level on any topic
- Uncertainty detection: recognize when entering unknown territory
- Anti-hallucination: when confidence is low, the model naturally hedges
  (this is LEARNED, not hardcoded — no "I'm not sure" template)
- Knowledge boundary mapping: track which domains have strong/weak patterns
- Honest admission: trained to be accurate about own capabilities

All behavior is LEARNED from training data — the architecture provides
the capacity for self-assessment, but the actual calibration emerges
from training signals that reward accurate confidence and penalize
overconfident errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# Knowledge domains for boundary tracking
KNOWLEDGE_DOMAINS = [
    "factual",          # concrete facts, dates, names
    "procedural",       # how-to knowledge, step-by-step
    "conceptual",       # abstract concepts, theories
    "reasoning",        # logical inference, deduction
    "creative",         # novel combinations, analogies
    "social",           # emotional intelligence, social cues
    "technical",        # code, math, engineering
    "meta",             # self-knowledge, epistemic reasoning
]

NUM_DOMAINS = len(KNOWLEDGE_DOMAINS)


class EpistemicConfidenceEstimator(nn.Module):
    """
    Estimates how confident the model should be about its current output.
    Trained to be well-calibrated: high confidence when correct, low when wrong.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Global confidence scorer
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Per-domain confidence (which areas do we know well?)
        self.domain_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, NUM_DOMAINS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, hidden_dim) hidden states

        Returns:
            confidence: (B, 1) overall epistemic confidence
            domain_confidence: (B, NUM_DOMAINS) per-domain confidence
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)
        confidence = self.confidence_scorer(summary)  # (B, 1)
        domain_conf = self.domain_scorer(summary)  # (B, NUM_DOMAINS)
        return confidence, domain_conf


class UncertaintyDetector(nn.Module):
    """
    Detects when the model is entering uncertain territory.
    This goes beyond simple confidence — it identifies the TYPE
    of uncertainty (lack of data, ambiguity, conflicting info, etc.).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Uncertainty type classifier
        # Types: data_sparse, ambiguous, conflicting, out_of_domain, temporal
        self.uncertainty_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 5),
        )

        # Overall uncertainty magnitude
        self.uncertainty_magnitude = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, hidden_dim) hidden states

        Returns:
            uncertainty_type: (B, 5) probability over uncertainty types
            uncertainty_level: (B, 1) overall uncertainty magnitude
        """
        summary = x.mean(dim=1)  # (B, hidden_dim)
        type_logits = self.uncertainty_classifier(summary)  # (B, 5)
        type_probs = F.softmax(type_logits, dim=-1)
        level = self.uncertainty_magnitude(summary)  # (B, 1)
        return type_probs, level


class KnowledgeBoundaryMapper(nn.Module):
    """
    Maps the boundaries of the model's knowledge — where it has strong
    patterns and where it has gaps. This helps the model route its
    processing appropriately and be honest about its capabilities.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Boundary detector: is this query near a knowledge boundary?
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Domain relevance: which domains does this query touch?
        self.domain_relevance = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, NUM_DOMAINS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, hidden_dim) hidden states

        Returns:
            boundary_score: (B, 1) how close to a knowledge boundary
            domain_relevance: (B, NUM_DOMAINS) which domains are relevant
        """
        summary = x.mean(dim=1)
        boundary = self.boundary_detector(summary)
        relevance = self.domain_relevance(summary)
        return boundary, relevance


class KnowledgeAwareness(nn.Module):
    """
    Complete knowledge awareness system for ALAN v4.

    Integrates epistemic confidence, uncertainty detection, and knowledge
    boundary mapping to enable honest, well-calibrated responses.

    The key insight: when confidence is low and uncertainty is high, the
    model's hidden states are modulated to naturally produce hedged,
    honest language. This is NOT hardcoded "I'm not sure" — instead,
    the modulation makes the model's internal state ACTUALLY uncertain,
    which produces naturally uncertain language during generation.

    Training signals:
    - Reward: high confidence on correct answers
    - Reward: low confidence on incorrect answers (well-calibrated)
    - Reward: accurate domain confidence mapping
    - Reward: honest uncertainty admission
    - Penalize: overconfident wrong answers (hallucination)
    - Penalize: underconfident correct answers (excessive hedging)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Epistemic confidence estimation
        self.confidence_estimator = EpistemicConfidenceEstimator(hidden_dim)

        # Uncertainty detection
        self.uncertainty_detector = UncertaintyDetector(hidden_dim)

        # Knowledge boundary mapping
        self.boundary_mapper = KnowledgeBoundaryMapper(hidden_dim)

        # Anti-hallucination gate: modulates hidden states based on confidence
        # When confidence is low, this gate dampens the model's certainty
        # in its internal representations, leading to naturally hedged output
        self.anti_hallucination_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1 + NUM_DOMAINS + 1, hidden_dim),
            nn.Sigmoid(),
        )

        # Uncertainty-aware transform: reshapes hidden states when uncertain
        self.uncertainty_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Honest admission modulation: encourages accurate self-assessment
        self.honesty_modulator = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Final blend gate
        self.blend_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        activation_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply knowledge awareness processing to hidden states.

        Args:
            x: (B, T, hidden_dim) hidden states
            activation_weight: (B,) optional gating weight

        Returns:
            x: modulated hidden states with confidence-aware adjustments
            metadata: dict with confidence, uncertainty, boundary info
        """
        B, T, D = x.shape
        summary = x.mean(dim=1)  # (B, hidden_dim)

        # 1. Estimate epistemic confidence
        confidence, domain_confidence = self.confidence_estimator(x)  # (B,1), (B,NUM_DOMAINS)

        # 2. Detect uncertainty
        uncertainty_type, uncertainty_level = self.uncertainty_detector(x)  # (B,5), (B,1)

        # 3. Map knowledge boundaries
        boundary_score, domain_relevance = self.boundary_mapper(x)  # (B,1), (B,NUM_DOMAINS)

        # 4. Anti-hallucination gate
        # Combine confidence, uncertainty, and domain info to gate hidden states
        gate_input = torch.cat([
            summary,               # (B, hidden_dim)
            confidence,            # (B, 1)
            domain_confidence,     # (B, NUM_DOMAINS)
            uncertainty_level,     # (B, 1)
        ], dim=-1)  # (B, hidden_dim + 1 + NUM_DOMAINS + 1)

        anti_hall_gate = self.anti_hallucination_gate(gate_input)  # (B, hidden_dim)
        anti_hall_gate = anti_hall_gate.unsqueeze(1)  # (B, 1, hidden_dim)

        # 5. Uncertainty-aware transform
        uncertain_repr = self.uncertainty_transform(x)  # (B, T, hidden_dim)

        # 6. Honesty modulation
        honesty_input = torch.cat([
            summary,
            confidence,
        ], dim=-1)  # (B, hidden_dim + 1)
        honesty_mod = self.honesty_modulator(honesty_input)  # (B, hidden_dim)
        honesty_mod = honesty_mod.unsqueeze(1)  # (B, 1, hidden_dim)

        # 7. Combine: blend between confident and uncertain representations
        # High confidence -> keep original states
        # Low confidence -> shift toward uncertainty-aware representation
        blend = self.blend_gate(summary).unsqueeze(1)  # (B, 1, 1)

        if activation_weight is not None:
            blend = blend * activation_weight.view(-1, 1, 1)

        # Apply anti-hallucination gate to the uncertain path
        modulated = anti_hall_gate * uncertain_repr + honesty_mod

        x = x + blend * (modulated - x)
        x = self.norm(x)

        # Uncertainty type labels
        uncertainty_labels = [
            "data_sparse", "ambiguous", "conflicting",
            "out_of_domain", "temporal",
        ]
        uncertainty_idx = uncertainty_type.argmax(dim=-1)  # (B,)

        # Domain confidence dict
        domain_conf_dict = {
            name: domain_confidence[0, i].item()
            for i, name in enumerate(KNOWLEDGE_DOMAINS)
        }

        return x, {
            "confidence": confidence.mean().item(),
            "confidence_tensor": confidence.squeeze(-1),  # (B,)
            "domain_confidence": domain_conf_dict,
            "domain_confidence_tensor": domain_confidence,  # (B, NUM_DOMAINS)
            "uncertainty_level": uncertainty_level.mean().item(),
            "uncertainty_level_tensor": uncertainty_level.squeeze(-1),  # (B,)
            "uncertainty_type": uncertainty_labels[uncertainty_idx[0].item()],
            "uncertainty_type_probs": uncertainty_type[0].detach().tolist(),
            "boundary_score": boundary_score.mean().item(),
            "boundary_score_tensor": boundary_score.squeeze(-1),  # (B,)
            "domain_relevance": {
                name: domain_relevance[0, i].item()
                for i, name in enumerate(KNOWLEDGE_DOMAINS)
            },
        }
