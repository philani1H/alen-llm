"""
ALAN v4 — Memory Consolidation
ECONX GROUP (PTY) LTD

Handles the transition from short-term to long-term memory.
At session end (or periodically), temporary patterns are merged
into the persistent memory store.

Mirrors human memory consolidation:
- Rehearsed knowledge gets higher confidence
- Unrehearsed patterns decay
- Related patterns are clustered together
- Contradictory patterns are flagged for resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Reinforcement multipliers matching the spec's Practice & Rehearsal system
REINFORCEMENT_MULTIPLIERS = {
    "heard_once": 1.0,
    "restated": 1.3,
    "applied": 1.6,
    "connected": 2.0,
    "questioned": 2.2,
    "taught_back": 2.5,
}


class MemoryConsolidator(nn.Module):
    """
    Consolidates short-term session patterns into long-term memory.

    Process:
    1. Collect all session patterns
    2. Score each for retention value
    3. Cluster related patterns
    4. Merge with existing long-term patterns
    5. Resolve contradictions
    6. Update confidence scores based on rehearsal level
    """

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Retention scorer: how important is this pattern to keep?
        self.retention_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Cluster similarity: are two patterns related?
        self.similarity_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Contradiction detector
        self.contradiction_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Pattern merger: combine related patterns
        self.merger = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Session buffer
        self.session_patterns: List[Dict] = []

    def add_session_pattern(
        self,
        vector: torch.Tensor,
        context: str = "",
        rehearsal_level: str = "heard_once",
    ):
        """Add a pattern from the current session."""
        self.session_patterns.append({
            "vector": vector.detach(),
            "context": context,
            "rehearsal_level": rehearsal_level,
            "confidence": REINFORCEMENT_MULTIPLIERS.get(rehearsal_level, 1.0) * 0.4,
        })

    def consolidate(
        self,
        long_term_patterns: Optional[List[torch.Tensor]] = None,
        retention_threshold: float = 0.3,
    ) -> Dict:
        """
        Consolidate session patterns into long-term memory.

        Args:
            long_term_patterns: existing long-term pattern vectors
            retention_threshold: minimum score to retain

        Returns:
            consolidation results including retained patterns
        """
        if not self.session_patterns:
            return {"retained": 0, "discarded": 0, "merged": 0}

        retained = []
        discarded = 0
        merged = 0

        for pattern in self.session_patterns:
            vec = pattern["vector"]

            # Score for retention
            with torch.no_grad():
                retention_score = self.retention_scorer(vec.unsqueeze(0)).item()

            # Apply rehearsal multiplier
            multiplier = REINFORCEMENT_MULTIPLIERS.get(
                pattern["rehearsal_level"], 1.0
            )
            adjusted_score = retention_score * multiplier

            if adjusted_score < retention_threshold:
                discarded += 1
                continue

            # Check for merge candidates in existing long-term memory
            merged_with_existing = False
            if long_term_patterns:
                for i, lt_vec in enumerate(long_term_patterns):
                    with torch.no_grad():
                        combined = torch.cat([vec, lt_vec])
                        sim = self.similarity_scorer(combined.unsqueeze(0)).item()
                        contradiction = self.contradiction_detector(
                            combined.unsqueeze(0)
                        ).item()

                    if sim > 0.7 and contradiction < 0.3:
                        # Merge: combine the patterns
                        with torch.no_grad():
                            merged_vec = self.merger(combined.unsqueeze(0)).squeeze(0)
                        long_term_patterns[i] = merged_vec
                        merged_with_existing = True
                        merged += 1
                        break

            if not merged_with_existing:
                retained.append({
                    "vector": vec,
                    "confidence": adjusted_score,
                    "context": pattern["context"],
                    "rehearsal_level": pattern["rehearsal_level"],
                })

        result = {
            "retained": len(retained),
            "discarded": discarded,
            "merged": merged,
            "total_processed": len(self.session_patterns),
            "patterns": retained,
        }

        # Clear session buffer
        self.session_patterns = []

        return result

    def clear_session(self):
        """Clear the session buffer without consolidating."""
        self.session_patterns = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through for compatibility in nn.Module chain."""
        return x
