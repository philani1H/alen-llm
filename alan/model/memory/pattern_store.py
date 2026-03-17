"""
ALAN v4 — Pattern Store (External Vector Memory)
ECONX GROUP (PTY) LTD

Two-tier memory system:

TIER 1: In-context memory (within the transformer's context window)
- Recent conversation history
- Current session patterns
- Managed by attention mechanism naturally

TIER 2: External persistent memory (vector store)
- Long-term patterns and learned knowledge
- User-specific preferences and history
- Cross-session pattern consolidation
- Retrieved via learned similarity matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A single stored knowledge pattern."""
    content_vector: torch.Tensor
    context: str = ""
    confidence: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    source: str = "interaction"
    domain: str = "general"
    connections: List[int] = field(default_factory=list)

    def reinforce(self, boost: float = 1.5):
        """Boost confidence when pattern is confirmed."""
        self.confidence = min(1.0, self.confidence * boost)
        self.access_count += 1
        self.last_accessed = time.time()

    def decay(self, factor: float = 0.995):
        """Apply time-based decay."""
        self.confidence *= factor


class PatternStore(nn.Module):
    """
    External vector memory for long-term pattern storage.
    Allows ALAN to store and retrieve relevant past patterns.

    Memory operations (all learned, not scripted):
    - STORE: After each interaction, decide what to remember
    - RETRIEVE: Before generating, query relevant past patterns
    - UPDATE: When corrections are made, modify associated patterns
    - CONSOLIDATE: At session end, merge temporary → long-term
    - FORGET: Patterns never retrieved gradually decay
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        max_patterns: int = 100000,
        retrieval_top_k: int = 10,
        decay_factor: float = 0.995,
        reinforcement_boost: float = 1.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_patterns = max_patterns
        self.retrieval_top_k = retrieval_top_k
        self.decay_factor = decay_factor
        self.reinforcement_boost = reinforcement_boost

        # Learned query projection for similarity matching
        self.query_proj = nn.Linear(hidden_dim, 512)
        self.key_proj = nn.Linear(hidden_dim, 512)

        # Storage relevance gate: should we store this?
        self.store_gate = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Memory integration: how to blend retrieved memories
        self.memory_integrator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        # Pattern storage (in-memory for now, can be backed by FAISS/vector DB)
        self.patterns: List[Pattern] = []
        self._key_cache: Optional[torch.Tensor] = None

    def store(
        self,
        vector: torch.Tensor,
        context: str = "",
        source: str = "interaction",
        domain: str = "general",
    ) -> bool:
        """
        Store a new pattern if the store gate deems it worth remembering.

        Args:
            vector: (hidden_dim,) pattern vector
            context: text description of when/why this was learned
            source: interaction, training, or inference
            domain: knowledge domain

        Returns:
            True if stored, False if filtered out
        """
        with torch.no_grad():
            relevance = self.store_gate(vector.unsqueeze(0)).item()

        if relevance < 0.3:
            return False

        pattern = Pattern(
            content_vector=vector.detach().cpu(),
            context=context,
            confidence=relevance,
            access_count=0,
            last_accessed=time.time(),
            source=source,
            domain=domain,
        )
        self.patterns.append(pattern)
        self._key_cache = None  # Invalidate cache

        # Trim if over capacity
        if len(self.patterns) > self.max_patterns:
            self._evict_least_valuable()

        return True

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        Retrieve top-k most relevant patterns for the query.

        Args:
            query: (B, hidden_dim) or (hidden_dim,) query vector
            top_k: number of patterns to retrieve

        Returns:
            retrieved: (B, k, hidden_dim) retrieved pattern vectors (or None)
            indices: list of pattern indices
        """
        if not self.patterns:
            return None, []

        k = top_k or self.retrieval_top_k

        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Project query
        q = self.query_proj(query)  # (B, 512)

        # Build key matrix from stored patterns
        if self._key_cache is None:
            keys = torch.stack([p.content_vector for p in self.patterns])
            self._key_cache = keys.to(query.device)

        keys = self._key_cache
        k_proj = self.key_proj(keys)  # (N, 512)

        # Similarity scores
        scores = torch.matmul(q, k_proj.T)  # (B, N)

        # Weight by confidence
        confidences = torch.tensor(
            [p.confidence for p in self.patterns],
            device=query.device,
        )
        scores = scores * confidences.unsqueeze(0)

        actual_k = min(k, scores.shape[1])
        top_scores, top_indices = scores.topk(actual_k, dim=-1)

        # Retrieve pattern vectors
        idx_list = top_indices[0].tolist()
        retrieved = keys[top_indices]  # (B, k, hidden_dim)

        # Update access counts
        for idx in idx_list:
            self.patterns[idx].access_count += 1
            self.patterns[idx].last_accessed = time.time()

        return retrieved, idx_list

    def update_pattern(self, index: int, new_vector: torch.Tensor, boost: bool = True):
        """Update a stored pattern (e.g., after correction)."""
        if 0 <= index < len(self.patterns):
            self.patterns[index].content_vector = new_vector.detach().cpu()
            if boost:
                self.patterns[index].reinforce(self.reinforcement_boost)
            self._key_cache = None

    def decay_all(self):
        """Apply time-based decay to all patterns."""
        for pattern in self.patterns:
            pattern.decay(self.decay_factor)

    def _evict_least_valuable(self):
        """Remove the least valuable patterns when over capacity."""
        if len(self.patterns) <= self.max_patterns:
            return

        # Score each pattern: confidence * log(access_count + 1)
        import math
        scores = [
            p.confidence * math.log(p.access_count + 1 + 1e-8)
            for p in self.patterns
        ]

        # Remove bottom 10%
        n_remove = len(self.patterns) - self.max_patterns
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        remove_set = set(sorted_indices[:n_remove])
        self.patterns = [p for i, p in enumerate(self.patterns) if i not in remove_set]
        self._key_cache = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate retrieved memories with current representation.

        Args:
            x: (B, T, hidden_dim) current hidden states

        Returns:
            x: memory-augmented hidden states
        """
        query = x.mean(dim=1)  # (B, hidden_dim)
        retrieved, _ = self.retrieve(query)

        if retrieved is None:
            return x

        # Attend over retrieved memories
        mem_summary = retrieved.mean(dim=1)  # (B, hidden_dim)
        gate_input = torch.cat([query, mem_summary], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)).unsqueeze(1)
        mem_integrated = self.memory_integrator(
            torch.cat([query, mem_summary], dim=-1)
        ).unsqueeze(1)

        x = x + gate * mem_integrated
        return self.norm(x)

    def get_stats(self) -> Dict:
        """Get memory store statistics."""
        if not self.patterns:
            return {"num_patterns": 0}

        return {
            "num_patterns": len(self.patterns),
            "avg_confidence": sum(p.confidence for p in self.patterns) / len(self.patterns),
            "avg_access_count": sum(p.access_count for p in self.patterns) / len(self.patterns),
            "domains": list(set(p.domain for p in self.patterns)),
        }
